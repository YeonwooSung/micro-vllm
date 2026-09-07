#include "coli_cuda.hpp"

#include "../model/family.hpp"
#include "../quant/native_act.hpp"
#include "../quant/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace mvllm {
namespace coli_cuda {
namespace {

int ceil_div(int a, int b) { return b > 0 ? (a + b - 1) / b : 0; }

struct Backend {
    bool inited = false;
    std::vector<int> devices;
    size_t tensor_count = 0;
    size_t tensor_bytes = 0;
    uint64_t g_calls = 0, g_experts = 0, g_rows = 0;
    bool have_e8 = false;
    bool have_fp8 = false;
    float fp8_lut[256]{};
    uint8_t e8_grid[256 * 4]{};
    std::map<int, std::vector<float>> issue_y;
    std::map<std::pair<int, int>, std::vector<char>> scratch;
    std::map<int, std::vector<void *>> allocs;
};

Backend g;

} // namespace

struct Tensor {
    int fmt = 0;
    int I = 0;
    int O = 0;
    int gs = 0;
    int device = 0;
    std::vector<float> f32;
    std::vector<int8_t> i8;
    std::vector<uint8_t> packed;
    std::vector<float> scale;
};

namespace {

void account_add(const Tensor *t) {
    if (!t)
        return;
    ++g.tensor_count;
    g.tensor_bytes += tensor_bytes(t);
}

void account_sub(const Tensor *t) {
    if (!t || g.tensor_count == 0)
        return;
    --g.tensor_count;
    const size_t b = tensor_bytes(t);
    g.tensor_bytes = g.tensor_bytes > b ? g.tensor_bytes - b : 0;
}

bool adopt(Tensor **slot, Tensor *t) {
    if (!slot || !t)
        return false;
    if (*slot)
        tensor_free(*slot);
    *slot = t;
    account_add(t);
    return true;
}

float dequant(const Tensor *t, int o, int i) {
    const int I = t->I;
    switch (t->fmt) {
    case 0:
        return t->f32[static_cast<size_t>(o) * I + i];
    case 1:
        return static_cast<float>(t->i8[static_cast<size_t>(o) * I + i]) * t->scale[static_cast<size_t>(o)];
    case 2: {
        const uint8_t byte = t->packed[static_cast<size_t>(o) * ceil_div(I, 2) + i / 2];
        const int nib = (i & 1) ? (byte >> 4) : (byte & 15);
        return static_cast<float>(nib - 8) * t->scale[static_cast<size_t>(o)];
    }
    case 3: {
        const uint8_t byte = t->packed[static_cast<size_t>(o) * ceil_div(I, 4) + i / 4];
        const int code = (byte >> (2 * (i & 3))) & 3;
        return static_cast<float>(code - 2) * t->scale[static_cast<size_t>(o)];
    }
    case 4: {
        const int gs = t->gs > 0 ? t->gs : 64;
        const uint8_t byte = t->packed[static_cast<size_t>(o) * ceil_div(I, 2) + i / 2];
        const int nib = (i & 1) ? (byte >> 4) : (byte & 15);
        const int sg = ceil_div(I, gs);
        return static_cast<float>(nib - 8) * t->scale[static_cast<size_t>(o) * sg + i / gs];
    }
    case 8: {
        const int si = std::max(ceil_div(I, 128), 1);
        const float sc = t->scale[static_cast<size_t>(o / 128) * si + i / 128];
        const uint8_t b = t->packed[static_cast<size_t>(o) * I + i];
        const float q = g.have_fp8 ? g.fp8_lut[b] : e4m3fn_decode(b);
        return q * sc;
    }
    default:
        return 0.f;
    }
}

bool matvec_one(const Tensor *t, float *y, const float *x) {
    if (!t || !y || !x)
        return false;
    if (t->fmt == 0) {
        quant::matmul_f32(y, x, t->f32.data(), 1, t->I, t->O);
        return true;
    }
    if (t->fmt == 1) {
        quant::matmul_int8_row(y, x, t->i8.data(), t->scale.data(), 1, t->I, t->O);
        return true;
    }
    if (t->fmt == 4 && t->gs == 64) {
        quant::matmul_int4_g64(y, x, t->packed.data(), t->scale.data(), 1, t->I, t->O);
        return true;
    }
    for (int o = 0; o < t->O; ++o) {
        float s = 0.f;
        for (int i = 0; i < t->I; ++i)
            s += x[i] * dequant(t, o, i);
        y[o] = s;
    }
    return true;
}

void apply_rope_pairs(float *v, int pos, int pairs, float theta) {
    if (!v || pairs <= 0 || theta <= 0.f)
        return;
    const int n = pairs * 2;
    for (int i = 0; i < pairs; ++i) {
        const float freq = std::pow(theta, -2.f * static_cast<float>(i) / static_cast<float>(n));
        const float ang = static_cast<float>(pos) * freq;
        const float c = std::cos(ang), s = std::sin(ang);
        const float a = v[2 * i], b = v[2 * i + 1];
        v[2 * i] = a * c - b * s;
        v[2 * i + 1] = a * s + b * c;
    }
}

} // namespace

bool init(const int *devices, int count) {
    if (g.inited)
        return true;
    g.devices.clear();
    if (devices && count > 0)
        g.devices.assign(devices, devices + count);
    else
        g.devices.push_back(0);
    g.inited = true;
    return true;
}

void shutdown() {
    for (auto &kv : g.allocs)
        for (void *p : kv.second)
            std::free(p);
    g = Backend{};
}

bool available() { return g.inited; }

int available_device_count() { return 0; }

int device_count() { return g.inited ? static_cast<int>(g.devices.size()) : 0; }

int device_at(int index) {
    if (!g.inited || index < 0 || index >= static_cast<int>(g.devices.size()))
        return -1;
    return g.devices[static_cast<size_t>(index)];
}

bool mem_info(int, size_t *free_bytes, size_t *total_bytes) {
    if (!g.inited)
        return false;
    if (free_bytes)
        *free_bytes = 0;
    if (total_bytes)
        *total_bytes = 0;
    return true;
}

bool device_integrated(int) { return true; }

void stats(int, size_t *tensor_count, size_t *tensor_bytes) {
    if (tensor_count)
        *tensor_count = g.tensor_count;
    if (tensor_bytes)
        *tensor_bytes = g.tensor_bytes;
}

void group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows, double *h2d_ms,
                 double *kernel_ms, double *d2h_ms) {
    if (calls)
        *calls = g.g_calls;
    if (experts)
        *experts = g.g_experts;
    if (rows)
        *rows = g.g_rows;
    if (h2d_ms)
        *h2d_ms = 0;
    if (kernel_ms)
        *kernel_ms = 0;
    if (d2h_ms)
        *d2h_ms = 0;
}

void group_stats_device(int, uint64_t *calls, uint64_t *experts, uint64_t *rows, double *h2d_ms,
                        double *kernel_ms, double *d2h_ms) {
    group_stats(calls, experts, rows, h2d_ms, kernel_ms, d2h_ms);
}

bool e8_set_grid(const void *grid) {
    if (!grid)
        return false;
    std::memcpy(g.e8_grid, grid, sizeof(g.e8_grid));
    g.have_e8 = true;
    return true;
}

bool fp8_set_lut(const float *lut) {
    if (!lut)
        return false;
    std::memcpy(g.fp8_lut, lut, sizeof(g.fp8_lut));
    g.have_fp8 = true;
    return true;
}

bool tensor_upload(Tensor **t, const void *weights, const float *scales, int fmt, int I, int O,
                   int device, int gs) {
    if (!weights || I <= 0 || O <= 0)
        return false;
    if (fmt == 6 && !g.have_e8)
        return false;
    if (fmt == 8 && !g.have_fp8)
        return false;
    if (fmt != 0 && fmt != 1 && fmt != 2 && fmt != 3 && fmt != 4 && fmt != 6 && fmt != 8)
        return false;
    auto *out = new Tensor();
    out->fmt = fmt;
    out->I = I;
    out->O = O;
    out->gs = (fmt == 4) ? (gs > 0 ? gs : 64) : 0;
    out->device = device;
    const size_t n = static_cast<size_t>(O) * static_cast<size_t>(I);
    if (fmt == 0) {
        const auto *w = static_cast<const float *>(weights);
        out->f32.assign(w, w + n);
    } else if (fmt == 1) {
        const auto *w = static_cast<const int8_t *>(weights);
        out->i8.assign(w, w + n);
        if (scales)
            out->scale.assign(scales, scales + O);
        else
            out->scale.assign(static_cast<size_t>(O), 1.f);
    } else if (fmt == 2 || fmt == 4) {
        const auto *w = static_cast<const uint8_t *>(weights);
        out->packed.assign(w, w + static_cast<size_t>(O) * ceil_div(I, 2));
        const int nsc = (fmt == 4) ? O * ceil_div(I, out->gs) : O;
        if (scales)
            out->scale.assign(scales, scales + nsc);
        else
            out->scale.assign(static_cast<size_t>(nsc), 1.f);
    } else if (fmt == 3) {
        const auto *w = static_cast<const uint8_t *>(weights);
        out->packed.assign(w, w + static_cast<size_t>(O) * ceil_div(I, 4));
        if (scales)
            out->scale.assign(scales, scales + O);
        else
            out->scale.assign(static_cast<size_t>(O), 1.f);
    } else if (fmt == 8) {
        const auto *w = static_cast<const uint8_t *>(weights);
        out->packed.assign(w, w + n);
        const int nsc = std::max(ceil_div(O, 128), 1) * std::max(ceil_div(I, 128), 1);
        if (scales)
            out->scale.assign(scales, scales + nsc);
        else
            out->scale.assign(static_cast<size_t>(nsc), 1.f);
    } else {
        const auto *w = static_cast<const uint8_t *>(weights);
        out->packed.assign(w, w + n);
    }
    return adopt(t, out);
}

bool tensor_upload_g(Tensor **t, const void *weights, const float *scales, int fmt, int I, int O,
                     int device, int gs) {
    return tensor_upload(t, weights, scales, fmt, I, O, device, gs);
}

void tensor_free(Tensor *t) {
    if (!t)
        return;
    account_sub(t);
    delete t;
}

size_t tensor_bytes(const Tensor *t) {
    if (!t)
        return 0;
    return t->f32.size() * sizeof(float) + t->i8.size() + t->packed.size() +
           t->scale.size() * sizeof(float);
}

int tensor_device(const Tensor *t) { return t ? t->device : -1; }
int tensor_fmt(const Tensor *t) { return t ? t->fmt : -1; }
int tensor_rows(const Tensor *t) { return t ? t->O : 0; }
int tensor_cols(const Tensor *t) { return t ? t->I : 0; }

bool tensor_update(Tensor *t, const void *weights, const float *scales) {
    if (!t || !weights)
        return false;
    Tensor *tmp = nullptr;
    if (!tensor_upload(&tmp, weights, scales, t->fmt, t->I, t->O, t->device, t->gs))
        return false;
    t->f32.swap(tmp->f32);
    t->i8.swap(tmp->i8);
    t->packed.swap(tmp->packed);
    t->scale.swap(tmp->scale);
    tensor_free(tmp);
    return true;
}

bool matmul(Tensor **t, float *y, const float *x, const void *weights, const float *scales, int fmt,
            int S, int I, int O, int device, int gs) {
    if (!t || !y || !x || S <= 0 || I <= 0 || O <= 0)
        return false;
    if (fmt == 6 || fmt == 7)
        return false;
    if (!*t) {
        if (!tensor_upload(t, weights, scales, fmt, I, O, device, gs))
            return false;
    }
    if ((*t)->fmt != fmt || (*t)->I != I || (*t)->O != O)
        return false;
    for (int s = 0; s < S; ++s) {
        if (!matvec_one(*t, y + static_cast<size_t>(s) * O, x + static_cast<size_t>(s) * I))
            return false;
    }
    return true;
}

bool matmul_mxfp4(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s, int S, int I,
                  int O) {
    if (!y || !x || !q4 || !e8s || S <= 0 || I <= 0 || O <= 0)
        return false;
    quant::matmul_mxfp4(y, x, q4, e8s, S, I, O);
    return true;
}

bool expert_mlp(Tensor *gate, Tensor *up, Tensor *down, float *y, const float *x, int S) {
    if (!gate || !up || !down || !y || !x || S <= 0)
        return false;
    const int I = gate->I;
    const int H = gate->O;
    if (up->I != I || up->O != H || down->I != H || down->O != I)
        return false;
    std::vector<float> g(static_cast<size_t>(H)), u(static_cast<size_t>(H));
    for (int s = 0; s < S; ++s) {
        const float *xs = x + static_cast<size_t>(s) * I;
        if (!matvec_one(gate, g.data(), xs) || !matvec_one(up, u.data(), xs))
            return false;
        quant::silu_mul(g.data(), u.data(), H);
        if (!matvec_one(down, y + static_cast<size_t>(s) * I, g.data()))
            return false;
    }
    return true;
}

bool shared_mlp_w4a16(Tensor *gate, Tensor *up, Tensor *down, float *y, const float *x, int S) {
    return expert_mlp(gate, up, down, y, x, S);
}

bool expert_group(Tensor *const *gates, Tensor *const *ups, Tensor *const *downs, const int *rows,
                  int count, float *y, const float *x) {
    if (!gates || !ups || !downs || !rows || !y || !x || count <= 0)
        return false;
    const int D = gates[0] ? gates[0]->I : 0;
    if (D <= 0)
        return false;
    int off = 0;
    for (int e = 0; e < count; ++e) {
        const int r = rows[e];
        if (r < 0)
            return false;
        if (r > 0) {
            if (!expert_mlp(gates[e], ups[e], downs[e], y + static_cast<size_t>(off) * D,
                            x + static_cast<size_t>(off) * D, r))
                return false;
        }
        off += r;
    }
    ++g.g_calls;
    g.g_experts += static_cast<uint64_t>(count);
    g.g_rows += static_cast<uint64_t>(off);
    return true;
}

bool expert_group_issue(Tensor *const *gates, Tensor *const *ups, Tensor *const *downs,
                        const int *rows, int count, const float *x) {
    if (!gates || !rows || count <= 0)
        return false;
    const int D = gates[0] ? gates[0]->I : 0;
    int tot = 0;
    for (int e = 0; e < count; ++e)
        tot += std::max(rows[e], 0);
    auto &buf = g.issue_y[0];
    buf.assign(static_cast<size_t>(tot) * std::max(D, 0), 0.f);
    return expert_group(gates, ups, downs, rows, count, buf.data(), x);
}

const float *expert_group_take(int device) {
    auto it = g.issue_y.find(device);
    if (it == g.issue_y.end() || it->second.empty())
        return nullptr;
    return it->second.data();
}

bool expert_group_pinned(Tensor *const *gates, Tensor *const *ups, Tensor *const *downs,
                         const int *rows, int count, float *y, const float *x, int) {
    return expert_group(gates, ups, downs, rows, count, y, x);
}

namespace {

bool absorb_one(const Tensor *kv_b, float *ctx, const float *q, const float *latent,
                const float *rope, int H, int Q, int R, int V, int K, int t0, int t1, float scale) {
    if (!kv_b || !ctx || !q || !latent || H <= 0 || Q <= 0 || V <= 0 || K <= 0)
        return false;
    const int T = std::max(t1 - t0, 0);
    for (int h = 0; h < H; ++h) {
        const float *qn = q + h * (Q + R);
        const float *qr = qn + Q;
        std::vector<float> scores(static_cast<size_t>(std::max(T, 1)), 0.f);
        for (int ti = 0; ti < T; ++ti) {
            const float *lat = latent + static_cast<size_t>(t0 + ti) * K;
            float kn_dot = 0.f;
            for (int qi = 0; qi < Q; ++qi) {
                float kt = 0.f;
                const int row = h * (Q + V) + qi;
                for (int k = 0; k < K; ++k)
                    kt += dequant(kv_b, row, k) * lat[k];
                kn_dot += qn[qi] * kt;
            }
            float rd = 0.f;
            if (rope && R > 0) {
                const float *rp = rope + static_cast<size_t>(t0 + ti) * R;
                for (int r = 0; r < R; ++r)
                    rd += qr[r] * rp[r];
            }
            scores[static_cast<size_t>(ti)] = scale * (kn_dot + rd);
        }
        if (T > 0) {
            float mx = scores[0];
            for (int ti = 1; ti < T; ++ti)
                if (scores[static_cast<size_t>(ti)] > mx)
                    mx = scores[static_cast<size_t>(ti)];
            float den = 0.f;
            for (int ti = 0; ti < T; ++ti) {
                scores[static_cast<size_t>(ti)] = std::exp(scores[static_cast<size_t>(ti)] - mx);
                den += scores[static_cast<size_t>(ti)];
            }
            for (int ti = 0; ti < T && den > 0.f; ++ti)
                scores[static_cast<size_t>(ti)] /= den;
        }
        float *ch = ctx + h * V;
        for (int v = 0; v < V; ++v)
            ch[v] = 0.f;
        for (int ti = 0; ti < T; ++ti) {
            const float *lat = latent + static_cast<size_t>(t0 + ti) * K;
            const float a = T > 0 ? scores[static_cast<size_t>(ti)] : 0.f;
            for (int v = 0; v < V; ++v) {
                float vv = 0.f;
                const int row = h * (Q + V) + Q + v;
                for (int k = 0; k < K; ++k)
                    vv += dequant(kv_b, row, k) * lat[k];
                ch[v] += a * vv;
            }
        }
    }
    return true;
}

} // namespace

bool attention_absorb(Tensor *kv_b, float *ctx, const float *q, const float *latent,
                      const float *rope, int H, int Q, int R, int V, int K, int T, float scale) {
    if (T < 0)
        return false;
    return absorb_one(kv_b, ctx, q, latent, rope, H, Q, R, V, K, 0, T, scale);
}

bool attention_absorb_batch(Tensor *kv_b, float *ctx, const float *q, const float *latent,
                            const float *rope, int S, int H, int Q, int R, int V, int K, int T,
                            float scale) {
    if (!kv_b || !ctx || !q || S <= 0 || T < 0)
        return false;
    const int qn = H * (Q + R);
    const int cn = H * V;
    for (int s = 0; s < S; ++s) {
        const int vis = T - S + s + 1;
        if (!absorb_one(kv_b, ctx + static_cast<size_t>(s) * cn, q + static_cast<size_t>(s) * qn,
                        latent, rope, H, Q, R, V, K, 0, std::max(vis, 0), scale))
            return false;
    }
    return true;
}

bool attention_project_batch(Tensor *kv_b, Tensor *o_proj, float *out, const float *q,
                             const float *latent, const float *rope, int S, int H, int Q, int R,
                             int V, int K, int T, float scale) {
    if (!o_proj || !out)
        return false;
    std::vector<float> ctx(static_cast<size_t>(S) * H * V, 0.f);
    if (!attention_absorb_batch(kv_b, ctx.data(), q, latent, rope, S, H, Q, R, V, K, T, scale))
        return false;
    return pipe_gemm(o_proj, out, ctx.data(), S);
}

bool attention_project_ragged(Tensor *kv_b, Tensor *o_proj, float *out, const float *q,
                              const void *const *, const float *const *latent,
                              const float *const *rope, const int *lengths, int S, int H, int Q,
                              int R, int V, int K, int max_t, float scale) {
    if (!kv_b || !o_proj || !out || !q || !latent || !lengths || S <= 0)
        return false;
    const int qn = H * (Q + R);
    const int cn = H * V;
    std::vector<float> ctx(static_cast<size_t>(S) * cn, 0.f);
    for (int s = 0; s < S; ++s) {
        const int t = std::min(std::max(lengths[s], 0), max_t);
        if (!absorb_one(kv_b, ctx.data() + static_cast<size_t>(s) * cn,
                        q + static_cast<size_t>(s) * qn, latent[s], rope ? rope[s] : nullptr, H, Q,
                        R, V, K, 0, t, scale))
            return false;
    }
    return pipe_gemm(o_proj, out, ctx.data(), S);
}

float *pipe_scratch(int device, int slot, size_t bytes) {
    auto &buf = g.scratch[std::make_pair(device, slot)];
    if (buf.size() < bytes)
        buf.resize(bytes);
    return reinterpret_cast<float *>(buf.data());
}

void *pipe_alloc(int device, size_t bytes) {
    void *p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p)
        g.allocs[device].push_back(p);
    return p;
}

void pipe_free(int device, void *p) {
    if (!p)
        return;
    auto &v = g.allocs[device];
    v.erase(std::remove(v.begin(), v.end(), p), v.end());
    std::free(p);
}

bool pipe_upload(int, void *dst, const void *src, size_t bytes) {
    if (!dst || (!src && bytes > 0))
        return false;
    if (bytes)
        std::memcpy(dst, src, bytes);
    return true;
}

bool pipe_download(int, const void *src, void *dst, size_t bytes) {
    if (!dst || (!src && bytes > 0))
        return false;
    if (bytes)
        std::memcpy(dst, src, bytes);
    return true;
}

bool pipe_rmsnorm(int, float *y, const float *x, const float *w, int S, int D, float eps) {
    if (!y || !x || S <= 0 || D <= 0)
        return false;
    for (int s = 0; s < S; ++s)
        quant::rmsnorm(x + static_cast<size_t>(s) * D, w, y + static_cast<size_t>(s) * D, D, eps);
    return true;
}

bool pipe_rope(int, float *v, const int *pos, int rows, int stride, int offset, int R, int heads,
               float theta) {
    if (!v || !pos || rows <= 0 || R <= 0)
        return false;
    const int pairs = R / 2;
    for (int r = 0; r < rows; ++r) {
        float *row = v + static_cast<size_t>(r) * stride + offset;
        const int nh = std::max(heads, 1);
        const int per = R;
        for (int h = 0; h < nh; ++h)
            apply_rope_pairs(row + h * per, pos[r], pairs, theta);
    }
    return true;
}

bool pipe_silu_mul(int, float *gate, const float *up, size_t n) {
    if (!gate || !up)
        return false;
    quant::silu_mul(gate, up, static_cast<int>(n));
    return true;
}

bool pipe_add(int, float *x, const float *t, size_t n) {
    if (!x || !t)
        return false;
    for (size_t i = 0; i < n; ++i)
        x[i] += t[i];
    return true;
}

bool pipe_rows_add(int, float *x, const float *partial, const int *rows, int nrows, int D) {
    if (!x || !partial || !rows || nrows <= 0 || D <= 0)
        return false;
    for (int i = 0; i < nrows; ++i) {
        float *dst = x + static_cast<size_t>(rows[i]) * D;
        const float *src = partial + static_cast<size_t>(i) * D;
        for (int d = 0; d < D; ++d)
            dst[d] += src[d];
    }
    return true;
}

bool pipe_gemm(Tensor *t, float *y, const float *x, int S) {
    if (!t || !y || !x || S <= 0)
        return false;
    for (int s = 0; s < S; ++s) {
        if (!matvec_one(t, y + static_cast<size_t>(s) * t->O, x + static_cast<size_t>(s) * t->I))
            return false;
    }
    return true;
}

bool pipe_router(int, const float *x, const void *rw, const void *rb, int D, int E, int Ksel,
                 float, int, float routed_scale, int *idx, float *w, int *keff) {
    if (!x || !rw || !idx || !w || D <= 0 || E <= 0)
        return false;
    std::vector<float> logits(static_cast<size_t>(E), 0.f);
    quant::matmul_f32(logits.data(), x, static_cast<const float *>(rw), 1, D, E);
    if (rb) {
        const auto *b = static_cast<const float *>(rb);
        for (int e = 0; e < E; ++e)
            logits[static_cast<size_t>(e)] += b[e];
    }
    const int k = moe_topk(logits.data(), E, Ksel, idx, w, nullptr);
    if (routed_scale != 1.f)
        for (int i = 0; i < k; ++i)
            w[i] *= routed_scale;
    if (keff)
        *keff = k;
    return true;
}

bool pipe_sync(int) { return g.inited; }

} // namespace coli_cuda
} // namespace mvllm
