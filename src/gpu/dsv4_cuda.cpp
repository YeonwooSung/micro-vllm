#include "dsv4_cuda.hpp"

#include "../model/family.hpp"
#include "../quant/bf16.hpp"
#include "../quant/native_act.hpp"
#include "../quant/quant.hpp"

#if defined(MVLLM_WITH_CUDA_GEMM)
#include "dsv4_cuda_device.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace mvllm {
namespace dsv4_cuda {
namespace {

int ceil_div(int a, int b) { return b > 0 ? (a + b - 1) / b : 0; }

float softplus(float z) {
    if (z > 20.f)
        return z;
    if (z < -20.f)
        return std::exp(z);
    return std::log1p(std::exp(z));
}

float sqrt_softplus(float z) { return std::sqrt(std::max(softplus(z), 0.f)); }

int infer_mhc_m(int n_rows) {
    for (int m = 1; m <= 16; ++m)
        if (2 * m + m * m == n_rows)
            return m;
    return 4;
}

struct LayerKv {
    int window = 0;
    int dim = 0;
    std::vector<float> ring;
    std::vector<float> comp;
};

struct Backend {
    bool inited = false;
    std::vector<int> devices;
    bool graph_open = false;
    int graph_dev = 0;
    std::map<int, std::pair<int, int>> decode;
    std::map<std::pair<int, int>, LayerKv> kv;
};

Backend g;

LayerKv &layer_kv(int device, int layer) { return g.kv[std::make_pair(device, layer)]; }

} // namespace

struct Tensor {
    Dtype dtype = Dtype::F32;
    int O = 0;
    int I = 0;
    int device = 0;
    std::vector<float> f32;
    std::vector<uint16_t> bf16;
    std::vector<uint8_t> packed;
    std::vector<uint8_t> scale;
};

struct Activation {
    int device = 0;
    std::vector<float> v;
};

struct KvCache {
    int device = 0;
    int window = 0;
    int head_dim = 0;
    int max_tokens = 0;
    int rope_pairs = 0;
    std::vector<float> rope_cos, rope_sin, compress_cos, compress_sin;
    std::vector<float> keys;
    int filled = 0;
};

struct ExpertSet {
    bool owns = false;
    int count = 0;
    int hidden = 0;
    int intermediate = 0;
    int device = 0;
    std::vector<Tensor *> gate, up, down;
    Tensor *sg = nullptr;
    Tensor *su = nullptr;
    Tensor *sd = nullptr;
    std::vector<int64_t> hash;
};

struct Graph {
    int device = 0;
};

namespace {

void assign_scale(Tensor *t, const uint8_t *scale, int n) {
    t->scale.assign(static_cast<size_t>(std::max(n, 0)), 127);
    if (scale && n > 0)
        std::memcpy(t->scale.data(), scale, static_cast<size_t>(n));
}

bool adopt(Tensor **slot, Tensor *t) {
    if (!slot || !t)
        return false;
    if (*slot)
        tensor_free(*slot);
    *slot = t;
    return true;
}

float dequant(const Tensor *t, int o, int i) {
    const int I = t->I;
    switch (t->dtype) {
    case Dtype::F32:
        return t->f32[static_cast<size_t>(o) * I + i];
    case Dtype::BF16:
        return bf16_decode(t->bf16[static_cast<size_t>(o) * I + i]);
    case Dtype::FP8:
    case Dtype::FP8BF16: {
        const int si = std::max(ceil_div(I, 128), 1);
        const size_t si_i = static_cast<size_t>(o / 128) * static_cast<size_t>(si) +
                            static_cast<size_t>(i / 128);
        const float sc = si_i < t->scale.size() ? e8m0_decode(t->scale[si_i]) : 1.f;
        const float q = e4m3fn_decode(t->packed[static_cast<size_t>(o) * I + i]) * sc;
        return t->dtype == Dtype::FP8BF16 ? bf16_round(q) : q;
    }
    case Dtype::FP4: {
        const int packed_i = ceil_div(I, 2);
        const int sg = std::max(ceil_div(I, 32), 1);
        const uint8_t byte = t->packed[static_cast<size_t>(o) * packed_i + i / 2];
        const uint8_t nib = (i & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 15);
        const size_t si = static_cast<size_t>(o) * static_cast<size_t>(sg) + static_cast<size_t>(i / 32);
        const float sc = si < t->scale.size() ? e8m0_decode(t->scale[si]) : 1.f;
        return e2m1_decode(nib) * sc;
    }
    }
    return 0.f;
}

const float *vec_data(const Tensor *t) { return t && !t->f32.empty() ? t->f32.data() : nullptr; }

int vec_n(const Tensor *t) {
    if (!t)
        return 0;
    if (t->O == 1)
        return t->I;
    if (t->I == 1)
        return t->O;
    return t->O * t->I;
}

bool one_expert(Tensor *gate, Tensor *up, Tensor *down, float weight, float limit, float *y,
                const float *x) {
    if (!gate || !up || !down || !y || !x)
        return false;
    const int I = gate->I;
    const int O = gate->O;
    if (up->I != I || up->O != O || down->I != O || down->O != I)
        return false;
    std::vector<float> g(static_cast<size_t>(O)), u(static_cast<size_t>(O)), h(static_cast<size_t>(O)),
        d(static_cast<size_t>(I));
    if (!matvec(gate, g.data(), x) || !matvec(up, u.data(), x))
        return false;
    for (int i = 0; i < O; ++i)
        h[static_cast<size_t>(i)] = quant::clamped_swiglu(g[static_cast<size_t>(i)],
                                                          u[static_cast<size_t>(i)], limit);
    if (!matvec(down, d.data(), h.data()))
        return false;
    for (int i = 0; i < I; ++i)
        y[i] += weight * d[static_cast<size_t>(i)];
    return true;
}

bool sparse_from_rows(const float *q, const float *vals, int value_rows, const float *sinks,
                      const int *meta, int comp_base, int heads, int dim, int tokens, float scale,
                      float *out) {
    if (!q || !vals || !meta || !out || heads <= 0 || dim <= 0 || tokens <= 0 || value_rows < 0)
        return false;
    const int hd = heads * dim;
    for (int t = 0; t < tokens; ++t) {
        const int woff = meta[3 * t + 0];
        const int wn = meta[3 * t + 1];
        const int cn = meta[3 * t + 2];
        for (int h = 0; h < heads; ++h) {
            const float *qh = q + static_cast<size_t>(t) * hd + static_cast<size_t>(h) * dim;
            std::vector<float> scores;
            std::vector<const float *> rows;
            scores.reserve(static_cast<size_t>(std::max(wn, 0) + std::max(cn, 0) + 1));
            rows.reserve(scores.capacity());
            for (int r = 0; r < wn; ++r) {
                const int idx = woff + r;
                if (idx < 0 || idx >= value_rows)
                    continue;
                const float *row = vals + static_cast<size_t>(idx) * dim;
                float s = 0.f;
                for (int d = 0; d < dim; ++d)
                    s += qh[d] * row[d];
                scores.push_back(scale * s);
                rows.push_back(row);
            }
            for (int c = 0; c < cn; ++c) {
                const int idx = comp_base + c;
                if (idx < 0 || idx >= value_rows)
                    continue;
                const float *row = vals + static_cast<size_t>(idx) * dim;
                float s = 0.f;
                for (int d = 0; d < dim; ++d)
                    s += qh[d] * row[d];
                scores.push_back(scale * s);
                rows.push_back(row);
            }
            const bool have_sink = sinks != nullptr;
            if (have_sink)
                scores.push_back(sinks[h]);
            float mx = scores.empty() ? 0.f : scores[0];
            for (float s : scores)
                if (s > mx)
                    mx = s;
            float den = 0.f;
            std::vector<float> w(scores.size());
            for (size_t i = 0; i < scores.size(); ++i) {
                w[i] = std::exp(scores[i] - mx);
                den += w[i];
            }
            float *oh = out + static_cast<size_t>(t) * hd + static_cast<size_t>(h) * dim;
            for (int d = 0; d < dim; ++d)
                oh[d] = 0.f;
            if (den <= 0.f)
                continue;
            const size_t nval = rows.size();
            for (size_t i = 0; i < nval; ++i) {
                const float a = w[i] / den;
                for (int d = 0; d < dim; ++d)
                    oh[d] += a * rows[i][d];
            }
        }
    }
    return true;
}

bool build_cached_vals(int device, int layer, const float *chunk, int chunk_start, int tokens,
                       int dim, int abs_base, int comp_limit, std::vector<float> &vals,
                       int &comp_base) {
    LayerKv &kv = layer_kv(device, layer);
    if (dim <= 0)
        return false;
    if (kv.dim == 0)
        kv.dim = dim;
    if (kv.dim != dim)
        return false;
    const int window = kv.window > 0 ? kv.window : 8;
    if (kv.ring.empty())
        kv.ring.assign(static_cast<size_t>(window) * dim, 0.f);
    std::vector<float> win;
    const int max_pos = chunk_start + std::max(tokens, 0);
    for (int p = abs_base; p < max_pos; ++p) {
        const float *src = nullptr;
        if (chunk && p >= chunk_start && p < chunk_start + tokens)
            src = chunk + static_cast<size_t>(p - chunk_start) * dim;
        else if (!kv.ring.empty() && kv.window > 0) {
            const int slot = ((p % kv.window) + kv.window) % kv.window;
            src = kv.ring.data() + static_cast<size_t>(slot) * dim;
        }
        if (!src)
            continue;
        win.insert(win.end(), src, src + dim);
    }
    if (win.empty() && chunk && tokens > 0)
        win.assign(chunk, chunk + static_cast<size_t>(tokens) * dim);
    comp_base = static_cast<int>(win.size() / static_cast<size_t>(std::max(dim, 1)));
    vals.swap(win);
    const int ncomp = std::min(comp_limit, static_cast<int>(kv.comp.size() / static_cast<size_t>(dim)));
    if (ncomp > 0)
        vals.insert(vals.end(), kv.comp.begin(), kv.comp.begin() + static_cast<size_t>(ncomp) * dim);
    return true;
}

int mhc_m_from_fn(const Tensor *fn, int H) {
    if (!fn || H <= 0)
        return 0;
    if (fn->I > 0 && fn->I % H == 0) {
        const int m = fn->I / H;
        if (2 * m + m * m == fn->O)
            return m;
    }
    return infer_mhc_m(fn->O);
}

bool mhc_pre_host(const float *residual, const float *fn, const float *scale, const float *base,
                  int M, int H, float rms_eps, float pre_eps, float sink_eps, float post_mult,
                  int sink_iters, float *state, float *input) {
    if (!residual || !fn || !scale || !base || !state || !input || M < 1 || H < 1)
        return false;
    const int MH = M * H;
    const int N = 2 * M + M * M;
    float ss = 0.f;
    for (int i = 0; i < MH; ++i)
        ss += residual[i] * residual[i];
    const float inv = 1.f / std::sqrt(ss / static_cast<float>(MH) + rms_eps);
    std::vector<float> mix(static_cast<size_t>(N), 0.f);
    for (int n = 0; n < N; ++n) {
        const float *row = fn + static_cast<size_t>(n) * MH;
        float v = 0.f;
        for (int i = 0; i < MH; ++i)
            v += residual[i] * row[i];
        mix[static_cast<size_t>(n)] = v * inv;
    }
    std::vector<float> pre(static_cast<size_t>(M), 0.f);
    float *post = state;
    float *comb = state + M;
    for (int i = 0; i < M; ++i) {
        pre[static_cast<size_t>(i)] =
            quant::sigmoid(mix[static_cast<size_t>(i)] * scale[0] + base[i]) + pre_eps;
        post[i] = quant::sigmoid(mix[static_cast<size_t>(M + i)] * scale[1] + base[M + i]) * post_mult;
    }
    for (int i = 0; i < M; ++i) {
        float mx = -1e30f;
        for (int j = 0; j < M; ++j) {
            const float v = mix[static_cast<size_t>(2 * M + i * M + j)] * scale[2] +
                            base[2 * M + i * M + j];
            comb[i * M + j] = v;
            if (v > mx)
                mx = v;
        }
        float z = 0.f;
        for (int j = 0; j < M; ++j) {
            const float e = std::exp(comb[i * M + j] - mx);
            comb[i * M + j] = e;
            z += e;
        }
        for (int j = 0; j < M; ++j)
            comb[i * M + j] = comb[i * M + j] / z + sink_eps;
    }
    const int iters = std::max(sink_iters, 0);
    for (int it = 0; it < iters; ++it) {
        if (it) {
            for (int i = 0; i < M; ++i) {
                float s = 0.f;
                for (int j = 0; j < M; ++j)
                    s += comb[i * M + j];
                for (int j = 0; j < M; ++j)
                    comb[i * M + j] /= s + sink_eps;
            }
        }
        for (int j = 0; j < M; ++j) {
            float s = 0.f;
            for (int i = 0; i < M; ++i)
                s += comb[i * M + j];
            for (int i = 0; i < M; ++i)
                comb[i * M + j] /= s + sink_eps;
        }
    }
    for (int h = 0; h < H; ++h) {
        float v = 0.f;
        for (int i = 0; i < M; ++i)
            v += pre[static_cast<size_t>(i)] * residual[i * H + h];
        input[h] = v;
    }
    return true;
}

bool mhc_post_host(const float *x, const float *residual, const float *state, int M, int H,
                   float *out) {
    if (!x || !residual || !state || !out || M < 1 || H < 1)
        return false;
    const float *post = state;
    const float *comb = state + M;
    for (int j = 0; j < M; ++j) {
        for (int h = 0; h < H; ++h) {
            float v = post[j] * x[h];
            for (int i = 0; i < M; ++i)
                v += comb[i * M + j] * residual[i * H + h];
            out[j * H + h] = v;
        }
    }
    return true;
}

void apply_rope_pairs(float *v, int dim, int qk_rope, int pos, const float *cos, const float *sin,
                      int pairs) {
    if (!v || qk_rope <= 0 || !cos || !sin || pairs <= 0)
        return;
    const int n = std::min(qk_rope / 2, pairs);
    const int base = dim - 2 * n;
    const int table = (pos > 0 && pairs >= 2 * n) ? n : 0;
    for (int p = 0; p < n; ++p) {
        const int o = base + 2 * p;
        if (o < 0 || o + 1 >= dim)
            continue;
        const float c = cos[table + p];
        const float s = sin[table + p];
        const float a = v[o], b = v[o + 1];
        v[o] = a * c - b * s;
        v[o + 1] = a * s + b * c;
    }
}

bool rmsnorm_tensor(float *y, const float *x, const Tensor *w, int n, float eps) {
    if (!y || !x || n <= 0)
        return false;
    const float *ww = nullptr;
    std::vector<float> ones;
    if (w && vec_data(w) && vec_n(w) >= n)
        ww = vec_data(w);
    else {
        ones.assign(static_cast<size_t>(n), 1.f);
        ww = ones.data();
    }
    quant::rmsnorm(x, ww, y, n, eps);
    return true;
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
#if defined(MVLLM_WITH_CUDA_GEMM)
    device::probe();
#endif
    return true;
}

void shutdown() {
#if defined(MVLLM_WITH_CUDA_GEMM)
    device::shutdown();
#endif
    g = Backend{};
}

bool available() { return g.inited; }

const char *backend_name() {
#if defined(MVLLM_WITH_CUDA_GEMM)
    if (device::probe())
        return "cuda";
#endif
    return "cpu";
}

bool backend_arch_ok(int device) { return g.inited && device >= 0; }

long long mem_free_mb(int id) {
#if defined(MVLLM_WITH_CUDA_GEMM)
    if (device::probe())
        return device::mem_free_mb(id);
#endif
    (void)id;
    return 0;
}

bool upload_fp8(Tensor **t, const uint8_t *w, const uint8_t *scale, int O, int I, int device) {
    if (!w || O <= 0 || I <= 0)
        return false;
    auto *out = new Tensor();
    out->dtype = Dtype::FP8;
    out->O = O;
    out->I = I;
    out->device = device;
    out->packed.assign(w, w + static_cast<size_t>(O) * I);
    assign_scale(out, scale, std::max(ceil_div(O, 128), 1) * std::max(ceil_div(I, 128), 1));
    return adopt(t, out);
}

bool upload_fp8_bf16(Tensor **t, const uint8_t *w, const uint8_t *scale, int O, int I, int device) {
    if (!upload_fp8(t, w, scale, O, I, device))
        return false;
    (*t)->dtype = Dtype::FP8BF16;
    return true;
}

bool upload_fp4(Tensor **t, const uint8_t *w, const uint8_t *scale, int O, int I, int device) {
    if (!w || O <= 0 || I <= 0)
        return false;
    auto *out = new Tensor();
    out->dtype = Dtype::FP4;
    out->O = O;
    out->I = I;
    out->device = device;
    out->packed.assign(w, w + static_cast<size_t>(O) * ceil_div(I, 2));
    assign_scale(out, scale, O * std::max(ceil_div(I, 32), 1));
    return adopt(t, out);
}

bool upload_bf16(Tensor **t, const uint16_t *w, int O, int I, int device) {
    if (!w || O <= 0 || I <= 0)
        return false;
    auto *out = new Tensor();
    out->dtype = Dtype::BF16;
    out->O = O;
    out->I = I;
    out->device = device;
    out->bf16.assign(w, w + static_cast<size_t>(O) * I);
    return adopt(t, out);
}

bool upload_f32(Tensor **t, const float *w, int O, int I, int device) {
    if (!w || O <= 0 || I <= 0)
        return false;
    auto *out = new Tensor();
    out->dtype = Dtype::F32;
    out->O = O;
    out->I = I;
    out->device = device;
    out->f32.assign(w, w + static_cast<size_t>(O) * I);
    return adopt(t, out);
}

bool tensor_refill_fp4(Tensor *t, const uint8_t *w, const uint8_t *scale, int O, int I, int) {
    if (!t || t->dtype != Dtype::FP4 || !w || O != t->O || I != t->I)
        return false;
    t->packed.assign(w, w + static_cast<size_t>(O) * ceil_div(I, 2));
    assign_scale(t, scale, O * std::max(ceil_div(I, 32), 1));
    return true;
}

void tensor_free(Tensor *t) { delete t; }

long long tensor_bytes(const Tensor *t) {
    if (!t)
        return 0;
    return static_cast<long long>(t->f32.size() * sizeof(float) + t->bf16.size() * sizeof(uint16_t) +
                                  t->packed.size() + t->scale.size());
}

int tensor_device(const Tensor *t) { return t ? t->device : -1; }
Dtype tensor_dtype(const Tensor *t) { return t ? t->dtype : Dtype::F32; }
int tensor_rows(const Tensor *t) { return t ? t->O : 0; }
int tensor_cols(const Tensor *t) { return t ? t->I : 0; }

bool matvec(Tensor *t, float *y, const float *x) {
    if (!t || !y || !x || t->O <= 0 || t->I <= 0)
        return false;
#if defined(MVLLM_WITH_CUDA_GEMM)
    if (device::probe()) {
        if (t->dtype == Dtype::FP4 &&
            device::try_fp4_matvec(y, x, t->packed.data(), t->scale.data(), t->O, t->I))
            return true;
        if ((t->dtype == Dtype::FP8 || t->dtype == Dtype::FP8BF16) &&
            device::try_fp8_matvec(y, x, t->packed.data(), t->scale.data(), t->O, t->I))
            return true;
        if (t->dtype == Dtype::F32 &&
            device::try_f32_matvec(y, x, t->f32.data(), t->O, t->I))
            return true;
    }
#endif
    if (t->dtype == Dtype::FP4) {
        quant::matmul_mxfp4(y, x, t->packed.data(), t->scale.data(), 1, t->I, t->O);
        return true;
    }
    if (t->dtype == Dtype::F32) {
        quant::matmul_f32(y, x, t->f32.data(), 1, t->I, t->O);
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

bool matmul_batch(Tensor *t, const Activation *input, int tokens, Activation *output) {
    if (!t || !input || !output || tokens <= 0)
        return false;
    if (static_cast<int>(input->v.size()) < tokens * t->I ||
        static_cast<int>(output->v.size()) < tokens * t->O)
        return false;
    for (int s = 0; s < tokens; ++s) {
        if (!matvec(t, output->v.data() + static_cast<size_t>(s) * t->O,
                    input->v.data() + static_cast<size_t>(s) * t->I))
            return false;
    }
    return true;
}

bool matmul_bf16_batch(Tensor *t, const float *x, int tokens, float *y) {
    if (!t || !x || !y || tokens <= 0)
        return false;
    for (int s = 0; s < tokens; ++s) {
        if (!matvec(t, y + static_cast<size_t>(s) * t->O, x + static_cast<size_t>(s) * t->I))
            return false;
    }
    return true;
}

bool matvec_grouped(Tensor *t, float *y, const float *x, int) { return matvec(t, y, x); }

bool sparse_attn_batch(int, const float *q, const float *vals, const float *sinks, const int *meta,
                       int value_rows, int comp_base, int heads, int dim, int tokens, float scale,
                       float *out) {
    return sparse_from_rows(q, vals, value_rows, sinks, meta, comp_base, heads, dim, tokens, scale,
                            out);
}

bool sparse_attn_batch_cached(int device, int layer, const float *q, const float *chunk,
                              int chunk_start, const float *sinks, const int *meta, int abs_base,
                              int comp_limit, int heads, int dim, int tokens, float scale,
                              float *out) {
    if (!q || !meta || !out)
        return false;
    std::vector<float> vals;
    int comp_base = 0;
    if (!build_cached_vals(device, layer, chunk, chunk_start, tokens, dim, abs_base, comp_limit,
                           vals, comp_base))
        return false;
    const int value_rows = dim > 0 ? static_cast<int>(vals.size() / static_cast<size_t>(dim)) : 0;
    return sparse_from_rows(q, vals.data(), value_rows, sinks, meta, comp_base, heads, dim, tokens,
                            scale, out);
}

bool sparse_attn_batch_cached_idx(int device, int layer, const float *q, const float *chunk,
                                  int chunk_start, const float *sinks, const int *meta,
                                  const int *sel, int selstride, int abs_base, int comp_limit,
                                  int heads, int dim, int tokens, float scale, float *out) {
    if (!q || !meta || !out || !sel)
        return false;
    std::vector<float> stored;
    int stored_comp = 0;
    if (!build_cached_vals(device, layer, chunk, chunk_start, tokens, dim, abs_base, comp_limit,
                           stored, stored_comp))
        return false;
    const int value_rows = dim > 0 ? static_cast<int>(stored.size() / static_cast<size_t>(dim)) : 0;
    std::vector<int> adj(static_cast<size_t>(tokens) * 3);
    std::vector<float> vals;
    int comp_base = 0;
    for (int t = 0; t < tokens; ++t) {
        const int woff = meta[3 * t + 0];
        const int wn = meta[3 * t + 1];
        const int cn = meta[3 * t + 2];
        adj[static_cast<size_t>(t) * 3 + 0] = static_cast<int>(vals.size() / static_cast<size_t>(dim));
        adj[static_cast<size_t>(t) * 3 + 1] = wn;
        for (int r = 0; r < wn; ++r) {
            const int idx = woff + r;
            if (idx < 0 || idx >= stored_comp)
                vals.insert(vals.end(), static_cast<size_t>(dim), 0.f);
            else
                vals.insert(vals.end(), stored.begin() + static_cast<size_t>(idx) * dim,
                            stored.begin() + static_cast<size_t>(idx + 1) * dim);
        }
        if (t == 0)
            comp_base = static_cast<int>(vals.size() / static_cast<size_t>(dim));
        adj[static_cast<size_t>(t) * 3 + 2] = cn;
        for (int c = 0; c < cn; ++c) {
            const int ord = sel[static_cast<size_t>(t) * selstride + c];
            const int idx = stored_comp + ord;
            if (ord < 0 || idx >= value_rows)
                vals.insert(vals.end(), static_cast<size_t>(dim), 0.f);
            else
                vals.insert(vals.end(), stored.begin() + static_cast<size_t>(idx) * dim,
                            stored.begin() + static_cast<size_t>(idx + 1) * dim);
        }
    }
    const int nrows = dim > 0 ? static_cast<int>(vals.size() / static_cast<size_t>(dim)) : 0;
    return sparse_from_rows(q, vals.data(), nrows, sinks, adj.data(), comp_base, heads, dim, tokens,
                            scale, out);
}

bool indexer_score_batch(int, const float *queries, const float *keys, const float *head_w,
                         const int *counts, int tokens, int heads, int dim, int count,
                         float *scores) {
    if (!queries || !keys || !head_w || !counts || !scores || tokens <= 0 || heads <= 0 || dim <= 0 ||
        count < 0)
        return false;
    for (int t = 0; t < tokens; ++t) {
        const int lim = counts[t];
        for (int c = 0; c < count; ++c) {
            float s = 0.f;
            if (c < lim) {
                for (int h = 0; h < heads; ++h) {
                    const float *qh =
                        queries + (static_cast<size_t>(t) * heads + h) * static_cast<size_t>(dim);
                    const float *kc = keys + static_cast<size_t>(c) * dim;
                    float dot = 0.f;
                    for (int d = 0; d < dim; ++d)
                        dot += qh[d] * kc[d];
                    if (dot < 0.f)
                        dot = 0.f;
                    s += dot * head_w[t * heads + h];
                }
            }
            scores[t * count + c] = s;
        }
    }
    return true;
}

bool fp8_ref_matmul(int, const uint8_t *w, const float *bscale, int rows, int cols,
                    int, const float *x, int tokens, float *y) {
    if (!w || !bscale || !x || !y || rows <= 0 || cols <= 0 || tokens <= 0)
        return false;
    const int sc = std::max(ceil_div(cols, 128), 1);
    for (int t = 0; t < tokens; ++t) {
        const float *xt = x + static_cast<size_t>(t) * cols;
        float *yt = y + static_cast<size_t>(t) * rows;
        for (int o = 0; o < rows; ++o) {
            float s = 0.f;
            for (int i = 0; i < cols; ++i) {
                const float scv = bscale[(o / 128) * sc + i / 128];
                s += xt[i] * e4m3fn_decode(w[static_cast<size_t>(o) * cols + i]) * scv;
            }
            yt[o] = s;
        }
    }
    return true;
}

bool stream_drain(int) { return g.inited; }

bool kv_ring_append(int device, int layer, const float *rows, int start_pos, int count, int window,
                    int dim) {
    if (!rows || count <= 0 || window <= 0 || dim <= 0)
        return false;
    LayerKv &kv = layer_kv(device, layer);
    kv.window = window;
    kv.dim = dim;
    kv.ring.assign(static_cast<size_t>(window) * dim, 0.f);
    for (int i = 0; i < count; ++i) {
        const int pos = start_pos + i;
        const int slot = ((pos % window) + window) % window;
        std::memcpy(kv.ring.data() + static_cast<size_t>(slot) * dim,
                    rows + static_cast<size_t>(i) * dim, static_cast<size_t>(dim) * sizeof(float));
    }
    return true;
}

bool kv_comp_append(int device, int layer, const float *rows, int start_idx, int count, int dim) {
    if (!rows || count <= 0 || dim <= 0)
        return false;
    LayerKv &kv = layer_kv(device, layer);
    kv.dim = dim;
    const size_t need = static_cast<size_t>(start_idx + count) * dim;
    if (kv.comp.size() < need)
        kv.comp.resize(need, 0.f);
    std::memcpy(kv.comp.data() + static_cast<size_t>(start_idx) * dim, rows,
                static_cast<size_t>(count) * dim * sizeof(float));
    return true;
}

bool head_argmax(Tensor *t, const float *x, int *id, float *value) {
    if (!t || !x || !id || !value)
        return false;
    std::vector<float> y(static_cast<size_t>(std::max(t->O, 0)), 0.f);
    if (!matvec(t, y.data(), x))
        return false;
    int best = 0;
    float bv = y[0];
    for (int i = 1; i < t->O; ++i) {
        if (y[static_cast<size_t>(i)] > bv) {
            bv = y[static_cast<size_t>(i)];
            best = i;
        }
    }
    *id = best;
    *value = bv;
    return true;
}

bool final_argmax(const Activation *residual, Tensor *, Tensor *, Tensor *, Tensor *norm,
                  Tensor *head, int, int H, float eps, float, int *id, float *value) {
    if (!residual || !head || !id || !value || H <= 0)
        return false;
    if (static_cast<int>(residual->v.size()) < H)
        return false;
    std::vector<float> n(static_cast<size_t>(H));
    if (!rmsnorm_tensor(n.data(), residual->v.data(), norm, H, eps))
        return false;
    return head_argmax(head, n.data(), id, value);
}

bool expert_group(Tensor *const *gate, Tensor *const *up, Tensor *const *down, const float *weights,
                  int count, float limit, float *y, const float *x) {
    if (!gate || !up || !down || !weights || !y || !x || count <= 0)
        return false;
    const int I = gate[0] ? gate[0]->I : 0;
    if (I <= 0)
        return false;
    std::fill(y, y + I, 0.f);
    for (int e = 0; e < count; ++e) {
        if (!one_expert(gate[e], up[e], down[e], weights[e], limit, y, x))
            return false;
    }
    return true;
}

bool expert_fp8(Tensor *gate, Tensor *up, Tensor *down, float limit, float *y, const float *x) {
    if (!gate || !y || gate->I <= 0)
        return false;
    std::fill(y, y + gate->I, 0.f);
    return one_expert(gate, up, down, 1.f, limit, y, x);
}

bool moe(Tensor *const *gate, Tensor *const *up, Tensor *const *down, const float *weights,
         int count, Tensor *shared_gate, Tensor *shared_up, Tensor *shared_down, float limit,
         float *y, const float *x) {
    if (!expert_group(gate, up, down, weights, count, limit, y, x))
        return false;
    if (!shared_gate || !shared_up || !shared_down)
        return true;
    std::vector<float> sh(static_cast<size_t>(shared_gate->I), 0.f);
    if (!expert_fp8(shared_gate, shared_up, shared_down, limit, sh.data(), x))
        return false;
    for (int i = 0; i < shared_gate->I; ++i)
        y[i] += sh[static_cast<size_t>(i)];
    return true;
}

bool qkv(Tensor *q_a, Tensor *q_norm, Tensor *q_b, Tensor *kv, float eps, float *q_out,
         float *kv_out, const float *x) {
    if (!q_a || !q_b || !kv || !q_out || !kv_out || !x)
        return false;
    std::vector<float> a(static_cast<size_t>(q_a->O), 0.f);
    if (!matvec(q_a, a.data(), x))
        return false;
    if (q_norm) {
        std::vector<float> n(a.size());
        if (!rmsnorm_tensor(n.data(), a.data(), q_norm, q_a->O, eps))
            return false;
        a.swap(n);
    }
    if (!matvec(q_b, q_out, a.data()))
        return false;
    return matvec(kv, kv_out, x);
}

bool wo(Tensor *wo_a, Tensor *wo_b, int, float *out, const float *context) {
    if (!wo_a || !out || !context || wo_a->I <= 0)
        return false;
    const int I = wo_a->I;
    const int O = wo_a->O;
    std::vector<float> tmp(static_cast<size_t>(O), 0.f);
    if (!matvec(wo_a, tmp.data(), context))
        return false;
    if (wo_b)
        return matvec(wo_b, out, tmp.data());
    std::memcpy(out, tmp.data(), static_cast<size_t>(O) * sizeof(float));
    return true;
}

Activation *activation_create(int device, long long elements) {
    if (elements <= 0)
        return nullptr;
    auto *a = new Activation();
    a->device = device;
    a->v.assign(static_cast<size_t>(elements), 0.f);
    return a;
}

void activation_free(Activation *a) { delete a; }

bool activation_upload(Activation *a, const float *x, long long elements) {
    if (!a || !x || elements <= 0)
        return false;
    const long long n = std::min(elements, static_cast<long long>(a->v.size()));
    std::memcpy(a->v.data(), x, static_cast<size_t>(n) * sizeof(float));
    return true;
}

bool activation_download(float *x, const Activation *a, long long elements) {
    if (!a || !x || elements <= 0)
        return false;
    const long long n = std::min(elements, static_cast<long long>(a->v.size()));
    std::memcpy(x, a->v.data(), static_cast<size_t>(n) * sizeof(float));
    return true;
}

bool activation_copy(Activation *dst, const Activation *src, long long elements) {
    if (!dst || !src || elements <= 0)
        return false;
    return activation_copy_range(dst, 0, src, 0, elements);
}

bool activation_copy_range(Activation *dst, long long dst_offset, const Activation *src,
                           long long src_offset, long long elements) {
    if (!dst || !src || elements <= 0 || dst_offset < 0 || src_offset < 0)
        return false;
    if (dst_offset + elements > static_cast<long long>(dst->v.size()) ||
        src_offset + elements > static_cast<long long>(src->v.size()))
        return false;
    std::memcpy(dst->v.data() + dst_offset, src->v.data() + src_offset,
                static_cast<size_t>(elements) * sizeof(float));
    return true;
}

bool activation_sync(const Activation *a) { return a != nullptr; }
int activation_device(const Activation *a) { return a ? a->device : -1; }
long long activation_elements(const Activation *a) {
    return a ? static_cast<long long>(a->v.size()) : 0;
}

bool decode_state_set(int device, int token, int position) {
    if (!g.inited)
        return false;
    g.decode[device] = {token, position};
    return true;
}

void profiler_start() {}
void profiler_stop() {}

bool graph_begin(int device) {
    g.graph_open = true;
    g.graph_dev = device;
    return true;
}

Graph *graph_end(int device) {
    if (!g.graph_open && device < 0)
        return nullptr;
    g.graph_open = false;
    auto *gr = new Graph();
    gr->device = device;
    return gr;
}

bool graph_end_pair(int primary, int, Graph **primary_graph, Graph **peer_graph) {
    if (!primary_graph || !peer_graph)
        return false;
    *primary_graph = graph_end(primary);
    *peer_graph = graph_end(primary);
    return *primary_graph && *peer_graph;
}

bool graph_launch(Graph *graph) { return graph != nullptr; }
void graph_free(Graph *graph) { delete graph; }

bool mhc_pre(const Activation *residual, Tensor *fn, Tensor *scale, Tensor *base, int M, int H,
             float rms_eps, float pre_eps, float sink_eps, float post_mult, int sink_iters,
             Activation *state, Activation *input) {
    if (!residual || !fn || !scale || !base || !state || !input || M < 1 || H < 1)
        return false;
    const int MH = M * H;
    const int N = 2 * M + M * M;
    const int stn = M + M * M;
    if (static_cast<int>(residual->v.size()) < MH || static_cast<int>(input->v.size()) < H ||
        static_cast<int>(state->v.size()) < stn || fn->O < N || fn->I < MH ||
        static_cast<int>(scale->f32.size()) < 3 || static_cast<int>(base->f32.size()) < N)
        return false;
    return mhc_pre_host(residual->v.data(), fn->f32.data(), scale->f32.data(), base->f32.data(), M, H,
                        rms_eps, pre_eps, sink_eps, post_mult, sink_iters, state->v.data(),
                        input->v.data());
}

bool mhc_pre_norm(const Activation *residual, Tensor *fn, Tensor *scale, Tensor *base, Tensor *norm,
                  int M, int H, float rms_eps, float pre_eps, float sink_eps, float post_mult,
                  int sink_iters, float norm_eps, Activation *state, Activation *input) {
    if (!residual || M < 1 || H < 1)
        return false;
    Activation tmp;
    tmp.v = residual->v;
    for (int i = 0; i < M; ++i) {
        float *row = tmp.v.data() + static_cast<size_t>(i) * H;
        std::vector<float> n(static_cast<size_t>(H));
        if (!rmsnorm_tensor(n.data(), row, norm, H, norm_eps))
            return false;
        std::memcpy(row, n.data(), static_cast<size_t>(H) * sizeof(float));
    }
    return mhc_pre(&tmp, fn, scale, base, M, H, rms_eps, pre_eps, sink_eps, post_mult, sink_iters,
                   state, input);
}

bool mhc_pre_batch(const Activation *residual, Tensor *fn, Tensor *scale, Tensor *base, int tokens,
                   int H, Activation *state, Activation *input) {
    const int M = mhc_m_from_fn(fn, H);
    if (M < 1 || tokens <= 0 || !residual || !state || !input)
        return false;
    const int MH = M * H;
    const int stn = M + M * M;
    for (int t = 0; t < tokens; ++t) {
        Activation r, st, in;
        r.v.assign(residual->v.begin() + static_cast<size_t>(t) * MH,
                   residual->v.begin() + static_cast<size_t>(t + 1) * MH);
        st.v.assign(static_cast<size_t>(stn), 0.f);
        in.v.assign(static_cast<size_t>(H), 0.f);
        if (!mhc_pre(&r, fn, scale, base, M, H, 1e-6f, 1e-6f, 1e-6f, 2.f, 4, &st, &in))
            return false;
        std::copy(st.v.begin(), st.v.end(), state->v.begin() + static_cast<size_t>(t) * stn);
        std::copy(in.v.begin(), in.v.end(), input->v.begin() + static_cast<size_t>(t) * H);
    }
    return true;
}

bool mhc_pre_norm_batch(const Activation *residual, Tensor *fn, Tensor *scale, Tensor *base,
                        Tensor *norm, int tokens, int H, Activation *state, Activation *input) {
    const int M = mhc_m_from_fn(fn, H);
    if (M < 1 || tokens <= 0 || !residual)
        return false;
    const int MH = M * H;
    const int stn = M + M * M;
    for (int t = 0; t < tokens; ++t) {
        Activation r, st, in;
        r.v.assign(residual->v.begin() + static_cast<size_t>(t) * MH,
                   residual->v.begin() + static_cast<size_t>(t + 1) * MH);
        st.v.assign(static_cast<size_t>(stn), 0.f);
        in.v.assign(static_cast<size_t>(H), 0.f);
        if (!mhc_pre_norm(&r, fn, scale, base, norm, M, H, 1e-6f, 1e-6f, 1e-6f, 2.f, 4, 1e-6f, &st,
                          &in))
            return false;
        std::copy(st.v.begin(), st.v.end(), state->v.begin() + static_cast<size_t>(t) * stn);
        std::copy(in.v.begin(), in.v.end(), input->v.begin() + static_cast<size_t>(t) * H);
    }
    return true;
}

bool mhc_post(const Activation *x, const Activation *residual, const Activation *state, int M,
              int H, Activation *out) {
    if (!x || !residual || !state || !out)
        return false;
    return mhc_post_host(x->v.data(), residual->v.data(), state->v.data(), M, H, out->v.data());
}

bool mhc_post_pre(const Activation *x, const Activation *residual, Activation *state, int M, int H,
                  Activation *out, Tensor *fn, Tensor *scale, Tensor *base, float rms_eps,
                  float pre_eps, float sink_eps, float post_mult, int sink_iters,
                  Activation *input) {
    if (!mhc_post(x, residual, state, M, H, out))
        return false;
    return mhc_pre(out, fn, scale, base, M, H, rms_eps, pre_eps, sink_eps, post_mult, sink_iters,
                   state, input);
}

bool mhc_post_pre_norm(const Activation *x, const Activation *residual, Activation *state, int M,
                       int H, Activation *out, Tensor *fn, Tensor *scale, Tensor *base,
                       Tensor *norm, float rms_eps, float pre_eps, float sink_eps, float post_mult,
                       int sink_iters, float norm_eps, Activation *input) {
    if (!mhc_post(x, residual, state, M, H, out))
        return false;
    return mhc_pre_norm(out, fn, scale, base, norm, M, H, rms_eps, pre_eps, sink_eps, post_mult,
                        sink_iters, norm_eps, state, input);
}

bool mhc_post_batch(const Activation *x, const Activation *residual, const Activation *state,
                    int tokens, int H, Activation *out) {
    if (!x || !residual || !state || !out || tokens <= 0)
        return false;
    const int stn = static_cast<int>(state->v.size() / tokens);
    const int M = 1;
    int m = 1;
    for (; m <= 16; ++m)
        if (m + m * m == stn)
            break;
    const int Muse = m <= 16 ? m : M;
    const int MH = Muse * H;
    for (int t = 0; t < tokens; ++t) {
        if (!mhc_post_host(x->v.data() + static_cast<size_t>(t) * H,
                           residual->v.data() + static_cast<size_t>(t) * MH,
                           state->v.data() + static_cast<size_t>(t) * stn, Muse, H,
                           out->v.data() + static_cast<size_t>(t) * MH))
            return false;
    }
    return true;
}

bool mhc_post_pre_norm_batch(const Activation *x, const Activation *residual, Activation *state,
                             int tokens, int H, Activation *out, Tensor *fn, Tensor *scale,
                             Tensor *base, Tensor *norm, Activation *input) {
    if (!mhc_post_batch(x, residual, state, tokens, H, out))
        return false;
    return mhc_pre_norm_batch(out, fn, scale, base, norm, tokens, H, state, input);
}

bool attention_first(const Activation *input, Tensor *attn_norm, Tensor *q_a, Tensor *q_norm,
                     Tensor *q_b, Tensor *wkv, Tensor *kv_norm, Tensor *sink, Tensor *wo_a,
                     Tensor *wo_b, int heads, int head_dim, int qk_rope, int groups, float eps,
                     Activation *output) {
    if (!input || !q_a || !q_b || !wkv || !wo_a || !output || heads <= 0 || head_dim <= 0)
        return false;
    const int H = static_cast<int>(input->v.size());
    const int qn = heads * head_dim;
    if (H <= 0 || static_cast<int>(output->v.size()) < H)
        return false;
    std::vector<float> n(static_cast<size_t>(H));
    if (!rmsnorm_tensor(n.data(), input->v.data(), attn_norm, H, eps))
        return false;
    std::vector<float> q(static_cast<size_t>(std::max(q_b->O, qn)), 0.f);
    std::vector<float> k(static_cast<size_t>(std::max(wkv->O, head_dim)), 0.f);
    if (!qkv(q_a, q_norm, q_b, wkv, eps, q.data(), k.data(), n.data()))
        return false;
    std::vector<float> kn(static_cast<size_t>(head_dim));
    if (!rmsnorm_tensor(kn.data(), k.data(), kv_norm, head_dim, eps))
        return false;
    const float *sk = sink && vec_data(sink) ? vec_data(sink) : nullptr;
    const float att_scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> ctx(static_cast<size_t>(qn), 0.f);
    for (int h = 0; h < heads; ++h) {
        float *qh = q.data() + static_cast<size_t>(h) * head_dim;
        std::vector<float> qnrm(static_cast<size_t>(head_dim));
        quant::rmsnorm(qh, nullptr, qnrm.data(), head_dim, eps);
        if (qk_rope > 0) {
            // first token: identity RoPE
        }
        float score = 0.f;
        for (int d = 0; d < head_dim; ++d)
            score += qnrm[static_cast<size_t>(d)] * kn[static_cast<size_t>(d)];
        score *= att_scale;
        const float sink_v = sk ? sk[h] : -1e9f;
        const float mx = std::max(score, sink_v);
        const float a = std::exp(score - mx);
        const float den = a + std::exp(sink_v - mx);
        float *ch = ctx.data() + static_cast<size_t>(h) * head_dim;
        for (int d = 0; d < head_dim; ++d)
            ch[d] = (den > 0.f) ? a * kn[static_cast<size_t>(d)] / den : 0.f;
    }
    (void)groups;
    return wo(wo_a, wo_b, groups, output->v.data(), ctx.data());
}

KvCache *kv_create(int device, int window, int head_dim, int max_tokens, int rope_pairs,
                   const float *rope_cos, const float *rope_sin, const float *compress_cos,
                   const float *compress_sin) {
    if (window <= 0 || head_dim <= 0)
        return nullptr;
    auto *c = new KvCache();
    c->device = device;
    c->window = window;
    c->head_dim = head_dim;
    c->max_tokens = max_tokens;
    c->rope_pairs = rope_pairs;
    if (rope_cos && rope_sin && rope_pairs > 0) {
        c->rope_cos.assign(rope_cos, rope_cos + 2 * std::max(rope_pairs, 1));
        c->rope_sin.assign(rope_sin, rope_sin + 2 * std::max(rope_pairs, 1));
    }
    if (compress_cos && compress_sin && rope_pairs > 0) {
        c->compress_cos.assign(compress_cos, compress_cos + 2 * std::max(rope_pairs, 1));
        c->compress_sin.assign(compress_sin, compress_sin + 2 * std::max(rope_pairs, 1));
    }
    return c;
}

void kv_free(KvCache *cache) { delete cache; }

namespace {

void allreduce_pair(Activation *a, Activation *b) {
    if (!a || !b)
        return;
    const size_t n = std::min(a->v.size(), b->v.size());
    for (size_t i = 0; i < n; ++i) {
        const float s = a->v[i] + b->v[i];
        a->v[i] = s;
        b->v[i] = s;
    }
}

bool attention_window_heads(const Activation *input, Tensor *attn_norm, Tensor *q_a, Tensor *q_norm,
                            Tensor *q_b, Tensor *wkv, Tensor *kv_norm, Tensor *sink, Tensor *wo_a,
                            Tensor *wo_b, int heads, int head_dim, int qk_rope, int groups, int pos,
                            float eps, KvCache *cache, Activation *output, int head_begin,
                            int n_heads) {
    if (!input || !cache || !output || heads <= 0 || head_dim <= 0 || n_heads <= 0)
        return false;
    if (head_begin < 0 || head_begin + n_heads > heads)
        return false;
    const int H = static_cast<int>(input->v.size());
    if (H <= 0)
        return false;
    std::vector<float> n(static_cast<size_t>(H));
    if (!rmsnorm_tensor(n.data(), input->v.data(), attn_norm, H, eps))
        return false;
    const int qn = heads * head_dim;
    const int q_rows = q_b ? std::max(q_b->O, qn) : qn;
    std::vector<float> q(static_cast<size_t>(q_rows), 0.f);
    std::vector<float> k(static_cast<size_t>(head_dim), 0.f);
    if (!qkv(q_a, q_norm, q_b, wkv, eps, q.data(), k.data(), n.data()))
        return false;
    std::vector<float> kn(static_cast<size_t>(head_dim));
    if (!rmsnorm_tensor(kn.data(), k.data(), kv_norm, head_dim, eps))
        return false;
    const float *rcos = cache->rope_cos.empty() ? nullptr : cache->rope_cos.data();
    const float *rsin = cache->rope_sin.empty() ? nullptr : cache->rope_sin.data();
    apply_rope_pairs(kn.data(), head_dim, qk_rope, pos, rcos, rsin, cache->rope_pairs);
    const int q_avail = head_dim > 0 ? static_cast<int>(q.size() / static_cast<size_t>(head_dim)) : 0;
    const bool q_sharded = q_avail < heads;
    for (int hi = 0; hi < n_heads; ++hi) {
        const int qh_i = q_sharded ? hi : head_begin + hi;
        if (qh_i < 0 || qh_i >= q_avail)
            continue;
        apply_rope_pairs(q.data() + static_cast<size_t>(qh_i) * head_dim, head_dim, qk_rope, pos, rcos,
                         rsin, cache->rope_pairs);
    }
    cache->keys.insert(cache->keys.end(), kn.begin(), kn.end());
    cache->filled += 1;
    const int T = cache->filled;
    const float att_scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    const float *sk = sink && vec_data(sink) ? vec_data(sink) : nullptr;
    const int sink_n = sink ? vec_n(sink) : 0;
    const int wo_I = wo_a ? wo_a->I : qn;
    std::vector<float> ctx(static_cast<size_t>(std::max(wo_I, qn)), 0.f);
    const bool wo_full = wo_I >= qn;
    for (int hi = 0; hi < n_heads; ++hi) {
        const int h = head_begin + hi;
        const int qh_i = q_sharded ? hi : h;
        if (qh_i < 0 || qh_i >= q_avail)
            continue;
        const float *qh = q.data() + static_cast<size_t>(qh_i) * head_dim;
        std::vector<float> scores(static_cast<size_t>(T) + 1, 0.f);
        for (int t = 0; t < T; ++t) {
            const float *kt = cache->keys.data() + static_cast<size_t>(t) * head_dim;
            float s = 0.f;
            for (int d = 0; d < head_dim; ++d)
                s += qh[d] * kt[d];
            scores[static_cast<size_t>(t)] = s * att_scale;
        }
        const int si = sink_n >= heads ? h : hi;
        scores[static_cast<size_t>(T)] = (sk && si >= 0 && si < sink_n) ? sk[si] : -1e9f;
        float mx = scores[0];
        for (float s : scores)
            if (s > mx)
                mx = s;
        float den = 0.f;
        for (float &s : scores) {
            s = std::exp(s - mx);
            den += s;
        }
        const int ctx_i = wo_full ? h : hi;
        float *ch = ctx.data() + static_cast<size_t>(ctx_i) * head_dim;
        for (int t = 0; t < T; ++t) {
            const float a = den > 0.f ? scores[static_cast<size_t>(t)] / den : 0.f;
            const float *kt = cache->keys.data() + static_cast<size_t>(t) * head_dim;
            for (int d = 0; d < head_dim; ++d)
                ch[d] += a * kt[d];
        }
    }
    return wo(wo_a, wo_b, groups, output->v.data(), ctx.data());
}

int map_ep_expert(const ExpertSet *set, int global_id, int parity) {
    if (!set || global_id < 0)
        return -1;
    if (parity >= 0 && (global_id % 2) != parity)
        return -1;
    if (global_id < set->count && set->gate[static_cast<size_t>(global_id)])
        return global_id;
    if (parity >= 0) {
        const int packed = global_id / 2;
        if (packed >= 0 && packed < set->count && set->gate[static_cast<size_t>(packed)])
            return packed;
    }
    return -1;
}

bool moe_routed_parity(const Activation *input, ExpertSet *set, const int *ids, const float *weights,
                       int parity, float limit, Activation *output, bool with_shared) {
    if (!input || !output || input->v.empty())
        return false;
    if (static_cast<int>(output->v.size()) < static_cast<int>(input->v.size()))
        return false;
    std::fill(output->v.begin(), output->v.begin() + static_cast<std::ptrdiff_t>(input->v.size()),
              0.f);
    Tensor *g[6] = {}, *u[6] = {}, *d[6] = {};
    float ww[6] = {};
    int n = 0;
    if (set && ids && weights) {
        for (int k = 0; k < 6; ++k) {
            const int local = map_ep_expert(set, ids[k], parity);
            if (local < 0 || !set->up[static_cast<size_t>(local)] ||
                !set->down[static_cast<size_t>(local)])
                continue;
            g[n] = set->gate[static_cast<size_t>(local)];
            u[n] = set->up[static_cast<size_t>(local)];
            d[n] = set->down[static_cast<size_t>(local)];
            ww[n] = weights[k];
            ++n;
        }
    }
    Tensor *sg = (with_shared && set) ? set->sg : nullptr;
    Tensor *su = (with_shared && set) ? set->su : nullptr;
    Tensor *sd = (with_shared && set) ? set->sd : nullptr;
    if (n > 0)
        return moe(g, u, d, ww, n, sg, su, sd, limit, output->v.data(), input->v.data());
    if (sg && su && sd) {
        std::vector<float> sh(input->v.size(), 0.f);
        if (!expert_fp8(sg, su, sd, limit, sh.data(), input->v.data()))
            return false;
        std::memcpy(output->v.data(), sh.data(), input->v.size() * sizeof(float));
    }
    return true;
}

} // namespace

bool attention_window(const Activation *input, Tensor *attn_norm, Tensor *q_a, Tensor *q_norm,
                      Tensor *q_b, Tensor *wkv, Tensor *kv_norm, Tensor *sink, Tensor *wo_a,
                      Tensor *wo_b, Tensor *, Tensor *, Tensor *, Tensor *, int, int heads,
                      int head_dim, int qk_rope, int groups, int pos, float eps, KvCache *cache,
                      Activation *output) {
    return attention_window_heads(input, attn_norm, q_a, q_norm, q_b, wkv, kv_norm, sink, wo_a, wo_b,
                                  heads, head_dim, qk_rope, groups, pos, eps, cache, output, 0,
                                  heads);
}

bool attention_sparse_batch(const Activation *input, Tensor *attn_norm, Tensor *qkv_t, Tensor *,
                            Tensor *q_b, Tensor *, Tensor *sink, int heads, int head_dim,
                            int start_pos, int tokens, float eps, KvCache *cache,
                            Activation *context) {
    if (!input || !qkv_t || !context || !cache || tokens <= 0 || heads <= 0 || head_dim <= 0)
        return false;
    const int H = qkv_t->I > 0 ? qkv_t->I : static_cast<int>(input->v.size() / tokens);
    const int qn = heads * head_dim;
    std::vector<float> q(static_cast<size_t>(tokens) * qn, 0.f);
    std::vector<float> vals;
    for (int t = 0; t < tokens; ++t) {
        const float *x = input->v.data() + static_cast<size_t>(t) * H;
        std::vector<float> n(static_cast<size_t>(H));
        rmsnorm_tensor(n.data(), x, attn_norm, H, eps);
        std::vector<float> qt(static_cast<size_t>(std::max(q_b ? q_b->O : qn, qn)), 0.f);
        std::vector<float> kt(static_cast<size_t>(head_dim), 0.f);
        if (q_b) {
            std::vector<float> a(static_cast<size_t>(qkv_t->O), 0.f);
            matvec(qkv_t, a.data(), n.data());
            matvec(q_b, qt.data(), a.data());
        } else
            matvec(qkv_t, qt.data(), n.data());
        std::copy(qt.begin(), qt.begin() + qn, q.begin() + static_cast<size_t>(t) * qn);
        vals.insert(vals.end(), n.begin(), n.begin() + std::min(H, head_dim));
        if (static_cast<int>(vals.size()) % head_dim)
            vals.resize((vals.size() / head_dim + 1) * head_dim, 0.f);
        (void)start_pos;
    }
    const int value_rows = tokens;
    std::vector<int> meta(static_cast<size_t>(tokens) * 3);
    for (int t = 0; t < tokens; ++t) {
        meta[static_cast<size_t>(t) * 3 + 0] = 0;
        meta[static_cast<size_t>(t) * 3 + 1] = t + 1;
        meta[static_cast<size_t>(t) * 3 + 2] = 0;
    }
    const float *sk = sink && vec_data(sink) ? vec_data(sink) : nullptr;
    const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    if (static_cast<int>(context->v.size()) < tokens * qn)
        return false;
    return sparse_from_rows(q.data(), vals.data(), value_rows, sk, meta.data(), value_rows, heads,
                            head_dim, tokens, scale, context->v.data());
}

bool attention_output_batch(const Activation *context, Tensor *wo_a, Tensor *wo_b, int groups,
                            int tokens, Activation *output) {
    if (!context || !wo_a || !output || tokens <= 0)
        return false;
    const int C = wo_a->I;
    const int O = wo_b ? wo_b->O : wo_a->O;
    if (static_cast<int>(output->v.size()) < tokens * O)
        return false;
    for (int t = 0; t < tokens; ++t) {
        if (!wo(wo_a, wo_b, groups, output->v.data() + static_cast<size_t>(t) * O,
                context->v.data() + static_cast<size_t>(t) * C))
            return false;
    }
    return true;
}

bool attention_window_tp2(const Activation *input, Activation *peer_input,
                          const AttentionWeights *primary, const AttentionWeights *peer,
                          int compress_ratio, int heads, int head_dim, int qk_rope, int groups,
                          int pos, float eps, KvCache *cache, KvCache *peer_cache,
                          Activation *output, Activation *peer_output) {
    if (!input || !peer_input || !primary || !peer || !cache || !peer_cache || !output ||
        !peer_output || heads <= 0 || head_dim <= 0)
        return false;
    auto run_rank = [&](const Activation *in, const AttentionWeights *w, KvCache *kv,
                        Activation *out, int rank) -> bool {
        if (!w)
            return false;
        Tensor *q_proj = w->q_a ? w->q_a : w->qkv;
        const int half = heads / 2;
        const int q_rows = w->q_b ? w->q_b->O : (w->qkv ? w->qkv->O : 0);
        const int wo_in = w->wo_a ? w->wo_a->I : 0;
        const bool sharded =
            half > 0 && (q_rows == half * head_dim || wo_in == half * head_dim);
        if (half <= 0) {
            return attention_window(in, w->attn_norm, q_proj, w->q_norm, w->q_b, w->wkv, w->kv_norm,
                                    w->sink, w->wo_a, w->wo_b, w->compress_wkv, w->compress_wgate,
                                    w->compress_ape, w->compress_norm, compress_ratio, heads,
                                    head_dim, qk_rope, groups, pos, eps, kv, out);
        }
        const int n0 = half;
        const int n1 = heads - half;
        const int n_heads = rank == 0 ? n0 : n1;
        const int head_begin = sharded ? 0 : (rank == 0 ? 0 : n0);
        const int use_heads = sharded ? n_heads : heads;
        const int g = groups >= 2 ? groups / 2 : groups;
        return attention_window_heads(in, w->attn_norm, q_proj, w->q_norm, w->q_b, w->wkv,
                                      w->kv_norm, w->sink, w->wo_a, w->wo_b, use_heads, head_dim,
                                      qk_rope, g, pos, eps, kv, out, head_begin, n_heads);
    };
    if (!run_rank(input, primary, cache, output, 0))
        return false;
    if (!run_rank(peer_input, peer, peer_cache, peer_output, 1))
        return false;
    if (heads >= 2)
        allreduce_pair(output, peer_output);
    return true;
}

bool route(const Activation *input, Tensor *gate, Tensor *bias, const int *fixed_ids,
           float routed_scale, int ids[6], float weights[6]) {
    if (!input || !gate || !ids || !weights || gate->O <= 0 || gate->I <= 0)
        return false;
    const int E = gate->O;
    const int H = gate->I;
    if (static_cast<int>(input->v.size()) < H)
        return false;
    std::vector<float> logits(static_cast<size_t>(E), 0.f);
    if (!matvec(gate, logits.data(), input->v.data()))
        return false;
    std::vector<float> score(static_cast<size_t>(E)), choice(static_cast<size_t>(E));
    const float *b = nullptr;
    if (bias && !bias->f32.empty())
        b = bias->f32.data();
    for (int e = 0; e < E; ++e) {
        score[static_cast<size_t>(e)] = sqrt_softplus(logits[static_cast<size_t>(e)]);
        const float bv = b && e < static_cast<int>(bias->f32.size()) ? b[e] : 0.f;
        choice[static_cast<size_t>(e)] = score[static_cast<size_t>(e)] + bv;
    }
    const int k = std::min(6, E);
    if (fixed_ids) {
        for (int i = 0; i < 6; ++i)
            ids[i] = i < k ? fixed_ids[i] : -1;
    } else {
        moe_topk(choice.data(), E, k, ids, weights, score.data());
        for (int i = k; i < 6; ++i)
            ids[i] = -1;
    }
    float sum = 0.f;
    for (int i = 0; i < 6; ++i) {
        const int e = ids[i];
        float w = (e >= 0 && e < E) ? score[static_cast<size_t>(e)] : 0.f;
        if (w < 0.f)
            w = 0.f;
        weights[i] = w;
        sum += w;
    }
    for (int i = 0; i < 6; ++i) {
        if (i >= k) {
            weights[i] = 0.f;
            continue;
        }
        weights[i] = (sum > 0.f ? weights[i] / sum : 1.f / static_cast<float>(k)) * routed_scale;
    }
    return true;
}

bool rmsnorm(Activation *x, Tensor *weight, float eps, int elements) {
    if (!x || elements <= 0 || static_cast<int>(x->v.size()) < elements)
        return false;
    std::vector<float> y(static_cast<size_t>(elements));
    if (!rmsnorm_tensor(y.data(), x->v.data(), weight, elements, eps))
        return false;
    std::memcpy(x->v.data(), y.data(), static_cast<size_t>(elements) * sizeof(float));
    return true;
}

bool moe_activation(Tensor *const *gate, Tensor *const *up, Tensor *const *down,
                    const float *weights, int count, Tensor *shared_gate, Tensor *shared_up,
                    Tensor *shared_down, float limit, const Activation *input, Activation *output) {
    if (!input || !output || input->v.empty())
        return false;
    if (static_cast<int>(output->v.size()) < static_cast<int>(input->v.size()))
        return false;
    return moe(gate, up, down, weights, count, shared_gate, shared_up, shared_down, limit,
               output->v.data(), input->v.data());
}

ExpertSet *expert_set_create(Tensor *const *gate, Tensor *const *up, Tensor *const *down, int count,
                             Tensor *shared_gate, Tensor *shared_up, Tensor *shared_down) {
    if (count < 0)
        return nullptr;
    auto *s = new ExpertSet();
    s->owns = false;
    s->count = count;
    s->gate.assign(gate, gate + count);
    s->up.assign(up, up + count);
    s->down.assign(down, down + count);
    s->sg = shared_gate;
    s->su = shared_up;
    s->sd = shared_down;
    return s;
}

ExpertSet *expert_bank_create(int count, int hidden, int intermediate, int device,
                              Tensor *shared_gate, Tensor *shared_up, Tensor *shared_down) {
    if (count <= 0 || hidden <= 0 || intermediate <= 0)
        return nullptr;
    auto *s = new ExpertSet();
    s->owns = true;
    s->count = count;
    s->hidden = hidden;
    s->intermediate = intermediate;
    s->device = device;
    s->gate.assign(static_cast<size_t>(count), nullptr);
    s->up.assign(static_cast<size_t>(count), nullptr);
    s->down.assign(static_cast<size_t>(count), nullptr);
    s->sg = shared_gate;
    s->su = shared_up;
    s->sd = shared_down;
    return s;
}

bool expert_bank_upload(ExpertSet *set, int expert, const uint8_t *gate_weight,
                        const uint8_t *gate_scale, const uint8_t *up_weight,
                        const uint8_t *up_scale, const uint8_t *down_weight,
                        const uint8_t *down_scale, Tensor **gate, Tensor **up, Tensor **down) {
    if (!set || expert < 0 || expert >= set->count || set->hidden <= 0)
        return false;
    const int H = set->hidden;
    const int O = set->intermediate;
    if (!upload_fp4(&set->gate[static_cast<size_t>(expert)], gate_weight, gate_scale, O, H,
                    set->device) ||
        !upload_fp4(&set->up[static_cast<size_t>(expert)], up_weight, up_scale, O, H, set->device) ||
        !upload_fp4(&set->down[static_cast<size_t>(expert)], down_weight, down_scale, H, O,
                    set->device))
        return false;
    if (gate)
        *gate = set->gate[static_cast<size_t>(expert)];
    if (up)
        *up = set->up[static_cast<size_t>(expert)];
    if (down)
        *down = set->down[static_cast<size_t>(expert)];
    return true;
}

bool expert_bank_set_shared(ExpertSet *set, Tensor *sg, Tensor *su, Tensor *sd) {
    if (!set)
        return false;
    set->sg = sg;
    set->su = su;
    set->sd = sd;
    return true;
}

bool expert_bank_upload_aux(ExpertSet *set, int expert, const uint8_t *gw, const uint8_t *gs,
                            const uint8_t *uw, const uint8_t *us, const uint8_t *dw,
                            const uint8_t *ds, Tensor **gate, Tensor **up, Tensor **down) {
    return expert_bank_upload(set, expert, gw, gs, uw, us, dw, ds, gate, up, down);
}

bool expert_bank_upload_tp2(ExpertSet *set, int expert, int rank, const uint8_t *gate_weight,
                            const uint8_t *gate_scale, const uint8_t *up_weight,
                            const uint8_t *up_scale, const uint8_t *down_weight,
                            const uint8_t *down_scale) {
    if (!set || expert < 0 || expert >= set->count || (rank != 0 && rank != 1) || !gate_weight ||
        !up_weight || !down_weight)
        return false;
    const int H = set->hidden;
    const int J = set->intermediate;
    if (H <= 0 || J <= 0 || (J & 1))
        return expert_bank_upload(set, expert, gate_weight, gate_scale, up_weight, up_scale,
                                  down_weight, down_scale, nullptr, nullptr, nullptr);
    const int packed_h = ceil_div(H, 2);
    const int scale_h = std::max(ceil_div(H, 32), 1);
    const uint8_t *gw = gate_weight + static_cast<size_t>(rank) * static_cast<size_t>(J) * packed_h;
    const uint8_t *uw = up_weight + static_cast<size_t>(rank) * static_cast<size_t>(J) * packed_h;
    const uint8_t *gs =
        gate_scale ? gate_scale + static_cast<size_t>(rank) * static_cast<size_t>(J) * scale_h
                   : nullptr;
    const uint8_t *us =
        up_scale ? up_scale + static_cast<size_t>(rank) * static_cast<size_t>(J) * scale_h : nullptr;
    const int packed_full = ceil_div(2 * J, 2);
    const int packed_half = ceil_div(J, 2);
    const int scale_full = std::max(ceil_div(2 * J, 32), 1);
    const int scale_half = std::max(ceil_div(J, 32), 1);
    std::vector<uint8_t> dw(static_cast<size_t>(H) * packed_half);
    std::vector<uint8_t> ds(static_cast<size_t>(H) * scale_half, 127);
    for (int o = 0; o < H; ++o) {
        std::memcpy(dw.data() + static_cast<size_t>(o) * packed_half,
                    down_weight + static_cast<size_t>(o) * packed_full +
                        static_cast<size_t>(rank) * packed_half,
                    static_cast<size_t>(packed_half));
        if (down_scale)
            std::memcpy(ds.data() + static_cast<size_t>(o) * scale_half,
                        down_scale + static_cast<size_t>(o) * scale_full +
                            static_cast<size_t>(rank) * scale_half,
                        static_cast<size_t>(scale_half));
    }
    if (!upload_fp4(&set->gate[static_cast<size_t>(expert)], gw, gs, J, H, set->device) ||
        !upload_fp4(&set->up[static_cast<size_t>(expert)], uw, us, J, H, set->device) ||
        !upload_fp4(&set->down[static_cast<size_t>(expert)], dw.data(), ds.data(), H, J,
                    set->device))
        return false;
    return true;
}

void expert_set_free(ExpertSet *set) {
    if (!set)
        return;
    if (set->owns) {
        for (int i = 0; i < set->count; ++i) {
            tensor_free(set->gate[static_cast<size_t>(i)]);
            tensor_free(set->up[static_cast<size_t>(i)]);
            tensor_free(set->down[static_cast<size_t>(i)]);
        }
    }
    delete set;
}

bool expert_set_upload_hash(ExpertSet *set, const int64_t *map, int vocab, int) {
    if (!set || !map || vocab < 0)
        return false;
    set->hash.assign(map, map + vocab);
    return true;
}

bool route_moe(const Activation *input, Tensor *gate, Tensor *bias, int, float routed_scale,
               ExpertSet *experts, float limit, Activation *output) {
    if (!input || !experts || !output)
        return false;
    int ids[6];
    float w[6];
    if (!route(input, gate, bias, nullptr, routed_scale, ids, w))
        return false;
    Tensor *g[6] = {}, *u[6] = {}, *d[6] = {};
    float ww[6] = {};
    int n = 0;
    for (int k = 0; k < 6; ++k) {
        const int e = ids[k];
        if (e < 0 || e >= experts->count || !experts->gate[static_cast<size_t>(e)])
            continue;
        g[n] = experts->gate[static_cast<size_t>(e)];
        u[n] = experts->up[static_cast<size_t>(e)];
        d[n] = experts->down[static_cast<size_t>(e)];
        ww[n] = w[k];
        ++n;
    }
    if (n == 0)
        return false;
    if (static_cast<int>(output->v.size()) < static_cast<int>(input->v.size()))
        return false;
    return moe(g, u, d, ww, n, experts->sg, experts->su, experts->sd, limit, output->v.data(),
               input->v.data());
}

bool route_top6_batch(const Activation *input, Tensor *gate, Tensor *bias, int count,
                      float routed_scale, int *ids, float *weights) {
    if (!input || !gate || !ids || !weights || count <= 0)
        return false;
    const int H = gate->I;
    for (int t = 0; t < count; ++t) {
        Activation row;
        row.v.assign(input->v.begin() + static_cast<size_t>(t) * H,
                     input->v.begin() + static_cast<size_t>(t + 1) * H);
        if (!route(&row, gate, bias, nullptr, routed_scale, ids + t * 6, weights + t * 6))
            return false;
    }
    return true;
}

bool route_moe_ids_batch(const Activation *input, const int *ids, const float *weights, int count,
                         ExpertSet *experts, float limit, Activation *output) {
    if (!input || !ids || !weights || !experts || !output || count <= 0)
        return false;
    const int H = static_cast<int>(input->v.size() / count);
    if (H <= 0 || static_cast<int>(output->v.size()) < count * H)
        return false;
    std::fill(output->v.begin(), output->v.end(), 0.f);
    for (int t = 0; t < count; ++t) {
        Tensor *g[6] = {}, *u[6] = {}, *d[6] = {};
        float ww[6] = {};
        int n = 0;
        for (int k = 0; k < 6; ++k) {
            const int e = ids[t * 6 + k];
            if (e < 0 || e >= experts->count || !experts->gate[static_cast<size_t>(e)])
                continue;
            g[n] = experts->gate[static_cast<size_t>(e)];
            u[n] = experts->up[static_cast<size_t>(e)];
            d[n] = experts->down[static_cast<size_t>(e)];
            ww[n] = weights[t * 6 + k];
            ++n;
        }
        if (n == 0)
            continue;
        if (!moe(g, u, d, ww, n, experts->sg, experts->su, experts->sd, limit,
                 output->v.data() + static_cast<size_t>(t) * H,
                 input->v.data() + static_cast<size_t>(t) * H))
            return false;
    }
    return true;
}

bool route_moe_batch(const Activation *input, Tensor *gate, Tensor *bias, const int *, int count,
                     float routed_scale, ExpertSet *experts, float limit, Activation *output) {
    if (count <= 0)
        return false;
    std::vector<int> ids(static_cast<size_t>(count) * 6, -1);
    std::vector<float> w(static_cast<size_t>(count) * 6, 0.f);
    if (!route_top6_batch(input, gate, bias, count, routed_scale, ids.data(), w.data()))
        return false;
    return route_moe_ids_batch(input, ids.data(), w.data(), count, experts, limit, output);
}

bool route_moe_ep2(const Activation *input, Tensor *gate, Tensor *bias,
                   const Activation *peer_input, Tensor *peer_gate, Tensor *peer_bias, int token,
                   float routed_scale, ExpertSet *local, ExpertSet *peer, float limit,
                   Activation *output, Activation *peer_output) {
    if (!input || !output || !peer_output || (!local && !peer))
        return false;
    const Activation *pin = peer_input ? peer_input : input;
    Tensor *pgate = peer_gate ? peer_gate : gate;
    Tensor *pbias = peer_bias ? peer_bias : bias;
    int ids[6] = {-1, -1, -1, -1, -1, -1};
    float w[6] = {};
    int ids_p[6] = {-1, -1, -1, -1, -1, -1};
    float w_p[6] = {};
    bool routed = false;
    if (gate && route(input, gate, bias, nullptr, routed_scale, ids, w))
        routed = true;
    if (pgate && route(pin, pgate, pbias, nullptr, routed_scale, ids_p, w_p))
        routed = true;
    else {
        std::memcpy(ids_p, ids, sizeof(ids));
        std::memcpy(w_p, w, sizeof(w));
    }
    if (!routed) {
        const bool a = local && route_moe(input, gate, bias, token, routed_scale, local, limit, output);
        const bool b =
            peer && route_moe(pin, pgate, pbias, token, routed_scale, peer, limit, peer_output);
        if (!a && !b)
            return false;
        if (!a)
            std::fill(output->v.begin(), output->v.end(), 0.f);
        if (!b)
            std::fill(peer_output->v.begin(), peer_output->v.end(), 0.f);
        allreduce_pair(output, peer_output);
        return true;
    }
    const bool have_local_shared = local && local->sg && local->su && local->sd;
    if (!moe_routed_parity(input, local, ids, w, 0, limit, output, have_local_shared))
        return false;
    if (!moe_routed_parity(pin, peer, ids_p, w_p, 1, limit, peer_output, !have_local_shared))
        return false;
    allreduce_pair(output, peer_output);
    return true;
}

} // namespace dsv4_cuda
} // namespace mvllm
