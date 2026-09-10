#include "h3_vision.hpp"
#include "family.hpp"
#include "../gpu/metal_h3.hpp"
#include "../io/safetensors.hpp"
#include "../quant/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <utility>

namespace mvllm {
namespace {

struct MergerW {
    std::vector<float> norm_w, norm_b;
    std::vector<float> fc1_w, fc1_b;
    std::vector<float> fc2_w, fc2_b;
};

struct BlockW {
    std::vector<float> norm1_w, norm1_b;
    std::vector<float> qkv_w, qkv_b;
    std::vector<float> proj_w, proj_b;
    std::vector<float> norm2_w, norm2_b;
    std::vector<float> fc1_w, fc1_b;
    std::vector<float> fc2_w, fc2_b;
};

struct VisionW {
    std::vector<float> pos_embed;
    std::vector<float> patch_w;
    std::vector<float> patch_b;
    std::vector<BlockW> blocks;
    MergerW merger;
    MergerW deepstack[kH3VisionDeepstacks];
    int patch_dim = 0;
    bool ready = false;
};

std::unordered_map<const H3VisionEncoder *, VisionW> &wmap() {
    static std::unordered_map<const H3VisionEncoder *, VisionW> m;
    return m;
}

VisionW &aw(const H3VisionEncoder *v) { return wmap()[v]; }

const VisionW *awc(const H3VisionEncoder *v) {
    auto it = wmap().find(v);
    return it == wmap().end() ? nullptr : &it->second;
}

int64_t numel_shape(const std::vector<int64_t> &sh) {
    int64_t n = 1;
    for (int64_t d : sh) {
        if (d <= 0)
            return 0;
        n *= d;
    }
    return n;
}

int isqrt_int(int n) {
    if (n <= 0)
        return 0;
    int r = static_cast<int>(std::sqrt(static_cast<double>(n)));
    while (r > 0 && r > n / r)
        --r;
    while (r + 1 <= n / (r + 1))
        ++r;
    return r;
}

bool perfect_square(int n, int &root) {
    root = isqrt_int(n);
    return root > 0 && root * root == n;
}

int pick_heads(int hidden) {
    if (hidden <= 0)
        return 1;
    if (hidden % 16 == 0 && hidden / 16 >= 4)
        return 16;
    int best_pref = 0;
    int best_any = 1;
    const int hmax = std::min(hidden, 16);
    for (int h = 1; h <= hmax; ++h) {
        if (hidden % h)
            continue;
        const int hd = hidden / h;
        if (hd % 2)
            continue;
        best_any = h;
        if (hd >= 4)
            best_pref = h;
    }
    return best_pref > 0 ? best_pref : best_any;
}

void ensure_metal_h3() {
    static const bool inited = metal_h3::init();
    (void)inited;
}

// Residual + RMS: x += skip; y = rmsnorm(x, w). False unless the site is RMS.
bool try_vae_rms_add(float *x, const float *skip, const float *w, float *y, int n, float eps) {
    ensure_metal_h3();
    if (!x || !y || !w || n < 1)
        return false;
    return metal_h3::vae_rms_add(x, skip, w, y, n, eps);
}

void add_bias(float *y, const float *b, int S, int O) {
    if (!y || !b || S <= 0 || O <= 0)
        return;
    for (int s = 0; s < S; ++s) {
        float *row = y + static_cast<size_t>(s) * O;
        for (int o = 0; o < O; ++o)
            row[o] += b[o];
    }
}

void linear(float *y, const float *x, const float *w, const float *b, int S, int I, int O) {
    if (!y || S <= 0 || O <= 0)
        return;
    if (!x || !w || I <= 0) {
        std::memset(y, 0, static_cast<size_t>(S) * static_cast<size_t>(O) * sizeof(float));
        add_bias(y, b, S, O);
        return;
    }
    quant::matmul_f32(y, x, w, S, I, O);
    add_bias(y, b, S, O);
}

bool weight_ok(const std::vector<float> &w, int O, int I) {
    return O > 0 && I > 0 && static_cast<int>(w.size()) >= O * I;
}

float gelu_tanh(float x) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    if (inner <= -10.f)
        return 0.f;
    if (inner >= 10.f)
        return x;
    return 0.5f * x * (1.f + std::tanh(inner));
}

float gelu_erf(float x) {
    if (x <= -10.f)
        return 0.f;
    if (x >= 10.f)
        return x;
    return 0.5f * x * (1.f + std::erff(x * 0.7071067811865475f));
}

void gelu_rows(float *x, int n, bool approx) {
    if (!x || n <= 0)
        return;
    if (approx) {
        for (int i = 0; i < n; ++i)
            x[i] = gelu_tanh(x[i]);
    } else {
        for (int i = 0; i < n; ++i)
            x[i] = gelu_erf(x[i]);
    }
}

io::StHit find_pref(const std::vector<io::StFile> &files, const std::string &P, const std::string &name) {
    io::StHit h = io::st_find_dir(files, P + name);
    if (h.tensor)
        return h;
    if (!P.empty())
        return io::st_find_dir(files, name);
    return h;
}

bool read_hit(const io::StHit &hit, std::vector<float> &dst, std::string &err) {
    if (!hit.tensor || !hit.file)
        return false;
    const int64_t n = numel_shape(hit.tensor->shape);
    if (n <= 0)
        return false;
    dst.assign(static_cast<size_t>(n), 0.f);
    return io::st_read_f32(*hit.file, *hit.tensor, dst.data(), n, err) == Status::Ok;
}

bool read_name(const std::vector<io::StFile> &files, const std::string &P, const std::string &name,
               std::vector<float> &dst, std::string &err) {
    return read_hit(find_pref(files, P, name), dst, err);
}

void infer_deepstack_after(H3VisionConfig &cfg) {
    const int L = cfg.layers;
    if (L > kH3VisionDeepstackAfter[2]) {
        cfg.deepstack_after[0] = kH3VisionDeepstackAfter[0];
        cfg.deepstack_after[1] = kH3VisionDeepstackAfter[1];
        cfg.deepstack_after[2] = kH3VisionDeepstackAfter[2];
        return;
    }
    const int n = std::min(3, std::max(L, 0));
    const int last = L > 0 ? L - 1 : 0;
    for (int i = 0; i < 3; ++i) {
        if (i < n)
            cfg.deepstack_after[i] = L - n + i;
        else
            cfg.deepstack_after[i] = last;
    }
}

void prepare_patch_rows(const float *rgb_hwc, int frames, int height, int width, int patch,
                        int temporal_patch, int merge, float *rows) {
    const int grid_h = height / patch;
    const int grid_w = width / patch;
    size_t out = 0;
    for (int block_h = 0; block_h < grid_h / merge; ++block_h)
        for (int block_w = 0; block_w < grid_w / merge; ++block_w)
            for (int inner_h = 0; inner_h < merge; ++inner_h)
                for (int inner_w = 0; inner_w < merge; ++inner_w)
                    for (int channel = 0; channel < 3; ++channel)
                        for (int temporal = 0; temporal < temporal_patch; ++temporal)
                            for (int patch_h = 0; patch_h < patch; ++patch_h)
                                for (int patch_w = 0; patch_w < patch; ++patch_w) {
                                    const int frame = frames == 1 ? 0 : temporal;
                                    const int y = (block_h * merge + inner_h) * patch + patch_h;
                                    const int x = (block_w * merge + inner_w) * patch + patch_w;
                                    const size_t idx = (static_cast<size_t>(frame) * height +
                                                        static_cast<size_t>(y)) *
                                                           static_cast<size_t>(width) +
                                                       static_cast<size_t>(x);
                                    rows[out++] = rgb_hwc[idx * 3u + static_cast<size_t>(channel)] * 2.f - 1.f;
                                }
}

void prepare_position_rows(const float *table, int pos_side, int hidden, int grid_h, int grid_w,
                           int merge, float *rows) {
    if (!table || !rows || hidden <= 0 || pos_side <= 0)
        return;
    size_t output_row = 0;
    const float hden = grid_h > 1 ? static_cast<float>(grid_h - 1) : 1.f;
    const float wden = grid_w > 1 ? static_cast<float>(grid_w - 1) : 1.f;
    for (int block_h = 0; block_h < grid_h / merge; ++block_h)
        for (int block_w = 0; block_w < grid_w / merge; ++block_w)
            for (int inner_h = 0; inner_h < merge; ++inner_h)
                for (int inner_w = 0; inner_w < merge; ++inner_w) {
                    const int row = block_h * merge + inner_h;
                    const int col = block_w * merge + inner_w;
                    const float hv = grid_h > 1 ? static_cast<float>(row) * static_cast<float>(pos_side - 1) / hden
                                                : 0.f;
                    const float wv = grid_w > 1 ? static_cast<float>(col) * static_cast<float>(pos_side - 1) / wden
                                                : 0.f;
                    int h0 = static_cast<int>(hv);
                    int w0 = static_cast<int>(wv);
                    if (h0 < 0)
                        h0 = 0;
                    if (w0 < 0)
                        w0 = 0;
                    if (h0 >= pos_side)
                        h0 = pos_side - 1;
                    if (w0 >= pos_side)
                        w0 = pos_side - 1;
                    const int h1 = h0 + 1 < pos_side ? h0 + 1 : h0;
                    const int w1 = w0 + 1 < pos_side ? w0 + 1 : w0;
                    const float dh = hv - static_cast<float>(h0);
                    const float dw = wv - static_cast<float>(w0);
                    const float coef[4] = {(1.f - dh) * (1.f - dw), (1.f - dh) * dw, dh * (1.f - dw), dh * dw};
                    const int idx[4] = {h0 * pos_side + w0, h0 * pos_side + w1, h1 * pos_side + w0,
                                        h1 * pos_side + w1};
                    float *dst = rows + output_row * static_cast<size_t>(hidden);
                    for (int ch = 0; ch < hidden; ++ch) {
                        float sum = 0.f;
                        for (int corner = 0; corner < 4; ++corner)
                            sum += table[static_cast<size_t>(idx[corner]) * hidden + static_cast<size_t>(ch)] *
                                   coef[corner];
                        dst[ch] = sum;
                    }
                    ++output_row;
                }
}

void prepare_rope(int grid_h, int grid_w, int merge, int rope_half, float *cosines, float *sines) {
    if (!cosines || !sines || rope_half <= 0)
        return;
    size_t row = 0;
    const int axis_freqs = rope_half / 2;
    for (int block_h = 0; block_h < grid_h / merge; ++block_h)
        for (int block_w = 0; block_w < grid_w / merge; ++block_w)
            for (int inner_h = 0; inner_h < merge; ++inner_h)
                for (int inner_w = 0; inner_w < merge; ++inner_w) {
                    const float positions[2] = {static_cast<float>(block_h * merge + inner_h),
                                                static_cast<float>(block_w * merge + inner_w)};
                    float *c = cosines + row * static_cast<size_t>(rope_half);
                    float *s = sines + row * static_cast<size_t>(rope_half);
                    std::memset(c, 0, static_cast<size_t>(rope_half) * sizeof(float));
                    std::memset(s, 0, static_cast<size_t>(rope_half) * sizeof(float));
                    if (axis_freqs > 0) {
                        for (int axis = 0; axis < 2; ++axis) {
                            for (int index = 0; index < axis_freqs; ++index) {
                                const float inv =
                                    std::pow(10000.f, -static_cast<float>(index * 2) / static_cast<float>(rope_half));
                                const float angle = positions[axis] * inv;
                                const size_t off = static_cast<size_t>(axis) * axis_freqs + static_cast<size_t>(index);
                                c[off] = std::cos(angle);
                                s[off] = std::sin(angle);
                            }
                        }
                    } else {
                        // rope_half==1: one frequency from the height axis
                        const float angle = positions[0];
                        c[0] = std::cos(angle);
                        s[0] = std::sin(angle);
                    }
                    ++row;
                }
}

void apply_qkv_rope(const float *qkv, const float *cosines, const float *sines, float *query, float *key,
                    float *value, int seq, int heads, int hd, int rope_half) {
    const int inner = heads * hd;
    for (int row = 0; row < seq; ++row) {
        const float *crow = cosines + static_cast<size_t>(row) * std::max(rope_half, 1);
        const float *srow = sines + static_cast<size_t>(row) * std::max(rope_half, 1);
        for (int head = 0; head < heads; ++head) {
            const float *qs = qkv + (static_cast<size_t>(row) * 3 + 0) * inner + static_cast<size_t>(head) * hd;
            const float *ks = qkv + (static_cast<size_t>(row) * 3 + 1) * inner + static_cast<size_t>(head) * hd;
            const float *vs = qkv + (static_cast<size_t>(row) * 3 + 2) * inner + static_cast<size_t>(head) * hd;
            float *qd = query + (static_cast<size_t>(row) * heads + head) * hd;
            float *kd = key + (static_cast<size_t>(row) * heads + head) * hd;
            float *vd = value + (static_cast<size_t>(row) * heads + head) * hd;
            for (int dim = 0; dim < hd; ++dim)
                vd[dim] = vs[dim];
            for (int dim = 0; dim < hd; ++dim) {
                const int half = rope_half;
                const int pair = dim < half ? dim + half : dim - half;
                if (half <= 0 || pair < 0 || pair >= hd) {
                    qd[dim] = qs[dim];
                    kd[dim] = ks[dim];
                    continue;
                }
                const int rope_index = dim % half;
                const float c = crow[rope_index];
                const float s = srow[rope_index];
                const float q0 = qs[dim], q1 = qs[pair];
                const float k0 = ks[dim], k1 = ks[pair];
                if (dim < half) {
                    qd[dim] = q0 * c - q1 * s;
                    kd[dim] = k0 * c - k1 * s;
                } else {
                    qd[dim] = q0 * c + q1 * s;
                    kd[dim] = k0 * c + k1 * s;
                }
            }
        }
    }
}

void sdpa(const float *query, const float *key, const float *value, float *out, int seq, int heads, int hd) {
    const float scale = 1.f / std::sqrt(static_cast<float>(std::max(hd, 1)));
    std::vector<float> scores(static_cast<size_t>(std::max(seq, 0)));
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < seq; ++i) {
            const float *qi = query + (static_cast<size_t>(i) * heads + h) * hd;
            for (int j = 0; j < seq; ++j) {
                const float *kj = key + (static_cast<size_t>(j) * heads + h) * hd;
                float dot = 0.f;
                for (int d = 0; d < hd; ++d)
                    dot += qi[d] * kj[d];
                scores[static_cast<size_t>(j)] = dot * scale;
            }
            quant::softmax_inplace(scores.data(), seq);
            float *oi = out + (static_cast<size_t>(i) * heads + h) * hd;
            std::memset(oi, 0, static_cast<size_t>(hd) * sizeof(float));
            for (int j = 0; j < seq; ++j) {
                const float *vj = value + (static_cast<size_t>(j) * heads + h) * hd;
                const float a = scores[static_cast<size_t>(j)];
                for (int d = 0; d < hd; ++d)
                    oi[d] += a * vj[d];
            }
        }
    }
}

void run_block(const BlockW &b, float *hidden, int rows, const H3VisionConfig &cfg, const float *rope_cos,
               const float *rope_sin) {
    const int H = cfg.hidden;
    const int I = cfg.intermediate;
    const int heads = cfg.heads;
    const int hd = cfg.head_dim;
    const int rope_half = cfg.rope_half;
    if (H <= 0 || rows <= 0)
        return;

    ensure_metal_h3();
    const float *n1w = b.norm1_w.size() >= static_cast<size_t>(H) ? b.norm1_w.data() : nullptr;
    const float *n1b = b.norm1_b.size() >= static_cast<size_t>(H) ? b.norm1_b.data() : nullptr;
    const float *n2w = b.norm2_w.size() >= static_cast<size_t>(H) ? b.norm2_w.data() : nullptr;
    const float *n2b = b.norm2_b.size() >= static_cast<size_t>(H) ? b.norm2_b.data() : nullptr;
    const float *qkv = weight_ok(b.qkv_w, 3 * H, H) ? b.qkv_w.data() : nullptr;
    const float *qkv_b = b.qkv_b.size() >= static_cast<size_t>(3 * H) ? b.qkv_b.data() : nullptr;
    const float *proj = weight_ok(b.proj_w, H, H) ? b.proj_w.data() : nullptr;
    const float *proj_b = b.proj_b.size() >= static_cast<size_t>(H) ? b.proj_b.data() : nullptr;
    const float *fc1 = (I > 0 && weight_ok(b.fc1_w, I, H)) ? b.fc1_w.data() : nullptr;
    const float *fc1_b = b.fc1_b.size() >= static_cast<size_t>(I) ? b.fc1_b.data() : nullptr;
    const float *fc2 = (I > 0 && weight_ok(b.fc2_w, H, I)) ? b.fc2_w.data() : nullptr;
    const float *fc2_b = b.fc2_b.size() >= static_cast<size_t>(H) ? b.fc2_b.data() : nullptr;
    if (metal_h3::vision_block(hidden, rows, H, heads, hd, I, n1w, n1b, qkv, qkv_b, proj, proj_b,
                               n2w, n2b, fc1, fc1_b, fc2, fc2_b, rope_cos, rope_sin, rope_half,
                               cfg.ln_eps))
        return;

    std::vector<float> norm(static_cast<size_t>(rows) * H);
    for (int r = 0; r < rows; ++r)
        layernorm(hidden + static_cast<size_t>(r) * H, n1w, n1b, norm.data() + static_cast<size_t>(r) * H, H,
                  cfg.ln_eps);

    if (weight_ok(b.qkv_w, 3 * H, H) && heads > 0 && hd > 0 && heads * hd == H) {
        std::vector<float> qkv(static_cast<size_t>(rows) * 3 * H);
        std::vector<float> query(static_cast<size_t>(rows) * H);
        std::vector<float> key(static_cast<size_t>(rows) * H);
        std::vector<float> value(static_cast<size_t>(rows) * H);
        std::vector<float> attn(static_cast<size_t>(rows) * H);
        std::vector<float> branch(static_cast<size_t>(rows) * H);
        linear(qkv.data(), norm.data(), b.qkv_w.data(),
               b.qkv_b.size() >= static_cast<size_t>(3 * H) ? b.qkv_b.data() : nullptr, rows, H, 3 * H);
        apply_qkv_rope(qkv.data(), rope_cos, rope_sin, query.data(), key.data(), value.data(), rows, heads, hd,
                       rope_half);
        sdpa(query.data(), key.data(), value.data(), attn.data(), rows, heads, hd);
        if (weight_ok(b.proj_w, H, H))
            linear(branch.data(), attn.data(), b.proj_w.data(),
                   b.proj_b.size() >= static_cast<size_t>(H) ? b.proj_b.data() : nullptr, rows, H, H);
        else
            std::memcpy(branch.data(), attn.data(), static_cast<size_t>(rows) * H * sizeof(float));
        // Attn residual then LayerNorm (mean/var + bias) — not RMS.
        if (!try_vae_rms_add(hidden, branch.data(), nullptr, nullptr, rows * H, cfg.ln_eps)) {
            for (size_t i = 0; i < static_cast<size_t>(rows) * H; ++i)
                hidden[i] += branch[i];
        }
    }

    for (int r = 0; r < rows; ++r)
        layernorm(hidden + static_cast<size_t>(r) * H, n2w, n2b, norm.data() + static_cast<size_t>(r) * H, H,
                  cfg.ln_eps);
    if (I > 0 && weight_ok(b.fc1_w, I, H) && weight_ok(b.fc2_w, H, I)) {
        std::vector<float> fc1(static_cast<size_t>(rows) * I);
        std::vector<float> branch(static_cast<size_t>(rows) * H);
        linear(fc1.data(), norm.data(), b.fc1_w.data(),
               b.fc1_b.size() >= static_cast<size_t>(I) ? b.fc1_b.data() : nullptr, rows, H, I);
        gelu_rows(fc1.data(), rows * I, true);
        linear(branch.data(), fc1.data(), b.fc2_w.data(),
               b.fc2_b.size() >= static_cast<size_t>(H) ? b.fc2_b.data() : nullptr, rows, I, H);
        // GELU MLP residual then LayerNorm — not nax / RMS.
        if (!try_vae_rms_add(hidden, branch.data(), nullptr, nullptr, rows * H, cfg.ln_eps)) {
            for (size_t i = 0; i < static_cast<size_t>(rows) * H; ++i)
                hidden[i] += branch[i];
        }
    }
}

void pool_to_out(const float *src, int rows, int in_w, int out_w, float *dst) {
    for (int r = 0; r < rows; ++r) {
        const float *in = src + static_cast<size_t>(r) * in_w;
        float *o = dst + static_cast<size_t>(r) * out_w;
        if (in_w == out_w) {
            std::memcpy(o, in, static_cast<size_t>(out_w) * sizeof(float));
            continue;
        }
        for (int j = 0; j < out_w; ++j) {
            if (in_w <= 0) {
                o[j] = 0.f;
                continue;
            }
            if (out_w == in_w) {
                o[j] = in[j];
            } else if (out_w < in_w) {
                const int i0 = (j * in_w) / out_w;
                int i1 = ((j + 1) * in_w) / out_w;
                if (i1 <= i0)
                    i1 = i0 + 1;
                float s = 0.f;
                for (int i = i0; i < i1 && i < in_w; ++i)
                    s += in[i];
                o[j] = s / static_cast<float>(i1 - i0);
            } else {
                o[j] = in[j % in_w];
            }
        }
    }
}

void run_merger(const MergerW &m, const float *hidden, int patch_rows, int hidden_w, int merge, int out_w,
                bool deepstack, float eps, float *dst) {
    const int group = merge * merge;
    if (patch_rows <= 0 || hidden_w <= 0 || group <= 0 || out_w <= 0 || !dst)
        return;
    const int merged_rows = patch_rows / group;
    if (merged_rows <= 0)
        return;
    const int packed = hidden_w * group;
    int merge_dim = packed;
    if (weight_ok(m.fc1_w, 1, 1)) {
        const int side = static_cast<int>(std::sqrt(static_cast<double>(m.fc1_w.size())));
        if (side > 0 && side * side == static_cast<int>(m.fc1_w.size()) && side <= packed)
            merge_dim = side;
    } else if (weight_ok(m.fc2_w, out_w, 1)) {
        const int in = static_cast<int>(m.fc2_w.size() / out_w);
        if (in > 0 && in <= packed)
            merge_dim = in;
    }

    const int norm_width = deepstack ? merge_dim : hidden_w;
    const int norm_rows = deepstack ? merged_rows : patch_rows;
    std::vector<float> normed(static_cast<size_t>(std::max(norm_rows, 0)) * std::max(norm_width, 0), 0.f);
    const float *nw = m.norm_w.size() >= static_cast<size_t>(norm_width) ? m.norm_w.data() : nullptr;
    const float *nb = m.norm_b.size() >= static_cast<size_t>(norm_width) ? m.norm_b.data() : nullptr;
    if (hidden && norm_width > 0) {
        for (int r = 0; r < norm_rows; ++r)
            layernorm(hidden + static_cast<size_t>(r) * norm_width, nw, nb,
                      normed.data() + static_cast<size_t>(r) * norm_width, norm_width, eps);
    }

    std::vector<float> act(static_cast<size_t>(merged_rows) * merge_dim, 0.f);
    const float *fc1_in = normed.data();
    if (weight_ok(m.fc1_w, merge_dim, merge_dim)) {
        linear(act.data(), fc1_in, m.fc1_w.data(),
               m.fc1_b.size() >= static_cast<size_t>(merge_dim) ? m.fc1_b.data() : nullptr, merged_rows, merge_dim,
               merge_dim);
        gelu_rows(act.data(), merged_rows * merge_dim, false);
    } else {
        const size_t nbytes =
            static_cast<size_t>(merged_rows) * static_cast<size_t>(merge_dim) * sizeof(float);
        if (!normed.empty())
            std::memcpy(act.data(), normed.data(), std::min(nbytes, normed.size() * sizeof(float)));
    }

    if (weight_ok(m.fc2_w, out_w, merge_dim)) {
        linear(dst, act.data(), m.fc2_w.data(), m.fc2_b.size() >= static_cast<size_t>(out_w) ? m.fc2_b.data() : nullptr,
               merged_rows, merge_dim, out_w);
    } else {
        pool_to_out(act.data(), merged_rows, merge_dim, out_w, dst);
    }
}

bool merger_present(const MergerW &m) {
    return !m.fc2_w.empty() || !m.fc1_w.empty() || !m.norm_w.empty();
}

void load_merger(const std::vector<io::StFile> &files, const std::string &P, const std::string &prefix, MergerW &m,
                 std::string &err) {
    read_name(files, P, prefix + "norm.weight", m.norm_w, err);
    read_name(files, P, prefix + "norm.bias", m.norm_b, err);
    read_name(files, P, prefix + "linear_fc1.weight", m.fc1_w, err);
    read_name(files, P, prefix + "linear_fc1.bias", m.fc1_b, err);
    read_name(files, P, prefix + "linear_fc2.weight", m.fc2_w, err);
    read_name(files, P, prefix + "linear_fc2.bias", m.fc2_b, err);
}

} // namespace

Status H3VisionEncoder::load(const std::string &model_dir, std::string &err) {
    ready_ = false;
    from_checkpoint_ = false;
    cfg_ = H3VisionConfig{};
    wmap().erase(this);
    err.clear();

    const char *tails[] = {"/FL2VA/text_encoder", "/Ref2VA/text_encoder", "/text_encoder", "/qwen", ""};
    std::vector<io::StFile> files;
    std::string prefix;
    bool hit = false;
    for (const char *tail : tails) {
        files.clear();
        std::string oerr;
        Status st = io::st_open_dir(model_dir + tail, files, oerr);
        if (st != Status::Ok || files.empty()) {
            if (!files.empty())
                io::st_close_dir(files);
            continue;
        }
        auto has = [&](const std::string &n) { return io::st_find_dir(files, n).tensor != nullptr; };
        if (has("model.visual.patch_embed.proj.weight")) {
            prefix = "model.visual.";
            hit = true;
            break;
        }
        if (has("visual.patch_embed.proj.weight")) {
            prefix = "visual.";
            hit = true;
            break;
        }
        if (has("patch_embed.proj.weight")) {
            prefix = "";
            hit = true;
            break;
        }
        io::st_close_dir(files);
    }
    if (!hit) {
        err.clear();
        return Status::Ok;
    }

    auto has = [&](const std::string &n) { return find_pref(files, prefix, n).tensor != nullptr; };

    io::StHit patch = find_pref(files, prefix, "patch_embed.proj.weight");
    if (!patch.tensor || patch.tensor->shape.empty()) {
        io::st_close_dir(files);
        err.clear();
        return Status::Ok;
    }

    H3VisionConfig cfg = H3VisionConfig{};
    cfg.hidden = static_cast<int>(patch.tensor->shape[0]);
    if (cfg.hidden <= 0) {
        io::st_close_dir(files);
        err = "h3 vision patch_embed has empty out dim";
        return Status::ParseError;
    }

    cfg.temporal_patch = kH3VisionTemporalPatch;
    cfg.patch = kH3VisionPatch;
    if (patch.tensor->shape.size() >= 5) {
        cfg.temporal_patch = static_cast<int>(patch.tensor->shape[2]);
        cfg.patch = static_cast<int>(patch.tensor->shape[3]);
        if (cfg.temporal_patch <= 0)
            cfg.temporal_patch = kH3VisionTemporalPatch;
        if (cfg.patch <= 0)
            cfg.patch = kH3VisionPatch;
    } else if (patch.tensor->shape.size() >= 2) {
        const int inner = static_cast<int>(patch.tensor->shape[1]);
        int p = 0;
        if (inner % 6 == 0 && perfect_square(inner / 6, p)) {
            cfg.temporal_patch = 2;
            cfg.patch = p;
        } else if (inner % 3 == 0 && perfect_square(inner / 3, p)) {
            cfg.temporal_patch = 1;
            cfg.patch = p;
        }
    }

    cfg.layers = 0;
    for (int i = 0; i < 64; ++i) {
        if (!has("blocks." + std::to_string(i) + ".attn.qkv.weight"))
            break;
        cfg.layers = i + 1;
    }

    cfg.heads = pick_heads(cfg.hidden);
    if (cfg.heads < 1)
        cfg.heads = 1;
    cfg.head_dim = cfg.hidden / cfg.heads;
    if (cfg.head_dim <= 0) {
        cfg.heads = 1;
        cfg.head_dim = cfg.hidden;
    }
    if (cfg.head_dim % 2 == 0)
        cfg.rope_half = cfg.head_dim / 2;
    else
        cfg.rope_half = cfg.head_dim;
    if (cfg.hidden == kH3VisionHidden && cfg.heads == kH3VisionHeads) {
        cfg.head_dim = kH3VisionHeadDim;
        cfg.rope_half = kH3VisionRopeHalf;
    }

    io::StHit fc1 = find_pref(files, prefix, "blocks.0.mlp.linear_fc1.weight");
    if (fc1.tensor && !fc1.tensor->shape.empty())
        cfg.intermediate = static_cast<int>(fc1.tensor->shape[0]);
    else
        cfg.intermediate = cfg.hidden;

    cfg.merge = kH3VisionMerge;
    io::StHit mfc1 = find_pref(files, prefix, "merger.linear_fc1.weight");
    if (mfc1.tensor && !mfc1.tensor->shape.empty() && cfg.hidden > 0) {
        const int md = static_cast<int>(mfc1.tensor->shape[0]);
        if (md > 0 && md % cfg.hidden == 0) {
            int root = 0;
            if (perfect_square(md / cfg.hidden, root))
                cfg.merge = root;
        }
    }
    if (cfg.merge <= 0)
        cfg.merge = kH3VisionMerge;

    cfg.out_width = cfg.hidden;
    io::StHit mfc2 = find_pref(files, prefix, "merger.linear_fc2.weight");
    if (mfc2.tensor && !mfc2.tensor->shape.empty())
        cfg.out_width = static_cast<int>(mfc2.tensor->shape[0]);
    else {
        io::StHit dfc2 = find_pref(files, prefix, "deepstack_merger_list.0.linear_fc2.weight");
        if (dfc2.tensor && !dfc2.tensor->shape.empty())
            cfg.out_width = static_cast<int>(dfc2.tensor->shape[0]);
    }
    if (cfg.out_width <= 0)
        cfg.out_width = cfg.hidden;

    cfg.pos_side = kH3VisionPosSide;
    io::StHit pos = find_pref(files, prefix, "pos_embed.weight");
    if (pos.tensor && !pos.tensor->shape.empty()) {
        int count = 0;
        if (pos.tensor->shape.size() >= 2)
            count = static_cast<int>(pos.tensor->shape[0]);
        else if (cfg.hidden > 0)
            count = static_cast<int>(numel_shape(pos.tensor->shape) / cfg.hidden);
        int side = 0;
        if (perfect_square(count, side))
            cfg.pos_side = side;
    }

    infer_deepstack_after(cfg);

    VisionW w;
    w.patch_dim = 3 * cfg.temporal_patch * cfg.patch * cfg.patch;
    std::string rerr;
    if (!read_hit(patch, w.patch_w, rerr) || w.patch_w.empty()) {
        io::st_close_dir(files);
        err = rerr.empty() ? "h3 vision cannot read patch_embed.proj.weight" : rerr;
        return Status::ParseError;
    }
    if (cfg.hidden > 0 && static_cast<int>(w.patch_w.size()) >= cfg.hidden)
        w.patch_dim = static_cast<int>(w.patch_w.size() / cfg.hidden);
    read_name(files, prefix, "patch_embed.proj.bias", w.patch_b, rerr);
    read_name(files, prefix, "pos_embed.weight", w.pos_embed, rerr);

    w.blocks.resize(static_cast<size_t>(std::max(cfg.layers, 0)));
    for (int i = 0; i < cfg.layers; ++i) {
        const std::string B = "blocks." + std::to_string(i) + ".";
        BlockW &b = w.blocks[static_cast<size_t>(i)];
        read_name(files, prefix, B + "norm1.weight", b.norm1_w, rerr);
        read_name(files, prefix, B + "norm1.bias", b.norm1_b, rerr);
        read_name(files, prefix, B + "attn.qkv.weight", b.qkv_w, rerr);
        read_name(files, prefix, B + "attn.qkv.bias", b.qkv_b, rerr);
        read_name(files, prefix, B + "attn.proj.weight", b.proj_w, rerr);
        read_name(files, prefix, B + "attn.proj.bias", b.proj_b, rerr);
        read_name(files, prefix, B + "norm2.weight", b.norm2_w, rerr);
        read_name(files, prefix, B + "norm2.bias", b.norm2_b, rerr);
        read_name(files, prefix, B + "mlp.linear_fc1.weight", b.fc1_w, rerr);
        read_name(files, prefix, B + "mlp.linear_fc1.bias", b.fc1_b, rerr);
        read_name(files, prefix, B + "mlp.linear_fc2.weight", b.fc2_w, rerr);
        read_name(files, prefix, B + "mlp.linear_fc2.bias", b.fc2_b, rerr);
    }

    load_merger(files, prefix, "merger.", w.merger, rerr);
    for (int k = 0; k < kH3VisionDeepstacks; ++k)
        load_merger(files, prefix, "deepstack_merger_list." + std::to_string(k) + ".", w.deepstack[k], rerr);

    io::st_close_dir(files);
    w.ready = true;
    aw(this) = std::move(w);
    cfg_ = cfg;
    ready_ = true;
    from_checkpoint_ = true;
    err.clear();
    return Status::Ok;
}

void H3VisionEncoder::encode(const float *rgb_hwc, int frames, int height, int width, H3VisionOut &out) const {
    ensure_metal_h3();
    out = H3VisionOut{};
    if (!ready_ || !rgb_hwc)
        return;
    if ((frames != 1 && frames != 2) || height < 32 || width < 32 || (height % 32) || (width % 32))
        return;

    const VisionW *wp = awc(this);
    if (!wp || !wp->ready)
        return;
    const VisionW &w = *wp;
    const H3VisionConfig &c = cfg_;
    const int patch = c.patch > 0 ? c.patch : kH3VisionPatch;
    const int merge = c.merge > 0 ? c.merge : kH3VisionMerge;
    const int temporal = c.temporal_patch > 0 ? c.temporal_patch : kH3VisionTemporalPatch;
    const int hidden = c.hidden;
    if (patch <= 0 || merge <= 0 || hidden <= 0 || height % patch || width % patch)
        return;
    const int grid_h = height / patch;
    const int grid_w = width / patch;
    if (grid_h % merge || grid_w % merge || grid_h <= 0 || grid_w <= 0)
        return;

    const int rows = grid_h * grid_w;
    const int tokens = (grid_h / merge) * (grid_w / merge);
    const int out_w = c.out_width > 0 ? c.out_width : hidden;
    const int patch_dim = w.patch_dim > 0 ? w.patch_dim : 3 * temporal * patch * patch;
    if (tokens <= 0 || rows <= 0 || patch_dim <= 0 || out_w <= 0)
        return;

    out.grid_h = grid_h;
    out.grid_w = grid_w;
    out.tokens = tokens;
    out.out_width = out_w;
    out.merged.assign(static_cast<size_t>(tokens) * out_w, 0.f);
    for (int i = 0; i < kH3VisionDeepstacks; ++i)
        out.deepstack[i].assign(static_cast<size_t>(tokens) * out_w, 0.f);

    std::vector<float> patches(static_cast<size_t>(rows) * patch_dim, 0.f);
    prepare_patch_rows(rgb_hwc, frames, height, width, patch, temporal, merge, patches.data());

    std::vector<float> hidden_s(static_cast<size_t>(rows) * hidden, 0.f);
    if (weight_ok(w.patch_w, hidden, patch_dim)) {
        linear(hidden_s.data(), patches.data(), w.patch_w.data(),
               w.patch_b.size() >= static_cast<size_t>(hidden) ? w.patch_b.data() : nullptr, rows, patch_dim, hidden);
    } else {
        pool_to_out(patches.data(), rows, patch_dim, hidden, hidden_s.data());
    }

    if (!w.pos_embed.empty() && c.pos_side > 0 &&
        static_cast<int>(w.pos_embed.size()) >= c.pos_side * c.pos_side * hidden) {
        std::vector<float> pos(static_cast<size_t>(rows) * hidden, 0.f);
        prepare_position_rows(w.pos_embed.data(), c.pos_side, hidden, grid_h, grid_w, merge, pos.data());
        for (size_t i = 0; i < hidden_s.size(); ++i)
            hidden_s[i] += pos[i];
    }

    const int rope_half = c.rope_half > 0 ? c.rope_half : std::max(c.head_dim / 2, 1);
    std::vector<float> rope_cos(static_cast<size_t>(rows) * std::max(rope_half, 1), 0.f);
    std::vector<float> rope_sin(static_cast<size_t>(rows) * std::max(rope_half, 1), 0.f);
    prepare_rope(grid_h, grid_w, merge, rope_half, rope_cos.data(), rope_sin.data());

    auto take_deepstack = [&](int layer) {
        for (int k = 0; k < kH3VisionDeepstacks; ++k) {
            if (c.deepstack_after[k] != layer)
                continue;
            if (!merger_present(w.deepstack[k]))
                continue;
            run_merger(w.deepstack[k], hidden_s.data(), rows, hidden, merge, out_w, true, c.ln_eps,
                       out.deepstack[k].data());
        }
    };

    if (c.layers <= 0)
        take_deepstack(0);
    for (int layer = 0; layer < c.layers; ++layer) {
        if (layer < static_cast<int>(w.blocks.size()))
            run_block(w.blocks[static_cast<size_t>(layer)], hidden_s.data(), rows, c, rope_cos.data(),
                      rope_sin.data());
        take_deepstack(layer);
    }

    if (merger_present(w.merger)) {
        run_merger(w.merger, hidden_s.data(), rows, hidden, merge, out_w, false, c.ln_eps, out.merged.data());
    } else {
        // LN over hidden, then view as merged groups and pool to out_width.
        std::vector<float> normed(hidden_s.size(), 0.f);
        for (int r = 0; r < rows; ++r)
            layernorm(hidden_s.data() + static_cast<size_t>(r) * hidden, nullptr, nullptr,
                      normed.data() + static_cast<size_t>(r) * hidden, hidden, c.ln_eps);
        const int merge_dim = hidden * merge * merge;
        pool_to_out(normed.data(), tokens, merge_dim, out_w, out.merged.data());
    }
}

} // namespace mvllm
