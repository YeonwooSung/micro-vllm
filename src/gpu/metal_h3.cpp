#include "metal_h3.hpp"

#if !defined(MVLLM_WITH_METAL)

#include "../model/family.hpp"
#include "../quant/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mvllm {
namespace metal_h3 {

bool init() { return true; }

void shutdown() {}

bool available() { return true; }

const char *backend_name() { return "cpu"; }

bool dit_residual(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                  int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                  int tokens, float eps, const float *adaln_mod, const float *q_norm,
                  const float *k_norm, const float *rope_cos, const float *rope_sin) {
    h3_dit_block_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn, head_dim,
                     x, tokens, eps, adaln_mod, q_norm, k_norm, rope_cos, rope_sin);
    return true;
}

bool gemm_int8(float *y, const float *x, const int8_t *w, const float *scale, int S, int I, int O) {
    if (!y || !x || !w || !scale || S < 1 || I < 1 || O < 1)
        return false;
    for (int s = 0; s < S; ++s) {
        const float *xs = x + static_cast<size_t>(s) * I;
        float *ys = y + static_cast<size_t>(s) * O;
        for (int o = 0; o < O; ++o) {
            const int8_t *wo = w + static_cast<size_t>(o) * I;
            float acc = 0.f;
            for (int i = 0; i < I; ++i)
                acc += xs[i] * static_cast<float>(wo[i]);
            ys[o] = acc * scale[o];
        }
    }
    return true;
}

bool nax_mlp(float *y, const float *x, const float *w_up, const float *w_down, int S, int D, int I) {
    if (!y || !x || !w_up || !w_down || S < 1 || D < 1 || I < 1)
        return false;
    std::vector<float> mid(static_cast<size_t>(S) * static_cast<size_t>(I), 0.f);
    quant::matmul_f32(mid.data(), x, w_up, S, D, I);
    for (int i = 0; i < S * I; ++i)
        mid[static_cast<size_t>(i)] =
            mid[static_cast<size_t>(i)] * quant::sigmoid(mid[static_cast<size_t>(i)]);
    std::vector<float> down(static_cast<size_t>(S) * static_cast<size_t>(D), 0.f);
    quant::matmul_f32(down.data(), mid.data(), w_down, S, I, D);
    for (int i = 0; i < S * D; ++i)
        y[i] += down[static_cast<size_t>(i)];
    return true;
}

bool vae_rms_add(float *x, const float *skip, const float *w, float *y, int n, float eps) {
    if (!x || !y || n < 1)
        return false;
    if (skip) {
        for (int i = 0; i < n; ++i)
            x[i] += skip[i];
    }
    quant::rmsnorm(x, w, y, n, eps);
    return true;
}

bool vae_transformer_block(float *x, int tokens, int hidden, int heads, int hd,
                           const float *norm1, const float *qkv_w, const float *qkv_b,
                           const float *out_w, const float *out_b, const float *scale1,
                           const float *norm2, const float *w1, const float *b1, const float *w2,
                           const float *b2, const float *scale2, int ffn, float eps) {
    if (!x || tokens < 1 || hidden < 1)
        return false;
    const int N = tokens;
    const int H = hidden;
    auto add_bias = [](float *y, const float *b, int S, int O) {
        if (!b)
            return;
        for (int s = 0; s < S; ++s) {
            float *row = y + static_cast<size_t>(s) * O;
            for (int o = 0; o < O; ++o)
                row[o] += b[o];
        }
    };
    auto linear = [&](float *y, const float *in, const float *w, const float *b, int S, int I,
                      int O) {
        quant::matmul_f32(y, in, w, S, I, O);
        add_bias(y, b, S, O);
    };
    std::vector<float> nrm(static_cast<size_t>(N) * H);
    for (int s = 0; s < N; ++s)
        quant::rmsnorm(x + static_cast<size_t>(s) * H, norm1, nrm.data() + static_cast<size_t>(s) * H,
                       H, eps);
    if (qkv_w) {
        std::vector<float> qkv(static_cast<size_t>(N) * 3 * H);
        linear(qkv.data(), nrm.data(), qkv_w, qkv_b, N, H, 3 * H);
        std::vector<float> q(static_cast<size_t>(N) * H), k(static_cast<size_t>(N) * H),
            v(static_cast<size_t>(N) * H), attn(static_cast<size_t>(N) * H);
        for (int i = 0; i < N; ++i) {
            std::memcpy(q.data() + static_cast<size_t>(i) * H,
                        qkv.data() + static_cast<size_t>(i) * 3 * H, static_cast<size_t>(H) * sizeof(float));
            std::memcpy(k.data() + static_cast<size_t>(i) * H,
                        qkv.data() + static_cast<size_t>(i) * 3 * H + H,
                        static_cast<size_t>(H) * sizeof(float));
            std::memcpy(v.data() + static_cast<size_t>(i) * H,
                        qkv.data() + static_cast<size_t>(i) * 3 * H + 2 * H,
                        static_cast<size_t>(H) * sizeof(float));
        }
        if (heads > 0 && hd > 0 && heads * hd == H) {
            const float scale = 1.f / std::sqrt(static_cast<float>(hd));
            std::vector<float> scores(static_cast<size_t>(N));
            for (int h = 0; h < heads; ++h) {
                for (int i = 0; i < N; ++i) {
                    const float *qi = q.data() + (static_cast<size_t>(i) * heads + h) * hd;
                    float mx = -1e30f;
                    for (int j = 0; j < N; ++j) {
                        const float *kj = k.data() + (static_cast<size_t>(j) * heads + h) * hd;
                        float dot = 0.f;
                        for (int d = 0; d < hd; ++d)
                            dot += qi[d] * kj[d];
                        scores[static_cast<size_t>(j)] = dot * scale;
                        if (scores[static_cast<size_t>(j)] > mx)
                            mx = scores[static_cast<size_t>(j)];
                    }
                    float sum = 0.f;
                    for (int j = 0; j < N; ++j) {
                        float e = std::exp(scores[static_cast<size_t>(j)] - mx);
                        scores[static_cast<size_t>(j)] = e;
                        sum += e;
                    }
                    const float inv = 1.f / (sum + 1e-12f);
                    float *oi = attn.data() + (static_cast<size_t>(i) * heads + h) * hd;
                    for (int d = 0; d < hd; ++d)
                        oi[d] = 0.f;
                    for (int j = 0; j < N; ++j) {
                        const float a = scores[static_cast<size_t>(j)] * inv;
                        const float *vj = v.data() + (static_cast<size_t>(j) * heads + h) * hd;
                        for (int d = 0; d < hd; ++d)
                            oi[d] += a * vj[d];
                    }
                }
            }
        } else {
            attn = q;
        }
        std::vector<float> ao(static_cast<size_t>(N) * H);
        if (out_w)
            linear(ao.data(), attn.data(), out_w, out_b, N, H, H);
        else
            ao.swap(attn);
        if (scale1) {
            for (int i = 0; i < N * H; ++i)
                ao[static_cast<size_t>(i)] *= scale1[i % H];
        }
        for (int i = 0; i < N * H; ++i)
            x[i] += ao[static_cast<size_t>(i)];
        for (int s = 0; s < N; ++s)
            quant::rmsnorm(x + static_cast<size_t>(s) * H, norm2,
                           nrm.data() + static_cast<size_t>(s) * H, H, eps);
    } else {
        for (int s = 0; s < N; ++s)
            quant::rmsnorm(x + static_cast<size_t>(s) * H, norm2,
                           nrm.data() + static_cast<size_t>(s) * H, H, eps);
    }
    if (w1 && ffn >= 1) {
        const int w1o = 2 * ffn;
        std::vector<float> gu(static_cast<size_t>(N) * w1o);
        linear(gu.data(), nrm.data(), w1, b1, N, H, w1o);
        for (int s = 0; s < N; ++s)
            quant::silu_mul(gu.data() + static_cast<size_t>(s) * w1o,
                            gu.data() + static_cast<size_t>(s) * w1o + ffn, ffn);
        std::vector<float> fo(static_cast<size_t>(N) * H, 0.f);
        if (w2) {
            std::vector<float> hid(static_cast<size_t>(N) * ffn);
            for (int s = 0; s < N; ++s)
                std::memcpy(hid.data() + static_cast<size_t>(s) * ffn,
                            gu.data() + static_cast<size_t>(s) * w1o,
                            static_cast<size_t>(ffn) * sizeof(float));
            linear(fo.data(), hid.data(), w2, b2, N, ffn, H);
        }
        for (int i = 0; i < N * H; ++i) {
            float s = scale2 ? scale2[i % H] : 1.f;
            x[i] += fo[static_cast<size_t>(i)] * s;
        }
    }
    return true;
}

namespace {

void add_bias_rows(float *y, const float *b, int S, int O) {
    if (!y || !b || S <= 0 || O <= 0)
        return;
    for (int s = 0; s < S; ++s) {
        float *row = y + static_cast<size_t>(s) * O;
        for (int o = 0; o < O; ++o)
            row[o] += b[o];
    }
}

void linear_rows(float *y, const float *x, const float *w, const float *b, int S, int I, int O) {
    quant::matmul_f32(y, x, w, S, I, O);
    add_bias_rows(y, b, S, O);
}

float gelu_tanh_vis(float x) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    if (inner <= -10.f)
        return 0.f;
    if (inner >= 10.f)
        return x;
    return 0.5f * x * (1.f + std::tanh(inner));
}

float gelu_audio(float x) {
    return 0.5f * x * (1.f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
}

void apply_vision_rope(const float *qkv, const float *cosines, const float *sines, float *query,
                       float *key, float *value, int seq, int heads, int hd, int rope_half) {
    const int inner = heads * hd;
    for (int row = 0; row < seq; ++row) {
        const float *crow = (cosines && rope_half > 0) ? cosines + static_cast<size_t>(row) * rope_half
                                                       : nullptr;
        const float *srow = (sines && rope_half > 0) ? sines + static_cast<size_t>(row) * rope_half
                                                     : nullptr;
        for (int head = 0; head < heads; ++head) {
            const float *qs = qkv + (static_cast<size_t>(row) * 3 + 0) * inner +
                              static_cast<size_t>(head) * hd;
            const float *ks = qkv + (static_cast<size_t>(row) * 3 + 1) * inner +
                              static_cast<size_t>(head) * hd;
            const float *vs = qkv + (static_cast<size_t>(row) * 3 + 2) * inner +
                              static_cast<size_t>(head) * hd;
            float *qd = query + (static_cast<size_t>(row) * heads + head) * hd;
            float *kd = key + (static_cast<size_t>(row) * heads + head) * hd;
            float *vd = value + (static_cast<size_t>(row) * heads + head) * hd;
            for (int dim = 0; dim < hd; ++dim)
                vd[dim] = vs[dim];
            for (int dim = 0; dim < hd; ++dim) {
                const int half = rope_half;
                const int pair = dim < half ? dim + half : dim - half;
                if (!crow || !srow || half <= 0 || pair < 0 || pair >= hd) {
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

void sdpa_full(const float *query, const float *key, const float *value, float *out, int seq,
               int heads, int hd) {
    const float scale = 1.f / std::sqrt(static_cast<float>(hd > 0 ? hd : 1));
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

void sdpa_causal_b(const float *q, const float *k, const float *v, float *out, int B, int T,
                   int heads, int hd) {
    const float scale = 1.f / std::sqrt(static_cast<float>(hd > 0 ? hd : 1));
    std::vector<float> scores(static_cast<size_t>(T));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int i = 0; i < T; ++i) {
                const float *qi = q + ((static_cast<size_t>(b) * T + i) * heads + h) * hd;
                float mx = -1e30f;
                int lim = i + 1;
                for (int j = 0; j < lim; ++j) {
                    const float *kj = k + ((static_cast<size_t>(b) * T + j) * heads + h) * hd;
                    float dot = 0.f;
                    for (int d = 0; d < hd; ++d)
                        dot += qi[d] * kj[d];
                    scores[static_cast<size_t>(j)] = dot * scale;
                    if (scores[static_cast<size_t>(j)] > mx)
                        mx = scores[static_cast<size_t>(j)];
                }
                float sum = 0.f;
                for (int j = 0; j < lim; ++j) {
                    float e = std::exp(scores[static_cast<size_t>(j)] - mx);
                    scores[static_cast<size_t>(j)] = e;
                    sum += e;
                }
                float inv = 1.f / (sum + 1e-12f);
                float *oi = out + ((static_cast<size_t>(b) * T + i) * heads + h) * hd;
                for (int d = 0; d < hd; ++d)
                    oi[d] = 0.f;
                for (int j = 0; j < lim; ++j) {
                    float a = scores[static_cast<size_t>(j)] * inv;
                    const float *vj = v + ((static_cast<size_t>(b) * T + j) * heads + h) * hd;
                    for (int d = 0; d < hd; ++d)
                        oi[d] += a * vj[d];
                }
            }
        }
    }
}

} // namespace

bool vision_block(float *x, int rows, int hidden, int heads, int hd, int intermediate,
                  const float *norm1_w, const float *norm1_b, const float *qkv_w,
                  const float *qkv_b, const float *proj_w, const float *proj_b,
                  const float *norm2_w, const float *norm2_b, const float *fc1_w,
                  const float *fc1_b, const float *fc2_w, const float *fc2_b,
                  const float *rope_cos, const float *rope_sin, int rope_half, float eps) {
    if (!x || rows < 1 || hidden < 1 || !qkv_w)
        return false;
    if (heads < 1 || hd < 1 || heads * hd != hidden)
        return false;
    const int H = hidden;
    const int I = intermediate;
    std::vector<float> norm(static_cast<size_t>(rows) * H);
    for (int r = 0; r < rows; ++r)
        layernorm(x + static_cast<size_t>(r) * H, norm1_w, norm1_b,
                  norm.data() + static_cast<size_t>(r) * H, H, eps);
    std::vector<float> qkv(static_cast<size_t>(rows) * 3 * H);
    std::vector<float> query(static_cast<size_t>(rows) * H);
    std::vector<float> key(static_cast<size_t>(rows) * H);
    std::vector<float> value(static_cast<size_t>(rows) * H);
    std::vector<float> attn(static_cast<size_t>(rows) * H);
    std::vector<float> branch(static_cast<size_t>(rows) * H);
    linear_rows(qkv.data(), norm.data(), qkv_w, qkv_b, rows, H, 3 * H);
    apply_vision_rope(qkv.data(), rope_cos, rope_sin, query.data(), key.data(), value.data(), rows,
                      heads, hd, rope_half);
    sdpa_full(query.data(), key.data(), value.data(), attn.data(), rows, heads, hd);
    if (proj_w)
        linear_rows(branch.data(), attn.data(), proj_w, proj_b, rows, H, H);
    else
        std::memcpy(branch.data(), attn.data(), static_cast<size_t>(rows) * H * sizeof(float));
    for (size_t i = 0; i < static_cast<size_t>(rows) * H; ++i)
        x[i] += branch[i];
    for (int r = 0; r < rows; ++r)
        layernorm(x + static_cast<size_t>(r) * H, norm2_w, norm2_b,
                  norm.data() + static_cast<size_t>(r) * H, H, eps);
    if (I > 0 && fc1_w && fc2_w) {
        std::vector<float> fc1(static_cast<size_t>(rows) * I);
        linear_rows(fc1.data(), norm.data(), fc1_w, fc1_b, rows, H, I);
        for (int i = 0; i < rows * I; ++i)
            fc1[static_cast<size_t>(i)] = gelu_tanh_vis(fc1[static_cast<size_t>(i)]);
        linear_rows(branch.data(), fc1.data(), fc2_w, fc2_b, rows, I, H);
        for (size_t i = 0; i < static_cast<size_t>(rows) * H; ++i)
            x[i] += branch[i];
    }
    return true;
}

bool audio_pre_block(float *base, const float *seq, int B, int L, int C, int ch, int heads,
                     const float *norm1_w, const float *norm1_b, const float *qkv_w,
                     const float *q_bias, const float *k_bias, const float *v_bias,
                     const float *proj_w, const float *proj_b, const float *norm2_w,
                     const float *norm2_b, const float *mlp_norm_w, const float *mlp_norm_b,
                     const float *w0, const float *b0, const float *w1, const float *b1,
                     const float *w2, const float *b2, float eps) {
    if (!base || !seq || !qkv_w || B < 1 || L < 1 || C < 1 || ch < 1 || heads < 1)
        return false;
    const int hd = C / heads;
    if (hd < 1 || heads * hd != C)
        return false;
    const int rows = B * L;
    std::vector<float> an(static_cast<size_t>(rows) * C);
    std::memcpy(an.data(), seq, static_cast<size_t>(rows) * C * sizeof(float));
    for (int r = 0; r < rows; ++r)
        layernorm(an.data() + static_cast<size_t>(r) * C, norm1_w, norm1_b,
                  an.data() + static_cast<size_t>(r) * C, C, eps);
    std::vector<float> qkv_o(static_cast<size_t>(rows) * 3 * C);
    linear_rows(qkv_o.data(), an.data(), qkv_w, nullptr, rows, C, 3 * C);
    std::vector<float> q(static_cast<size_t>(rows) * C), k(static_cast<size_t>(rows) * C),
        v(static_cast<size_t>(rows) * C), att(static_cast<size_t>(rows) * C);
    for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < C; ++d) {
            q[static_cast<size_t>(r) * C + d] =
                qkv_o[static_cast<size_t>(r) * 3 * C + d] + (q_bias ? q_bias[d] : 0.f);
            k[static_cast<size_t>(r) * C + d] =
                qkv_o[static_cast<size_t>(r) * 3 * C + C + d] + (k_bias ? k_bias[d] : 0.f);
            v[static_cast<size_t>(r) * C + d] =
                qkv_o[static_cast<size_t>(r) * 3 * C + 2 * C + d] + (v_bias ? v_bias[d] : 0.f);
        }
    }
    sdpa_causal_b(q.data(), k.data(), v.data(), att.data(), B, L, heads, hd);
    std::vector<float> pooled(static_cast<size_t>(rows) * ch, 0.f);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < ch; ++c)
            pooled[static_cast<size_t>(r) * ch + c] = att[static_cast<size_t>(r) * C + (c % C)];
    std::vector<float> ap(static_cast<size_t>(rows) * ch, 0.f);
    if (proj_w)
        linear_rows(ap.data(), pooled.data(), proj_w, proj_b, rows, ch, ch);
    else
        ap.swap(pooled);
    for (int i = 0; i < rows * ch; ++i)
        base[i] += ap[static_cast<size_t>(i)];
    if (w0 && w1 && w2) {
        std::vector<float> n2(static_cast<size_t>(rows) * ch);
        std::memcpy(n2.data(), base, static_cast<size_t>(rows) * ch * sizeof(float));
        for (int r = 0; r < rows; ++r)
            layernorm(n2.data() + static_cast<size_t>(r) * ch, norm2_w, norm2_b,
                      n2.data() + static_cast<size_t>(r) * ch, ch, eps);
        for (int r = 0; r < rows; ++r)
            layernorm(n2.data() + static_cast<size_t>(r) * ch, mlp_norm_w, mlp_norm_b,
                      n2.data() + static_cast<size_t>(r) * ch, ch, eps);
        const int mid = 2 * ch;
        std::vector<float> gate(static_cast<size_t>(rows) * mid),
            lin(static_cast<size_t>(rows) * mid), gg(static_cast<size_t>(rows) * mid);
        linear_rows(gate.data(), n2.data(), w0, b0, rows, ch, mid);
        linear_rows(lin.data(), n2.data(), w1, b1, rows, ch, mid);
        for (size_t i = 0; i < gg.size(); ++i)
            gg[i] = gelu_audio(gate[i]) * lin[i];
        std::vector<float> br(static_cast<size_t>(rows) * ch);
        linear_rows(br.data(), gg.data(), w2, b2, rows, mid, ch);
        for (int i = 0; i < rows * ch; ++i)
            base[i] += br[static_cast<size_t>(i)];
    }
    return true;
}

} // namespace metal_h3
} // namespace mvllm

#endif // !MVLLM_WITH_METAL
