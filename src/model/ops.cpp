#include "family.hpp"
#include "../quant/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mvllm {
namespace {

void silu_vec(float *x, int n) {
    for (int i = 0; i < n; ++i)
        x[i] = x[i] * quant::sigmoid(x[i]);
}

void l2_normalize(float *x, int n, float eps) {
    float acc = 0.f;
    for (int i = 0; i < n; ++i)
        acc += x[i] * x[i];
    float inv = 1.f / std::sqrt(acc + eps);
    for (int i = 0; i < n; ++i)
        x[i] *= inv;
}

} // namespace

void kda_short_conv(float *x, const float *taps, float *window, int channels, int k) {
    if (!x || !taps || !window || channels <= 0 || k <= 1)
        return;
    for (int p = 0; p < channels; ++p) {
        float *w = window + static_cast<size_t>(p) * k;
        for (int i = 0; i < k - 1; ++i)
            w[i] = w[i + 1];
        w[k - 1] = x[p];
        float acc = 0.f;
        const float *t = taps + static_cast<size_t>(p) * k;
        for (int i = 0; i < k; ++i)
            acc += w[i] * t[i];
        x[p] = acc;
    }
}

void kda_step(const float *x, int hidden, const KdaConfig &kda, const quant::QuantMat *w_q,
              const quant::QuantMat *w_k, const quant::QuantMat *w_v, const quant::QuantMat *w_b,
              const quant::QuantMat *w_fa, const quant::QuantMat *w_fb, const float *dt_bias,
              int dt_n, const float *a_log, const quant::QuantMat *w_g, const quant::QuantMat *w_o,
              const float *out_norm, float *S, float *y, float eps, const float *conv_q,
              const float *conv_k, const float *conv_v, float *win_q, float *win_k, float *win_v) {
    const int H = kda.heads;
    const int D = kda.head_dim;
    const int P = H * D;
    auto orows = [](const quant::QuantMat *w, int fb) {
        return (w && !w->empty() && w->O > fb) ? w->O : fb;
    };
    const int qn = orows(w_q, P);
    const int kn = orows(w_k, P);
    const int vn = orows(w_v, P);
    const int bn = orows(w_b, P);
    const int gn = orows(w_g, P);
    const int zn = orows(w_fb, P);
    const int on = (w_o && !w_o->empty() && w_o->I > P) ? w_o->I : P;
    std::vector<float> q(qn, 0.f), k(kn, 0.f), v(vn, 0.f), b(bn, 0.f), z(zn, 0.f), g(gn, 0.f),
        o(on, 0.f), mix(P);

    if (w_q)
        w_q->gemm(q.data(), x, 1);
    if (w_k)
        w_k->gemm(k.data(), x, 1);
    if (w_v)
        w_v->gemm(v.data(), x, 1);
    const int K = kda.conv_k > 0 ? kda.conv_k : 4;
    kda_short_conv(q.data(), conv_q, win_q, P, K);
    kda_short_conv(k.data(), conv_k, win_k, P, K);
    kda_short_conv(v.data(), conv_v, win_v, P, K);
    silu_vec(q.data(), P);
    silu_vec(k.data(), P);
    silu_vec(v.data(), P);

    const float qscale = 1.f / std::sqrt(static_cast<float>(D));
    for (int h = 0; h < H; ++h) {
        l2_normalize(q.data() + h * D, D, eps);
        l2_normalize(k.data() + h * D, D, eps);
        for (int i = 0; i < D; ++i)
            q[h * D + i] *= qscale;
    }

    // z = W_fb(W_fa x) + dt_bias   (low-rank dt)
    std::vector<float> fa(orows(w_fa, D > 0 ? D : 1), 0.f);
    if (w_fa && w_fb && !w_fa->empty() && !w_fb->empty()) {
        w_fa->gemm(fa.data(), x, 1);
        w_fb->gemm(z.data(), fa.data(), 1);
    } else {
        std::fill(z.begin(), z.end(), 0.f);
    }
    if (dt_bias && dt_n >= P) {
        for (int i = 0; i < P; ++i)
            z[i] += dt_bias[i];
    } else if (dt_bias && dt_n >= H) {
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < D; ++i)
                z[h * D + i] += dt_bias[h];
    } else if (dt_bias && dt_n > 0) {
        for (int i = 0; i < P; ++i)
            z[i] += dt_bias[i % dt_n];
    }

    if (w_b && !w_b->empty())
        w_b->gemm(b.data(), x, 1);
    else
        std::fill(b.begin(), b.end(), 0.f);
    if (w_g && !w_g->empty())
        w_g->gemm(g.data(), x, 1);
    else
        std::fill(g.begin(), g.end(), 0.f);

    const float gmin = kda.gate_lower_bound;
    for (int h = 0; h < H; ++h) {
        float *Sh = S + static_cast<size_t>(h) * D * D;
        const float *qh = q.data() + h * D;
        const float *kh = k.data() + h * D;
        const float *vh = v.data() + h * D;
        const float *zh = z.data() + h * D;
        const float *bh = b.data() + h * D;
        float *oh = o.data() + h * D;

        float alog = 0.f;
        if (a_log)
            alog = a_log[h];
        float eA = std::exp(alog);

        // S <- Diag(alpha) S
        for (int i = 0; i < D; ++i) {
            float gk = gmin * quant::sigmoid(eA * zh[i]);
            float alpha = std::exp(gk);
            for (int j = 0; j < D; ++j)
                Sh[i * D + j] *= alpha;
        }
        // S <- (I - beta k k^T) S + beta k v^T
        // beta is per-channel in the reference; we use per-head mean of σ(b).
        float beta = 0.f;
        for (int i = 0; i < D; ++i)
            beta += quant::sigmoid(bh[i]);
        beta /= static_cast<float>(D);

        std::vector<float> ktS(D, 0.f);
        for (int j = 0; j < D; ++j) {
            float acc = 0.f;
            for (int i = 0; i < D; ++i)
                acc += kh[i] * Sh[i * D + j];
            ktS[j] = acc;
        }
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j)
                Sh[i * D + j] -= beta * kh[i] * ktS[j];
            for (int j = 0; j < D; ++j)
                Sh[i * D + j] += beta * kh[i] * vh[j];
        }
        // o = S^T q
        for (int j = 0; j < D; ++j) {
            float acc = 0.f;
            for (int i = 0; i < D; ++i)
                acc += Sh[i * D + j] * qh[i];
            oh[j] = acc;
        }
        quant::rmsnorm(oh, out_norm, oh, D, eps);
        if (!out_norm) {
            // rmsnorm with w=1
        }
        float gate = quant::sigmoid(g[h * D]); // first channel as scalar if full-rank unused
        // Full-rank gate is per-channel.
        for (int i = 0; i < D; ++i) {
            float gi = quant::sigmoid(g[h * D + i]);
            oh[i] *= gi;
            (void)gate;
        }
    }

    if (w_o && !w_o->empty())
        w_o->gemm(y, o.data(), 1);
    else
        std::memcpy(y, o.data(), std::min(P, hidden) * sizeof(float));
}

void attnres_mix(const std::vector<std::vector<float>> &snapshots, const float *prefix,
                 const float *res_norm, const float *res_proj, float *hidden, int n, float eps) {
    const int ns = static_cast<int>(snapshots.size()) + 1;
    std::vector<const float *> vecs(ns);
    for (int i = 0; i < static_cast<int>(snapshots.size()); ++i)
        vecs[i] = snapshots[i].data();
    vecs[ns - 1] = prefix;

    std::vector<float> scores(ns, 0.f);
    std::vector<float> rw(n, 1.f);
    if (res_norm && res_proj) {
        for (int i = 0; i < n; ++i)
            rw[i] = res_norm[i] * res_proj[i];
    } else if (res_norm) {
        std::memcpy(rw.data(), res_norm, n * sizeof(float));
    }

    for (int i = 0; i < ns; ++i) {
        float acc = 0.f, ss = 0.f;
        const float *v = vecs[i];
        for (int j = 0; j < n; ++j) {
            acc += v[j] * rw[j];
            ss += v[j] * v[j];
        }
        scores[i] = acc / std::sqrt(ss / static_cast<float>(n) + eps);
    }
    quant::softmax_inplace(scores.data(), ns);
    std::fill(hidden, hidden + n, 0.f);
    for (int i = 0; i < ns; ++i) {
        const float *v = vecs[i];
        for (int j = 0; j < n; ++j)
            hidden[j] += scores[i] * v[j];
    }
}

void mhc_mix(float *streams, int hidden, int mult, const float *alpha, int iters, float eps) {
    // Sinkhorn-normalize an [mult, mult] mix if alpha provided; otherwise average.
    if (mult <= 1)
        return;
    std::vector<float> A(mult * mult, 1.f / static_cast<float>(mult));
    if (alpha)
        std::memcpy(A.data(), alpha, mult * mult * sizeof(float));
    for (int t = 0; t < iters; ++t) {
        for (int r = 0; r < mult; ++r) {
            float s = eps;
            for (int c = 0; c < mult; ++c)
                s += A[r * mult + c];
            for (int c = 0; c < mult; ++c)
                A[r * mult + c] /= s;
        }
        for (int c = 0; c < mult; ++c) {
            float s = eps;
            for (int r = 0; r < mult; ++r)
                s += A[r * mult + c];
            for (int r = 0; r < mult; ++r)
                A[r * mult + c] /= s;
        }
    }
    std::vector<float> out(static_cast<size_t>(mult) * hidden, 0.f);
    for (int r = 0; r < mult; ++r) {
        for (int c = 0; c < mult; ++c) {
            float a = A[r * mult + c];
            const float *src = streams + static_cast<size_t>(c) * hidden;
            float *dst = out.data() + static_cast<size_t>(r) * hidden;
            for (int i = 0; i < hidden; ++i)
                dst[i] += a * src[i];
        }
    }
    std::memcpy(streams, out.data(), out.size() * sizeof(float));
}

int moe_topk(const float *scores, int n, int k, int *idx, float *w) {
    if (k > n)
        k = n;
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i)
        order[i] = i;
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b) {
                          if (scores[a] == scores[b])
                              return a < b;
                          return scores[a] > scores[b];
                      });
    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
        idx[i] = order[i];
        w[i] = scores[order[i]];
        if (w[i] < 0.f)
            w[i] = 0.f;
        sum += w[i];
    }
    if (sum > 0.f) {
        for (int i = 0; i < k; ++i)
            w[i] /= sum;
    } else {
        for (int i = 0; i < k; ++i)
            w[i] = 1.f / static_cast<float>(k);
    }
    return k;
}

int moe_union_ids(const int *idx, int n_tok, int topk, int *out, int out_cap) {
    if (!idx || !out || n_tok <= 0 || topk <= 0 || out_cap <= 0)
        return 0;
    std::vector<int> all(static_cast<size_t>(n_tok) * topk);
    int n = 0;
    for (int i = 0; i < n_tok * topk; ++i)
        if (idx[i] >= 0)
            all[static_cast<size_t>(n++)] = idx[i];
    if (n <= 0)
        return 0;
    std::sort(all.begin(), all.begin() + n);
    int u = 0;
    for (int i = 0; i < n && u < out_cap; ++i) {
        if (u == 0 || all[static_cast<size_t>(i)] != out[u - 1])
            out[u++] = all[static_cast<size_t>(i)];
    }
    return u;
}

void bf16_to_f32(const uint16_t *src, float *dst, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(dst + i, &bits, sizeof(float));
    }
}

void h3_dit_block_cpu(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                      int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                      int tokens, float eps) {
    if (!blob || !x || hidden <= 0 || inner <= 0 || ffn <= 0 || tokens <= 0)
        return;
    const int64_t qkv_n = qkv_bytes / 2;
    const int64_t out_n = out_bytes / 2;
    const int64_t fc1_n = fc1_bytes / 2;
    const int64_t fc2_n = fc2_bytes / 2;
    const uint16_t *qkv_b = reinterpret_cast<const uint16_t *>(blob);
    const uint16_t *out_b = reinterpret_cast<const uint16_t *>(blob + qkv_bytes);
    const uint16_t *fc1_b = reinterpret_cast<const uint16_t *>(blob + qkv_bytes + out_bytes);
    const uint16_t *fc2_b =
        reinterpret_cast<const uint16_t *>(blob + qkv_bytes + out_bytes + fc1_bytes);
    std::vector<float> Wqkv(static_cast<size_t>(qkv_n));
    std::vector<float> Wout(static_cast<size_t>(out_n));
    std::vector<float> Wfc1(static_cast<size_t>(fc1_n));
    std::vector<float> Wfc2(static_cast<size_t>(fc2_n));
    bf16_to_f32(qkv_b, Wqkv.data(), qkv_n);
    bf16_to_f32(out_b, Wout.data(), out_n);
    bf16_to_f32(fc1_b, Wfc1.data(), fc1_n);
    bf16_to_f32(fc2_b, Wfc2.data(), fc2_n);

    const int T = tokens;
    const int I = inner;
    const int H = hidden;
    std::vector<float> xn(static_cast<size_t>(T) * H);
    std::vector<float> ones(H, 1.f);
    for (int t = 0; t < T; ++t)
        quant::rmsnorm(x + t * H, ones.data(), xn.data() + t * H, H, eps);

    std::vector<float> qkv(static_cast<size_t>(T) * 3 * I);
    quant::matmul_f32(qkv.data(), xn.data(), Wqkv.data(), T, H, 3 * I);

    int hd = head_dim > 0 ? head_dim : I;
    int heads = I / hd;
    if (heads < 1) {
        heads = 1;
        hd = I;
    }
    std::vector<float> ctx(static_cast<size_t>(T) * I, 0.f);
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    for (int h = 0; h < heads; ++h) {
        std::vector<float> scores(static_cast<size_t>(T) * T, 0.f);
        for (int qi = 0; qi < T; ++qi) {
            const float *q = qkv.data() + qi * 3 * I + h * hd;
            for (int ki = 0; ki < T; ++ki) {
                const float *k = qkv.data() + ki * 3 * I + I + h * hd;
                float acc = 0.f;
                for (int d = 0; d < hd; ++d)
                    acc += q[d] * k[d];
                scores[qi * T + ki] = acc * scale;
            }
            quant::softmax_inplace(scores.data() + qi * T, T);
        }
        for (int qi = 0; qi < T; ++qi) {
            float *o = ctx.data() + qi * I + h * hd;
            for (int d = 0; d < hd; ++d)
                o[d] = 0.f;
            for (int vi = 0; vi < T; ++vi) {
                const float *v = qkv.data() + vi * 3 * I + 2 * I + h * hd;
                float a = scores[qi * T + vi];
                for (int d = 0; d < hd; ++d)
                    o[d] += a * v[d];
            }
        }
    }

    std::vector<float> attn(static_cast<size_t>(T) * H);
    quant::matmul_f32(attn.data(), ctx.data(), Wout.data(), T, I, H);
    for (int i = 0; i < T * H; ++i)
        x[i] += attn[i];

    for (int t = 0; t < T; ++t)
        quant::rmsnorm(x + t * H, ones.data(), xn.data() + t * H, H, eps);
    std::vector<float> h1(static_cast<size_t>(T) * (2 * ffn));
    quant::matmul_f32(h1.data(), xn.data(), Wfc1.data(), T, H, 2 * ffn);
    for (int t = 0; t < T; ++t) {
        float *g = h1.data() + t * (2 * ffn);
        float *u = g + ffn;
        for (int i = 0; i < ffn; ++i)
            g[i] = g[i] * quant::sigmoid(g[i]) * u[i];
    }
    // down-proj reads the first ffn of each row (SiLU-gated)
    std::vector<float> gated(static_cast<size_t>(T) * ffn);
    for (int t = 0; t < T; ++t)
        std::memcpy(gated.data() + t * ffn, h1.data() + t * (2 * ffn),
                    static_cast<size_t>(ffn) * sizeof(float));
    std::vector<float> down(static_cast<size_t>(T) * H);
    quant::matmul_f32(down.data(), gated.data(), Wfc2.data(), T, ffn, H);
    for (int i = 0; i < T * H; ++i)
        x[i] += down[i];
}

void mla_absorb_kvb(const float *kv_b, int n_heads, int qk_nope, int v_head, int kv_lora,
                    quant::QuantMat &w_kt, quant::QuantMat &w_v, int bits) {
    if (!kv_b || n_heads <= 0 || qk_nope <= 0 || v_head <= 0 || kv_lora <= 0)
        return;
    std::vector<float> kt(static_cast<size_t>(n_heads) * kv_lora * qk_nope);
    std::vector<float> vv(static_cast<size_t>(n_heads) * v_head * kv_lora);
    for (int h = 0; h < n_heads; ++h) {
        const float *block = kv_b + static_cast<size_t>(h) * (qk_nope + v_head) * kv_lora;
        for (int d = 0; d < kv_lora; ++d)
            for (int i = 0; i < qk_nope; ++i)
                kt[(static_cast<size_t>(h) * kv_lora + d) * qk_nope + i] =
                    block[static_cast<size_t>(i) * kv_lora + d];
        std::memcpy(vv.data() + static_cast<size_t>(h) * v_head * kv_lora,
                    block + static_cast<size_t>(qk_nope) * kv_lora,
                    static_cast<size_t>(v_head) * kv_lora * sizeof(float));
    }
    w_kt.from_f32(kt.data(), n_heads * kv_lora, qk_nope, bits);
    w_v.from_f32(vv.data(), n_heads * v_head, kv_lora, bits);
}

void layernorm(const float *x, const float *w, const float *b, float *y, int n, float eps) {
    if (!x || !y || n <= 0)
        return;
    float mean = 0.f;
    for (int i = 0; i < n; ++i)
        mean += x[i];
    mean /= static_cast<float>(n);
    float var = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv = 1.f / std::sqrt(var / static_cast<float>(n) + eps);
    for (int i = 0; i < n; ++i) {
        float v = (x[i] - mean) * inv;
        if (w)
            v *= w[i];
        if (b)
            v += b[i];
        y[i] = v;
    }
}

int dsa_index_width(const DsaConfig &dsa) {
    int pool = dsa.kpool > 0 ? dsa.kpool : 1;
    int topk = dsa.topk > 0 ? dsa.topk : 0;
    return dsa.always_select_tail ? topk + pool - 1 : topk;
}

int dsa_select(int *out, const float *queries, const float *keys, const float *gates,
               const float *head_w, const float *ape, int seq, const DsaConfig &dsa) {
    const int width = dsa_index_width(dsa);
    if (!out || width <= 0)
        return 0;
    for (int i = 0; i < width; ++i)
        out[i] = -1;
    if (!queries || !keys || seq <= 0)
        return width;
    const int H = dsa.n_heads > 0 ? dsa.n_heads : 1;
    const int D = dsa.head_dim > 0 ? dsa.head_dim : 1;
    const int pool = dsa.kpool > 0 ? dsa.kpool : 1;
    const int topk = dsa.topk > 0 ? dsa.topk : 0;
    if (topk < pool || topk % pool != 0)
        return width;
    const int npool = seq / pool;
    const int want = topk / pool;
    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    std::vector<float> scores(static_cast<size_t>(std::max(npool, 0)), -1e30f);
    std::vector<float> pooled(static_cast<size_t>(D));
    std::vector<float> logit(static_cast<size_t>(pool));
    for (int p = 0; p < npool; ++p) {
        for (int d = 0; d < D; ++d) {
            float mx = -1e30f;
            for (int j = 0; j < pool; ++j) {
                float g = gates ? gates[static_cast<size_t>(p * pool + j) * D + d] : 0.f;
                float a = ape ? ape[static_cast<size_t>(j) * D + d] : 0.f;
                logit[static_cast<size_t>(j)] = g + a;
                if (logit[static_cast<size_t>(j)] > mx)
                    mx = logit[static_cast<size_t>(j)];
            }
            float z = 0.f;
            for (int j = 0; j < pool; ++j) {
                logit[static_cast<size_t>(j)] = std::exp(logit[static_cast<size_t>(j)] - mx);
                z += logit[static_cast<size_t>(j)];
            }
            float acc = 0.f;
            for (int j = 0; j < pool; ++j)
                acc += (logit[static_cast<size_t>(j)] / z) *
                       keys[static_cast<size_t>(p * pool + j) * D + d];
            pooled[static_cast<size_t>(d)] = acc;
        }
        float s = 0.f;
        for (int h = 0; h < H; ++h) {
            float dot = 0.f;
            const float *qh = queries + static_cast<size_t>(h) * D;
            for (int d = 0; d < D; ++d)
                dot += qh[d] * pooled[static_cast<size_t>(d)];
            if (dot < 0.f)
                dot = 0.f;
            float hw = head_w ? head_w[h] : 1.f;
            s += hw * dot * scale;
        }
        scores[static_cast<size_t>(p)] = s;
    }
    std::vector<int> order(static_cast<size_t>(npool));
    for (int i = 0; i < npool; ++i)
        order[static_cast<size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (scores[static_cast<size_t>(a)] != scores[static_cast<size_t>(b)])
            return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
        return a < b;
    });
    int used = 0;
    const int take = std::min(want, npool);
    for (int r = 0; r < take; ++r) {
        int p = order[static_cast<size_t>(r)];
        for (int j = 0; j < pool && used < topk; ++j)
            out[used++] = p * pool + j;
    }
    if (dsa.always_select_tail) {
        int rem = seq % pool;
        int start = seq - rem;
        for (int j = 0; j < rem && topk + j < width; ++j)
            out[topk + j] = start + j;
    }
    return width;
}

void mla_step(const float *x, int hidden, const MlaConfig &mla, const quant::QuantMat *w_qa,
              const float *qa_ln, const quant::QuantMat *w_qb, const quant::QuantMat *w_kva,
              const float *kva_ln, const quant::QuantMat *w_kt, const quant::QuantMat *w_v,
              const quant::QuantMat *w_o, const quant::QuantMat *w_g, float *cache, int pos,
              float *y, float eps, const int *selected, int n_sel) {
    if (!x || !y || hidden <= 0 || pos < 0)
        return;
    const int H = mla.n_heads > 0 ? mla.n_heads : 1;
    const int QK = mla.qk_nope > 0 ? mla.qk_nope : 1;
    const int R = mla.qk_rope > 0 ? mla.qk_rope : 0;
    const int QH = QK + R;
    const int Vh = mla.v_head > 0 ? mla.v_head : QK;
    const int L = mla.kv_lora > 0 ? mla.kv_lora : 0;
    const int QL = mla.q_lora > 0 ? mla.q_lora : 0;
    const int stride = L + R;
    const bool absorbed = L > 0 && w_kva && !w_kva->empty() && w_kt && !w_kt->empty() && w_v &&
                          !w_v->empty() && cache;
    if (!absorbed) {
        // Dense Q/O stand-in when MLA tensors are missing.
        if (w_qa && !w_qa->empty() && w_o && !w_o->empty()) {
            std::vector<float> q(static_cast<size_t>(w_qa->O));
            w_qa->gemm(q.data(), x, 1);
            w_o->gemm(y, q.data(), 1);
            return;
        }
        if (w_o && !w_o->empty()) {
            w_o->gemm(y, x, 1);
            return;
        }
        std::memset(y, 0, static_cast<size_t>(hidden) * sizeof(float));
        return;
    }

    std::vector<float> qa(static_cast<size_t>(QL > 0 ? QL : hidden));
    const float *q_in = x;
    if (w_qa && !w_qa->empty() && QL > 0) {
        w_qa->gemm(qa.data(), x, 1);
        if (qa_ln)
            quant::rmsnorm(qa.data(), qa_ln, qa.data(), QL, eps);
        q_in = qa.data();
    }
    std::vector<float> q(static_cast<size_t>(H) * QH, 0.f);
    if (w_qb && !w_qb->empty())
        w_qb->gemm(q.data(), q_in, 1);
    else if (w_qa && !w_qa->empty())
        std::memcpy(q.data(), qa.data(), std::min(q.size(), qa.size()) * sizeof(float));

    float *crow = cache + static_cast<size_t>(pos) * stride;
    w_kva->gemm(crow, x, 1);
    if (kva_ln)
        quant::rmsnorm(crow, kva_ln, crow, L, eps);

    std::vector<int> ts;
    if (selected && n_sel > 0) {
        ts.reserve(static_cast<size_t>(n_sel));
        for (int i = 0; i < n_sel; ++i)
            if (selected[i] >= 0 && selected[i] <= pos)
                ts.push_back(selected[i]);
    } else {
        ts.resize(static_cast<size_t>(pos + 1));
        for (int t = 0; t <= pos; ++t)
            ts[static_cast<size_t>(t)] = t;
    }
    const int nt = static_cast<int>(ts.size());
    const float scale = 1.f / std::sqrt(static_cast<float>(QH));
    std::vector<float> ctx(static_cast<size_t>(H) * Vh, 0.f);
    std::vector<float> qabs(static_cast<size_t>(L));
    std::vector<float> scores(static_cast<size_t>(std::max(nt, 1)));
    std::vector<float> pooled(static_cast<size_t>(L));
    for (int h = 0; h < H; ++h) {
        const float *qh = q.data() + static_cast<size_t>(h) * QH;
        w_kt->gemm_rows(qabs.data(), qh, 1, h * L, L);
        for (int ti = 0; ti < nt; ++ti) {
            const int t = ts[static_cast<size_t>(ti)];
            const float *ct = cache + static_cast<size_t>(t) * stride;
            float s = 0.f;
            for (int i = 0; i < L; ++i)
                s += qabs[i] * ct[i];
            for (int i = 0; i < R; ++i)
                s += qh[QK + i] * ct[L + i];
            scores[static_cast<size_t>(ti)] = s * scale;
        }
        if (nt > 0)
            quant::softmax_inplace(scores.data(), nt);
        std::fill(pooled.begin(), pooled.end(), 0.f);
        for (int ti = 0; ti < nt; ++ti) {
            const int t = ts[static_cast<size_t>(ti)];
            const float *ct = cache + static_cast<size_t>(t) * stride;
            const float a = scores[static_cast<size_t>(ti)];
            for (int i = 0; i < L; ++i)
                pooled[i] += a * ct[i];
        }
        w_v->gemm_rows(ctx.data() + static_cast<size_t>(h) * Vh, pooled.data(), 1, h * Vh, Vh);
    }
    if (mla.output_gate && w_g && !w_g->empty()) {
        std::vector<float> g(static_cast<size_t>(H) * Vh, 0.f);
        w_g->gemm(g.data(), x, 1);
        for (int i = 0; i < H * Vh; ++i)
            ctx[i] *= quant::sigmoid(g[i]);
    }
    if (w_o && !w_o->empty())
        w_o->gemm(y, ctx.data(), 1);
    else
        std::memset(y, 0, static_cast<size_t>(hidden) * sizeof(float));
}

} // namespace mvllm
