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
              int dt_n, const float *a_log, const quant::QuantMat *w_g,
              const quant::QuantMat *w_gb, const quant::QuantMat *w_o, const float *out_norm,
              float *S, float *y, float eps, const float *conv_q,
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
    const int bn = (w_b && !w_b->empty()) ? w_b->O : H;
    const int gn = (w_gb && !w_gb->empty()) ? orows(w_gb, P) : orows(w_g, P);
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
        l2_normalize(q.data() + h * D, D, 1e-6f);
        l2_normalize(k.data() + h * D, D, 1e-6f);
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
    if (w_g && !w_g->empty() && w_gb && !w_gb->empty()) {
        std::vector<float> ga(static_cast<size_t>(std::max(w_g->O, 1)), 0.f);
        w_g->gemm(ga.data(), x, 1);
        w_gb->gemm(g.data(), ga.data(), 1);
    } else if (w_g && !w_g->empty()) {
        w_g->gemm(g.data(), x, 1);
    } else {
        std::fill(g.begin(), g.end(), 0.f);
    }

    const float gmin = kda.gate_lower_bound;
    for (int h = 0; h < H; ++h) {
        float *Sh = S + static_cast<size_t>(h) * D * D;
        const float *qh = q.data() + h * D;
        const float *kh = k.data() + h * D;
        const float *vh = v.data() + h * D;
        const float *zh = z.data() + h * D;
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
        // Official b_proj is [heads, hidden] → β_h = σ(b[h]). Packed [P] keeps a mean.
        float beta = 0.f;
        if (bn >= P && D > 0) {
            const float *bh = b.data() + h * D;
            for (int i = 0; i < D; ++i)
                beta += quant::sigmoid(bh[i]);
            beta /= static_cast<float>(D);
        } else if (h < bn) {
            beta = quant::sigmoid(b[h]);
        } else {
            beta = 0.5f;
        }

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

int mhc_pre(float *collapsed, float *post, float *comb, const float *streams, const float *fn,
            const float *scale, const float *base, int mult, int hidden, int iters, float norm_eps,
            float hc_eps) {
    if (!collapsed || !post || !comb || !streams || !fn || !scale || !base || mult < 1 ||
        hidden < 1)
        return -1;
    const int flat = mult * hidden;
    const int mix_n = (2 + mult) * mult;
    float ms = 0.f;
    for (int i = 0; i < flat; ++i)
        ms += streams[i] * streams[i];
    const float inv = 1.f / std::sqrt(ms / static_cast<float>(flat) + norm_eps);
    std::vector<float> mixes(static_cast<size_t>(mix_n), 0.f);
    std::vector<float> pre(static_cast<size_t>(mult), 0.f);
    for (int r = 0; r < mix_n; ++r) {
        float s = 0.f;
        const float *row = fn + static_cast<size_t>(r) * flat;
        for (int c = 0; c < flat; ++c)
            s += row[c] * streams[c];
        mixes[static_cast<size_t>(r)] = s * inv;
    }
    for (int i = 0; i < mult; ++i) {
        pre[static_cast<size_t>(i)] =
            quant::sigmoid(mixes[static_cast<size_t>(i)] * scale[0] + base[i]) + hc_eps;
        post[i] = 2.f * quant::sigmoid(mixes[static_cast<size_t>(mult + i)] * scale[1] +
                                       base[mult + i]);
    }
    const int off = 2 * mult;
    for (int r = 0; r < mult; ++r) {
        float mx = -1e30f;
        for (int c = 0; c < mult; ++c) {
            float v = mixes[static_cast<size_t>(off + r * mult + c)] * scale[2] +
                      base[off + r * mult + c];
            comb[r * mult + c] = v;
            if (v > mx)
                mx = v;
        }
        float z = 0.f;
        for (int c = 0; c < mult; ++c) {
            float e = std::exp(comb[r * mult + c] - mx);
            comb[r * mult + c] = e;
            z += e;
        }
        for (int c = 0; c < mult; ++c)
            comb[r * mult + c] = comb[r * mult + c] / z + hc_eps;
    }
    std::vector<float> sums(static_cast<size_t>(mult), 0.f);
    for (int c = 0; c < mult; ++c) {
        float s = 0.f;
        for (int r = 0; r < mult; ++r)
            s += comb[r * mult + c];
        sums[static_cast<size_t>(c)] = s;
    }
    for (int r = 0; r < mult; ++r)
        for (int c = 0; c < mult; ++c)
            comb[r * mult + c] /= sums[static_cast<size_t>(c)] + hc_eps;
    for (int it = 1; it < iters; ++it) {
        for (int r = 0; r < mult; ++r) {
            float s = 0.f;
            for (int c = 0; c < mult; ++c)
                s += comb[r * mult + c];
            sums[static_cast<size_t>(r)] = s;
        }
        for (int r = 0; r < mult; ++r)
            for (int c = 0; c < mult; ++c)
                comb[r * mult + c] /= sums[static_cast<size_t>(r)] + hc_eps;
        for (int c = 0; c < mult; ++c) {
            float s = 0.f;
            for (int r = 0; r < mult; ++r)
                s += comb[r * mult + c];
            sums[static_cast<size_t>(c)] = s;
        }
        for (int r = 0; r < mult; ++r)
            for (int c = 0; c < mult; ++c)
                comb[r * mult + c] /= sums[static_cast<size_t>(c)] + hc_eps;
    }
    for (int c = 0; c < hidden; ++c) {
        float s = 0.f;
        for (int m = 0; m < mult; ++m)
            s += pre[static_cast<size_t>(m)] * streams[m * hidden + c];
        collapsed[c] = s;
    }
    return 0;
}

int mhc_post(float *streams, const float *branch, const float *residual, const float *post,
             const float *comb, int mult, int hidden) {
    if (!streams || !branch || !residual || !post || !comb || mult < 1 || hidden < 1)
        return -1;
    std::vector<float> out(static_cast<size_t>(mult) * hidden, 0.f);
    for (int dst = 0; dst < mult; ++dst) {
        for (int c = 0; c < hidden; ++c) {
            float v = 0.f;
            for (int src = 0; src < mult; ++src)
                v += comb[src * mult + dst] * residual[src * hidden + c];
            v += post[dst] * branch[c];
            out[static_cast<size_t>(dst) * hidden + c] = v;
        }
    }
    std::memcpy(streams, out.data(), out.size() * sizeof(float));
    return 0;
}

int moe_topk(const float *choice, int n, int k, int *idx, float *w, const float *mix) {
    if (k > n)
        k = n;
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i)
        order[i] = i;
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b) {
                          if (choice[a] == choice[b])
                              return a < b;
                          return choice[a] > choice[b];
                      });
    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
        idx[i] = order[i];
        w[i] = mix ? mix[order[i]] : choice[order[i]];
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

void apply_rope(float *x, int n, int pos, float theta) {
    if (!x || n < 2 || theta <= 0.f)
        return;
    const int pairs = n / 2;
    for (int i = 0; i < pairs; ++i) {
        float freq = std::pow(theta, -2.f * static_cast<float>(i) / static_cast<float>(n));
        float ang = static_cast<float>(pos) * freq;
        float c = std::cos(ang), s = std::sin(ang);
        float a = x[2 * i], b = x[2 * i + 1];
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

void apply_penalties(float *logits, int vocab, const int *hist, int hist_n, float frequency,
                     float presence) {
    if (!logits || vocab <= 0 || !hist || hist_n <= 0)
        return;
    if (frequency == 0.f && presence == 0.f)
        return;
    std::vector<int> count(static_cast<size_t>(vocab), 0);
    bool any = false;
    for (int i = 0; i < hist_n; ++i) {
        int t = hist[i];
        if (t >= 0 && t < vocab) {
            ++count[static_cast<size_t>(t)];
            any = true;
        }
    }
    if (!any)
        return;
    for (int t = 0; t < vocab; ++t) {
        int c = count[static_cast<size_t>(t)];
        if (c > 0)
            logits[t] -= frequency * static_cast<float>(c) + presence;
    }
}

void apply_repetition_penalty(float *logits, int vocab, const int *hist, int hist_n, float penalty) {
    if (!logits || vocab <= 0 || !hist || hist_n <= 0)
        return;
    if (penalty == 1.f || penalty <= 0.f)
        return;
    std::vector<uint8_t> seen(static_cast<size_t>(vocab), 0);
    for (int i = 0; i < hist_n; ++i) {
        int t = hist[i];
        if (t < 0 || t >= vocab || seen[static_cast<size_t>(t)])
            continue;
        seen[static_cast<size_t>(t)] = 1;
        if (logits[t] < 0.f)
            logits[t] *= penalty;
        else
            logits[t] /= penalty;
    }
}

void apply_top_k(float *logits, int vocab, int k) {
    if (!logits || vocab <= 0 || k <= 0 || k >= vocab)
        return;
    std::vector<int> order(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i)
        order[static_cast<size_t>(i)] = i;
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) {
        if (logits[a] != logits[b])
            return logits[a] > logits[b];
        return a < b;
    });
    std::vector<uint8_t> keep(static_cast<size_t>(vocab), 0);
    for (int i = 0; i < k; ++i)
        keep[static_cast<size_t>(order[static_cast<size_t>(i)])] = 1;
    for (int i = 0; i < vocab; ++i)
        if (!keep[static_cast<size_t>(i)])
            logits[i] = -1e30f;
}

void apply_min_p(float *logits, int vocab, float min_p) {
    if (!logits || vocab <= 0 || min_p <= 0.f || min_p >= 1.f)
        return;
    float m = logits[0];
    for (int i = 1; i < vocab; ++i)
        if (logits[i] > m)
            m = logits[i];
    std::vector<float> p(static_cast<size_t>(vocab));
    float sum = 0.f;
    for (int i = 0; i < vocab; ++i) {
        p[static_cast<size_t>(i)] = std::exp(logits[i] - m);
        sum += p[static_cast<size_t>(i)];
    }
    if (!(sum > 0.f))
        return;
    float p_max = 0.f;
    for (int i = 0; i < vocab; ++i) {
        p[static_cast<size_t>(i)] /= sum;
        if (p[static_cast<size_t>(i)] > p_max)
            p_max = p[static_cast<size_t>(i)];
    }
    const float thresh = min_p * p_max;
    for (int i = 0; i < vocab; ++i)
        if (p[static_cast<size_t>(i)] < thresh)
            logits[i] = -1e30f;
}

void apply_logit_bias(float *logits, int vocab, const std::pair<int, float> *bias, int n_bias) {
    if (!logits || !bias || vocab <= 0)
        return;
    for (int i = 0; i < n_bias; ++i) {
        int t = bias[i].first;
        if (t >= 0 && t < vocab)
            logits[t] += bias[i].second;
    }
}

float token_logprob(const float *logits, int vocab, int token, const uint8_t *allow) {
    if (!logits || vocab <= 0 || token < 0 || token >= vocab)
        return -1e30f;
    if (allow && !allow[token])
        return -1e30f;
    float m = 0.f;
    bool have = false;
    for (int i = 0; i < vocab; ++i) {
        if (allow && !allow[i])
            continue;
        if (!have || logits[i] > m) {
            m = logits[i];
            have = true;
        }
    }
    if (!have)
        return -1e30f;
    float sum = 0.f;
    for (int i = 0; i < vocab; ++i) {
        if (allow && !allow[i])
            continue;
        sum += std::exp(logits[i] - m);
    }
    if (!(sum > 0.f))
        return -1e30f;
    return (logits[token] - m) - std::log(sum);
}

void top_logprobs(const float *logits, int vocab, int k, GenLogprob *out, int *n_out,
                  const uint8_t *allow) {
    if (n_out)
        *n_out = 0;
    if (!logits || !out || vocab <= 0 || k < 1)
        return;
    std::vector<int> idx;
    idx.reserve(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) {
        if (allow && !allow[i])
            continue;
        idx.push_back(i);
    }
    if (idx.empty())
        return;
    const int take = std::min(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + take, idx.end(), [&](int a, int b) {
        if (logits[a] != logits[b])
            return logits[a] > logits[b];
        return a < b;
    });
    for (int i = 0; i < take; ++i) {
        out[i].token = idx[static_cast<size_t>(i)];
        out[i].logprob = token_logprob(logits, vocab, idx[static_cast<size_t>(i)], allow);
    }
    if (n_out)
        *n_out = take;
}

int sample_token(const float *logits, int vocab, float temperature, float top_p, uint64_t *rng,
                 const uint8_t *allow, float *out_logprob) {
    int chosen = 0;
    if (!logits || vocab <= 0) {
        if (out_logprob)
            *out_logprob = -1e30f;
        return 0;
    }
    auto ok = [&](int i) { return !allow || allow[i]; };
    if (temperature <= 0.f) {
        int best = -1;
        for (int i = 0; i < vocab; ++i)
            if (ok(i) && (best < 0 || logits[i] > logits[best]))
                best = i;
        chosen = best < 0 ? 0 : best;
    } else {
        std::vector<float> p(static_cast<size_t>(vocab));
        float m = 0.f;
        bool have = false;
        for (int i = 0; i < vocab; ++i) {
            if (!ok(i))
                continue;
            if (!have || logits[i] > m) {
                m = logits[i];
                have = true;
            }
        }
        if (!have) {
            if (out_logprob)
                *out_logprob = token_logprob(logits, vocab, 0, allow);
            return 0;
        }
        float sum = 0.f;
        for (int i = 0; i < vocab; ++i) {
            if (!ok(i)) {
                p[static_cast<size_t>(i)] = 0.f;
                continue;
            }
            p[static_cast<size_t>(i)] = std::exp((logits[i] - m) / temperature);
            sum += p[static_cast<size_t>(i)];
        }
        if (!(sum > 0.f)) {
            if (out_logprob)
                *out_logprob = token_logprob(logits, vocab, 0, allow);
            return 0;
        }
        for (int i = 0; i < vocab; ++i)
            p[static_cast<size_t>(i)] /= sum;
        if (top_p > 0.f && top_p < 1.f) {
            std::vector<int> ord(static_cast<size_t>(vocab));
            for (int i = 0; i < vocab; ++i)
                ord[static_cast<size_t>(i)] = i;
            std::sort(ord.begin(), ord.end(), [&](int a, int b) {
                return p[static_cast<size_t>(a)] > p[static_cast<size_t>(b)];
            });
            float acc = 0.f;
            for (int i = 0; i < vocab; ++i) {
                acc += p[static_cast<size_t>(ord[static_cast<size_t>(i)])];
                if (acc >= top_p) {
                    for (int j = i + 1; j < vocab; ++j)
                        p[static_cast<size_t>(ord[static_cast<size_t>(j)])] = 0.f;
                    break;
                }
            }
            float s2 = 0.f;
            for (int i = 0; i < vocab; ++i)
                s2 += p[static_cast<size_t>(i)];
            if (s2 > 0.f) {
                for (int i = 0; i < vocab; ++i)
                    p[static_cast<size_t>(i)] /= s2;
            }
        }
        uint64_t s = rng ? *rng : 1ull;
        s += 0x9E3779B97F4A7C15ull;
        s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ull;
        s = (s ^ (s >> 27)) * 0x94D049BB133111EBull;
        s ^= s >> 31;
        if (rng)
            *rng = s;
        float u = static_cast<float>((s >> 11) * (1.0 / 9007199254740992.0));
        float c = 0.f;
        chosen = vocab - 1;
        for (int i = 0; i < vocab; ++i) {
            c += p[static_cast<size_t>(i)];
            if (u <= c) {
                chosen = i;
                break;
            }
        }
    }
    if (out_logprob)
        *out_logprob = token_logprob(logits, vocab, chosen, allow);
    return chosen;
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
                      int tokens, float eps, const float *adaln_mod, const float *q_norm,
                      const float *k_norm, const float *rope_cos, const float *rope_sin) {
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
    auto apply_adaln = [&](const float *src, float *dst, int scale_slot, int shift_slot) {
        for (int t = 0; t < T; ++t) {
            quant::rmsnorm(src + t * H, ones.data(), dst + t * H, H, eps);
            if (!adaln_mod)
                continue;
            const float *s = adaln_mod + scale_slot * H;
            const float *b = adaln_mod + shift_slot * H;
            for (int i = 0; i < H; ++i)
                dst[t * H + i] = dst[t * H + i] * (1.f + s[i]) + b[i];
        }
    };
    apply_adaln(x, xn.data(), 0, 1);

    std::vector<float> qkv(static_cast<size_t>(T) * 3 * I);
    quant::matmul_f32(qkv.data(), xn.data(), Wqkv.data(), T, H, 3 * I);

    int hd = head_dim > 0 ? head_dim : I;
    int heads = I / hd;
    if (heads < 1) {
        heads = 1;
        hd = I;
    }
    if (q_norm || k_norm) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < heads; ++h) {
                float *q = qkv.data() + t * 3 * I + h * hd;
                float *k = qkv.data() + t * 3 * I + I + h * hd;
                if (q_norm)
                    quant::rmsnorm(q, q_norm, q, hd, eps);
                if (k_norm)
                    quant::rmsnorm(k, k_norm, k, hd, eps);
            }
        }
    }
    if (rope_cos && rope_sin && hd >= 96) {
        const int half = 48;
        for (int t = 0; t < T; ++t) {
            const float *c = rope_cos + static_cast<size_t>(t) * half;
            const float *s = rope_sin + static_cast<size_t>(t) * half;
            for (int h = 0; h < heads; ++h) {
                float *q = qkv.data() + t * 3 * I + h * hd;
                float *k = qkv.data() + t * 3 * I + I + h * hd;
                for (int d = 0; d < half; ++d) {
                    float qa = q[d], qb = q[d + half];
                    q[d] = qa * c[d] - qb * s[d];
                    q[d + half] = qa * s[d] + qb * c[d];
                    float ka = k[d], kb = k[d + half];
                    k[d] = ka * c[d] - kb * s[d];
                    k[d + half] = ka * s[d] + kb * c[d];
                }
            }
        }
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
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < H; ++i) {
            float g = adaln_mod ? adaln_mod[2 * H + i] : 1.f;
            x[t * H + i] += g * attn[t * H + i];
        }

    apply_adaln(x, xn.data(), 3, 4);
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
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < H; ++i) {
            float g = adaln_mod ? adaln_mod[5 * H + i] : 1.f;
            x[t * H + i] += g * down[t * H + i];
        }
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
    if (R > 0 && !mla.nope && mla.rope_theta > 0.f)
        apply_rope(crow + L, R, pos, mla.rope_theta);
    if (R > 0 && !mla.nope && mla.rope_theta > 0.f) {
        for (int h = 0; h < H; ++h)
            apply_rope(q.data() + static_cast<size_t>(h) * QH + QK, R, pos, mla.rope_theta);
    }

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

namespace {

void vit_matvec(float *out, const float *w, const float *b, const float *in, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float s = b ? b[r] : 0.f;
        const float *wr = w + static_cast<size_t>(r) * cols;
        for (int c = 0; c < cols; ++c)
            s += wr[c] * in[c];
        out[r] = s;
    }
}

float vit_gelu(float x) {
    return 0.5f * x * (1.f + std::erff(x * 0.70710678118654752f));
}

} // namespace

int glm_vit_forward(const GlmVitTower &tower, const float *pixels, int grid_h, int grid_w,
                    float *out) {
    const VisionConfig &c = tower.cfg;
    if (!out || !pixels || grid_h < 1 || grid_w < 1 || !tower.ready())
        return -1;
    const int merge = c.merge > 0 ? c.merge : 1;
    if (grid_h % merge || grid_w % merge)
        return -1;
    const int hidden = c.hidden;
    const int heads = c.heads > 0 ? c.heads : 1;
    const int hd = hidden / heads;
    if (hidden <= 0 || hd <= 0 || hd % 4)
        return -1;
    const int tokens = grid_h * grid_w;
    const int pin = c.in_channels * c.temporal * c.patch * c.patch;
    const int rot = hd / 2;
    const int half = rot / 2;
    std::vector<float> state(static_cast<size_t>(tokens) * hidden, 0.f);
    std::vector<float> cost(static_cast<size_t>(tokens) * hd, 0.f);
    std::vector<float> sint(static_cast<size_t>(tokens) * hd, 0.f);
    std::vector<float> qkv(static_cast<size_t>(tokens) * 3 * hidden, 0.f);
    std::vector<float> scores(static_cast<size_t>(tokens), 0.f);
    std::vector<float> scratch(static_cast<size_t>(std::max(hidden, c.intermediate)), 0.f);
    std::vector<float> branch(static_cast<size_t>(tokens) * hidden, 0.f);
    if (static_cast<int>(tower.patch_w.size()) < hidden * pin)
        return -1;
    for (int t = 0; t < tokens; ++t)
        vit_matvec(state.data() + static_cast<size_t>(t) * hidden, tower.patch_w.data(),
                   tower.patch_b.empty() ? nullptr : tower.patch_b.data(),
                   pixels + static_cast<size_t>(t) * pin, hidden, pin);
    int index = 0;
    const float theta = c.rope_theta > 0.f ? c.rope_theta : 10000.f;
    for (int bh = 0; bh < grid_h / merge; ++bh)
        for (int bw = 0; bw < grid_w / merge; ++bw)
            for (int ih = 0; ih < merge; ++ih)
                for (int iw = 0; iw < merge; ++iw, ++index) {
                    float pos[2] = {static_cast<float>(bh * merge + ih),
                                    static_cast<float>(bw * merge + iw)};
                    float *cosine = cost.data() + static_cast<size_t>(index) * hd;
                    float *sine = sint.data() + static_cast<size_t>(index) * hd;
                    for (int axis = 0; axis < 2; ++axis)
                        for (int j = 0; j < half; ++j) {
                            float inv = std::pow(theta, -2.f * static_cast<float>(j) /
                                                            static_cast<float>(rot));
                            float ang = pos[axis] * inv;
                            int slot = axis * half + j;
                            cosine[slot] = std::cos(ang);
                            sine[slot] = std::sin(ang);
                            cosine[slot + rot] = cosine[slot];
                            sine[slot + rot] = sine[slot];
                        }
                }
    const float eps = c.eps > 0.f ? c.eps : 1e-5f;
    const float slim = c.swiglu_limit > 0.f ? c.swiglu_limit : 10.f;
    for (const GlmVitBlock &block : tower.blocks) {
        if (block.qkv_w.empty() || block.norm1.empty())
            continue;
        for (int t = 0; t < tokens; ++t) {
            quant::rmsnorm(state.data() + static_cast<size_t>(t) * hidden, block.norm1.data(),
                           scratch.data(), hidden, eps);
            vit_matvec(qkv.data() + static_cast<size_t>(t) * 3 * hidden, block.qkv_w.data(),
                       block.qkv_b.empty() ? nullptr : block.qkv_b.data(), scratch.data(),
                       3 * hidden, hidden);
        }
        for (int t = 0; t < tokens; ++t) {
            float *row = qkv.data() + static_cast<size_t>(t) * 3 * hidden;
            const float *cosine = cost.data() + static_cast<size_t>(t) * hd;
            const float *sine = sint.data() + static_cast<size_t>(t) * hd;
            for (int h = 0; h < heads; ++h) {
                float *q = row + h * hd;
                float *k = row + hidden + h * hd;
                if (!block.q_norm.empty())
                    quant::rmsnorm(q, block.q_norm.data(), q, hd, eps);
                if (!block.k_norm.empty())
                    quant::rmsnorm(k, block.k_norm.data(), k, hd, eps);
                for (int pass = 0; pass < 2; ++pass) {
                    float *vec = pass ? k : q;
                    std::vector<float> rotated(static_cast<size_t>(hd));
                    for (int i = 0; i < hd; ++i)
                        rotated[static_cast<size_t>(i)] =
                            i < hd / 2 ? -vec[i + hd / 2] : vec[i - hd / 2];
                    for (int i = 0; i < hd; ++i)
                        vec[i] = vec[i] * cosine[i] + rotated[static_cast<size_t>(i)] * sine[i];
                }
            }
        }
        const float scale = 1.f / std::sqrt(static_cast<float>(hd));
        for (int t = 0; t < tokens; ++t) {
            float *result = branch.data() + static_cast<size_t>(t) * hidden;
            for (int h = 0; h < heads; ++h) {
                const float *q = qkv.data() + static_cast<size_t>(t) * 3 * hidden + h * hd;
                float mx = -1e30f;
                for (int s = 0; s < tokens; ++s) {
                    const float *k = qkv.data() + static_cast<size_t>(s) * 3 * hidden + hidden +
                                     h * hd;
                    float dot = 0.f;
                    for (int i = 0; i < hd; ++i)
                        dot += q[i] * k[i];
                    scores[static_cast<size_t>(s)] = dot * scale;
                    if (scores[static_cast<size_t>(s)] > mx)
                        mx = scores[static_cast<size_t>(s)];
                }
                float z = 0.f;
                for (int s = 0; s < tokens; ++s) {
                    scores[static_cast<size_t>(s)] = std::exp(scores[static_cast<size_t>(s)] - mx);
                    z += scores[static_cast<size_t>(s)];
                }
                float *slot = result + h * hd;
                std::fill(slot, slot + hd, 0.f);
                for (int s = 0; s < tokens; ++s) {
                    const float *vv = qkv.data() + static_cast<size_t>(s) * 3 * hidden +
                                      2 * hidden + h * hd;
                    float a = scores[static_cast<size_t>(s)] / z;
                    for (int i = 0; i < hd; ++i)
                        slot[i] += a * vv[i];
                }
            }
        }
        for (int t = 0; t < tokens; ++t) {
            if (!block.proj_w.empty())
                vit_matvec(scratch.data(), block.proj_w.data(),
                           block.proj_b.empty() ? nullptr : block.proj_b.data(),
                           branch.data() + static_cast<size_t>(t) * hidden, hidden, hidden);
            else
                std::memcpy(scratch.data(), branch.data() + static_cast<size_t>(t) * hidden,
                            static_cast<size_t>(hidden) * sizeof(float));
            float *st = state.data() + static_cast<size_t>(t) * hidden;
            for (int i = 0; i < hidden; ++i)
                st[i] += scratch[i];
        }
        const int I = c.intermediate > 0 ? c.intermediate : hidden;
        std::vector<float> gate(static_cast<size_t>(I) * 2, 0.f);
        for (int t = 0; t < tokens; ++t) {
            float *up = gate.data() + I;
            if (!block.norm2.empty())
                quant::rmsnorm(state.data() + static_cast<size_t>(t) * hidden, block.norm2.data(),
                               scratch.data(), hidden, eps);
            else
                std::memcpy(scratch.data(), state.data() + static_cast<size_t>(t) * hidden,
                            static_cast<size_t>(hidden) * sizeof(float));
            if (!block.gate_w.empty())
                vit_matvec(gate.data(), block.gate_w.data(),
                           block.gate_b.empty() ? nullptr : block.gate_b.data(), scratch.data(), I,
                           hidden);
            if (!block.up_w.empty())
                vit_matvec(up, block.up_w.data(), block.up_b.empty() ? nullptr : block.up_b.data(),
                           scratch.data(), I, hidden);
            for (int i = 0; i < I; ++i) {
                float g = gate[static_cast<size_t>(i)] > slim ? slim : gate[static_cast<size_t>(i)];
                float u = up[i];
                if (u > slim)
                    u = slim;
                if (u < -slim)
                    u = -slim;
                gate[static_cast<size_t>(i)] = g * quant::sigmoid(g) * u;
            }
            if (!block.down_w.empty())
                vit_matvec(scratch.data(), block.down_w.data(),
                           block.down_b.empty() ? nullptr : block.down_b.data(), gate.data(), hidden,
                           I);
            float *st = state.data() + static_cast<size_t>(t) * hidden;
            for (int i = 0; i < hidden; ++i)
                st[i] += scratch[i];
        }
    }
    if (!tower.post_norm.empty()) {
        for (int t = 0; t < tokens; ++t)
            quant::rmsnorm(state.data() + static_cast<size_t>(t) * hidden, tower.post_norm.data(),
                           state.data() + static_cast<size_t>(t) * hidden, hidden, eps);
    }
    const int nblk = (grid_h / merge) * (grid_w / merge);
    const int oh = c.out_hidden > 0 ? c.out_hidden : hidden;
    const int pi = c.proj_intermediate > 0 ? c.proj_intermediate : c.intermediate;
    std::vector<float> merged(static_cast<size_t>(oh), 0.f);
    std::vector<float> gated(static_cast<size_t>(std::max(pi, 1)) * 2, 0.f);
    for (int n = 0; n < nblk; ++n) {
        if (!tower.down_w.empty()) {
            for (int o = 0; o < oh; ++o) {
                float s = o < static_cast<int>(tower.down_b.size()) ? tower.down_b[static_cast<size_t>(o)]
                                                                    : 0.f;
                for (int ch = 0; ch < hidden; ++ch)
                    for (int kh = 0; kh < merge; ++kh)
                        for (int kw = 0; kw < merge; ++kw) {
                            size_t wi = (((static_cast<size_t>(o) * hidden + ch) * merge + kh) *
                                             merge +
                                         kw);
                            size_t tok = static_cast<size_t>(n) * merge * merge +
                                         static_cast<size_t>(kh) * merge + kw;
                            if (wi < tower.down_w.size())
                                s += tower.down_w[wi] * state[tok * hidden + ch];
                        }
                merged[static_cast<size_t>(o)] = s;
            }
        } else {
            std::fill(merged.begin(), merged.end(), 0.f);
            int cnt = 0;
            for (int kh = 0; kh < merge; ++kh)
                for (int kw = 0; kw < merge; ++kw) {
                    size_t tok = static_cast<size_t>(n) * merge * merge +
                                 static_cast<size_t>(kh) * merge + kw;
                    for (int i = 0; i < std::min(oh, hidden); ++i)
                        merged[static_cast<size_t>(i)] += state[tok * hidden + i];
                    ++cnt;
                }
            if (cnt > 0)
                for (int i = 0; i < oh; ++i)
                    merged[static_cast<size_t>(i)] /= static_cast<float>(cnt);
        }
        if (!tower.merger_proj.empty()) {
            std::vector<float> tmp(static_cast<size_t>(oh), 0.f);
            vit_matvec(tmp.data(), tower.merger_proj.data(), nullptr, merged.data(), oh, oh);
            layernorm(tmp.data(),
                      tower.merger_norm_w.empty() ? nullptr : tower.merger_norm_w.data(),
                      tower.merger_norm_b.empty() ? nullptr : tower.merger_norm_b.data(),
                      merged.data(), oh, 1e-5f);
            for (int i = 0; i < oh; ++i)
                merged[static_cast<size_t>(i)] = vit_gelu(merged[static_cast<size_t>(i)]);
        }
        if (!tower.merger_gate.empty() && !tower.merger_up.empty() && !tower.merger_down.empty() &&
            pi > 0) {
            float *up = gated.data() + pi;
            vit_matvec(gated.data(), tower.merger_gate.data(), nullptr, merged.data(), pi, oh);
            vit_matvec(up, tower.merger_up.data(), nullptr, merged.data(), pi, oh);
            for (int i = 0; i < pi; ++i) {
                float g = gated[static_cast<size_t>(i)] > slim ? slim : gated[static_cast<size_t>(i)];
                float u = up[i];
                if (u > slim)
                    u = slim;
                if (u < -slim)
                    u = -slim;
                gated[static_cast<size_t>(i)] = g * quant::sigmoid(g) * u;
            }
            vit_matvec(out + static_cast<size_t>(n) * oh, tower.merger_down.data(), nullptr,
                       gated.data(), oh, pi);
        } else {
            std::memcpy(out + static_cast<size_t>(n) * oh, merged.data(),
                        static_cast<size_t>(oh) * sizeof(float));
        }
    }
    return nblk;
}

int glm_vit_embed(const float *rgb, int width, int height, const VisionConfig &v,
                  const quant::QuantMat *patch, const quant::QuantMat *proj, float *out,
                  int out_cap, const GlmVitTower *tower) {
    if (!rgb || !out || out_cap <= 0 || width <= 0 || height <= 0)
        return 0;
    const int P = v.patch > 0 ? v.patch : 14;
    if (tower && tower->ready()) {
        const int merge = v.merge > 0 ? v.merge : tower->cfg.merge;
        const int isz = v.image_size > 0 ? v.image_size : std::max(width, P * std::max(merge, 1));
        const int step = P * std::max(merge, 1);
        int side = (isz / step) * step;
        if (side < step)
            side = step;
        const int gh = side / P, gw = side / P;
        const int T = v.temporal > 0 ? v.temporal : 2;
        const int C = v.in_channels > 0 ? v.in_channels : 3;
        const int pin = C * T * P * P;
        std::vector<float> pixels(static_cast<size_t>(gh) * gw * pin, 0.f);
        const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
        const float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
        int idx = 0;
        for (int bh = 0; bh < gh / std::max(merge, 1); ++bh)
            for (int bw = 0; bw < gw / std::max(merge, 1); ++bw)
                for (int ih = 0; ih < std::max(merge, 1); ++ih)
                    for (int iw = 0; iw < std::max(merge, 1); ++iw, ++idx) {
                        const int py = bh * merge + ih;
                        const int px = bw * merge + iw;
                        float *dst = pixels.data() + static_cast<size_t>(idx) * pin;
                        for (int tt = 0; tt < T; ++tt)
                            for (int dy = 0; dy < P; ++dy)
                                for (int dx = 0; dx < P; ++dx)
                                    for (int ch = 0; ch < C; ++ch) {
                                        int sy = (py * P + dy) * height / side;
                                        int sx = (px * P + dx) * width / side;
                                        if (sy >= height)
                                            sy = height - 1;
                                        if (sx >= width)
                                            sx = width - 1;
                                        if (sy < 0)
                                            sy = 0;
                                        if (sx < 0)
                                            sx = 0;
                                        float val = rgb[(static_cast<size_t>(sy) * width + sx) * 3 +
                                                        ch];
                                        val = (val - mean[ch]) / stdv[ch];
                                        dst[((tt * P + dy) * P + dx) * C + ch] = val;
                                    }
                    }
        const int nout = (gh / std::max(merge, 1)) * (gw / std::max(merge, 1));
        const int take = std::min(nout, out_cap);
        std::vector<float> full(static_cast<size_t>(std::max(nout, 1)) *
                                std::max(v.out_hidden > 0 ? v.out_hidden : v.hidden, 1));
        if (glm_vit_forward(*tower, pixels.data(), gh, gw, full.data()) < 0)
            return 0;
        const int od = v.out_hidden > 0 ? v.out_hidden : v.hidden;
        std::memcpy(out, full.data(), static_cast<size_t>(take) * od * sizeof(float));
        return take;
    }
    const int isz = v.image_size > 0 ? v.image_size : std::max(width, P);
    const int merge = v.merge > 0 ? v.merge : 1;
    const int gh = std::max(isz / P, 1);
    const int gw = std::max(isz / P, 1);
    const int Vh = (patch && !patch->empty()) ? patch->O : (v.hidden > 0 ? v.hidden : 32);
    const int pin = 3 * P * P;
    std::vector<float> patches(static_cast<size_t>(gh) * gw * Vh, 0.f);
    std::vector<float> raw(static_cast<size_t>(pin), 0.f);
    for (int py = 0; py < gh; ++py) {
        for (int px = 0; px < gw; ++px) {
            for (int dy = 0; dy < P; ++dy) {
                int sy = (py * P + dy) * height / std::max(isz, 1);
                if (sy >= height)
                    sy = height - 1;
                for (int dx = 0; dx < P; ++dx) {
                    int sx = (px * P + dx) * width / std::max(isz, 1);
                    if (sx >= width)
                        sx = width - 1;
                    const float *pix = rgb + (static_cast<size_t>(sy) * width + sx) * 3;
                    const int o = (dy * P + dx) * 3;
                    raw[static_cast<size_t>(o)] = pix[0];
                    raw[static_cast<size_t>(o + 1)] = pix[1];
                    raw[static_cast<size_t>(o + 2)] = pix[2];
                }
            }
            float *dst = patches.data() + static_cast<size_t>(py * gw + px) * Vh;
            if (patch && !patch->empty() && patch->I == pin)
                patch->gemm(dst, raw.data(), 1);
            else {
                float m = 0.f;
                for (int i = 0; i < pin; ++i)
                    m += raw[static_cast<size_t>(i)];
                m /= static_cast<float>(pin);
                for (int i = 0; i < Vh; ++i)
                    dst[i] = m;
            }
        }
    }
    std::vector<float> merged = patches;
    int nh = gh, nw = gw;
    if (merge > 1) {
        const int mh = std::max(gh / merge, 1);
        const int mw = std::max(gw / merge, 1);
        merged.assign(static_cast<size_t>(mh) * mw * Vh, 0.f);
        for (int y = 0; y < mh; ++y) {
            for (int x = 0; x < mw; ++x) {
                float *d = merged.data() + static_cast<size_t>(y * mw + x) * Vh;
                int cnt = 0;
                for (int dy = 0; dy < merge; ++dy) {
                    for (int dx = 0; dx < merge; ++dx) {
                        const int sy = y * merge + dy;
                        const int sx = x * merge + dx;
                        if (sy >= gh || sx >= gw)
                            continue;
                        const float *s = patches.data() + static_cast<size_t>(sy * gw + sx) * Vh;
                        for (int i = 0; i < Vh; ++i)
                            d[i] += s[i];
                        ++cnt;
                    }
                }
                if (cnt > 0) {
                    for (int i = 0; i < Vh; ++i)
                        d[i] /= static_cast<float>(cnt);
                }
            }
        }
        nh = mh;
        nw = mw;
    }
    const int ntok = nh * nw;
    const int take = std::min(ntok, out_cap);
    const int Od = (proj && !proj->empty()) ? proj->O : (v.out_hidden > 0 ? v.out_hidden : Vh);
    for (int t = 0; t < take; ++t) {
        float *dst = out + static_cast<size_t>(t) * Od;
        const float *src = merged.data() + static_cast<size_t>(t) * Vh;
        if (proj && !proj->empty())
            proj->gemm(dst, src, 1);
        else {
            const int n = std::min(Od, Vh);
            std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
            if (Od > n)
                std::memset(dst + n, 0, static_cast<size_t>(Od - n) * sizeof(float));
        }
    }
    return take;
}

int h3_dit_patchify(const float *z, int ch, int t, int h, int w, float *rows) {
    if (!z || !rows || ch < 1 || t < 1 || h < 2 || w < 2 || (h & 1) || (w & 1))
        return 0;
    int n = 0;
    for (int ti = 0; ti < t; ++ti)
        for (int y = 0; y < h; y += 2)
            for (int x = 0; x < w; x += 2)
                for (int c = 0; c < ch; ++c)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            size_t in = (((static_cast<size_t>(c) * t + ti) * h + (y + dy)) * w) +
                                        (x + dx);
                            rows[n++] = z[in];
                        }
    return n;
}

int h3_dit_unpatchify(const float *rows, int ch, int t, int h, int w, float *z) {
    if (!rows || !z || ch < 1 || t < 1 || h < 2 || w < 2 || (h & 1) || (w & 1))
        return 0;
    int n = 0;
    for (int ti = 0; ti < t; ++ti)
        for (int y = 0; y < h; y += 2)
            for (int x = 0; x < w; x += 2)
                for (int c = 0; c < ch; ++c)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            size_t out = (((static_cast<size_t>(c) * t + ti) * h + (y + dy)) * w) +
                                         (x + dx);
                            z[out] = rows[n++];
                        }
    return n;
}

void h3_sigma_video(int steps, float *sigmas, float shift) {
    if (!sigmas || steps < 1)
        return;
    if (shift <= 0.f)
        shift = 12.f;
    for (int i = 0; i < steps; ++i) {
        int base_index = (i * 1000) / steps;
        float base = static_cast<float>(1000 - base_index) / 1000.f;
        sigmas[i] = shift * base / (1.f + (shift - 1.f) * base);
    }
    sigmas[steps] = 0.f;
}

void h3_time_features(float t, float *out, int dim) {
    if (!out || dim < 2)
        return;
    const int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        float freq = std::exp(-std::log(10000.f) * static_cast<float>(i) / static_cast<float>(half));
        float ang = t * freq;
        out[i] = std::cos(ang);
        out[half + i] = std::sin(ang);
    }
}

int h3_euler_step(float *sample, const float *velocity, int n, float sigma, float sigma_next) {
    if (!sample || !velocity || n <= 0 || !(sigma > sigma_next) || sigma_next < 0.f)
        return 0;
    const float d = sigma - sigma_next;
    for (int i = 0; i < n; ++i)
        sample[i] += d * velocity[i];
    return 1;
}

} // namespace mvllm
