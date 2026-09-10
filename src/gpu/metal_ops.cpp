#include "metal_ops.hpp"

#include "../model/family.hpp"
#include "../quant/quant.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace mvllm {
namespace metal_ops {
namespace {

bool g_inited = false;

void cpu_rmsnorm(float *y, const float *x, const float *w, int nrows, int D, float eps) {
    for (int r = 0; r < nrows; ++r)
        quant::rmsnorm(x + static_cast<size_t>(r) * D, w, y + static_cast<size_t>(r) * D, D, eps);
}

void cpu_add(float *y, const float *a, size_t n) {
    for (size_t i = 0; i < n; ++i)
        y[i] += a[i];
}

void cpu_silu_mul(float *g, const float *u, size_t n) {
    quant::silu_mul(g, u, static_cast<int>(n));
}

void l2_normalize(float *x, int n, float eps) {
    float acc = 0.f;
    for (int i = 0; i < n; ++i)
        acc += x[i] * x[i];
    const float inv = 1.f / std::sqrt(acc + eps);
    for (int i = 0; i < n; ++i)
        x[i] *= inv;
}

// Matches ops.cpp kda_step after short-conv / SiLU / L2: Diag(alpha) S, then
// S -= beta k (k^T S) + beta k v^T, oh = S^T q. alpha is precomputed [H*hd].
void cpu_kda_fused(float *win_q, float *qt, float *win_k, float *kt, float *win_v, float *tv,
                   const float *taps_q, const float *taps_k, const float *taps_v, float *S,
                   const float *alpha, const float *beta, float *oh, int P, int K, int H, int hd) {
    kda_short_conv(qt, taps_q, win_q, P, K);
    kda_short_conv(kt, taps_k, win_k, P, K);
    kda_short_conv(tv, taps_v, win_v, P, K);

    for (int i = 0; i < P; ++i) {
        qt[i] *= quant::sigmoid(qt[i]);
        kt[i] *= quant::sigmoid(kt[i]);
        tv[i] *= quant::sigmoid(tv[i]);
    }

    const float qscale = 1.f / std::sqrt(static_cast<float>(hd));
    for (int h = 0; h < H; ++h) {
        l2_normalize(qt + h * hd, hd, 1e-6f);
        l2_normalize(kt + h * hd, hd, 1e-6f);
        for (int i = 0; i < hd; ++i)
            qt[h * hd + i] *= qscale;
    }

    std::vector<float> ktS(static_cast<size_t>(hd), 0.f);
    for (int h = 0; h < H; ++h) {
        float *Sh = S + static_cast<size_t>(h) * hd * hd;
        const float *qh = qt + h * hd;
        const float *kh = kt + h * hd;
        const float *vh = tv + h * hd;
        float *outh = oh + h * hd;
        const float b = beta[h];

        for (int i = 0; i < hd; ++i) {
            const float al = alpha[h * hd + i];
            for (int j = 0; j < hd; ++j)
                Sh[i * hd + j] *= al;
        }

        for (int j = 0; j < hd; ++j) {
            float acc = 0.f;
            for (int i = 0; i < hd; ++i)
                acc += kh[i] * Sh[i * hd + j];
            ktS[static_cast<size_t>(j)] = acc;
        }
        for (int i = 0; i < hd; ++i) {
            for (int j = 0; j < hd; ++j)
                Sh[i * hd + j] -= b * kh[i] * ktS[static_cast<size_t>(j)];
            for (int j = 0; j < hd; ++j)
                Sh[i * hd + j] += b * kh[i] * vh[j];
        }
        for (int j = 0; j < hd; ++j) {
            float acc = 0.f;
            for (int i = 0; i < hd; ++i)
                acc += Sh[i * hd + j] * qh[i];
            outh[j] = acc;
        }
    }
}

bool kda_args_ok(const float *win_q, const float *qt, const float *win_k, const float *kt,
                 const float *win_v, const float *tv, const float *taps_q, const float *taps_k,
                 const float *taps_v, const float *S, const float *alpha, const float *beta,
                 const float *oh, int P, int K, int H, int hd) {
    return win_q && qt && win_k && kt && win_v && tv && taps_q && taps_k && taps_v && S && alpha &&
           beta && oh && P == H * hd && K >= 1 && H >= 1 && hd >= 1;
}

void apply_act(float *g, const float *u, int n, Act act, float a, float b) {
    if (!g || !u || n < 1)
        return;
    if (act == Act::ClampSwiGLU) {
        for (int i = 0; i < n; ++i)
            g[i] = quant::clamped_swiglu(g[i], u[i], a);
        return;
    }
    if (act == Act::Situ) {
        for (int i = 0; i < n; ++i)
            g[i] = quant::situ_glu(g[i], u[i], a, b);
        return;
    }
    quant::silu_mul(g, u, n);
}

void cpu_route(const float *nrm, const float *router_w, const float *router_bias, int D, int E,
               int K, float rscale, int *idx_out, float *w_out) {
    if (!nrm || !router_w || !idx_out || !w_out || D < 1 || E < 1 || K < 1)
        return;
    std::vector<float> scores(static_cast<size_t>(E), 0.f);
    std::vector<float> choice(static_cast<size_t>(E), 0.f);
    quant::matmul_f32(scores.data(), nrm, router_w, 1, D, E);
    for (int i = 0; i < E; ++i) {
        scores[static_cast<size_t>(i)] = quant::sigmoid(scores[static_cast<size_t>(i)]);
        const float b = router_bias && i >= 0 ? (i < E ? router_bias[i] : 0.f) : 0.f;
        choice[static_cast<size_t>(i)] = scores[static_cast<size_t>(i)] + b;
    }
    moe_topk(choice.data(), E, K, idx_out, w_out, scores.data());
    if (rscale != 1.f) {
        const int kk = K > E ? E : K;
        for (int i = 0; i < kk; ++i)
            w_out[i] *= rscale;
    }
}

void cpu_layer_decode_full(float *x, const float *attn, const float *in_ln, const float *post_ln,
                           const float *shg, const float *shu, const float *shd, int D, int Iinter,
                           float eps, Act act, float act_a, float act_b, const float *router_w,
                           const float *router_bias, int E, int K, float rscale, float *inrm_out,
                           float *nrm_out, float *sh_out, int *idx_out, float *w_out) {
    if (in_ln && inrm_out)
        quant::rmsnorm(x, in_ln, inrm_out, D, eps);
    if (attn) {
        for (int i = 0; i < D; ++i)
            x[i] += attn[i];
    }
    quant::rmsnorm(x, post_ln, nrm_out, D, eps);
    if (shg && shu && shd && Iinter > 0 && sh_out) {
        std::vector<float> gate(static_cast<size_t>(Iinter));
        std::vector<float> up(static_cast<size_t>(Iinter));
        quant::matmul_f32(gate.data(), nrm_out, shg, 1, D, Iinter);
        quant::matmul_f32(up.data(), nrm_out, shu, 1, D, Iinter);
        apply_act(gate.data(), up.data(), Iinter, act, act_a, act_b);
        quant::matmul_f32(sh_out, gate.data(), shd, 1, Iinter, D);
    }
    cpu_route(nrm_out, router_w, router_bias, D, E, K, rscale, idx_out, w_out);
}

void cpu_layer_decode(float *x, const float *attn, const float *post_ln, const float *shg,
                      const float *shu, const float *shd, int D, int Iinter, float eps,
                      float *nrm_out, float *sh_out) {
    cpu_layer_decode_full(x, attn, nullptr, post_ln, shg, shu, shd, D, Iinter, eps, Act::Silu, 0.f,
                          0.f, nullptr, nullptr, 0, 0, 1.f, nullptr, nrm_out, sh_out, nullptr,
                          nullptr);
}

void cpu_moe_block(int nb, int D, int Iinter, int fmt, int qgs, const void *const *g,
                   const void *const *u, const void *const *d, const float *const *gs,
                   const float *const *us, const float *const *ds, const float *xg, const int *xoff,
                   const int *nr, const int *rows, const float *rw, float *out, int S, Act act,
                   float act_a, float act_b) {
    (void)qgs;
    int base = 0;
    std::vector<float> gate, up, hh;
    for (int e = 0; e < nb; ++e) {
        const int nre = nr[e];
        if (nre <= 0)
            continue;
        if (Iinter < 1 || !g || !u || !d || !g[e] || !u[e] || !d[e] || !xoff) {
            base += nre;
            continue;
        }
        if (fmt == 4 && (!gs || !us || !ds || !gs[e] || !us[e] || !ds[e])) {
            base += nre;
            continue;
        }
        const float *xe = xg + static_cast<size_t>(xoff[e]) * static_cast<size_t>(D);
        gate.assign(static_cast<size_t>(nre) * static_cast<size_t>(Iinter), 0.f);
        up.assign(static_cast<size_t>(nre) * static_cast<size_t>(Iinter), 0.f);
        hh.assign(static_cast<size_t>(nre) * static_cast<size_t>(D), 0.f);
        if (fmt == 0) {
            quant::matmul_f32(gate.data(), xe, static_cast<const float *>(g[e]), nre, D, Iinter);
            quant::matmul_f32(up.data(), xe, static_cast<const float *>(u[e]), nre, D, Iinter);
            apply_act(gate.data(), up.data(), nre * Iinter, act, act_a, act_b);
            quant::matmul_f32(hh.data(), gate.data(), static_cast<const float *>(d[e]), nre, Iinter,
                              D);
        } else {
            quant::matmul_int4_g64(gate.data(), xe, static_cast<const uint8_t *>(g[e]), gs[e], nre,
                                   D, Iinter);
            quant::matmul_int4_g64(up.data(), xe, static_cast<const uint8_t *>(u[e]), us[e], nre, D,
                                   Iinter);
            apply_act(gate.data(), up.data(), nre * Iinter, act, act_a, act_b);
            quant::matmul_int4_g64(hh.data(), gate.data(), static_cast<const uint8_t *>(d[e]), ds[e],
                                   nre, Iinter, D);
        }
        if (rows && rw) {
            for (int r = 0; r < nre; ++r) {
                const int dest = rows[base + r];
                if (dest < 0 || (S > 0 && dest >= S))
                    continue;
                const float w = rw[base + r];
                const float *hr = hh.data() + static_cast<size_t>(r) * static_cast<size_t>(D);
                float *orow = out + static_cast<size_t>(dest) * static_cast<size_t>(D);
                for (int i = 0; i < D; ++i)
                    orow[i] += w * hr[i];
            }
        }
        base += nre;
    }
}

} // namespace

bool init() {
    g_inited = true;
    return true;
}

void shutdown() { g_inited = false; }

bool available() { return g_inited; }

const char *backend_name() { return "cpu"; }

bool rmsnorm(float *y, const float *x, const float *w, int nrows, int D, float eps) {
    if (!y || !x || nrows < 1 || D < 1)
        return false;
    cpu_rmsnorm(y, x, w, nrows, D, eps);
    return true;
}

bool add(float *y, const float *a, size_t n) {
    if (!y || !a)
        return false;
    cpu_add(y, a, n);
    return true;
}

bool silu_mul(float *g, const float *u, size_t n) {
    if (!g || !u)
        return false;
    cpu_silu_mul(g, u, n);
    return true;
}

bool kda_fused_token(float *win_q, float *qt, float *win_k, float *kt, float *win_v, float *tv,
                     const float *taps_q, const float *taps_k, const float *taps_v, float *S,
                     const float *alpha, const float *beta, float *oh, int P, int K, int H,
                     int hd) {
    if (!kda_args_ok(win_q, qt, win_k, kt, win_v, tv, taps_q, taps_k, taps_v, S, alpha, beta, oh, P,
                     K, H, hd))
        return false;
    cpu_kda_fused(win_q, qt, win_k, kt, win_v, tv, taps_q, taps_k, taps_v, S, alpha, beta, oh, P, K,
                  H, hd);
    return true;
}

bool layer_decode(float *x, const float *attn, const float *post_ln, const float *shg,
                  const float *shu, const float *shd, int D, int Iinter, float eps, float *nrm_out,
                  float *sh_out) {
    return layer_decode_full(x, attn, nullptr, post_ln, shg, shu, shd, D, Iinter, eps, Act::Silu,
                             0.f, 0.f, nullptr, nullptr, 0, 0, 1.f, nullptr, nrm_out, sh_out,
                             nullptr, nullptr);
}

bool layer_decode_full(float *x, const float *attn, const float *in_ln, const float *post_ln,
                       const float *shg, const float *shu, const float *shd, int D, int Iinter,
                       float eps, Act act, float act_a, float act_b, const float *router_w,
                       const float *router_bias, int E, int K, float rscale, float *inrm_out,
                       float *nrm_out, float *sh_out, int *idx_out, float *w_out) {
    if (!x || !post_ln || !nrm_out || D < 1)
        return false;
    cpu_layer_decode_full(x, attn, in_ln, post_ln, shg, shu, shd, D, Iinter, eps, act, act_a, act_b,
                          router_w, router_bias, E, K, rscale, inrm_out, nrm_out, sh_out, idx_out,
                          w_out);
    return true;
}

bool layer_decode_kda(const KdaToken &kda, float *x, const float *in_ln, const float *post_ln,
                      const float *shg, const float *shu, const float *shd, int D, int Iinter,
                      float eps, Act act, float act_a, float act_b, const float *router_w,
                      const float *router_bias, int E, int K, float rscale, float *inrm_out,
                      float *nrm_out, float *sh_out, int *idx_out, float *w_out) {
    if (!kda_args_ok(kda.win_q, kda.qt, kda.win_k, kda.kt, kda.win_v, kda.tv, kda.taps_q, kda.taps_k,
                     kda.taps_v, kda.S, kda.alpha, kda.beta, kda.oh, kda.P, kda.K, kda.H, kda.hd) ||
        !x || !post_ln || !nrm_out || D < 1 || kda.P < D)
        return false;
    cpu_kda_fused(kda.win_q, kda.qt, kda.win_k, kda.kt, kda.win_v, kda.tv, kda.taps_q, kda.taps_k,
                  kda.taps_v, kda.S, kda.alpha, kda.beta, kda.oh, kda.P, kda.K, kda.H, kda.hd);
    cpu_layer_decode_full(x, kda.oh, in_ln, post_ln, shg, shu, shd, D, Iinter, eps, act, act_a,
                          act_b, router_w, router_bias, E, K, rscale, inrm_out, nrm_out, sh_out,
                          idx_out, w_out);
    return true;
}

bool moe_block(int nb, int D, int Iinter, int fmt, int qgs, const void *const *g,
               const void *const *u, const void *const *d, const float *const *gs,
               const float *const *us, const float *const *ds, const float *xg, const int *xoff,
               const int *nr, const int *rows, const float *rw, float *out, int S, Act act,
               float act_a, float act_b) {
    if (nb < 1 || D < 1 || !out || !xg || (fmt != 0 && fmt != 4))
        return false;
    if (!xoff || !nr)
        return false;
    cpu_moe_block(nb, D, Iinter, fmt, qgs, g, u, d, gs, us, ds, xg, xoff, nr, rows, rw, out, S, act,
                  act_a, act_b);
    return true;
}

} // namespace metal_ops
} // namespace mvllm
