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

void cpu_layer_decode(float *x, const float *attn, const float *post_ln, const float *shg,
                      const float *shu, const float *shd, int D, int Iinter, float eps,
                      float *nrm_out, float *sh_out) {
    for (int i = 0; i < D; ++i)
        x[i] += attn[i];
    quant::rmsnorm(x, post_ln, nrm_out, D, eps);
    if (!shg || !shu || !shd || Iinter < 1)
        return;
    std::vector<float> gate(static_cast<size_t>(Iinter));
    std::vector<float> up(static_cast<size_t>(Iinter));
    quant::matmul_f32(gate.data(), nrm_out, shg, 1, D, Iinter);
    quant::matmul_f32(up.data(), nrm_out, shu, 1, D, Iinter);
    quant::silu_mul(gate.data(), up.data(), Iinter);
    if (sh_out)
        quant::matmul_f32(sh_out, gate.data(), shd, 1, Iinter, D);
}

void cpu_moe_block(int nb, int D, int Iinter, int fmt, int qgs, const void *const *g,
                   const void *const *u, const void *const *d, const float *const *gs,
                   const float *const *us, const float *const *ds, const float *xg, const int *xoff,
                   const int *nr, const int *rows, const float *rw, float *out, int S) {
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
            quant::silu_mul(gate.data(), up.data(), nre * Iinter);
            quant::matmul_f32(hh.data(), gate.data(), static_cast<const float *>(d[e]), nre, Iinter,
                              D);
        } else {
            quant::matmul_int4_g64(gate.data(), xe, static_cast<const uint8_t *>(g[e]), gs[e], nre,
                                   D, Iinter);
            quant::matmul_int4_g64(up.data(), xe, static_cast<const uint8_t *>(u[e]), us[e], nre, D,
                                   Iinter);
            quant::silu_mul(gate.data(), up.data(), nre * Iinter);
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
    if (!x || !attn || !post_ln || !nrm_out || D < 1)
        return false;
    cpu_layer_decode(x, attn, post_ln, shg, shu, shd, D, Iinter, eps, nrm_out, sh_out);
    return true;
}

bool moe_block(int nb, int D, int Iinter, int fmt, int qgs, const void *const *g,
               const void *const *u, const void *const *d, const float *const *gs,
               const float *const *us, const float *const *ds, const float *xg, const int *xoff,
               const int *nr, const int *rows, const float *rw, float *out, int S) {
    if (nb < 1 || D < 1 || !out || !xg || (fmt != 0 && fmt != 4))
        return false;
    if (!xoff || !nr)
        return false;
    cpu_moe_block(nb, D, Iinter, fmt, qgs, g, u, d, gs, us, ds, xg, xoff, nr, rows, rw, out, S);
    return true;
}

} // namespace metal_ops
} // namespace mvllm
