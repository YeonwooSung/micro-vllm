#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ops.hpp"

#include "../model/family.hpp"
#include "../quant/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace mvllm {
namespace metal_ops {
namespace {

bool g_inited = false;
bool g_use_metal = false;

id<MTLDevice> g_dev = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLComputePipelineState> g_p_rms = nil;
id<MTLComputePipelineState> g_p_add = nil;
id<MTLComputePipelineState> g_p_silu = nil;
id<MTLComputePipelineState> g_p_act = nil;
id<MTLComputePipelineState> g_p_kda = nil;
id<MTLComputePipelineState> g_p_gdn = nil;
id<MTLComputePipelineState> g_p_gemm_f32 = nil;
id<MTLComputePipelineState> g_p_gemm_i4 = nil;
id<MTLComputePipelineState> g_p_gemm_mx = nil;
id<MTLLibrary> g_vendor_lib = nil;
bool g_vendor = false;
char g_vendor_st[256] = "off";

static const char *kMetalSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct RmsArgs { int nrows; int D; float eps; int has_w; };
struct ElemArgs { int n; };
struct ActArgs { int n; float a; float b; int kind; };
struct KdaArgs { int P; int K; int H; int hd; };
struct GdnArgs { int nq; int nkv; int hd; int group; };
struct GemmArgs { int S; int I; int O; int qgs; };

inline float silu_from(float x) {
    float sig;
    if (x >= 0.0f)
        sig = 1.0f / (1.0f + exp(-x));
    else {
        float z = exp(x);
        sig = z / (1.0f + z);
    }
    return x * sig;
}

void kda_conv_one(device float *x, device float *win, device const float *taps, int p, int K) {
    device float *w = win + (ulong)p * (uint)K;
    for (int i = 0; i < K - 1; ++i)
        w[i] = w[i + 1];
    w[K - 1] = x[p];
    float acc = 0.0f;
    const device float *t = taps + (ulong)p * (uint)K;
    for (int i = 0; i < K; ++i)
        acc += w[i] * t[i];
    x[p] = acc;
}

kernel void op_rmsnorm(device const float *x [[buffer(0)]],
                       device const float *w [[buffer(1)]],
                       device float *y [[buffer(2)]],
                       constant RmsArgs &a [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.nrows) return;
    const device float *xs = x + (ulong)gid * (uint)a.D;
    device float *ys = y + (ulong)gid * (uint)a.D;
    float ss = 0.0f;
    for (int i = 0; i < a.D; ++i)
        ss += xs[i] * xs[i];
    float inv = 1.0f / sqrt(ss / (float)a.D + a.eps);
    if (a.has_w != 0) {
        for (int i = 0; i < a.D; ++i)
            ys[i] = xs[i] * inv * w[i];
    } else {
        for (int i = 0; i < a.D; ++i)
            ys[i] = xs[i] * inv;
    }
}

kernel void op_add(device float *y [[buffer(0)]],
                   device const float *a [[buffer(1)]],
                   constant ElemArgs &e [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)e.n) return;
    y[gid] += a[gid];
}

kernel void op_silu_mul(device float *g [[buffer(0)]],
                        device const float *u [[buffer(1)]],
                        constant ElemArgs &e [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)e.n) return;
    g[gid] = silu_from(g[gid]) * u[gid];
}

kernel void op_act_mul(device float *g [[buffer(0)]],
                       device const float *u [[buffer(1)]],
                       constant ActArgs &e [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)e.n) return;
    float gv = g[gid];
    float uv = u[gid];
    if (e.kind == 1) {
        float lim = e.a;
        if (lim > 0.0f && gv > lim)
            gv = lim;
        float s = silu_from(gv);
        if (lim > 0.0f) {
            if (uv > lim)
                uv = lim;
            if (uv < -lim)
                uv = -lim;
        }
        g[gid] = s * uv;
    } else if (e.kind == 2) {
        float b1 = e.a;
        float b2 = e.b;
        if (b1 == 0.0f)
            b1 = 1.0f;
        if (b2 == 0.0f)
            b2 = 1.0f;
        float sg;
        if (gv >= 0.0f)
            sg = 1.0f / (1.0f + exp(-gv));
        else {
            float z = exp(gv);
            sg = z / (1.0f + z);
        }
        g[gid] = b1 * tanh(gv / b1) * sg * b2 * tanh(uv / b2);
    } else {
        g[gid] = silu_from(gv) * uv;
    }
}

kernel void op_kda_fused(device float *win_q [[buffer(0)]],
                         device float *qt [[buffer(1)]],
                         device float *win_k [[buffer(2)]],
                         device float *kt [[buffer(3)]],
                         device float *win_v [[buffer(4)]],
                         device float *tv [[buffer(5)]],
                         device const float *taps_q [[buffer(6)]],
                         device const float *taps_k [[buffer(7)]],
                         device const float *taps_v [[buffer(8)]],
                         device float *S [[buffer(9)]],
                         device const float *alpha [[buffer(10)]],
                         device const float *beta [[buffer(11)]],
                         device float *oh [[buffer(12)]],
                         constant KdaArgs &a [[buffer(13)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.H) return;
    const int D = a.hd;
    const int K = a.K;
    const int base = (int)gid * D;

    if (K > 1) {
        for (int d = 0; d < D; ++d) {
            const int p = base + d;
            kda_conv_one(qt, win_q, taps_q, p, K);
            kda_conv_one(kt, win_k, taps_k, p, K);
            kda_conv_one(tv, win_v, taps_v, p, K);
        }
    }

    device float *qh = qt + base;
    device float *kh = kt + base;
    device float *vh = tv + base;
    for (int d = 0; d < D; ++d) {
        qh[d] = silu_from(qh[d]);
        kh[d] = silu_from(kh[d]);
        vh[d] = silu_from(vh[d]);
    }

    float accq = 0.0f, acck = 0.0f;
    for (int d = 0; d < D; ++d) {
        accq += qh[d] * qh[d];
        acck += kh[d] * kh[d];
    }
    const float invq = 1.0f / sqrt(accq + 1.0e-6f);
    const float invk = 1.0f / sqrt(acck + 1.0e-6f);
    const float qscale = 1.0f / sqrt((float)D);
    for (int d = 0; d < D; ++d) {
        qh[d] *= invq * qscale;
        kh[d] *= invk;
    }

    device float *Sh = S + (ulong)gid * (uint)(D * D);
    device float *outh = oh + base;
    const float b = beta[gid];

    for (int i = 0; i < D; ++i) {
        const float al = alpha[base + i];
        for (int j = 0; j < D; ++j)
            Sh[i * D + j] *= al;
    }

    // Scratch ktS in oh (caller-zeroed), then overwrite with S^T q.
    for (int j = 0; j < D; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < D; ++i)
            acc += kh[i] * Sh[i * D + j];
        outh[j] = acc;
    }
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j)
            Sh[i * D + j] -= b * kh[i] * outh[j];
        for (int j = 0; j < D; ++j)
            Sh[i * D + j] += b * kh[i] * vh[j];
    }
    for (int j = 0; j < D; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < D; ++i)
            acc += Sh[i * D + j] * qh[i];
        outh[j] = acc;
    }
}

kernel void op_gdn_delta(device float *S [[buffer(0)]],
                         device float *ctx [[buffer(1)]],
                         device const float *q [[buffer(2)]],
                         device const float *k [[buffer(3)]],
                         device const float *v [[buffer(4)]],
                         constant GdnArgs &a [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.nq) return;
    const int D = a.hd;
    const int g = a.group > 0 ? a.group : 1;
    int kh = (int)gid / g;
    if (kh > a.nkv - 1)
        kh = a.nkv - 1;
    const device float *kk = k + (ulong)kh * (uint)D;
    const device float *vv = v + (ulong)kh * (uint)D;
    const device float *qq = q + (ulong)gid * (uint)D;
    device float *Sh = S + (ulong)gid * (uint)(D * D);
    device float *ch = ctx + (ulong)gid * (uint)D;
    const float beta = 1.0f / (1.0f + exp(-kk[0]));
    // Stash S^T k in this head's ctx slice, then overwrite with S^T q.
    for (int j = 0; j < D; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < D; ++i)
            acc += Sh[i * D + j] * kk[i];
        ch[j] = acc;
    }
    for (int i = 0; i < D; ++i) {
        const float ki = kk[i];
        for (int j = 0; j < D; ++j)
            Sh[i * D + j] += beta * (ki * vv[j] - ki * ch[j]);
    }
    for (int j = 0; j < D; ++j) {
        float o = 0.0f;
        for (int i = 0; i < D; ++i)
            o += qq[i] * Sh[i * D + j];
        ch[j] = o;
    }
}

kernel void op_gemm_f32(device const float *x [[buffer(0)]],
                        device const float *w [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant GemmArgs &a [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint o = gid.x;
    uint s = gid.y;
    if (o >= (uint)a.O || s >= (uint)a.S) return;
    const device float *xs = x + (ulong)s * (uint)a.I;
    const device float *wo = w + (ulong)o * (uint)a.I;
    float acc = 0.0f;
    for (int i = 0; i < a.I; ++i)
        acc += xs[i] * wo[i];
    y[(ulong)s * (uint)a.O + o] = acc;
}

kernel void op_gemm_int4_g64(device const float *x [[buffer(0)]],
                             device const uchar *packed [[buffer(1)]],
                             device const float *scales [[buffer(2)]],
                             device float *y [[buffer(3)]],
                             constant GemmArgs &a [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint o = gid.x;
    uint s = gid.y;
    if (o >= (uint)a.O || s >= (uint)a.S) return;
    const int I = a.I;
    const int gs = a.qgs > 0 ? a.qgs : 64;
    const int stride = (I + 1) / 2;
    const device float *xs = x + (ulong)s * (uint)I;
    const device uchar *prow = packed + (ulong)o * (uint)stride;
    const device float *srow = scales + (ulong)o * (uint)((I + gs - 1) / gs);
    float acc = 0.0f;
    for (int i = 0; i < I; ++i) {
        uchar b = prow[i >> 1];
        uint nib = (i & 1) ? (b >> 4) : (b & 0x0f);
        int q = (int)nib - 8;
        acc += xs[i] * ((float)q * srow[i / gs]);
    }
    y[(ulong)s * (uint)a.O + o] = acc;
}

constant float MX4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

kernel void op_gemm_mxfp4(device const float *x [[buffer(0)]],
                          device const uchar *packed [[buffer(1)]],
                          device const uchar *scales [[buffer(2)]],
                          device float *y [[buffer(3)]],
                          constant GemmArgs &a [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint o = gid.x;
    uint s = gid.y;
    if (o >= (uint)a.O || s >= (uint)a.S) return;
    const int I = a.I;
    const int stride = (I + 1) / 2;
    const int ng = (I + 31) / 32;
    const device float *xs = x + (ulong)s * (uint)I;
    const device uchar *prow = packed + (ulong)o * (uint)stride;
    const device uchar *srow = scales + (ulong)o * (uint)ng;
    float acc = 0.0f;
    for (int i = 0; i < I; ++i) {
        uchar b = prow[i >> 1];
        uint nib = (i & 1) ? (b >> 4) : (b & 0x0f);
        uint e = srow[i / 32];
        float scale = as_type<float>(e << 23);
        acc += xs[i] * (MX4_LUT[nib] * scale);
    }
    y[(ulong)s * (uint)a.O + o] = acc;
}
)MSL";

struct RmsArgs {
    int nrows, D;
    float eps;
    int has_w;
};
struct ElemArgs {
    int n;
};
struct ActArgs {
    int n;
    float a, b;
    int kind;
};
struct KdaArgs {
    int P, K, H, hd;
};
struct GdnArgs {
    int nq, nkv, hd, group;
};
struct GemmArgs {
    int S, I, O, qgs;
};

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

void cpu_gdn_delta(float *S, float *ctx, const float *q, const float *k, const float *v, int nq,
                   int nkv, int hd, int group) {
    const int g = std::max(group, 1);
    std::vector<float> attn(static_cast<size_t>(hd), 0.f);
    for (int hh = 0; hh < nq; ++hh) {
        const int kh = std::min(hh / g, nkv - 1);
        const float *kk = k + kh * hd;
        const float *vv = v + kh * hd;
        const float *qq = q + hh * hd;
        float *Sh = S + static_cast<size_t>(hh) * hd * hd;
        const float beta = 1.f / (1.f + std::exp(-kk[0]));
        for (int j = 0; j < hd; ++j) {
            float a = 0.f;
            for (int i = 0; i < hd; ++i)
                a += Sh[static_cast<size_t>(i) * hd + j] * kk[i];
            attn[static_cast<size_t>(j)] = a;
        }
        for (int i = 0; i < hd; ++i)
            for (int j = 0; j < hd; ++j)
                Sh[static_cast<size_t>(i) * hd + j] +=
                    beta * (kk[i] * vv[j] - kk[i] * attn[static_cast<size_t>(j)]);
        for (int j = 0; j < hd; ++j) {
            float o = 0.f;
            for (int i = 0; i < hd; ++i)
                o += qq[i] * Sh[static_cast<size_t>(i) * hd + j];
            ctx[hh * hd + j] = o;
        }
    }
}

bool gdn_args_ok(const float *S, const float *ctx, const float *q, const float *k, const float *v,
                 int nq, int nkv, int hd) {
    return S && ctx && q && k && v && nq >= 1 && nkv >= 1 && hd >= 1;
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
        const float b = router_bias ? router_bias[i] : 0.f;
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

void cpu_layer_decode_kda(const KdaToken &kda, float *x, const float *in_ln, const float *post_ln,
                          const float *shg, const float *shu, const float *shd, int D, int Iinter,
                          float eps, Act act, float act_a, float act_b, const float *router_w,
                          const float *router_bias, int E, int K, float rscale, float *inrm_out,
                          float *nrm_out, float *sh_out, int *idx_out, float *w_out) {
    cpu_kda_fused(kda.win_q, kda.qt, kda.win_k, kda.kt, kda.win_v, kda.tv, kda.taps_q, kda.taps_k,
                  kda.taps_v, kda.S, kda.alpha, kda.beta, kda.oh, kda.P, kda.K, kda.H, kda.hd);
    cpu_layer_decode_full(x, kda.oh, in_ln, post_ln, shg, shu, shd, D, Iinter, eps, act, act_a,
                          act_b, router_w, router_bias, E, K, rscale, inrm_out, nrm_out, sh_out,
                          idx_out, w_out);
}

bool mla_args_ok(const MlaAbsorb &m, const float *x, const float *post_ln, int D,
                 const float *nrm_out) {
    return m.q && m.cache && m.w_kt && m.w_v && m.oh && x && post_ln && nrm_out && m.H >= 1 &&
           m.QK >= 1 && m.Vh >= 1 && m.L >= 1 && m.R >= 0 && m.T >= 1 && m.stride >= m.L + m.R &&
           D >= 1;
}

// score_j = (W_kt q_nope)·c_j + q_rot·R_j; ctx = W_v (Σ a_j c_j).
void cpu_mla_absorb(float *ctx, const MlaAbsorb &m) {
    const int QH = m.QK + m.R;
    const float scale = 1.f / std::sqrt(static_cast<float>(QH > 0 ? QH : 1));
    std::vector<float> qabs(static_cast<size_t>(m.L));
    std::vector<float> scores(static_cast<size_t>(m.T));
    std::vector<float> pooled(static_cast<size_t>(m.L));
    for (int h = 0; h < m.H; ++h) {
        const float *qh = m.q + static_cast<size_t>(h) * QH;
        const float *wkt = m.w_kt + static_cast<size_t>(h) * m.L * m.QK;
        for (int k = 0; k < m.L; ++k) {
            float acc = 0.f;
            const float *row = wkt + static_cast<size_t>(k) * m.QK;
            for (int i = 0; i < m.QK; ++i)
                acc += row[i] * qh[i];
            qabs[static_cast<size_t>(k)] = acc;
        }
        for (int t = 0; t < m.T; ++t) {
            const float *ct = m.cache + static_cast<size_t>(t) * m.stride;
            float s = 0.f;
            for (int i = 0; i < m.L; ++i)
                s += qabs[static_cast<size_t>(i)] * ct[i];
            for (int i = 0; i < m.R; ++i)
                s += qh[m.QK + i] * ct[m.L + i];
            scores[static_cast<size_t>(t)] = s * scale;
        }
        quant::softmax_inplace(scores.data(), m.T);
        std::fill(pooled.begin(), pooled.end(), 0.f);
        for (int t = 0; t < m.T; ++t) {
            const float *ct = m.cache + static_cast<size_t>(t) * m.stride;
            const float a = scores[static_cast<size_t>(t)];
            for (int i = 0; i < m.L; ++i)
                pooled[static_cast<size_t>(i)] += a * ct[i];
        }
        float *ch = ctx + static_cast<size_t>(h) * m.Vh;
        const float *wv = m.w_v + static_cast<size_t>(h) * m.Vh * m.L;
        for (int v = 0; v < m.Vh; ++v) {
            float acc = 0.f;
            const float *row = wv + static_cast<size_t>(v) * m.L;
            for (int i = 0; i < m.L; ++i)
                acc += row[i] * pooled[static_cast<size_t>(i)];
            ch[v] = acc;
        }
    }
}

void cpu_layer_decode_mla(const MlaAbsorb &m, float *x, const float *post_ln, int D, float eps,
                          float *nrm_out) {
    const int hv = m.H * m.Vh;
    std::vector<float> ctx(static_cast<size_t>(std::max(hv, D)), 0.f);
    cpu_mla_absorb(ctx.data(), m);
    if (m.w_o)
        quant::matmul_f32(m.oh, ctx.data(), m.w_o, 1, hv, D);
    else
        std::memcpy(m.oh, ctx.data(), static_cast<size_t>(std::max(hv, D)) * sizeof(float));
    cpu_layer_decode_full(x, m.oh, nullptr, post_ln, nullptr, nullptr, nullptr, D, 0, eps,
                          Act::Silu, 0.f, 0.f, nullptr, nullptr, 0, 0, 1.f, nullptr, nrm_out,
                          nullptr, nullptr, nullptr);
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
        if ((fmt == 4 || fmt == 7) && (!gs || !us || !ds || !gs[e] || !us[e] || !ds[e])) {
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
        } else if (fmt == 7) {
            const uint8_t *gsc = reinterpret_cast<const uint8_t *>(gs[e]);
            const uint8_t *usc = reinterpret_cast<const uint8_t *>(us[e]);
            const uint8_t *dsc = reinterpret_cast<const uint8_t *>(ds[e]);
            quant::matmul_mxfp4(gate.data(), xe, static_cast<const uint8_t *>(g[e]), gsc, nre, D,
                                Iinter);
            quant::matmul_mxfp4(up.data(), xe, static_cast<const uint8_t *>(u[e]), usc, nre, D,
                                Iinter);
            apply_act(gate.data(), up.data(), nre * Iinter, act, act_a, act_b);
            quant::matmul_mxfp4(hh.data(), gate.data(), static_cast<const uint8_t *>(d[e]), dsc, nre,
                                Iinter, D);
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

id<MTLBuffer> buf_bytes(const void *p, size_t n) {
    if (n == 0)
        n = 1;
    id<MTLBuffer> b = [g_dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    if (p && n)
        std::memcpy([b contents], p, n);
    return b;
}

id<MTLBuffer> buf_empty(size_t n) {
    if (n == 0)
        n = 1;
    return [g_dev newBufferWithLength:n options:MTLResourceStorageModeShared];
}

void dispatch1(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, uint n,
               NSArray *bufs, const void *uni, size_t uni_n) {
    [enc setComputePipelineState:pso];
    NSUInteger i = 0;
    for (id b in bufs)
        [enc setBuffer:b offset:0 atIndex:i++];
    if (uni && uni_n)
        [enc setBytes:uni length:uni_n atIndex:i];
    MTLSize tg = MTLSizeMake(64, 1, 1);
    MTLSize grid = MTLSizeMake(((n + 63u) / 64u) * 64u, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

void dispatch2(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, uint gx, uint gy,
               NSArray *bufs, const void *uni, size_t uni_n) {
    [enc setComputePipelineState:pso];
    NSUInteger i = 0;
    for (id b in bufs)
        [enc setBuffer:b offset:0 atIndex:i++];
    if (uni && uni_n)
        [enc setBytes:uni length:uni_n atIndex:i];
    MTLSize tg = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake(((gx + 15u) / 16u) * 16u, ((gy + 15u) / 16u) * 16u, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

bool commit_wait(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    if (!cb || !enc)
        return false;
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return cb.error == nil;
}

void drop_metal() {
    g_p_rms = nil;
    g_p_add = nil;
    g_p_silu = nil;
    g_p_act = nil;
    g_p_kda = nil;
    g_p_gdn = nil;
    g_p_gemm_f32 = nil;
    g_p_gemm_i4 = nil;
    g_p_gemm_mx = nil;
    g_queue = nil;
    g_dev = nil;
    g_use_metal = false;
}

} // namespace

bool init() {
    @autoreleasepool {
        if (g_inited)
            return true;
        g_inited = true;
        g_use_metal = false;
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev)
            return true;
        g_queue = [g_dev newCommandQueue];
        if (!g_queue) {
            g_dev = nil;
            return true;
        }
        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:kMetalSrc];
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            drop_metal();
            return true;
        }
        auto pso = [&](const char *name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (!fn)
                return nil;
            NSError *e = nil;
            return [g_dev newComputePipelineStateWithFunction:fn error:&e];
        };
        g_p_rms = pso("op_rmsnorm");
        g_p_add = pso("op_add");
        g_p_silu = pso("op_silu_mul");
        g_p_act = pso("op_act_mul");
        g_p_kda = pso("op_kda_fused");
        g_p_gdn = pso("op_gdn_delta");
        g_p_gemm_f32 = pso("op_gemm_f32");
        g_p_gemm_i4 = pso("op_gemm_int4_g64");
        g_p_gemm_mx = pso("op_gemm_mxfp4");
        if (g_p_rms && g_p_add && g_p_silu)
            g_use_metal = true;
        else
            drop_metal();
#if defined(MVLLM_VENDOR_METAL)
        g_vendor = false;
        std::snprintf(g_vendor_st, sizeof(g_vendor_st), "off");
        if (g_dev) {
            NSString *path = @MVLLM_VENDOR_METAL_DIR "/coli_metal_kernels.metal";
            NSError *verr = nil;
            NSString *vsrc = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding
                                                          error:&verr];
            if (!vsrc) {
                std::snprintf(g_vendor_st, sizeof(g_vendor_st), "err: read");
            } else {
                MTLCompileOptions *opt = [MTLCompileOptions new];
                id<MTLLibrary> vlib = [g_dev newLibraryWithSource:vsrc options:opt error:&verr];
                if (!vlib) {
                    std::snprintf(g_vendor_st, sizeof(g_vendor_st), "err: compile");
                } else {
                    g_vendor_lib = vlib;
                    g_vendor = true;
                    std::snprintf(g_vendor_st, sizeof(g_vendor_st), "coli");
                }
            }
        }
#endif
        return true;
    }
}

void shutdown() {
    @autoreleasepool {
        drop_metal();
        g_vendor_lib = nil;
        g_vendor = false;
        std::snprintf(g_vendor_st, sizeof(g_vendor_st), "off");
        g_inited = false;
    }
}

bool available() { return g_inited; }

const char *backend_name() { return g_use_metal ? "metal" : "cpu"; }

bool vendor_loaded() { return g_vendor; }

const char *vendor_status() { return g_vendor_st; }

bool rmsnorm(float *y, const float *x, const float *w, int nrows, int D, float eps) {
    if (!y || !x || nrows < 1 || D < 1)
        return false;
    if (!g_use_metal || !g_p_rms) {
        cpu_rmsnorm(y, x, w, nrows, D, eps);
        return true;
    }
    @autoreleasepool {
        const size_t xb = sizeof(float) * static_cast<size_t>(nrows) * D;
        const size_t wb = w ? sizeof(float) * static_cast<size_t>(D) : 0;
        id<MTLBuffer> bx = buf_bytes(x, xb);
        id<MTLBuffer> bw = buf_bytes(w, wb);
        id<MTLBuffer> by = buf_empty(xb);
        if (!bx || !bw || !by) {
            cpu_rmsnorm(y, x, w, nrows, D, eps);
            return true;
        }
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) {
            cpu_rmsnorm(y, x, w, nrows, D, eps);
            return true;
        }
        RmsArgs a{nrows, D, eps, w ? 1 : 0};
        dispatch1(enc, g_p_rms, static_cast<uint>(nrows), @[ bx, bw, by ], &a, sizeof(a));
        if (!commit_wait(cb, enc)) {
            cpu_rmsnorm(y, x, w, nrows, D, eps);
            return true;
        }
        std::memcpy(y, [by contents], xb);
    }
    return true;
}

bool add(float *y, const float *a, size_t n) {
    if (!y || !a)
        return false;
    if (n == 0)
        return true;
    if (!g_use_metal || !g_p_add) {
        cpu_add(y, a, n);
        return true;
    }
    @autoreleasepool {
        const size_t nb = sizeof(float) * n;
        id<MTLBuffer> by = buf_bytes(y, nb);
        id<MTLBuffer> ba = buf_bytes(a, nb);
        if (!by || !ba) {
            cpu_add(y, a, n);
            return true;
        }
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) {
            cpu_add(y, a, n);
            return true;
        }
        ElemArgs e{static_cast<int>(n)};
        dispatch1(enc, g_p_add, static_cast<uint>(n), @[ by, ba ], &e, sizeof(e));
        if (!commit_wait(cb, enc)) {
            cpu_add(y, a, n);
            return true;
        }
        std::memcpy(y, [by contents], nb);
    }
    return true;
}

bool silu_mul(float *g, const float *u, size_t n) {
    if (!g || !u)
        return false;
    if (n == 0)
        return true;
    if (!g_use_metal || !g_p_silu) {
        cpu_silu_mul(g, u, n);
        return true;
    }
    @autoreleasepool {
        const size_t nb = sizeof(float) * n;
        id<MTLBuffer> bg = buf_bytes(g, nb);
        id<MTLBuffer> bu = buf_bytes(u, nb);
        if (!bg || !bu) {
            cpu_silu_mul(g, u, n);
            return true;
        }
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) {
            cpu_silu_mul(g, u, n);
            return true;
        }
        ElemArgs e{static_cast<int>(n)};
        dispatch1(enc, g_p_silu, static_cast<uint>(n), @[ bg, bu ], &e, sizeof(e));
        if (!commit_wait(cb, enc)) {
            cpu_silu_mul(g, u, n);
            return true;
        }
        std::memcpy(g, [bg contents], nb);
    }
    return true;
}

bool kda_fused_token(float *win_q, float *qt, float *win_k, float *kt, float *win_v, float *tv,
                     const float *taps_q, const float *taps_k, const float *taps_v, float *S,
                     const float *alpha, const float *beta, float *oh, int P, int K, int H,
                     int hd) {
    if (!kda_args_ok(win_q, qt, win_k, kt, win_v, tv, taps_q, taps_k, taps_v, S, alpha, beta, oh, P,
                     K, H, hd))
        return false;
    if (!g_use_metal || !g_p_kda) {
        cpu_kda_fused(win_q, qt, win_k, kt, win_v, tv, taps_q, taps_k, taps_v, S, alpha, beta, oh, P,
                      K, H, hd);
        return true;
    }
    @autoreleasepool {
        const size_t pb = sizeof(float) * static_cast<size_t>(P);
        const size_t wb = sizeof(float) * static_cast<size_t>(P) * K;
        const size_t sb = sizeof(float) * static_cast<size_t>(H) * hd * hd;
        const size_t ab = sizeof(float) * static_cast<size_t>(H) * hd;
        const size_t bb = sizeof(float) * static_cast<size_t>(H);
        id<MTLBuffer> b_win_q = buf_bytes(win_q, wb);
        id<MTLBuffer> b_qt = buf_bytes(qt, pb);
        id<MTLBuffer> b_win_k = buf_bytes(win_k, wb);
        id<MTLBuffer> b_kt = buf_bytes(kt, pb);
        id<MTLBuffer> b_win_v = buf_bytes(win_v, wb);
        id<MTLBuffer> b_tv = buf_bytes(tv, pb);
        id<MTLBuffer> b_tq = buf_bytes(taps_q, wb);
        id<MTLBuffer> b_tk = buf_bytes(taps_k, wb);
        id<MTLBuffer> b_tvt = buf_bytes(taps_v, wb);
        id<MTLBuffer> b_S = buf_bytes(S, sb);
        id<MTLBuffer> b_al = buf_bytes(alpha, ab);
        id<MTLBuffer> b_be = buf_bytes(beta, bb);
        id<MTLBuffer> b_oh = buf_bytes(oh, pb);
        if (!b_win_q || !b_qt || !b_win_k || !b_kt || !b_win_v || !b_tv || !b_tq || !b_tk ||
            !b_tvt || !b_S || !b_al || !b_be || !b_oh) {
            cpu_kda_fused(win_q, qt, win_k, kt, win_v, tv, taps_q, taps_k, taps_v, S, alpha, beta,
                          oh, P, K, H, hd);
            return true;
        }
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) {
            cpu_kda_fused(win_q, qt, win_k, kt, win_v, tv, taps_q, taps_k, taps_v, S, alpha, beta,
                          oh, P, K, H, hd);
            return true;
        }
        KdaArgs args{P, K, H, hd};
        NSArray *bufs = @[
            b_win_q, b_qt, b_win_k, b_kt, b_win_v, b_tv, b_tq, b_tk, b_tvt, b_S, b_al, b_be, b_oh
        ];
        dispatch1(enc, g_p_kda, static_cast<uint>(H), bufs, &args, sizeof(args));
        if (!commit_wait(cb, enc)) {
            cpu_kda_fused(win_q, qt, win_k, kt, win_v, tv, taps_q, taps_k, taps_v, S, alpha, beta,
                          oh, P, K, H, hd);
            return true;
        }
        std::memcpy(win_q, [b_win_q contents], wb);
        std::memcpy(qt, [b_qt contents], pb);
        std::memcpy(win_k, [b_win_k contents], wb);
        std::memcpy(kt, [b_kt contents], pb);
        std::memcpy(win_v, [b_win_v contents], wb);
        std::memcpy(tv, [b_tv contents], pb);
        std::memcpy(S, [b_S contents], sb);
        std::memcpy(oh, [b_oh contents], pb);
    }
    return true;
}

bool gdn_delta(float *S, float *ctx, const float *q, const float *k, const float *v, int nq,
               int nkv, int hd, int group) {
    if (!gdn_args_ok(S, ctx, q, k, v, nq, nkv, hd))
        return false;
    if (!g_use_metal || !g_p_gdn) {
        cpu_gdn_delta(S, ctx, q, k, v, nq, nkv, hd, group);
        return true;
    }
    @autoreleasepool {
        const size_t sb = sizeof(float) * static_cast<size_t>(nq) * hd * hd;
        const size_t qb = sizeof(float) * static_cast<size_t>(nq) * hd;
        const size_t kb = sizeof(float) * static_cast<size_t>(nkv) * hd;
        id<MTLBuffer> b_S = buf_bytes(S, sb);
        id<MTLBuffer> b_ctx = buf_bytes(ctx, qb);
        id<MTLBuffer> b_q = buf_bytes(q, qb);
        id<MTLBuffer> b_k = buf_bytes(k, kb);
        id<MTLBuffer> b_v = buf_bytes(v, kb);
        if (!b_S || !b_ctx || !b_q || !b_k || !b_v) {
            cpu_gdn_delta(S, ctx, q, k, v, nq, nkv, hd, group);
            return true;
        }
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) {
            cpu_gdn_delta(S, ctx, q, k, v, nq, nkv, hd, group);
            return true;
        }
        GdnArgs args{nq, nkv, hd, group};
        dispatch1(enc, g_p_gdn, static_cast<uint>(nq), @[ b_S, b_ctx, b_q, b_k, b_v ], &args,
                  sizeof(args));
        if (!commit_wait(cb, enc)) {
            cpu_gdn_delta(S, ctx, q, k, v, nq, nkv, hd, group);
            return true;
        }
        std::memcpy(S, [b_S contents], sb);
        std::memcpy(ctx, [b_ctx contents], qb);
    }
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
    const bool want_sh = shg && shu && shd && Iinter > 0 && sh_out;
    const bool want_rt = router_w && E > 0 && K > 0 && idx_out && w_out;
    const bool need_act = want_sh && act != Act::Silu;
    if (!attn || !g_use_metal || !g_p_add || !g_p_rms || (want_sh && !g_p_gemm_f32) ||
        (need_act && !g_p_act) || (want_rt && !g_p_gemm_f32)) {
        cpu_layer_decode_full(x, attn, in_ln, post_ln, shg, shu, shd, D, Iinter, eps, act, act_a,
                              act_b, router_w, router_bias, E, K, rscale, inrm_out, nrm_out, sh_out,
                              idx_out, w_out);
        return true;
    }
    if (in_ln && inrm_out)
        quant::rmsnorm(x, in_ln, inrm_out, D, eps);
    @autoreleasepool {
        const size_t db = sizeof(float) * static_cast<size_t>(D);
        const size_t ib = want_sh ? sizeof(float) * static_cast<size_t>(Iinter) : 0;
        const size_t shg_b = want_sh ? sizeof(float) * static_cast<size_t>(Iinter) * D : 0;
        const size_t shd_b = want_sh ? sizeof(float) * static_cast<size_t>(D) * Iinter : 0;
        const size_t rb = want_rt ? sizeof(float) * static_cast<size_t>(E) * D : 0;
        const size_t sb = want_rt ? sizeof(float) * static_cast<size_t>(E) : 0;
        id<MTLBuffer> bx = buf_bytes(x, db);
        id<MTLBuffer> ba = buf_bytes(attn, db);
        id<MTLBuffer> bw = buf_bytes(post_ln, db);
        id<MTLBuffer> bn = buf_empty(db);
        id<MTLBuffer> bshg = want_sh ? buf_bytes(shg, shg_b) : nil;
        id<MTLBuffer> bshu = want_sh ? buf_bytes(shu, shg_b) : nil;
        id<MTLBuffer> bshd = want_sh ? buf_bytes(shd, shd_b) : nil;
        id<MTLBuffer> bgate = want_sh ? buf_empty(ib) : nil;
        id<MTLBuffer> bup = want_sh ? buf_empty(ib) : nil;
        id<MTLBuffer> bsho = want_sh ? buf_empty(db) : nil;
        id<MTLBuffer> brw = want_rt ? buf_bytes(router_w, rb) : nil;
        id<MTLBuffer> bsc = want_rt ? buf_empty(sb) : nil;
        if (!bx || !ba || !bw || !bn ||
            (want_sh && (!bshg || !bshu || !bshd || !bgate || !bup || !bsho)) ||
            (want_rt && (!brw || !bsc))) {
            cpu_layer_decode_full(x, attn, in_ln, post_ln, shg, shu, shd, D, Iinter, eps, act, act_a,
                                  act_b, router_w, router_bias, E, K, rscale, inrm_out, nrm_out,
                                  sh_out, idx_out, w_out);
            return true;
        }
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) {
            cpu_layer_decode_full(x, attn, in_ln, post_ln, shg, shu, shd, D, Iinter, eps, act, act_a,
                                  act_b, router_w, router_bias, E, K, rscale, inrm_out, nrm_out,
                                  sh_out, idx_out, w_out);
            return true;
        }
        ElemArgs ea{D};
        dispatch1(enc, g_p_add, static_cast<uint>(D), @[ bx, ba ], &ea, sizeof(ea));
        RmsArgs ra{1, D, eps, 1};
        dispatch1(enc, g_p_rms, 1u, @[ bx, bw, bn ], &ra, sizeof(ra));
        if (want_sh) {
            GemmArgs g1{1, D, Iinter, 0};
            dispatch2(enc, g_p_gemm_f32, static_cast<uint>(Iinter), 1u, @[ bn, bshg, bgate ], &g1,
                      sizeof(g1));
            dispatch2(enc, g_p_gemm_f32, static_cast<uint>(Iinter), 1u, @[ bn, bshu, bup ], &g1,
                      sizeof(g1));
            ActArgs aa{Iinter, act_a, act_b, static_cast<int>(act)};
            if (g_p_act)
                dispatch1(enc, g_p_act, static_cast<uint>(Iinter), @[ bgate, bup ], &aa, sizeof(aa));
            else {
                ElemArgs se{Iinter};
                dispatch1(enc, g_p_silu, static_cast<uint>(Iinter), @[ bgate, bup ], &se, sizeof(se));
            }
            GemmArgs g2{1, Iinter, D, 0};
            dispatch2(enc, g_p_gemm_f32, static_cast<uint>(D), 1u, @[ bgate, bshd, bsho ], &g2,
                      sizeof(g2));
        }
        if (want_rt) {
            GemmArgs gr{1, D, E, 0};
            dispatch2(enc, g_p_gemm_f32, static_cast<uint>(E), 1u, @[ bn, brw, bsc ], &gr,
                      sizeof(gr));
        }
        if (!commit_wait(cb, enc)) {
            cpu_layer_decode_full(x, attn, in_ln, post_ln, shg, shu, shd, D, Iinter, eps, act, act_a,
                                  act_b, router_w, router_bias, E, K, rscale, inrm_out, nrm_out,
                                  sh_out, idx_out, w_out);
            return true;
        }
        std::memcpy(x, [bx contents], db);
        std::memcpy(nrm_out, [bn contents], db);
        if (want_sh)
            std::memcpy(sh_out, [bsho contents], db);
        if (want_rt) {
            std::vector<float> scores(static_cast<size_t>(E));
            std::memcpy(scores.data(), [bsc contents], sb);
            std::vector<float> choice(static_cast<size_t>(E));
            for (int i = 0; i < E; ++i) {
                scores[static_cast<size_t>(i)] = quant::sigmoid(scores[static_cast<size_t>(i)]);
                const float b = router_bias ? router_bias[i] : 0.f;
                choice[static_cast<size_t>(i)] = scores[static_cast<size_t>(i)] + b;
            }
            moe_topk(choice.data(), E, K, idx_out, w_out, scores.data());
            if (rscale != 1.f) {
                const int kk = K > E ? E : K;
                for (int i = 0; i < kk; ++i)
                    w_out[i] *= rscale;
            }
        }
    }
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
    const bool want_sh = shg && shu && shd && Iinter > 0 && sh_out;
    const bool want_rt = router_w && E > 0 && K > 0 && idx_out && w_out;
    const bool need_act = want_sh && act != Act::Silu;
    if (!g_use_metal || !g_p_kda || !g_p_add || !g_p_rms || (want_sh && !g_p_gemm_f32) ||
        (need_act && !g_p_act) || (want_rt && !g_p_gemm_f32)) {
        cpu_layer_decode_kda(kda, x, in_ln, post_ln, shg, shu, shd, D, Iinter, eps, act, act_a,
                             act_b, router_w, router_bias, E, K, rscale, inrm_out, nrm_out, sh_out,
                             idx_out, w_out);
        return true;
    }
    if (in_ln && inrm_out)
        quant::rmsnorm(x, in_ln, inrm_out, D, eps);
    @autoreleasepool {
        const size_t pb = sizeof(float) * static_cast<size_t>(kda.P);
        const size_t wb = sizeof(float) * static_cast<size_t>(kda.P) * kda.K;
        const size_t sbk = sizeof(float) * static_cast<size_t>(kda.H) * kda.hd * kda.hd;
        const size_t ab = sizeof(float) * static_cast<size_t>(kda.H) * kda.hd;
        const size_t bb = sizeof(float) * static_cast<size_t>(kda.H);
        const size_t db = sizeof(float) * static_cast<size_t>(D);
        const size_t ib = want_sh ? sizeof(float) * static_cast<size_t>(Iinter) : 0;
        const size_t shg_b = want_sh ? sizeof(float) * static_cast<size_t>(Iinter) * D : 0;
        const size_t shd_b = want_sh ? sizeof(float) * static_cast<size_t>(D) * Iinter : 0;
        const size_t rb = want_rt ? sizeof(float) * static_cast<size_t>(E) * D : 0;
        const size_t sb = want_rt ? sizeof(float) * static_cast<size_t>(E) : 0;
        id<MTLBuffer> b_win_q = buf_bytes(kda.win_q, wb);
        id<MTLBuffer> b_qt = buf_bytes(kda.qt, pb);
        id<MTLBuffer> b_win_k = buf_bytes(kda.win_k, wb);
        id<MTLBuffer> b_kt = buf_bytes(kda.kt, pb);
        id<MTLBuffer> b_win_v = buf_bytes(kda.win_v, wb);
        id<MTLBuffer> b_tv = buf_bytes(kda.tv, pb);
        id<MTLBuffer> b_tq = buf_bytes(kda.taps_q, wb);
        id<MTLBuffer> b_tk = buf_bytes(kda.taps_k, wb);
        id<MTLBuffer> b_tvt = buf_bytes(kda.taps_v, wb);
        id<MTLBuffer> b_S = buf_bytes(kda.S, sbk);
        id<MTLBuffer> b_al = buf_bytes(kda.alpha, ab);
        id<MTLBuffer> b_be = buf_bytes(kda.beta, bb);
        id<MTLBuffer> b_oh = buf_bytes(kda.oh, pb);
        id<MTLBuffer> bx = buf_bytes(x, db);
        id<MTLBuffer> bw = buf_bytes(post_ln, db);
        id<MTLBuffer> bn = buf_empty(db);
        id<MTLBuffer> bshg = want_sh ? buf_bytes(shg, shg_b) : nil;
        id<MTLBuffer> bshu = want_sh ? buf_bytes(shu, shg_b) : nil;
        id<MTLBuffer> bshd = want_sh ? buf_bytes(shd, shd_b) : nil;
        id<MTLBuffer> bgate = want_sh ? buf_empty(ib) : nil;
        id<MTLBuffer> bup = want_sh ? buf_empty(ib) : nil;
        id<MTLBuffer> bsho = want_sh ? buf_empty(db) : nil;
        id<MTLBuffer> brw = want_rt ? buf_bytes(router_w, rb) : nil;
        id<MTLBuffer> bsc = want_rt ? buf_empty(sb) : nil;
        if (!b_win_q || !b_qt || !b_win_k || !b_kt || !b_win_v || !b_tv || !b_tq || !b_tk ||
            !b_tvt || !b_S || !b_al || !b_be || !b_oh || !bx || !bw || !bn ||
            (want_sh && (!bshg || !bshu || !bshd || !bgate || !bup || !bsho)) ||
            (want_rt && (!brw || !bsc))) {
            cpu_layer_decode_kda(kda, x, in_ln, post_ln, shg, shu, shd, D, Iinter, eps, act, act_a,
                                 act_b, router_w, router_bias, E, K, rscale, inrm_out, nrm_out,
                                 sh_out, idx_out, w_out);
            return true;
        }
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) {
            cpu_layer_decode_kda(kda, x, in_ln, post_ln, shg, shu, shd, D, Iinter, eps, act, act_a,
                                 act_b, router_w, router_bias, E, K, rscale, inrm_out, nrm_out,
                                 sh_out, idx_out, w_out);
            return true;
        }
        KdaArgs kargs{kda.P, kda.K, kda.H, kda.hd};
        NSArray *kbufs = @[
            b_win_q, b_qt, b_win_k, b_kt, b_win_v, b_tv, b_tq, b_tk, b_tvt, b_S, b_al, b_be, b_oh
        ];
        dispatch1(enc, g_p_kda, static_cast<uint>(kda.H), kbufs, &kargs, sizeof(kargs));
        ElemArgs ea{D};
        dispatch1(enc, g_p_add, static_cast<uint>(D), @[ bx, b_oh ], &ea, sizeof(ea));
        RmsArgs ra{1, D, eps, 1};
        dispatch1(enc, g_p_rms, 1u, @[ bx, bw, bn ], &ra, sizeof(ra));
        if (want_sh) {
            GemmArgs g1{1, D, Iinter, 0};
            dispatch2(enc, g_p_gemm_f32, static_cast<uint>(Iinter), 1u, @[ bn, bshg, bgate ], &g1,
                      sizeof(g1));
            dispatch2(enc, g_p_gemm_f32, static_cast<uint>(Iinter), 1u, @[ bn, bshu, bup ], &g1,
                      sizeof(g1));
            ActArgs aa{Iinter, act_a, act_b, static_cast<int>(act)};
            if (g_p_act)
                dispatch1(enc, g_p_act, static_cast<uint>(Iinter), @[ bgate, bup ], &aa, sizeof(aa));
            else {
                ElemArgs se{Iinter};
                dispatch1(enc, g_p_silu, static_cast<uint>(Iinter), @[ bgate, bup ], &se, sizeof(se));
            }
            GemmArgs g2{1, Iinter, D, 0};
            dispatch2(enc, g_p_gemm_f32, static_cast<uint>(D), 1u, @[ bgate, bshd, bsho ], &g2,
                      sizeof(g2));
        }
        if (want_rt) {
            GemmArgs gr{1, D, E, 0};
            dispatch2(enc, g_p_gemm_f32, static_cast<uint>(E), 1u, @[ bn, brw, bsc ], &gr,
                      sizeof(gr));
        }
        if (!commit_wait(cb, enc)) {
            cpu_layer_decode_kda(kda, x, in_ln, post_ln, shg, shu, shd, D, Iinter, eps, act, act_a,
                                 act_b, router_w, router_bias, E, K, rscale, inrm_out, nrm_out,
                                 sh_out, idx_out, w_out);
            return true;
        }
        std::memcpy(kda.win_q, [b_win_q contents], wb);
        std::memcpy(kda.qt, [b_qt contents], pb);
        std::memcpy(kda.win_k, [b_win_k contents], wb);
        std::memcpy(kda.kt, [b_kt contents], pb);
        std::memcpy(kda.win_v, [b_win_v contents], wb);
        std::memcpy(kda.tv, [b_tv contents], pb);
        std::memcpy(kda.S, [b_S contents], sbk);
        std::memcpy(kda.oh, [b_oh contents], pb);
        std::memcpy(x, [bx contents], db);
        std::memcpy(nrm_out, [bn contents], db);
        if (want_sh)
            std::memcpy(sh_out, [bsho contents], db);
        if (want_rt) {
            std::vector<float> scores(static_cast<size_t>(E));
            std::memcpy(scores.data(), [bsc contents], sb);
            std::vector<float> choice(static_cast<size_t>(E));
            for (int i = 0; i < E; ++i) {
                scores[static_cast<size_t>(i)] = quant::sigmoid(scores[static_cast<size_t>(i)]);
                const float b = router_bias ? router_bias[i] : 0.f;
                choice[static_cast<size_t>(i)] = scores[static_cast<size_t>(i)] + b;
            }
            moe_topk(choice.data(), E, K, idx_out, w_out, scores.data());
            if (rscale != 1.f) {
                const int kk = K > E ? E : K;
                for (int i = 0; i < kk; ++i)
                    w_out[i] *= rscale;
            }
        }
    }
    return true;
}

bool layer_decode_mla(const MlaAbsorb &mla, float *x, const float *post_ln, int D, float eps,
                      float *nrm_out) {
    if (!mla_args_ok(mla, x, post_ln, D, nrm_out))
        return false;
    const int hv = mla.H * mla.Vh;
    const bool want_wo = mla.w_o != nullptr;
    if (!g_use_metal || !g_p_add || !g_p_rms || (want_wo && !g_p_gemm_f32)) {
        cpu_layer_decode_mla(mla, x, post_ln, D, eps, nrm_out);
        return true;
    }
    std::vector<float> ctx(static_cast<size_t>(std::max(hv, D)), 0.f);
    cpu_mla_absorb(ctx.data(), mla);
    if (!want_wo)
        std::memcpy(mla.oh, ctx.data(), static_cast<size_t>(std::max(hv, D)) * sizeof(float));
    @autoreleasepool {
        const size_t db = sizeof(float) * static_cast<size_t>(D);
        const size_t hb = sizeof(float) * static_cast<size_t>(hv);
        const size_t wob = want_wo ? sizeof(float) * static_cast<size_t>(D) * hv : 0;
        id<MTLBuffer> bx = buf_bytes(x, db);
        id<MTLBuffer> bw = buf_bytes(post_ln, db);
        id<MTLBuffer> bn = buf_empty(db);
        id<MTLBuffer> bctx = want_wo ? buf_bytes(ctx.data(), hb) : buf_bytes(mla.oh, db);
        id<MTLBuffer> bwo = want_wo ? buf_bytes(mla.w_o, wob) : nil;
        id<MTLBuffer> boh = want_wo ? buf_empty(db) : bctx;
        if (!bx || !bw || !bn || !bctx || (want_wo && (!bwo || !boh))) {
            cpu_layer_decode_mla(mla, x, post_ln, D, eps, nrm_out);
            return true;
        }
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) {
            cpu_layer_decode_mla(mla, x, post_ln, D, eps, nrm_out);
            return true;
        }
        if (want_wo) {
            GemmArgs go{1, hv, D, 0};
            dispatch2(enc, g_p_gemm_f32, static_cast<uint>(D), 1u, @[ bctx, bwo, boh ], &go,
                      sizeof(go));
        }
        ElemArgs ea{D};
        dispatch1(enc, g_p_add, static_cast<uint>(D), @[ bx, boh ], &ea, sizeof(ea));
        RmsArgs ra{1, D, eps, 1};
        dispatch1(enc, g_p_rms, 1u, @[ bx, bw, bn ], &ra, sizeof(ra));
        if (!commit_wait(cb, enc)) {
            cpu_layer_decode_mla(mla, x, post_ln, D, eps, nrm_out);
            return true;
        }
        std::memcpy(mla.oh, [boh contents], db);
        std::memcpy(x, [bx contents], db);
        std::memcpy(nrm_out, [bn contents], db);
    }
    return true;
}

bool moe_block(int nb, int D, int Iinter, int fmt, int qgs, const void *const *g,
               const void *const *u, const void *const *d, const float *const *gs,
               const float *const *us, const float *const *ds, const float *xg, const int *xoff,
               const int *nr, const int *rows, const float *rw, float *out, int S, Act act,
               float act_a, float act_b) {
    if (nb < 1 || D < 1 || !out || !xg || (fmt != 0 && fmt != 4 && fmt != 7))
        return false;
    if (!xoff || !nr)
        return false;
    const bool use_i4 = fmt == 4;
    const bool use_mx = fmt == 7;
    const bool need_act = act != Act::Silu;
    const bool metal_ok = g_use_metal && (need_act ? g_p_act : g_p_silu) &&
                          ((use_i4 && g_p_gemm_i4) || (use_mx && g_p_gemm_mx) ||
                           (!use_i4 && !use_mx && g_p_gemm_f32));
    if (!metal_ok || Iinter < 1) {
        cpu_moe_block(nb, D, Iinter, fmt, qgs, g, u, d, gs, us, ds, xg, xoff, nr, rows, rw, out, S,
                      act, act_a, act_b);
        return true;
    }
    @autoreleasepool {
        struct ExpertJob {
            int nre;
            int base;
            id<MTLBuffer> bhh;
        };
        std::vector<ExpertJob> jobs;
        int base = 0;
        int active = 0;
        for (int e = 0; e < nb; ++e) {
            const int nre = nr[e];
            if (nre <= 0)
                continue;
            ++active;
            base += nre;
        }
        if (active == 0)
            return true;

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) {
            cpu_moe_block(nb, D, Iinter, fmt, qgs, g, u, d, gs, us, ds, xg, xoff, nr, rows, rw, out,
                          S, act, act_a, act_b);
            return true;
        }

        const int group = qgs > 0 ? qgs : 64;
        const size_t pack_go = static_cast<size_t>(Iinter) * static_cast<size_t>((D + 1) / 2);
        const size_t sc_go =
            sizeof(float) * static_cast<size_t>(Iinter) * static_cast<size_t>((D + group - 1) / group);
        const size_t pack_d = static_cast<size_t>(D) * static_cast<size_t>((Iinter + 1) / 2);
        const size_t sc_d =
            sizeof(float) * static_cast<size_t>(D) * static_cast<size_t>((Iinter + group - 1) / group);
        const size_t mx_sc_go = static_cast<size_t>(Iinter) * static_cast<size_t>((D + 31) / 32);
        const size_t mx_sc_d = static_cast<size_t>(D) * static_cast<size_t>((Iinter + 31) / 32);
        const size_t f32_go = sizeof(float) * static_cast<size_t>(Iinter) * static_cast<size_t>(D);
        const size_t f32_d = sizeof(float) * static_cast<size_t>(D) * static_cast<size_t>(Iinter);

        bool ok = true;
        base = 0;
        std::vector<id<MTLBuffer>> keep;
        keep.reserve(static_cast<size_t>(active) * 8u);
        for (int e = 0; e < nb && ok; ++e) {
            const int nre = nr[e];
            if (nre <= 0)
                continue;
            if (!g || !u || !d || !g[e] || !u[e] || !d[e] ||
                ((use_i4 || use_mx) && (!gs || !us || !ds || !gs[e] || !us[e] || !ds[e]))) {
                base += nre;
                continue;
            }
            const float *xe = xg + static_cast<size_t>(xoff[e]) * static_cast<size_t>(D);
            const size_t xb = sizeof(float) * static_cast<size_t>(nre) * static_cast<size_t>(D);
            const size_t hb = sizeof(float) * static_cast<size_t>(nre) * static_cast<size_t>(Iinter);
            id<MTLBuffer> bxe = buf_bytes(xe, xb);
            id<MTLBuffer> bgate = buf_empty(hb);
            id<MTLBuffer> bup = buf_empty(hb);
            id<MTLBuffer> bhh = buf_empty(xb);
            id<MTLBuffer> bg = nil, bu = nil, bd = nil, bgs = nil, bus = nil, bds = nil;
            if (use_i4) {
                bg = buf_bytes(g[e], pack_go);
                bu = buf_bytes(u[e], pack_go);
                bd = buf_bytes(d[e], pack_d);
                bgs = buf_bytes(gs[e], sc_go);
                bus = buf_bytes(us[e], sc_go);
                bds = buf_bytes(ds[e], sc_d);
            } else if (use_mx) {
                bg = buf_bytes(g[e], pack_go);
                bu = buf_bytes(u[e], pack_go);
                bd = buf_bytes(d[e], pack_d);
                bgs = buf_bytes(gs[e], mx_sc_go);
                bus = buf_bytes(us[e], mx_sc_go);
                bds = buf_bytes(ds[e], mx_sc_d);
            } else {
                bg = buf_bytes(g[e], f32_go);
                bu = buf_bytes(u[e], f32_go);
                bd = buf_bytes(d[e], f32_d);
            }
            if (!bxe || !bgate || !bup || !bhh || !bg || !bu || !bd ||
                ((use_i4 || use_mx) && (!bgs || !bus || !bds))) {
                ok = false;
                break;
            }
            keep.push_back(bxe);
            keep.push_back(bgate);
            keep.push_back(bup);
            keep.push_back(bg);
            keep.push_back(bu);
            keep.push_back(bd);
            if (use_i4 || use_mx) {
                keep.push_back(bgs);
                keep.push_back(bus);
                keep.push_back(bds);
            }
            GemmArgs ag{nre, D, Iinter, group};
            GemmArgs ad{nre, Iinter, D, group};
            if (use_i4) {
                dispatch2(enc, g_p_gemm_i4, static_cast<uint>(Iinter), static_cast<uint>(nre),
                          @[ bxe, bg, bgs, bgate ], &ag, sizeof(ag));
                dispatch2(enc, g_p_gemm_i4, static_cast<uint>(Iinter), static_cast<uint>(nre),
                          @[ bxe, bu, bus, bup ], &ag, sizeof(ag));
            } else if (use_mx) {
                dispatch2(enc, g_p_gemm_mx, static_cast<uint>(Iinter), static_cast<uint>(nre),
                          @[ bxe, bg, bgs, bgate ], &ag, sizeof(ag));
                dispatch2(enc, g_p_gemm_mx, static_cast<uint>(Iinter), static_cast<uint>(nre),
                          @[ bxe, bu, bus, bup ], &ag, sizeof(ag));
            } else {
                dispatch2(enc, g_p_gemm_f32, static_cast<uint>(Iinter), static_cast<uint>(nre),
                          @[ bxe, bg, bgate ], &ag, sizeof(ag));
                dispatch2(enc, g_p_gemm_f32, static_cast<uint>(Iinter), static_cast<uint>(nre),
                          @[ bxe, bu, bup ], &ag, sizeof(ag));
            }
            if (g_p_act) {
                ActArgs aa{nre * Iinter, act_a, act_b, static_cast<int>(act)};
                dispatch1(enc, g_p_act, static_cast<uint>(nre * Iinter), @[ bgate, bup ], &aa,
                          sizeof(aa));
            } else {
                ElemArgs se{nre * Iinter};
                dispatch1(enc, g_p_silu, static_cast<uint>(nre * Iinter), @[ bgate, bup ], &se,
                          sizeof(se));
            }
            if (use_i4) {
                dispatch2(enc, g_p_gemm_i4, static_cast<uint>(D), static_cast<uint>(nre),
                          @[ bgate, bd, bds, bhh ], &ad, sizeof(ad));
            } else if (use_mx) {
                dispatch2(enc, g_p_gemm_mx, static_cast<uint>(D), static_cast<uint>(nre),
                          @[ bgate, bd, bds, bhh ], &ad, sizeof(ad));
            } else {
                dispatch2(enc, g_p_gemm_f32, static_cast<uint>(D), static_cast<uint>(nre),
                          @[ bgate, bd, bhh ], &ad, sizeof(ad));
            }
            jobs.push_back(ExpertJob{nre, base, bhh});
            base += nre;
        }
        if (!ok || !commit_wait(cb, enc)) {
            cpu_moe_block(nb, D, Iinter, fmt, qgs, g, u, d, gs, us, ds, xg, xoff, nr, rows, rw, out,
                          S, act, act_a, act_b);
            return true;
        }
        if (!rows || !rw)
            return true;
        for (const ExpertJob &job : jobs) {
            const float *hh = static_cast<const float *>([job.bhh contents]);
            for (int r = 0; r < job.nre; ++r) {
                const int dest = rows[job.base + r];
                if (dest < 0 || (S > 0 && dest >= S))
                    continue;
                const float w = rw[job.base + r];
                const float *hr = hh + static_cast<size_t>(r) * static_cast<size_t>(D);
                float *orow = out + static_cast<size_t>(dest) * static_cast<size_t>(D);
                for (int i = 0; i < D; ++i)
                    orow[i] += w * hr[i];
            }
        }
    }
    return true;
}

} // namespace metal_ops
} // namespace mvllm
