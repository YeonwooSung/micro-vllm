#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ops.hpp"

#include "../model/family.hpp"
#include "../quant/quant.hpp"

#include <cmath>
#include <cstdint>
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
id<MTLComputePipelineState> g_p_kda = nil;
id<MTLComputePipelineState> g_p_gemm_f32 = nil;
id<MTLComputePipelineState> g_p_gemm_i4 = nil;

static const char *kMetalSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct RmsArgs { int nrows; int D; float eps; int has_w; };
struct ElemArgs { int n; };
struct KdaArgs { int P; int K; int H; int hd; };
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
)MSL";

struct RmsArgs {
    int nrows, D;
    float eps;
    int has_w;
};
struct ElemArgs {
    int n;
};
struct KdaArgs {
    int P, K, H, hd;
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
    g_p_kda = nil;
    g_p_gemm_f32 = nil;
    g_p_gemm_i4 = nil;
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
        g_p_kda = pso("op_kda_fused");
        g_p_gemm_f32 = pso("op_gemm_f32");
        g_p_gemm_i4 = pso("op_gemm_int4_g64");
        if (g_p_rms && g_p_add && g_p_silu)
            g_use_metal = true;
        else
            drop_metal();
        return true;
    }
}

void shutdown() {
    @autoreleasepool {
        drop_metal();
        g_inited = false;
    }
}

bool available() { return g_inited; }

const char *backend_name() { return g_use_metal ? "metal" : "cpu"; }

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

bool layer_decode(float *x, const float *attn, const float *post_ln, const float *shg,
                  const float *shu, const float *shd, int D, int Iinter, float eps, float *nrm_out,
                  float *sh_out) {
    if (!x || !attn || !post_ln || !nrm_out || D < 1)
        return false;
    const bool want_sh = shg && shu && shd && Iinter > 0 && sh_out;
    if (!g_use_metal || !g_p_add || !g_p_rms || (want_sh && !g_p_gemm_f32)) {
        cpu_layer_decode(x, attn, post_ln, shg, shu, shd, D, Iinter, eps, nrm_out, sh_out);
        return true;
    }
    @autoreleasepool {
        const size_t db = sizeof(float) * static_cast<size_t>(D);
        const size_t ib = want_sh ? sizeof(float) * static_cast<size_t>(Iinter) : 0;
        const size_t shg_b = want_sh ? sizeof(float) * static_cast<size_t>(Iinter) * D : 0;
        const size_t shd_b = want_sh ? sizeof(float) * static_cast<size_t>(D) * Iinter : 0;
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
        if (!bx || !ba || !bw || !bn || (want_sh && (!bshg || !bshu || !bshd || !bgate || !bup || !bsho))) {
            cpu_layer_decode(x, attn, post_ln, shg, shu, shd, D, Iinter, eps, nrm_out, sh_out);
            return true;
        }
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc) {
            cpu_layer_decode(x, attn, post_ln, shg, shu, shd, D, Iinter, eps, nrm_out, sh_out);
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
            ElemArgs se{Iinter};
            dispatch1(enc, g_p_silu, static_cast<uint>(Iinter), @[ bgate, bup ], &se, sizeof(se));
            GemmArgs g2{1, Iinter, D, 0};
            dispatch2(enc, g_p_gemm_f32, static_cast<uint>(D), 1u, @[ bgate, bshd, bsho ], &g2,
                      sizeof(g2));
        }
        if (!commit_wait(cb, enc)) {
            cpu_layer_decode(x, attn, post_ln, shg, shu, shd, D, Iinter, eps, nrm_out, sh_out);
            return true;
        }
        std::memcpy(x, [bx contents], db);
        std::memcpy(nrm_out, [bn contents], db);
        if (want_sh)
            std::memcpy(sh_out, [bsho contents], db);
    }
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
    const bool use_i4 = fmt == 4;
    const bool metal_ok =
        g_use_metal && g_p_silu && ((use_i4 && g_p_gemm_i4) || (!use_i4 && g_p_gemm_f32));
    if (!metal_ok || Iinter < 1) {
        cpu_moe_block(nb, D, Iinter, fmt, qgs, g, u, d, gs, us, ds, xg, xoff, nr, rows, rw, out, S);
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
                          S);
            return true;
        }

        const int group = qgs > 0 ? qgs : 64;
        const size_t pack_go = static_cast<size_t>(Iinter) * static_cast<size_t>((D + 1) / 2);
        const size_t sc_go =
            sizeof(float) * static_cast<size_t>(Iinter) * static_cast<size_t>((D + group - 1) / group);
        const size_t pack_d = static_cast<size_t>(D) * static_cast<size_t>((Iinter + 1) / 2);
        const size_t sc_d =
            sizeof(float) * static_cast<size_t>(D) * static_cast<size_t>((Iinter + group - 1) / group);
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
                (use_i4 && (!gs || !us || !ds || !gs[e] || !us[e] || !ds[e]))) {
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
            } else {
                bg = buf_bytes(g[e], f32_go);
                bu = buf_bytes(u[e], f32_go);
                bd = buf_bytes(d[e], f32_d);
            }
            if (!bxe || !bgate || !bup || !bhh || !bg || !bu || !bd || (use_i4 && (!bgs || !bus || !bds))) {
                ok = false;
                break;
            }
            keep.push_back(bxe);
            keep.push_back(bgate);
            keep.push_back(bup);
            keep.push_back(bg);
            keep.push_back(bu);
            keep.push_back(bd);
            if (use_i4) {
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
            } else {
                dispatch2(enc, g_p_gemm_f32, static_cast<uint>(Iinter), static_cast<uint>(nre),
                          @[ bxe, bg, bgate ], &ag, sizeof(ag));
                dispatch2(enc, g_p_gemm_f32, static_cast<uint>(Iinter), static_cast<uint>(nre),
                          @[ bxe, bu, bup ], &ag, sizeof(ag));
            }
            ElemArgs se{nre * Iinter};
            dispatch1(enc, g_p_silu, static_cast<uint>(nre * Iinter), @[ bgate, bup ], &se,
                      sizeof(se));
            if (use_i4) {
                dispatch2(enc, g_p_gemm_i4, static_cast<uint>(D), static_cast<uint>(nre),
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
                          S);
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
