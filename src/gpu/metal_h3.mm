#include "metal_h3.hpp"

#include "../model/family.hpp"
#include "../quant/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(MVLLM_WITH_METAL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace mvllm {
namespace metal_h3 {
namespace {

static const char *kMetalSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct GemmArgs { int S; int I; int O; };
struct AdalnArgs { int T; int H; float eps; int has_mod; int scale_slot; int shift_slot; };
struct AttnArgs { int T; int I; int hd; int heads; };
struct ElemArgs { int n; };
struct QkNormArgs { int T; int I; int hd; int heads; int do_q; int do_k; float eps; };
struct RopeArgs { int T; int I; int hd; int heads; };
struct GateArgs { int T; int H; int has_mod; int slot; };

kernel void gemm_f32(device const float *x [[buffer(0)]],
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

kernel void bf16_to_f32(device const ushort *src [[buffer(0)]],
                        device float *dst [[buffer(1)]],
                        constant ElemArgs &a [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.n) return;
    dst[gid] = as_type<float>((uint)src[gid] << 16);
}

// RMSNorm then y = y * (1 + scale) + shift. Slots match h3_dit_block_cpu.
kernel void adaln(device const float *x [[buffer(0)]],
                  device const float *mod [[buffer(1)]],
                  device float *y [[buffer(2)]],
                  constant AdalnArgs &a [[buffer(3)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.T) return;
    const int H = a.H;
    const device float *xs = x + (ulong)gid * (uint)H;
    device float *ys = y + (ulong)gid * (uint)H;
    float ss = 0.0f;
    for (int i = 0; i < H; ++i)
        ss += xs[i] * xs[i];
    float inv = rsqrt(ss / (float)H + a.eps);
    if (a.has_mod != 0) {
        const device float *s = mod + (ulong)a.scale_slot * (uint)H;
        const device float *b = mod + (ulong)a.shift_slot * (uint)H;
        for (int i = 0; i < H; ++i)
            ys[i] = xs[i] * inv * (1.0f + s[i]) + b[i];
    } else {
        for (int i = 0; i < H; ++i)
            ys[i] = xs[i] * inv;
    }
}

kernel void qk_rmsnorm(device float *qkv [[buffer(0)]],
                       device const float *q_norm [[buffer(1)]],
                       device const float *k_norm [[buffer(2)]],
                       constant QkNormArgs &a [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint h = gid.x;
    uint t = gid.y;
    if (h >= (uint)a.heads || t >= (uint)a.T) return;
    const int I = a.I;
    const int hd = a.hd;
    if (a.do_q != 0) {
        device float *q = qkv + (ulong)t * (uint)(3 * I) + h * (uint)hd;
        float ss = 0.0f;
        for (int i = 0; i < hd; ++i)
            ss += q[i] * q[i];
        float inv = rsqrt(ss / (float)hd + a.eps);
        for (int i = 0; i < hd; ++i)
            q[i] = q[i] * inv * q_norm[i];
    }
    if (a.do_k != 0) {
        device float *k = qkv + (ulong)t * (uint)(3 * I) + I + h * (uint)hd;
        float ss = 0.0f;
        for (int i = 0; i < hd; ++i)
            ss += k[i] * k[i];
        float inv = rsqrt(ss / (float)hd + a.eps);
        for (int i = 0; i < hd; ++i)
            k[i] = k[i] * inv * k_norm[i];
    }
}

kernel void rope_qk(device float *qkv [[buffer(0)]],
                    device const float *cos [[buffer(1)]],
                    device const float *sin [[buffer(2)]],
                    constant RopeArgs &a [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    uint h = gid.x;
    uint t = gid.y;
    if (h >= (uint)a.heads || t >= (uint)a.T) return;
    if (a.hd < 96) return;
    const int nhalf = 48;
    const int I = a.I;
    const int hd = a.hd;
    const device float *c = cos + (ulong)t * (uint)nhalf;
    const device float *s = sin + (ulong)t * (uint)nhalf;
    device float *q = qkv + (ulong)t * (uint)(3 * I) + h * (uint)hd;
    device float *k = qkv + (ulong)t * (uint)(3 * I) + I + h * (uint)hd;
    for (int d = 0; d < nhalf; ++d) {
        float qa = q[d], qb = q[d + nhalf];
        q[d] = qa * c[d] - qb * s[d];
        q[d + nhalf] = qa * s[d] + qb * c[d];
        float ka = k[d], kb = k[d + nhalf];
        k[d] = ka * c[d] - kb * s[d];
        k[d + nhalf] = ka * s[d] + kb * c[d];
    }
}

kernel void dit_attn(device const float *qkv [[buffer(0)]],
                     device float *ctx [[buffer(1)]],
                     constant AttnArgs &a [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint h = gid.x;
    uint qi = gid.y;
    if (h >= (uint)a.heads || qi >= (uint)a.T) return;
    const int T = a.T;
    const int I = a.I;
    const int hd = a.hd;
    const float scale = rsqrt((float)hd);
    thread float scores[256];
    if (T > 256) return;
    const device float *q = qkv + (ulong)qi * (uint)(3 * I) + h * (uint)hd;
    float m = -1e30f;
    for (int ki = 0; ki < T; ++ki) {
        const device float *k = qkv + (ulong)ki * (uint)(3 * I) + I + h * (uint)hd;
        float acc = 0.0f;
        for (int d = 0; d < hd; ++d)
            acc += q[d] * k[d];
        scores[ki] = acc * scale;
        if (scores[ki] > m) m = scores[ki];
    }
    float sum = 0.0f;
    for (int ki = 0; ki < T; ++ki) {
        scores[ki] = exp(scores[ki] - m);
        sum += scores[ki];
    }
    float inv = (sum == 0.0f) ? 0.0f : 1.0f / sum;
    device float *o = ctx + (ulong)qi * (uint)I + h * (uint)hd;
    for (int d = 0; d < hd; ++d)
        o[d] = 0.0f;
    for (int vi = 0; vi < T; ++vi) {
        const device float *v = qkv + (ulong)vi * (uint)(3 * I) + 2 * I + h * (uint)hd;
        float w = scores[vi] * inv;
        for (int d = 0; d < hd; ++d)
            o[d] += w * v[d];
    }
}

// x += (has_mod ? mod[slot] : 1) * y. Attn gate is slot 2, MLP gate is slot 5.
kernel void residual_gate(device float *x [[buffer(0)]],
                          device const float *y [[buffer(1)]],
                          device const float *mod [[buffer(2)]],
                          constant GateArgs &a [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    const int n = a.T * a.H;
    if (gid >= (uint)n) return;
    int i = (int)gid % a.H;
    float g = (a.has_mod != 0) ? mod[(ulong)a.slot * (uint)a.H + (uint)i] : 1.0f;
    x[gid] += g * y[gid];
}

kernel void swiglu_pack(device const float *h1 [[buffer(0)]],
                        device float *gated [[buffer(1)]],
                        constant GemmArgs &a [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint t = gid.y;
    if (t >= (uint)a.S || i >= (uint)a.O) return;
    float g = h1[(ulong)t * (uint)(2 * a.O) + i];
    float u = h1[(ulong)t * (uint)(2 * a.O) + a.O + i];
    float sig = 1.0f / (1.0f + exp(-g));
    gated[(ulong)t * (uint)a.O + i] = g * sig * u;
}

// y[s,o] = scale[o] * dot(x[s,:], int8 w[o,:]).
kernel void gemm_int8(device const float *x [[buffer(0)]],
                      device const char *w [[buffer(1)]],
                      device const float *scale [[buffer(2)]],
                      device float *y [[buffer(3)]],
                      constant GemmArgs &a [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint o = gid.x;
    uint s = gid.y;
    if (o >= (uint)a.O || s >= (uint)a.S) return;
    const device float *xs = x + (ulong)s * (uint)a.I;
    const device char *wo = w + (ulong)o * (uint)a.I;
    float acc = 0.0f;
    for (int i = 0; i < a.I; ++i)
        acc += xs[i] * (float)wo[i];
    y[(ulong)s * (uint)a.O + o] = acc * scale[o];
}

// In-place silu(x) = x * sigmoid(x); sigmoid matches quant::sigmoid.
kernel void silu_f32(device float *x [[buffer(0)]],
                     constant ElemArgs &a [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.n) return;
    float v = x[gid];
    float sig;
    if (v >= 0.0f)
        sig = 1.0f / (1.0f + exp(-v));
    else {
        float z = exp(v);
        sig = z / (1.0f + z);
    }
    x[gid] = v * sig;
}

kernel void add_f32(device float *y [[buffer(0)]],
                    device const float *a [[buffer(1)]],
                    constant ElemArgs &e [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)e.n) return;
    y[gid] += a[gid];
}

struct RmsArgs { int n; float eps; int has_w; };

// y = rmsnorm(x, w) over n elements; has_w=0 uses scale 1.
kernel void rms_f32(device const float *x [[buffer(0)]],
                    device const float *w [[buffer(1)]],
                    device float *y [[buffer(2)]],
                    constant RmsArgs &a [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    float ss = 0.0f;
    for (int i = 0; i < a.n; ++i)
        ss += x[i] * x[i];
    float inv = 1.0f / sqrt(ss / (float)a.n + a.eps);
    if (a.has_w != 0) {
        for (int i = 0; i < a.n; ++i)
            y[i] = x[i] * inv * w[i];
    } else {
        for (int i = 0; i < a.n; ++i)
            y[i] = x[i] * inv;
    }
}

struct RmsRowArgs { int T; int H; float eps; int has_w; };

// Per-row RMS: y[t] = rmsnorm(x[t], w).
kernel void rms_rows(device const float *x [[buffer(0)]],
                     device const float *w [[buffer(1)]],
                     device float *y [[buffer(2)]],
                     constant RmsRowArgs &a [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.T) return;
    const int H = a.H;
    const device float *xs = x + (ulong)gid * (uint)H;
    device float *ys = y + (ulong)gid * (uint)H;
    float ss = 0.0f;
    for (int i = 0; i < H; ++i)
        ss += xs[i] * xs[i];
    float inv = rsqrt(ss / (float)H + a.eps);
    if (a.has_w != 0) {
        for (int i = 0; i < H; ++i)
            ys[i] = xs[i] * inv * w[i];
    } else {
        for (int i = 0; i < H; ++i)
            ys[i] = xs[i] * inv;
    }
}

// y[s, o] += b[o]
kernel void add_bias_f32(device float *y [[buffer(0)]],
                         device const float *b [[buffer(1)]],
                         constant GemmArgs &a [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint o = gid.x;
    uint s = gid.y;
    if (o >= (uint)a.O || s >= (uint)a.S) return;
    y[(ulong)s * (uint)a.O + o] += b[o];
}

struct LnRowArgs { int T; int H; float eps; int has_w; int has_b; };

// Per-row LayerNorm: y[t] = ((x[t]-mean)/sqrt(var+eps)) * w + b.
kernel void ln_rows(device const float *x [[buffer(0)]],
                    device const float *w [[buffer(1)]],
                    device const float *b [[buffer(2)]],
                    device float *y [[buffer(3)]],
                    constant LnRowArgs &a [[buffer(4)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.T) return;
    const int H = a.H;
    const device float *xs = x + (ulong)gid * (uint)H;
    device float *ys = y + (ulong)gid * (uint)H;
    float mean = 0.0f;
    for (int i = 0; i < H; ++i)
        mean += xs[i];
    mean /= (float)(H > 0 ? H : 1);
    float var = 0.0f;
    for (int i = 0; i < H; ++i) {
        float d = xs[i] - mean;
        var += d * d;
    }
    var /= (float)(H > 0 ? H : 1);
    float inv = 1.0f / sqrt(var + a.eps);
    for (int i = 0; i < H; ++i) {
        float v = (xs[i] - mean) * inv;
        if (a.has_w != 0)
            v *= w[i];
        if (a.has_b != 0)
            v += b[i];
        ys[i] = v;
    }
}

// In-place GELU-tanh matching h3_vision::gelu_tanh.
kernel void gelu_tanh_f32(device float *x [[buffer(0)]],
                          constant ElemArgs &a [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.n) return;
    float v = x[gid];
    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    if (inner <= -10.0f) {
        x[gid] = 0.0f;
        return;
    }
    if (inner >= 10.0f) {
        x[gid] = v;
        return;
    }
    x[gid] = 0.5f * v * (1.0f + tanh(inner));
}

// out = gelu_audio(gate) * lin. Audio GELU has no clamp.
kernel void gelu_mul_f32(device const float *gate [[buffer(0)]],
                         device const float *lin [[buffer(1)]],
                         device float *out [[buffer(2)]],
                         constant ElemArgs &a [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.n) return;
    float v = gate[gid];
    float g = 0.5f * v * (1.0f + tanh(0.79788456f * (v + 0.044715f * v * v * v)));
    out[gid] = g * lin[gid];
}

struct VisionRopeArgs { int T; int I; int hd; int heads; int rope_half; };

// Vision apply_qkv_rope on packed QKV [T, 3*I] → packed out. V is copied.
kernel void vision_rope(device const float *qkv_in [[buffer(0)]],
                        device float *qkv_out [[buffer(1)]],
                        device const float *cos [[buffer(2)]],
                        device const float *sin [[buffer(3)]],
                        constant VisionRopeArgs &a [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint h = gid.x;
    uint t = gid.y;
    if (h >= (uint)a.heads || t >= (uint)a.T) return;
    const int I = a.I;
    const int hd = a.hd;
    const int nhalf = a.rope_half;
    const device float *qs = qkv_in + (ulong)t * (uint)(3 * I) + h * (uint)hd;
    const device float *ks = qkv_in + (ulong)t * (uint)(3 * I) + I + h * (uint)hd;
    const device float *vs = qkv_in + (ulong)t * (uint)(3 * I) + 2 * I + h * (uint)hd;
    device float *qd = qkv_out + (ulong)t * (uint)(3 * I) + h * (uint)hd;
    device float *kd = qkv_out + (ulong)t * (uint)(3 * I) + I + h * (uint)hd;
    device float *vd = qkv_out + (ulong)t * (uint)(3 * I) + 2 * I + h * (uint)hd;
    const device float *crow = (nhalf > 0) ? cos + (ulong)t * (uint)nhalf : cos;
    const device float *srow = (nhalf > 0) ? sin + (ulong)t * (uint)nhalf : sin;
    for (int dim = 0; dim < hd; ++dim)
        vd[dim] = vs[dim];
    for (int dim = 0; dim < hd; ++dim) {
        int pair = dim < nhalf ? dim + nhalf : dim - nhalf;
        if (nhalf <= 0 || pair < 0 || pair >= hd) {
            qd[dim] = qs[dim];
            kd[dim] = ks[dim];
            continue;
        }
        int ri = dim % nhalf;
        float c = crow[ri];
        float s = srow[ri];
        float q0 = qs[dim], q1 = qs[pair];
        float k0 = ks[dim], k1 = ks[pair];
        if (dim < nhalf) {
            qd[dim] = q0 * c - q1 * s;
            kd[dim] = k0 * c - k1 * s;
        } else {
            qd[dim] = q0 * c + q1 * s;
            kd[dim] = k0 * c + k1 * s;
        }
    }
}

struct CausalAttnArgs { int B; int T; int I; int hd; int heads; };

// Causal SDPA over B sequences of length T. qkv/ctx laid out as [B*T, 3*I] / [B*T, I].
kernel void dit_attn_causal(device const float *qkv [[buffer(0)]],
                            device float *ctx [[buffer(1)]],
                            constant CausalAttnArgs &a [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint h = gid.x;
    uint idx = gid.y;
    const int B = a.B;
    const int T = a.T;
    const int I = a.I;
    const int hd = a.hd;
    if (h >= (uint)a.heads || idx >= (uint)(B * T)) return;
    if (T > 256) return;
    int b = (int)idx / T;
    int qi = (int)idx % T;
    const float scale = rsqrt((float)hd);
    thread float scores[256];
    const int base = b * T;
    const device float *q = qkv + (ulong)(base + qi) * (uint)(3 * I) + h * (uint)hd;
    float m = -1e30f;
    int lim = qi + 1;
    for (int ki = 0; ki < lim; ++ki) {
        const device float *k = qkv + (ulong)(base + ki) * (uint)(3 * I) + I + h * (uint)hd;
        float acc = 0.0f;
        for (int d = 0; d < hd; ++d)
            acc += q[d] * k[d];
        scores[ki] = acc * scale;
        if (scores[ki] > m) m = scores[ki];
    }
    float sum = 0.0f;
    for (int ki = 0; ki < lim; ++ki) {
        scores[ki] = exp(scores[ki] - m);
        sum += scores[ki];
    }
    float inv = 1.0f / (sum + 1e-12f);
    device float *o = ctx + (ulong)(base + qi) * (uint)I + h * (uint)hd;
    for (int d = 0; d < hd; ++d)
        o[d] = 0.0f;
    for (int vi = 0; vi < lim; ++vi) {
        const device float *v = qkv + (ulong)(base + vi) * (uint)(3 * I) + 2 * I + h * (uint)hd;
        float w = scores[vi] * inv;
        for (int d = 0; d < hd; ++d)
            o[d] += w * v[d];
    }
}

struct PoolArgs { int S; int Cin; int Cout; };

// y[s, c] = x[s, c % Cin] — first Cout of each head-concat row (wrap if Cout > Cin).
kernel void pool_prefix(device const float *x [[buffer(0)]],
                        device float *y [[buffer(1)]],
                        constant PoolArgs &a [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x;
    uint s = gid.y;
    if (c >= (uint)a.Cout || s >= (uint)a.S) return;
    int cin = a.Cin > 0 ? a.Cin : 1;
    y[(ulong)s * (uint)a.Cout + c] = x[(ulong)s * (uint)a.Cin + (c % (uint)cin)];
}
)MSL";

struct GemmArgs {
    int S, I, O;
};
struct AdalnArgs {
    int T, H;
    float eps;
    int has_mod, scale_slot, shift_slot;
};
struct AttnArgs {
    int T, I, hd, heads;
};
struct ElemArgs {
    int n;
};
struct QkNormArgs {
    int T, I, hd, heads, do_q, do_k;
    float eps;
};
struct RopeArgs {
    int T, I, hd, heads;
};
struct GateArgs {
    int T, H, has_mod, slot;
};
struct RmsArgs {
    int n;
    float eps;
    int has_w;
};
struct RmsRowArgs {
    int T, H;
    float eps;
    int has_w;
};
struct LnRowArgs {
    int T, H;
    float eps;
    int has_w, has_b;
};
struct VisionRopeArgs {
    int T, I, hd, heads, rope_half;
};
struct CausalAttnArgs {
    int B, T, I, hd, heads;
};
struct PoolArgs {
    int S, Cin, Cout;
};

id<MTLBuffer> buf_bytes(id<MTLDevice> dev, const void *p, size_t n) {
    if (n == 0)
        n = 1;
    id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    if (p && n)
        std::memcpy([b contents], p, n);
    return b;
}

id<MTLBuffer> buf_empty(id<MTLDevice> dev, size_t n) {
    if (n == 0)
        n = 1;
    return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
}

struct MetalH3 {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> p_gemm = nil;
    id<MTLComputePipelineState> p_bf16 = nil;
    id<MTLComputePipelineState> p_adaln = nil;
    id<MTLComputePipelineState> p_qknorm = nil;
    id<MTLComputePipelineState> p_rope = nil;
    id<MTLComputePipelineState> p_attn = nil;
    id<MTLComputePipelineState> p_gate = nil;
    id<MTLComputePipelineState> p_swiglu = nil;
    id<MTLComputePipelineState> p_gemm_i8 = nil;
    id<MTLComputePipelineState> p_silu = nil;
    id<MTLComputePipelineState> p_add = nil;
    id<MTLComputePipelineState> p_rms = nil;
    id<MTLComputePipelineState> p_rms_rows = nil;
    id<MTLComputePipelineState> p_bias = nil;
    id<MTLComputePipelineState> p_ln = nil;
    id<MTLComputePipelineState> p_gelu = nil;
    id<MTLComputePipelineState> p_gelu_mul = nil;
    id<MTLComputePipelineState> p_vrope = nil;
    id<MTLComputePipelineState> p_attn_c = nil;
    id<MTLComputePipelineState> p_pool = nil;
    bool ready = false;

    bool setup() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device)
                return false;
            queue = [device newCommandQueue];
            if (!queue)
                return false;
            NSError *err = nil;
            NSString *src = [NSString stringWithUTF8String:kMetalSrc];
            id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
            if (!lib)
                return false;
            auto pso = [&](const char *name) -> id<MTLComputePipelineState> {
                id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
                if (!fn)
                    return nil;
                NSError *e = nil;
                return [device newComputePipelineStateWithFunction:fn error:&e];
            };
            p_gemm = pso("gemm_f32");
            p_bf16 = pso("bf16_to_f32");
            p_adaln = pso("adaln");
            p_qknorm = pso("qk_rmsnorm");
            p_rope = pso("rope_qk");
            p_attn = pso("dit_attn");
            p_gate = pso("residual_gate");
            p_swiglu = pso("swiglu_pack");
            p_gemm_i8 = pso("gemm_int8");
            p_silu = pso("silu_f32");
            p_add = pso("add_f32");
            p_rms = pso("rms_f32");
            p_rms_rows = pso("rms_rows");
            p_bias = pso("add_bias_f32");
            p_ln = pso("ln_rows");
            p_gelu = pso("gelu_tanh_f32");
            p_gelu_mul = pso("gelu_mul_f32");
            p_vrope = pso("vision_rope");
            p_attn_c = pso("dit_attn_causal");
            p_pool = pso("pool_prefix");
            ready = p_gemm && p_bf16 && p_adaln && p_qknorm && p_rope && p_attn && p_gate && p_swiglu;
            return ready;
        }
    }

    void teardown() {
        p_gemm = p_bf16 = p_adaln = p_qknorm = p_rope = p_attn = p_gate = p_swiglu = p_gemm_i8 =
            p_silu = p_add = p_rms = p_rms_rows = p_bias = p_ln = p_gelu = p_gelu_mul = p_vrope =
                p_attn_c = p_pool = nil;
        queue = nil;
        device = nil;
        ready = false;
    }
};

MetalH3 g;
std::once_flag g_once;

bool cpu_dit(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
             int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x, int tokens,
             float eps, const float *adaln_mod, const float *q_norm, const float *k_norm,
             const float *rope_cos, const float *rope_sin) {
    h3_dit_block_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn, head_dim,
                     x, tokens, eps, adaln_mod, q_norm, k_norm, rope_cos, rope_sin);
    return true;
}

bool cpu_gemm_int8(float *y, const float *x, const int8_t *w, const float *scale, int S, int I,
                   int O) {
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

bool cpu_nax_mlp(float *y, const float *x, const float *w_up, const float *w_down, int S, int D,
                 int I) {
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

bool cpu_vae_rms_add(float *x, const float *skip, const float *w, float *y, int n, float eps) {
    if (skip) {
        for (int i = 0; i < n; ++i)
            x[i] += skip[i];
    }
    quant::rmsnorm(x, w, y, n, eps);
    return true;
}

bool cpu_vae_transformer_block(float *x, int tokens, int hidden, int heads, int hd,
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
                        qkv.data() + static_cast<size_t>(i) * 3 * H,
                        static_cast<size_t>(H) * sizeof(float));
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

bool cpu_vision_block(float *x, int rows, int hidden, int heads, int hd, int intermediate,
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

bool cpu_audio_pre_block(float *base, const float *seq, int B, int L, int C, int ch, int heads,
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

void enc1(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, uint n, NSArray *bufs,
          const void *uni, size_t uni_n) {
    [enc setComputePipelineState:p];
    NSUInteger i = 0;
    for (id b in bufs)
        [enc setBuffer:b offset:0 atIndex:i++];
    [enc setBytes:uni length:uni_n atIndex:i];
    MTLSize tg = MTLSizeMake(64, 1, 1);
    MTLSize grid = MTLSizeMake(((n + 63) / 64) * 64, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

void enc2(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, uint gx, uint gy,
          NSArray *bufs, const void *uni, size_t uni_n) {
    [enc setComputePipelineState:p];
    NSUInteger i = 0;
    for (id b in bufs)
        [enc setBuffer:b offset:0 atIndex:i++];
    [enc setBytes:uni length:uni_n atIndex:i];
    MTLSize tg = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake(((gx + 15) / 16) * 16, ((gy + 15) / 16) * 16, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

} // namespace

bool init() {
    std::call_once(g_once, []() { g.setup(); });
    return g.ready;
}

void shutdown() { g.teardown(); }

bool available() { return g.ready; }

const char *backend_name() { return g.ready ? "metal" : "cpu"; }

bool dit_residual(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                  int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                  int tokens, float eps, const float *adaln_mod, const float *q_norm,
                  const float *k_norm, const float *rope_cos, const float *rope_sin) {
    init();
    if (!g.ready || tokens > 256)
        return cpu_dit(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                       head_dim, x, tokens, eps, adaln_mod, q_norm, k_norm, rope_cos, rope_sin);
    if (!blob || !x || hidden <= 0 || inner <= 0 || ffn <= 0 || tokens <= 0)
        return cpu_dit(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                       head_dim, x, tokens, eps, adaln_mod, q_norm, k_norm, rope_cos, rope_sin);

    @autoreleasepool {
        const int T = tokens;
        const int H = hidden;
        const int I = inner;
        int hd = head_dim > 0 ? head_dim : I;
        int heads = I / hd;
        if (heads < 1) {
            heads = 1;
            hd = I;
        }
        const int64_t qkv_n = qkv_bytes / 2;
        const int64_t out_n = out_bytes / 2;
        const int64_t fc1_n = fc1_bytes / 2;
        const int64_t fc2_n = fc2_bytes / 2;
        const int has_mod = adaln_mod ? 1 : 0;
        const int do_q = q_norm ? 1 : 0;
        const int do_k = k_norm ? 1 : 0;
        const bool do_rope = rope_cos && rope_sin && hd >= 96;

        id<MTLBuffer> bqkv_bf = buf_bytes(g.device, blob, (size_t)qkv_bytes);
        id<MTLBuffer> bout_bf = buf_bytes(g.device, blob + qkv_bytes, (size_t)out_bytes);
        id<MTLBuffer> bfc1_bf =
            buf_bytes(g.device, blob + qkv_bytes + out_bytes, (size_t)fc1_bytes);
        id<MTLBuffer> bfc2_bf =
            buf_bytes(g.device, blob + qkv_bytes + out_bytes + fc1_bytes, (size_t)fc2_bytes);
        id<MTLBuffer> bx = buf_bytes(g.device, x, sizeof(float) * (size_t)T * H);
        id<MTLBuffer> bxn = buf_empty(g.device, sizeof(float) * (size_t)T * H);
        id<MTLBuffer> wqkv = buf_empty(g.device, sizeof(float) * (size_t)qkv_n);
        id<MTLBuffer> wout = buf_empty(g.device, sizeof(float) * (size_t)out_n);
        id<MTLBuffer> wfc1 = buf_empty(g.device, sizeof(float) * (size_t)fc1_n);
        id<MTLBuffer> wfc2 = buf_empty(g.device, sizeof(float) * (size_t)fc2_n);
        id<MTLBuffer> bqkv = buf_empty(g.device, sizeof(float) * (size_t)T * 3 * I);
        id<MTLBuffer> bctx = buf_empty(g.device, sizeof(float) * (size_t)T * I);
        id<MTLBuffer> battn = buf_empty(g.device, sizeof(float) * (size_t)T * H);
        id<MTLBuffer> bh1 = buf_empty(g.device, sizeof(float) * (size_t)T * 2 * ffn);
        id<MTLBuffer> bgated = buf_empty(g.device, sizeof(float) * (size_t)T * ffn);
        id<MTLBuffer> bdown = buf_empty(g.device, sizeof(float) * (size_t)T * H);
        id<MTLBuffer> bmod = buf_bytes(g.device, adaln_mod, sizeof(float) * (size_t)6 * H);
        id<MTLBuffer> bqn = buf_bytes(g.device, q_norm, sizeof(float) * (size_t)hd);
        id<MTLBuffer> bkn = buf_bytes(g.device, k_norm, sizeof(float) * (size_t)hd);
        id<MTLBuffer> bcos = buf_bytes(g.device, rope_cos, sizeof(float) * (size_t)T * 48);
        id<MTLBuffer> bsin = buf_bytes(g.device, rope_sin, sizeof(float) * (size_t)T * 48);

        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        if (!cb || !bqkv_bf || !bout_bf || !bfc1_bf || !bfc2_bf || !bx)
            return cpu_dit(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                           head_dim, x, tokens, eps, adaln_mod, q_norm, k_norm, rope_cos,
                           rope_sin);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc)
            return cpu_dit(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                           head_dim, x, tokens, eps, adaln_mod, q_norm, k_norm, rope_cos,
                           rope_sin);

        auto enc1 = [&](id<MTLComputePipelineState> p, uint n, NSArray *bufs, const void *uni,
                        size_t uni_n) {
            [enc setComputePipelineState:p];
            NSUInteger i = 0;
            for (id b in bufs)
                [enc setBuffer:b offset:0 atIndex:i++];
            [enc setBytes:uni length:uni_n atIndex:i];
            MTLSize tg = MTLSizeMake(64, 1, 1);
            MTLSize grid = MTLSizeMake(((n + 63) / 64) * 64, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        };
        auto enc2 = [&](id<MTLComputePipelineState> p, uint gx, uint gy, NSArray *bufs,
                        const void *uni, size_t uni_n) {
            [enc setComputePipelineState:p];
            NSUInteger i = 0;
            for (id b in bufs)
                [enc setBuffer:b offset:0 atIndex:i++];
            [enc setBytes:uni length:uni_n atIndex:i];
            MTLSize tg = MTLSizeMake(16, 16, 1);
            MTLSize grid = MTLSizeMake(((gx + 15) / 16) * 16, ((gy + 15) / 16) * 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        };

        auto launch_bf16 = [&](id<MTLBuffer> src, id<MTLBuffer> dst, int n) {
            ElemArgs e{n};
            enc1(g.p_bf16, (uint)n, @[ src, dst ], &e, sizeof(e));
        };
        launch_bf16(bqkv_bf, wqkv, (int)qkv_n);
        launch_bf16(bout_bf, wout, (int)out_n);
        launch_bf16(bfc1_bf, wfc1, (int)fc1_n);
        launch_bf16(bfc2_bf, wfc2, (int)fc2_n);

        AdalnArgs a0{T, H, eps, has_mod, 0, 1};
        enc1(g.p_adaln, (uint)T, @[ bx, bmod, bxn ], &a0, sizeof(a0));
        GemmArgs gq{T, H, 3 * I};
        enc2(g.p_gemm, (uint)(3 * I), (uint)T, @[ bxn, wqkv, bqkv ], &gq, sizeof(gq));
        if (do_q || do_k) {
            QkNormArgs qn{T, I, hd, heads, do_q, do_k, eps};
            enc2(g.p_qknorm, (uint)heads, (uint)T, @[ bqkv, bqn, bkn ], &qn, sizeof(qn));
        }
        if (do_rope) {
            RopeArgs ra{T, I, hd, heads};
            enc2(g.p_rope, (uint)heads, (uint)T, @[ bqkv, bcos, bsin ], &ra, sizeof(ra));
        }
        AttnArgs aa{T, I, hd, heads};
        enc2(g.p_attn, (uint)heads, (uint)T, @[ bqkv, bctx ], &aa, sizeof(aa));
        GemmArgs go{T, I, H};
        enc2(g.p_gemm, (uint)H, (uint)T, @[ bctx, wout, battn ], &go, sizeof(go));
        GateArgs ga{T, H, has_mod, 2};
        enc1(g.p_gate, (uint)(T * H), @[ bx, battn, bmod ], &ga, sizeof(ga));

        AdalnArgs a1{T, H, eps, has_mod, 3, 4};
        enc1(g.p_adaln, (uint)T, @[ bx, bmod, bxn ], &a1, sizeof(a1));
        GemmArgs g1{T, H, 2 * ffn};
        enc2(g.p_gemm, (uint)(2 * ffn), (uint)T, @[ bxn, wfc1, bh1 ], &g1, sizeof(g1));
        GemmArgs gs{T, 0, ffn};
        enc2(g.p_swiglu, (uint)ffn, (uint)T, @[ bh1, bgated ], &gs, sizeof(gs));
        GemmArgs g2{T, ffn, H};
        enc2(g.p_gemm, (uint)H, (uint)T, @[ bgated, wfc2, bdown ], &g2, sizeof(g2));
        GateArgs gm{T, H, has_mod, 5};
        enc1(g.p_gate, (uint)(T * H), @[ bx, bdown, bmod ], &gm, sizeof(gm));

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error)
            return cpu_dit(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                           head_dim, x, tokens, eps, adaln_mod, q_norm, k_norm, rope_cos,
                           rope_sin);
        std::memcpy(x, [bx contents], sizeof(float) * (size_t)T * H);
        return true;
    }
}

bool gemm_int8(float *y, const float *x, const int8_t *w, const float *scale, int S, int I, int O) {
    if (!y || !x || !w || !scale || S < 1 || I < 1 || O < 1)
        return false;
    init();
    if (!g.queue || !g.p_gemm_i8)
        return cpu_gemm_int8(y, x, w, scale, S, I, O);
    @autoreleasepool {
        const size_t xb = sizeof(float) * static_cast<size_t>(S) * static_cast<size_t>(I);
        const size_t wb = sizeof(int8_t) * static_cast<size_t>(O) * static_cast<size_t>(I);
        const size_t sb = sizeof(float) * static_cast<size_t>(O);
        const size_t yb = sizeof(float) * static_cast<size_t>(S) * static_cast<size_t>(O);
        id<MTLBuffer> bx = buf_bytes(g.device, x, xb);
        id<MTLBuffer> bw = buf_bytes(g.device, w, wb);
        id<MTLBuffer> bsc = buf_bytes(g.device, scale, sb);
        id<MTLBuffer> by = buf_empty(g.device, yb);
        if (!bx || !bw || !bsc || !by)
            return cpu_gemm_int8(y, x, w, scale, S, I, O);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc)
            return cpu_gemm_int8(y, x, w, scale, S, I, O);
        GemmArgs a{S, I, O};
        enc2(enc, g.p_gemm_i8, (uint)O, (uint)S, @[ bx, bw, bsc, by ], &a, sizeof(a));
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error)
            return cpu_gemm_int8(y, x, w, scale, S, I, O);
        std::memcpy(y, [by contents], yb);
        return true;
    }
}

bool nax_mlp(float *y, const float *x, const float *w_up, const float *w_down, int S, int D, int I) {
    if (!y || !x || !w_up || !w_down || S < 1 || D < 1 || I < 1)
        return false;
    init();
    if (!g.queue || !g.p_gemm || !g.p_silu || !g.p_add)
        return cpu_nax_mlp(y, x, w_up, w_down, S, D, I);
    @autoreleasepool {
        const size_t xb = sizeof(float) * static_cast<size_t>(S) * static_cast<size_t>(D);
        const size_t upb = sizeof(float) * static_cast<size_t>(I) * static_cast<size_t>(D);
        const size_t dnb = sizeof(float) * static_cast<size_t>(D) * static_cast<size_t>(I);
        const size_t midb = sizeof(float) * static_cast<size_t>(S) * static_cast<size_t>(I);
        id<MTLBuffer> bx = buf_bytes(g.device, x, xb);
        id<MTLBuffer> bup = buf_bytes(g.device, w_up, upb);
        id<MTLBuffer> bdnw = buf_bytes(g.device, w_down, dnb);
        id<MTLBuffer> by = buf_bytes(g.device, y, xb);
        id<MTLBuffer> bmid = buf_empty(g.device, midb);
        id<MTLBuffer> bdown = buf_empty(g.device, xb);
        if (!bx || !bup || !bdnw || !by || !bmid || !bdown)
            return cpu_nax_mlp(y, x, w_up, w_down, S, D, I);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc)
            return cpu_nax_mlp(y, x, w_up, w_down, S, D, I);
        GemmArgs gu{S, D, I};
        enc2(enc, g.p_gemm, (uint)I, (uint)S, @[ bx, bup, bmid ], &gu, sizeof(gu));
        ElemArgs se{S * I};
        enc1(enc, g.p_silu, (uint)(S * I), @[ bmid ], &se, sizeof(se));
        GemmArgs gd{S, I, D};
        enc2(enc, g.p_gemm, (uint)D, (uint)S, @[ bmid, bdnw, bdown ], &gd, sizeof(gd));
        ElemArgs ae{S * D};
        enc1(enc, g.p_add, (uint)(S * D), @[ by, bdown ], &ae, sizeof(ae));
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error)
            return cpu_nax_mlp(y, x, w_up, w_down, S, D, I);
        std::memcpy(y, [by contents], xb);
        return true;
    }
}

bool vae_rms_add(float *x, const float *skip, const float *w, float *y, int n, float eps) {
    if (!x || !y || n < 1)
        return false;
    init();
    if (!g.queue || !g.p_rms || (skip && !g.p_add))
        return cpu_vae_rms_add(x, skip, w, y, n, eps);
    @autoreleasepool {
        const size_t nb = sizeof(float) * static_cast<size_t>(n);
        id<MTLBuffer> bx = buf_bytes(g.device, x, nb);
        id<MTLBuffer> by = buf_empty(g.device, nb);
        id<MTLBuffer> bw = buf_bytes(g.device, w, w ? nb : 0);
        id<MTLBuffer> bsk = skip ? buf_bytes(g.device, skip, nb) : nil;
        if (!bx || !by || !bw || (skip && !bsk))
            return cpu_vae_rms_add(x, skip, w, y, n, eps);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc)
            return cpu_vae_rms_add(x, skip, w, y, n, eps);
        if (skip) {
            ElemArgs ae{n};
            enc1(enc, g.p_add, (uint)n, @[ bx, bsk ], &ae, sizeof(ae));
        }
        RmsArgs ra{n, eps, w ? 1 : 0};
        enc1(enc, g.p_rms, 1u, @[ bx, bw, by ], &ra, sizeof(ra));
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error)
            return cpu_vae_rms_add(x, skip, w, y, n, eps);
        std::memcpy(x, [bx contents], nb);
        std::memcpy(y, [by contents], nb);
        return true;
    }
}

bool vae_transformer_block(float *x, int tokens, int hidden, int heads, int hd,
                           const float *norm1, const float *qkv_w, const float *qkv_b,
                           const float *out_w, const float *out_b, const float *scale1,
                           const float *norm2, const float *w1, const float *b1, const float *w2,
                           const float *b2, const float *scale2, int ffn, float eps) {
    if (!x || tokens < 1 || hidden < 1)
        return false;
    init();
    const bool need_bias = qkv_b || out_b || b1 || b2;
    const bool metal_ok = g.queue && g.p_gemm && g.p_attn && g.p_swiglu && g.p_gate && g.p_rms_rows &&
                          (!need_bias || g.p_bias) && tokens <= 256 && heads > 0 && hd > 0 &&
                          heads * hd == hidden && qkv_w;
    if (!metal_ok)
        return cpu_vae_transformer_block(x, tokens, hidden, heads, hd, norm1, qkv_w, qkv_b, out_w,
                                         out_b, scale1, norm2, w1, b1, w2, b2, scale2, ffn, eps);
    @autoreleasepool {
        const int T = tokens;
        const int H = hidden;
        const int I = H;
        const bool do_ffn = w1 && ffn >= 1;
        const size_t xnb = sizeof(float) * static_cast<size_t>(T) * H;
        id<MTLBuffer> bx = buf_bytes(g.device, x, xnb);
        id<MTLBuffer> bnrm = buf_empty(g.device, xnb);
        id<MTLBuffer> bn1 = buf_bytes(g.device, norm1, norm1 ? sizeof(float) * (size_t)H : 0);
        id<MTLBuffer> bn2 = buf_bytes(g.device, norm2, norm2 ? sizeof(float) * (size_t)H : 0);
        id<MTLBuffer> bwqkv = buf_bytes(g.device, qkv_w, sizeof(float) * (size_t)3 * H * H);
        id<MTLBuffer> bqkv = buf_empty(g.device, sizeof(float) * (size_t)T * 3 * H);
        id<MTLBuffer> bctx = buf_empty(g.device, xnb);
        id<MTLBuffer> bao = out_w ? buf_empty(g.device, xnb) : bctx;
        id<MTLBuffer> bwout =
            out_w ? buf_bytes(g.device, out_w, sizeof(float) * (size_t)H * H) : nil;
        id<MTLBuffer> bqkvb =
            qkv_b ? buf_bytes(g.device, qkv_b, sizeof(float) * (size_t)3 * H) : nil;
        id<MTLBuffer> boutb = out_b ? buf_bytes(g.device, out_b, sizeof(float) * (size_t)H) : nil;
        id<MTLBuffer> bs1 =
            scale1 ? buf_bytes(g.device, scale1, sizeof(float) * (size_t)H) : buf_empty(g.device, 4);
        id<MTLBuffer> bs2 =
            scale2 ? buf_bytes(g.device, scale2, sizeof(float) * (size_t)H) : buf_empty(g.device, 4);
        id<MTLBuffer> bw1 = nil, bb1 = nil, bh1 = nil, bgated = nil, bw2 = nil, bb2 = nil,
                      bdown = nil;
        if (do_ffn) {
            bw1 = buf_bytes(g.device, w1, sizeof(float) * (size_t)2 * ffn * H);
            bh1 = buf_empty(g.device, sizeof(float) * (size_t)T * 2 * ffn);
            bgated = buf_empty(g.device, sizeof(float) * (size_t)T * ffn);
            if (b1)
                bb1 = buf_bytes(g.device, b1, sizeof(float) * (size_t)2 * ffn);
            if (w2) {
                bw2 = buf_bytes(g.device, w2, sizeof(float) * (size_t)H * ffn);
                bdown = buf_empty(g.device, xnb);
                if (b2)
                    bb2 = buf_bytes(g.device, b2, sizeof(float) * (size_t)H);
            }
        }
        if (!bx || !bnrm || !bn1 || !bn2 || !bwqkv || !bqkv || !bctx || !bao || !bs1 || !bs2 ||
            (out_w && !bwout) || (qkv_b && !bqkvb) || (out_b && !boutb) ||
            (do_ffn && (!bw1 || !bh1 || !bgated)) || (do_ffn && b1 && !bb1) ||
            (do_ffn && w2 && (!bw2 || !bdown)) || (do_ffn && w2 && b2 && !bb2))
            return cpu_vae_transformer_block(x, tokens, hidden, heads, hd, norm1, qkv_w, qkv_b,
                                             out_w, out_b, scale1, norm2, w1, b1, w2, b2, scale2,
                                             ffn, eps);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc)
            return cpu_vae_transformer_block(x, tokens, hidden, heads, hd, norm1, qkv_w, qkv_b,
                                             out_w, out_b, scale1, norm2, w1, b1, w2, b2, scale2,
                                             ffn, eps);

        RmsRowArgs r0{T, H, eps, norm1 ? 1 : 0};
        enc1(enc, g.p_rms_rows, (uint)T, @[ bx, bn1, bnrm ], &r0, sizeof(r0));
        GemmArgs gq{T, H, 3 * I};
        enc2(enc, g.p_gemm, (uint)(3 * I), (uint)T, @[ bnrm, bwqkv, bqkv ], &gq, sizeof(gq));
        if (qkv_b) {
            GemmArgs gb{T, 0, 3 * I};
            enc2(enc, g.p_bias, (uint)(3 * I), (uint)T, @[ bqkv, bqkvb ], &gb, sizeof(gb));
        }
        AttnArgs aa{T, I, hd, heads};
        enc2(enc, g.p_attn, (uint)heads, (uint)T, @[ bqkv, bctx ], &aa, sizeof(aa));
        id<MTLBuffer> ao = bctx;
        if (out_w) {
            GemmArgs go{T, I, H};
            enc2(enc, g.p_gemm, (uint)H, (uint)T, @[ bctx, bwout, bao ], &go, sizeof(go));
            if (out_b) {
                GemmArgs gb{T, 0, H};
                enc2(enc, g.p_bias, (uint)H, (uint)T, @[ bao, boutb ], &gb, sizeof(gb));
            }
            ao = bao;
        }
        GateArgs ga{T, H, scale1 ? 1 : 0, 0};
        enc1(enc, g.p_gate, (uint)(T * H), @[ bx, ao, bs1 ], &ga, sizeof(ga));
        RmsRowArgs r1{T, H, eps, norm2 ? 1 : 0};
        enc1(enc, g.p_rms_rows, (uint)T, @[ bx, bn2, bnrm ], &r1, sizeof(r1));
        if (do_ffn) {
            GemmArgs g1{T, H, 2 * ffn};
            enc2(enc, g.p_gemm, (uint)(2 * ffn), (uint)T, @[ bnrm, bw1, bh1 ], &g1, sizeof(g1));
            if (b1) {
                GemmArgs gb{T, 0, 2 * ffn};
                enc2(enc, g.p_bias, (uint)(2 * ffn), (uint)T, @[ bh1, bb1 ], &gb, sizeof(gb));
            }
            GemmArgs gs{T, 0, ffn};
            enc2(enc, g.p_swiglu, (uint)ffn, (uint)T, @[ bh1, bgated ], &gs, sizeof(gs));
            if (w2) {
                GemmArgs g2{T, ffn, H};
                enc2(enc, g.p_gemm, (uint)H, (uint)T, @[ bgated, bw2, bdown ], &g2, sizeof(g2));
                if (b2) {
                    GemmArgs gb{T, 0, H};
                    enc2(enc, g.p_bias, (uint)H, (uint)T, @[ bdown, bb2 ], &gb, sizeof(gb));
                }
                GateArgs gm{T, H, scale2 ? 1 : 0, 0};
                enc1(enc, g.p_gate, (uint)(T * H), @[ bx, bdown, bs2 ], &gm, sizeof(gm));
            }
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error)
            return cpu_vae_transformer_block(x, tokens, hidden, heads, hd, norm1, qkv_w, qkv_b,
                                             out_w, out_b, scale1, norm2, w1, b1, w2, b2, scale2,
                                             ffn, eps);
        std::memcpy(x, [bx contents], xnb);
        return true;
    }
}

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
    init();
    const bool do_rope = rope_cos && rope_sin && rope_half > 0;
    const bool do_mlp = intermediate > 0 && fc1_w && fc2_w;
    const bool need_bias = qkv_b || proj_b || fc1_b || fc2_b;
    const bool metal_ok = g.queue && g.p_gemm && g.p_attn && g.p_add && g.p_ln &&
                          (!do_mlp || g.p_gelu) && (!need_bias || g.p_bias) &&
                          (!do_rope || g.p_vrope) && rows <= 256;
    if (!metal_ok)
        return cpu_vision_block(x, rows, hidden, heads, hd, intermediate, norm1_w, norm1_b, qkv_w,
                                qkv_b, proj_w, proj_b, norm2_w, norm2_b, fc1_w, fc1_b, fc2_w,
                                fc2_b, rope_cos, rope_sin, rope_half, eps);
    @autoreleasepool {
        const int T = rows;
        const int H = hidden;
        const int I = intermediate;
        const size_t xnb = sizeof(float) * static_cast<size_t>(T) * H;
        id<MTLBuffer> bx = buf_bytes(g.device, x, xnb);
        id<MTLBuffer> bnrm = buf_empty(g.device, xnb);
        id<MTLBuffer> bn1w = buf_bytes(g.device, norm1_w, norm1_w ? sizeof(float) * (size_t)H : 0);
        id<MTLBuffer> bn1b = buf_bytes(g.device, norm1_b, norm1_b ? sizeof(float) * (size_t)H : 0);
        id<MTLBuffer> bn2w = buf_bytes(g.device, norm2_w, norm2_w ? sizeof(float) * (size_t)H : 0);
        id<MTLBuffer> bn2b = buf_bytes(g.device, norm2_b, norm2_b ? sizeof(float) * (size_t)H : 0);
        id<MTLBuffer> bwqkv = buf_bytes(g.device, qkv_w, sizeof(float) * (size_t)3 * H * H);
        id<MTLBuffer> bqkv = buf_empty(g.device, sizeof(float) * (size_t)T * 3 * H);
        id<MTLBuffer> bqkv_r = do_rope ? buf_empty(g.device, sizeof(float) * (size_t)T * 3 * H) : bqkv;
        id<MTLBuffer> bqkvb =
            qkv_b ? buf_bytes(g.device, qkv_b, sizeof(float) * (size_t)3 * H) : nil;
        id<MTLBuffer> bctx = buf_empty(g.device, xnb);
        id<MTLBuffer> bbranch = proj_w ? buf_empty(g.device, xnb) : bctx;
        id<MTLBuffer> bwproj =
            proj_w ? buf_bytes(g.device, proj_w, sizeof(float) * (size_t)H * H) : nil;
        id<MTLBuffer> bprojb = proj_b ? buf_bytes(g.device, proj_b, sizeof(float) * (size_t)H) : nil;
        id<MTLBuffer> bcos =
            do_rope ? buf_bytes(g.device, rope_cos, sizeof(float) * (size_t)T * rope_half) : nil;
        id<MTLBuffer> bsin =
            do_rope ? buf_bytes(g.device, rope_sin, sizeof(float) * (size_t)T * rope_half) : nil;
        id<MTLBuffer> bwfc1 = nil, bfc1b = nil, bmid = nil, bwfc2 = nil, bfc2b = nil, bmlp = nil;
        if (do_mlp) {
            bwfc1 = buf_bytes(g.device, fc1_w, sizeof(float) * (size_t)I * H);
            bmid = buf_empty(g.device, sizeof(float) * (size_t)T * I);
            bwfc2 = buf_bytes(g.device, fc2_w, sizeof(float) * (size_t)H * I);
            bmlp = buf_empty(g.device, xnb);
            if (fc1_b)
                bfc1b = buf_bytes(g.device, fc1_b, sizeof(float) * (size_t)I);
            if (fc2_b)
                bfc2b = buf_bytes(g.device, fc2_b, sizeof(float) * (size_t)H);
        }
        if (!bx || !bnrm || !bn1w || !bn1b || !bn2w || !bn2b || !bwqkv || !bqkv || !bqkv_r ||
            !bctx || !bbranch || (qkv_b && !bqkvb) || (proj_w && !bwproj) || (proj_b && !bprojb) ||
            (do_rope && (!bcos || !bsin)) ||
            (do_mlp && (!bwfc1 || !bmid || !bwfc2 || !bmlp)) || (do_mlp && fc1_b && !bfc1b) ||
            (do_mlp && fc2_b && !bfc2b))
            return cpu_vision_block(x, rows, hidden, heads, hd, intermediate, norm1_w, norm1_b,
                                    qkv_w, qkv_b, proj_w, proj_b, norm2_w, norm2_b, fc1_w, fc1_b,
                                    fc2_w, fc2_b, rope_cos, rope_sin, rope_half, eps);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc)
            return cpu_vision_block(x, rows, hidden, heads, hd, intermediate, norm1_w, norm1_b,
                                    qkv_w, qkv_b, proj_w, proj_b, norm2_w, norm2_b, fc1_w, fc1_b,
                                    fc2_w, fc2_b, rope_cos, rope_sin, rope_half, eps);

        LnRowArgs ln0{T, H, eps, norm1_w ? 1 : 0, norm1_b ? 1 : 0};
        enc1(enc, g.p_ln, (uint)T, @[ bx, bn1w, bn1b, bnrm ], &ln0, sizeof(ln0));
        GemmArgs gq{T, H, 3 * H};
        enc2(enc, g.p_gemm, (uint)(3 * H), (uint)T, @[ bnrm, bwqkv, bqkv ], &gq, sizeof(gq));
        if (qkv_b) {
            GemmArgs gb{T, 0, 3 * H};
            enc2(enc, g.p_bias, (uint)(3 * H), (uint)T, @[ bqkv, bqkvb ], &gb, sizeof(gb));
        }
        id<MTLBuffer> bqkv_attn = bqkv;
        if (do_rope) {
            VisionRopeArgs ra{T, H, hd, heads, rope_half};
            enc2(enc, g.p_vrope, (uint)heads, (uint)T, @[ bqkv, bqkv_r, bcos, bsin ], &ra,
                 sizeof(ra));
            bqkv_attn = bqkv_r;
        }
        AttnArgs aa{T, H, hd, heads};
        enc2(enc, g.p_attn, (uint)heads, (uint)T, @[ bqkv_attn, bctx ], &aa, sizeof(aa));
        id<MTLBuffer> ao = bctx;
        if (proj_w) {
            GemmArgs go{T, H, H};
            enc2(enc, g.p_gemm, (uint)H, (uint)T, @[ bctx, bwproj, bbranch ], &go, sizeof(go));
            if (proj_b) {
                GemmArgs gb{T, 0, H};
                enc2(enc, g.p_bias, (uint)H, (uint)T, @[ bbranch, bprojb ], &gb, sizeof(gb));
            }
            ao = bbranch;
        }
        ElemArgs ae{T * H};
        enc1(enc, g.p_add, (uint)(T * H), @[ bx, ao ], &ae, sizeof(ae));
        LnRowArgs ln1{T, H, eps, norm2_w ? 1 : 0, norm2_b ? 1 : 0};
        enc1(enc, g.p_ln, (uint)T, @[ bx, bn2w, bn2b, bnrm ], &ln1, sizeof(ln1));
        if (do_mlp) {
            GemmArgs g1{T, H, I};
            enc2(enc, g.p_gemm, (uint)I, (uint)T, @[ bnrm, bwfc1, bmid ], &g1, sizeof(g1));
            if (fc1_b) {
                GemmArgs gb{T, 0, I};
                enc2(enc, g.p_bias, (uint)I, (uint)T, @[ bmid, bfc1b ], &gb, sizeof(gb));
            }
            ElemArgs ge{T * I};
            enc1(enc, g.p_gelu, (uint)(T * I), @[ bmid ], &ge, sizeof(ge));
            GemmArgs g2{T, I, H};
            enc2(enc, g.p_gemm, (uint)H, (uint)T, @[ bmid, bwfc2, bmlp ], &g2, sizeof(g2));
            if (fc2_b) {
                GemmArgs gb{T, 0, H};
                enc2(enc, g.p_bias, (uint)H, (uint)T, @[ bmlp, bfc2b ], &gb, sizeof(gb));
            }
            enc1(enc, g.p_add, (uint)(T * H), @[ bx, bmlp ], &ae, sizeof(ae));
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error)
            return cpu_vision_block(x, rows, hidden, heads, hd, intermediate, norm1_w, norm1_b,
                                    qkv_w, qkv_b, proj_w, proj_b, norm2_w, norm2_b, fc1_w, fc1_b,
                                    fc2_w, fc2_b, rope_cos, rope_sin, rope_half, eps);
        std::memcpy(x, [bx contents], xnb);
        return true;
    }
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
    init();
    const int rows = B * L;
    const bool do_mlp = w0 && w1 && w2;
    const bool need_bias = q_bias || k_bias || v_bias || proj_b || b0 || b1 || b2;
    const bool metal_ok = g.queue && g.p_gemm && g.p_attn_c && g.p_add && g.p_ln && g.p_pool &&
                          (!do_mlp || g.p_gelu_mul) && (!need_bias || g.p_bias) && L <= 256;
    if (!metal_ok)
        return cpu_audio_pre_block(base, seq, B, L, C, ch, heads, norm1_w, norm1_b, qkv_w, q_bias,
                                   k_bias, v_bias, proj_w, proj_b, norm2_w, norm2_b, mlp_norm_w,
                                   mlp_norm_b, w0, b0, w1, b1, w2, b2, eps);
    @autoreleasepool {
        const size_t seqb = sizeof(float) * static_cast<size_t>(rows) * C;
        const size_t baseb = sizeof(float) * static_cast<size_t>(rows) * ch;
        id<MTLBuffer> bseq = buf_bytes(g.device, seq, seqb);
        id<MTLBuffer> bbase = buf_bytes(g.device, base, baseb);
        id<MTLBuffer> ban = buf_empty(g.device, seqb);
        id<MTLBuffer> bn1w = buf_bytes(g.device, norm1_w, norm1_w ? sizeof(float) * (size_t)C : 0);
        id<MTLBuffer> bn1b = buf_bytes(g.device, norm1_b, norm1_b ? sizeof(float) * (size_t)C : 0);
        id<MTLBuffer> bwqkv = buf_bytes(g.device, qkv_w, sizeof(float) * (size_t)3 * C * C);
        id<MTLBuffer> bqkv = buf_empty(g.device, sizeof(float) * (size_t)rows * 3 * C);
        std::vector<float> qkv_bias;
        id<MTLBuffer> bqkvb = nil;
        if (q_bias || k_bias || v_bias) {
            qkv_bias.assign(static_cast<size_t>(3) * C, 0.f);
            if (q_bias)
                std::memcpy(qkv_bias.data(), q_bias, sizeof(float) * (size_t)C);
            if (k_bias)
                std::memcpy(qkv_bias.data() + C, k_bias, sizeof(float) * (size_t)C);
            if (v_bias)
                std::memcpy(qkv_bias.data() + 2 * C, v_bias, sizeof(float) * (size_t)C);
            bqkvb = buf_bytes(g.device, qkv_bias.data(), sizeof(float) * (size_t)3 * C);
        }
        id<MTLBuffer> bctx = buf_empty(g.device, seqb);
        id<MTLBuffer> bpool = buf_empty(g.device, baseb);
        id<MTLBuffer> bap = proj_w ? buf_empty(g.device, baseb) : bpool;
        id<MTLBuffer> bwproj =
            proj_w ? buf_bytes(g.device, proj_w, sizeof(float) * (size_t)ch * ch) : nil;
        id<MTLBuffer> bprojb = proj_b ? buf_bytes(g.device, proj_b, sizeof(float) * (size_t)ch) : nil;
        id<MTLBuffer> bn2w = buf_bytes(g.device, norm2_w, norm2_w ? sizeof(float) * (size_t)ch : 0);
        id<MTLBuffer> bn2b = buf_bytes(g.device, norm2_b, norm2_b ? sizeof(float) * (size_t)ch : 0);
        id<MTLBuffer> bnmw =
            buf_bytes(g.device, mlp_norm_w, mlp_norm_w ? sizeof(float) * (size_t)ch : 0);
        id<MTLBuffer> bnmb =
            buf_bytes(g.device, mlp_norm_b, mlp_norm_b ? sizeof(float) * (size_t)ch : 0);
        id<MTLBuffer> bnrm = do_mlp ? buf_empty(g.device, baseb) : nil;
        const int mid = 2 * ch;
        id<MTLBuffer> bw0 = nil, bb0 = nil, bw1 = nil, bb1 = nil, bw2 = nil, bb2 = nil;
        id<MTLBuffer> bgate = nil, blin = nil, bgg = nil, bbr = nil;
        if (do_mlp) {
            bw0 = buf_bytes(g.device, w0, sizeof(float) * (size_t)mid * ch);
            bw1 = buf_bytes(g.device, w1, sizeof(float) * (size_t)mid * ch);
            bw2 = buf_bytes(g.device, w2, sizeof(float) * (size_t)ch * mid);
            bgate = buf_empty(g.device, sizeof(float) * (size_t)rows * mid);
            blin = buf_empty(g.device, sizeof(float) * (size_t)rows * mid);
            bgg = buf_empty(g.device, sizeof(float) * (size_t)rows * mid);
            bbr = buf_empty(g.device, baseb);
            if (b0)
                bb0 = buf_bytes(g.device, b0, sizeof(float) * (size_t)mid);
            if (b1)
                bb1 = buf_bytes(g.device, b1, sizeof(float) * (size_t)mid);
            if (b2)
                bb2 = buf_bytes(g.device, b2, sizeof(float) * (size_t)ch);
        }
        if (!bseq || !bbase || !ban || !bn1w || !bn1b || !bwqkv || !bqkv || !bctx || !bpool ||
            !bap || !bn2w || !bn2b || !bnmw || !bnmb || ((q_bias || k_bias || v_bias) && !bqkvb) ||
            (proj_w && !bwproj) || (proj_b && !bprojb) ||
            (do_mlp && (!bnrm || !bw0 || !bw1 || !bw2 || !bgate || !blin || !bgg || !bbr)) ||
            (do_mlp && b0 && !bb0) || (do_mlp && b1 && !bb1) || (do_mlp && b2 && !bb2))
            return cpu_audio_pre_block(base, seq, B, L, C, ch, heads, norm1_w, norm1_b, qkv_w,
                                       q_bias, k_bias, v_bias, proj_w, proj_b, norm2_w, norm2_b,
                                       mlp_norm_w, mlp_norm_b, w0, b0, w1, b1, w2, b2, eps);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoder] : nil;
        if (!cb || !enc)
            return cpu_audio_pre_block(base, seq, B, L, C, ch, heads, norm1_w, norm1_b, qkv_w,
                                       q_bias, k_bias, v_bias, proj_w, proj_b, norm2_w, norm2_b,
                                       mlp_norm_w, mlp_norm_b, w0, b0, w1, b1, w2, b2, eps);

        LnRowArgs ln0{rows, C, eps, norm1_w ? 1 : 0, norm1_b ? 1 : 0};
        enc1(enc, g.p_ln, (uint)rows, @[ bseq, bn1w, bn1b, ban ], &ln0, sizeof(ln0));
        GemmArgs gq{rows, C, 3 * C};
        enc2(enc, g.p_gemm, (uint)(3 * C), (uint)rows, @[ ban, bwqkv, bqkv ], &gq, sizeof(gq));
        if (bqkvb) {
            GemmArgs gb{rows, 0, 3 * C};
            enc2(enc, g.p_bias, (uint)(3 * C), (uint)rows, @[ bqkv, bqkvb ], &gb, sizeof(gb));
        }
        CausalAttnArgs aa{B, L, C, hd, heads};
        enc2(enc, g.p_attn_c, (uint)heads, (uint)rows, @[ bqkv, bctx ], &aa, sizeof(aa));
        PoolArgs pa{rows, C, ch};
        enc2(enc, g.p_pool, (uint)ch, (uint)rows, @[ bctx, bpool ], &pa, sizeof(pa));
        id<MTLBuffer> ao = bpool;
        if (proj_w) {
            GemmArgs go{rows, ch, ch};
            enc2(enc, g.p_gemm, (uint)ch, (uint)rows, @[ bpool, bwproj, bap ], &go, sizeof(go));
            if (proj_b) {
                GemmArgs gb{rows, 0, ch};
                enc2(enc, g.p_bias, (uint)ch, (uint)rows, @[ bap, bprojb ], &gb, sizeof(gb));
            }
            ao = bap;
        }
        ElemArgs ae{rows * ch};
        enc1(enc, g.p_add, (uint)(rows * ch), @[ bbase, ao ], &ae, sizeof(ae));
        if (do_mlp) {
            LnRowArgs ln1{rows, ch, eps, norm2_w ? 1 : 0, norm2_b ? 1 : 0};
            enc1(enc, g.p_ln, (uint)rows, @[ bbase, bn2w, bn2b, bnrm ], &ln1, sizeof(ln1));
            LnRowArgs ln2{rows, ch, eps, mlp_norm_w ? 1 : 0, mlp_norm_b ? 1 : 0};
            enc1(enc, g.p_ln, (uint)rows, @[ bnrm, bnmw, bnmb, bnrm ], &ln2, sizeof(ln2));
            GemmArgs g0{rows, ch, mid};
            enc2(enc, g.p_gemm, (uint)mid, (uint)rows, @[ bnrm, bw0, bgate ], &g0, sizeof(g0));
            if (b0) {
                GemmArgs gb{rows, 0, mid};
                enc2(enc, g.p_bias, (uint)mid, (uint)rows, @[ bgate, bb0 ], &gb, sizeof(gb));
            }
            GemmArgs g1{rows, ch, mid};
            enc2(enc, g.p_gemm, (uint)mid, (uint)rows, @[ bnrm, bw1, blin ], &g1, sizeof(g1));
            if (b1) {
                GemmArgs gb{rows, 0, mid};
                enc2(enc, g.p_bias, (uint)mid, (uint)rows, @[ blin, bb1 ], &gb, sizeof(gb));
            }
            ElemArgs ge{rows * mid};
            enc1(enc, g.p_gelu_mul, (uint)(rows * mid), @[ bgate, blin, bgg ], &ge, sizeof(ge));
            GemmArgs g2{rows, mid, ch};
            enc2(enc, g.p_gemm, (uint)ch, (uint)rows, @[ bgg, bw2, bbr ], &g2, sizeof(g2));
            if (b2) {
                GemmArgs gb{rows, 0, ch};
                enc2(enc, g.p_bias, (uint)ch, (uint)rows, @[ bbr, bb2 ], &gb, sizeof(gb));
            }
            enc1(enc, g.p_add, (uint)(rows * ch), @[ bbase, bbr ], &ae, sizeof(ae));
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error)
            return cpu_audio_pre_block(base, seq, B, L, C, ch, heads, norm1_w, norm1_b, qkv_w,
                                       q_bias, k_bias, v_bias, proj_w, proj_b, norm2_w, norm2_b,
                                       mlp_norm_w, mlp_norm_b, w0, b0, w1, b1, w2, b2, eps);
        std::memcpy(base, [bbase contents], baseb);
        return true;
    }
}

} // namespace metal_h3
} // namespace mvllm

#else // !MVLLM_WITH_METAL

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
                        qkv.data() + static_cast<size_t>(i) * 3 * H,
                        static_cast<size_t>(H) * sizeof(float));
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
    auto add_bias = [](float *y, const float *b, int S, int O) {
        if (!b)
            return;
        for (int s = 0; s < S; ++s) {
            float *row = y + static_cast<size_t>(s) * O;
            for (int o = 0; o < O; ++o)
                row[o] += b[o];
        }
    };
    auto linear = [&](float *y, const float *in, const float *w, const float *b, int S, int II,
                      int O) {
        quant::matmul_f32(y, in, w, S, II, O);
        add_bias(y, b, S, O);
    };
    auto gelu = [](float v) {
        const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        if (inner <= -10.f)
            return 0.f;
        if (inner >= 10.f)
            return v;
        return 0.5f * v * (1.f + std::tanh(inner));
    };
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
    linear(qkv.data(), norm.data(), qkv_w, qkv_b, rows, H, 3 * H);
    const int inner = heads * hd;
    for (int row = 0; row < rows; ++row) {
        const float *crow = (rope_cos && rope_half > 0)
                                ? rope_cos + static_cast<size_t>(row) * rope_half
                                : nullptr;
        const float *srow = (rope_sin && rope_half > 0)
                                ? rope_sin + static_cast<size_t>(row) * rope_half
                                : nullptr;
        for (int head = 0; head < heads; ++head) {
            const float *qs = qkv.data() + (static_cast<size_t>(row) * 3 + 0) * inner +
                              static_cast<size_t>(head) * hd;
            const float *ks = qkv.data() + (static_cast<size_t>(row) * 3 + 1) * inner +
                              static_cast<size_t>(head) * hd;
            const float *vs = qkv.data() + (static_cast<size_t>(row) * 3 + 2) * inner +
                              static_cast<size_t>(head) * hd;
            float *qd = query.data() + (static_cast<size_t>(row) * heads + head) * hd;
            float *kd = key.data() + (static_cast<size_t>(row) * heads + head) * hd;
            float *vd = value.data() + (static_cast<size_t>(row) * heads + head) * hd;
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
    const float scale = 1.f / std::sqrt(static_cast<float>(hd > 0 ? hd : 1));
    std::vector<float> scores(static_cast<size_t>(std::max(rows, 0)));
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < rows; ++i) {
            const float *qi = query.data() + (static_cast<size_t>(i) * heads + h) * hd;
            for (int j = 0; j < rows; ++j) {
                const float *kj = key.data() + (static_cast<size_t>(j) * heads + h) * hd;
                float dot = 0.f;
                for (int d = 0; d < hd; ++d)
                    dot += qi[d] * kj[d];
                scores[static_cast<size_t>(j)] = dot * scale;
            }
            quant::softmax_inplace(scores.data(), rows);
            float *oi = attn.data() + (static_cast<size_t>(i) * heads + h) * hd;
            std::memset(oi, 0, static_cast<size_t>(hd) * sizeof(float));
            for (int j = 0; j < rows; ++j) {
                const float *vj = value.data() + (static_cast<size_t>(j) * heads + h) * hd;
                const float a = scores[static_cast<size_t>(j)];
                for (int d = 0; d < hd; ++d)
                    oi[d] += a * vj[d];
            }
        }
    }
    if (proj_w)
        linear(branch.data(), attn.data(), proj_w, proj_b, rows, H, H);
    else
        std::memcpy(branch.data(), attn.data(), static_cast<size_t>(rows) * H * sizeof(float));
    for (size_t i = 0; i < static_cast<size_t>(rows) * H; ++i)
        x[i] += branch[i];
    for (int r = 0; r < rows; ++r)
        layernorm(x + static_cast<size_t>(r) * H, norm2_w, norm2_b,
                  norm.data() + static_cast<size_t>(r) * H, H, eps);
    if (I > 0 && fc1_w && fc2_w) {
        std::vector<float> fc1(static_cast<size_t>(rows) * I);
        linear(fc1.data(), norm.data(), fc1_w, fc1_b, rows, H, I);
        for (int i = 0; i < rows * I; ++i)
            fc1[static_cast<size_t>(i)] = gelu(fc1[static_cast<size_t>(i)]);
        linear(branch.data(), fc1.data(), fc2_w, fc2_b, rows, I, H);
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
    auto add_bias = [](float *y, const float *b, int S, int O) {
        if (!b)
            return;
        for (int s = 0; s < S; ++s) {
            float *row = y + static_cast<size_t>(s) * O;
            for (int o = 0; o < O; ++o)
                row[o] += b[o];
        }
    };
    auto linear = [&](float *y, const float *in, const float *w, const float *b, int S, int II,
                      int O) {
        quant::matmul_f32(y, in, w, S, II, O);
        add_bias(y, b, S, O);
    };
    auto gelu = [](float v) {
        return 0.5f * v * (1.f + std::tanh(0.79788456f * (v + 0.044715f * v * v * v)));
    };
    std::vector<float> an(static_cast<size_t>(rows) * C);
    std::memcpy(an.data(), seq, static_cast<size_t>(rows) * C * sizeof(float));
    for (int r = 0; r < rows; ++r)
        layernorm(an.data() + static_cast<size_t>(r) * C, norm1_w, norm1_b,
                  an.data() + static_cast<size_t>(r) * C, C, eps);
    std::vector<float> qkv_o(static_cast<size_t>(rows) * 3 * C);
    linear(qkv_o.data(), an.data(), qkv_w, nullptr, rows, C, 3 * C);
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
    const float scale = 1.f / std::sqrt(static_cast<float>(hd > 0 ? hd : 1));
    std::vector<float> scores(static_cast<size_t>(L));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int i = 0; i < L; ++i) {
                const float *qi = q.data() + ((static_cast<size_t>(b) * L + i) * heads + h) * hd;
                float mx = -1e30f;
                int lim = i + 1;
                for (int j = 0; j < lim; ++j) {
                    const float *kj = k.data() + ((static_cast<size_t>(b) * L + j) * heads + h) * hd;
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
                float *oi = att.data() + ((static_cast<size_t>(b) * L + i) * heads + h) * hd;
                for (int d = 0; d < hd; ++d)
                    oi[d] = 0.f;
                for (int j = 0; j < lim; ++j) {
                    float a = scores[static_cast<size_t>(j)] * inv;
                    const float *vj = v.data() + ((static_cast<size_t>(b) * L + j) * heads + h) * hd;
                    for (int d = 0; d < hd; ++d)
                        oi[d] += a * vj[d];
                }
            }
        }
    }
    std::vector<float> pooled(static_cast<size_t>(rows) * ch, 0.f);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < ch; ++c)
            pooled[static_cast<size_t>(r) * ch + c] = att[static_cast<size_t>(r) * C + (c % C)];
    std::vector<float> ap(static_cast<size_t>(rows) * ch, 0.f);
    if (proj_w)
        linear(ap.data(), pooled.data(), proj_w, proj_b, rows, ch, ch);
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
        linear(gate.data(), n2.data(), w0, b0, rows, ch, mid);
        linear(lin.data(), n2.data(), w1, b1, rows, ch, mid);
        for (size_t i = 0; i < gg.size(); ++i)
            gg[i] = gelu(gate[i]) * lin[i];
        std::vector<float> br(static_cast<size_t>(rows) * ch);
        linear(br.data(), gg.data(), w2, b2, rows, mid, ch);
        for (int i = 0; i < rows * ch; ++i)
            base[i] += br[static_cast<size_t>(i)];
    }
    return true;
}

} // namespace metal_h3
} // namespace mvllm

#endif // MVLLM_WITH_METAL
