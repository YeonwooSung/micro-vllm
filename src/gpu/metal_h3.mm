#include "metal_h3.hpp"

#include "../model/family.hpp"
#include "../quant/quant.hpp"

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
            ready = p_gemm && p_bf16 && p_adaln && p_qknorm && p_rope && p_attn && p_gate && p_swiglu;
            return ready;
        }
    }

    void teardown() {
        p_gemm = p_bf16 = p_adaln = p_qknorm = p_rope = p_attn = p_gate = p_swiglu = p_gemm_i8 =
            p_silu = p_add = p_rms = nil;
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

} // namespace metal_h3
} // namespace mvllm

#endif // MVLLM_WITH_METAL
