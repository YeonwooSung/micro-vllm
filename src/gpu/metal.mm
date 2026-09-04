#if defined(MVLLM_WITH_METAL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backend.hpp"

#include "../model/family.hpp"
#include "../quant/quant.hpp"

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace mvllm {
namespace gpu {
namespace {

static const char *kMetalSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant float MX4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

struct GemmArgs { int S; int I; int O; };
struct NormArgs { int T; int H; float eps; };
struct AttnArgs { int T; int I; int hd; int heads; };
struct ElemArgs { int n; };

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

kernel void gemm_int4_g64(device const float *x [[buffer(0)]],
                          device const uchar *packed [[buffer(1)]],
                          device const float *scales [[buffer(2)]],
                          device float *y [[buffer(3)]],
                          constant GemmArgs &a [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint o = gid.x;
    uint s = gid.y;
    if (o >= (uint)a.O || s >= (uint)a.S) return;
    const int I = a.I;
    const int stride = (I + 1) / 2;
    const int ng = (I + 63) / 64;
    const device float *xs = x + (ulong)s * (uint)I;
    const device uchar *prow = packed + (ulong)o * (uint)stride;
    const device float *srow = scales + (ulong)o * (uint)ng;
    float acc = 0.0f;
    for (int i = 0; i < I; ++i) {
        uchar b = prow[i >> 1];
        uint nib = (i & 1) ? (b >> 4) : (b & 0x0f);
        int q = (int)nib - 8;
        acc += xs[i] * ((float)q * srow[i / 64]);
    }
    y[(ulong)s * (uint)a.O + o] = acc;
}

kernel void gemm_mxfp4(device const float *x [[buffer(0)]],
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

kernel void bf16_to_f32(device const ushort *src [[buffer(0)]],
                        device float *dst [[buffer(1)]],
                        constant ElemArgs &a [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.n) return;
    dst[gid] = as_type<float>((uint)src[gid] << 16);
}

kernel void rmsnorm_ones(device const float *x [[buffer(0)]],
                         device float *y [[buffer(1)]],
                         constant NormArgs &a [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.T) return;
    const device float *xs = x + (ulong)gid * (uint)a.H;
    device float *ys = y + (ulong)gid * (uint)a.H;
    float ss = 0.0f;
    for (int i = 0; i < a.H; ++i)
        ss += xs[i] * xs[i];
    float inv = rsqrt(ss / (float)a.H + a.eps);
    for (int i = 0; i < a.H; ++i)
        ys[i] = xs[i] * inv;
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

kernel void add_inplace(device float *y [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        constant ElemArgs &a [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)a.n) return;
    y[gid] += x[gid];
}

kernel void swiglu_pack(device const float *h1 [[buffer(0)]],
                        device float *gated [[buffer(1)]],
                        constant GemmArgs &a [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    // a.S = T, a.I unused, a.O = ffn. h1 is [T, 2*ffn]
    uint i = gid.x;
    uint t = gid.y;
    if (t >= (uint)a.S || i >= (uint)a.O) return;
    float g = h1[(ulong)t * (uint)(2 * a.O) + i];
    float u = h1[(ulong)t * (uint)(2 * a.O) + a.O + i];
    float sig = 1.0f / (1.0f + exp(-g));
    gated[(ulong)t * (uint)a.O + i] = g * sig * u;
}
)MSL";

struct GemmArgs {
    int S, I, O;
};
struct NormArgs {
    int T, H;
    float eps;
};
struct AttnArgs {
    int T, I, hd, heads;
};
struct ElemArgs {
    int n;
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

class MetalBackend final : public Backend {
public:
    bool init() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_)
                return false;
            queue_ = [device_ newCommandQueue];
            if (!queue_)
                return false;
            NSError *err = nil;
            NSString *src = [NSString stringWithUTF8String:kMetalSrc];
            id<MTLLibrary> lib = [device_ newLibraryWithSource:src options:nil error:&err];
            if (!lib)
                return false;
            auto pso = [&](const char *name) -> id<MTLComputePipelineState> {
                id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
                if (!fn)
                    return nil;
                NSError *e = nil;
                return [device_ newComputePipelineStateWithFunction:fn error:&e];
            };
            p_gemm_f32_ = pso("gemm_f32");
            p_gemm_i4_ = pso("gemm_int4_g64");
            p_gemm_mx_ = pso("gemm_mxfp4");
            p_bf16_ = pso("bf16_to_f32");
            p_rms_ = pso("rmsnorm_ones");
            p_attn_ = pso("dit_attn");
            p_add_ = pso("add_inplace");
            p_swiglu_ = pso("swiglu_pack");
            return p_gemm_f32_ && p_gemm_i4_ && p_gemm_mx_ && p_bf16_ && p_rms_ && p_attn_ &&
                   p_add_ && p_swiglu_;
        }
    }

    Device device() const override { return Device::Metal; }
    const char *name() const override { return "metal"; }

    void gemm_f32(float *y, const float *x, const float *w, int S, int I, int O) override {
        if (!y || !x || !w || S <= 0 || I <= 0 || O <= 0)
            return;
        @autoreleasepool {
            const size_t xb = sizeof(float) * (size_t)S * I;
            const size_t wb = sizeof(float) * (size_t)O * I;
            const size_t yb = sizeof(float) * (size_t)S * O;
            id<MTLBuffer> bx = buf_bytes(device_, x, xb);
            id<MTLBuffer> bw = buf_bytes(device_, w, wb);
            id<MTLBuffer> by = buf_empty(device_, yb);
            GemmArgs a{S, I, O};
            if (!dispatch2(p_gemm_f32_, O, S, bx, bw, by, &a, sizeof(a))) {
                quant::matmul_f32(y, x, w, S, I, O);
                return;
            }
            std::memcpy(y, [by contents], yb);
        }
    }

    void gemm_int4_g64(float *y, const float *x, const uint8_t *packed, const float *scales, int S,
                       int I, int O) override {
        if (!y || !x || !packed || !scales || S <= 0 || I <= 0 || O <= 0)
            return;
        @autoreleasepool {
            const int stride = (I + 1) / 2;
            const int ng = (I + 63) / 64;
            id<MTLBuffer> bx = buf_bytes(device_, x, sizeof(float) * (size_t)S * I);
            id<MTLBuffer> bp = buf_bytes(device_, packed, (size_t)O * stride);
            id<MTLBuffer> bs = buf_bytes(device_, scales, sizeof(float) * (size_t)O * ng);
            id<MTLBuffer> by = buf_empty(device_, sizeof(float) * (size_t)S * O);
            GemmArgs a{S, I, O};
            if (!dispatch2(p_gemm_i4_, O, S, bx, bp, bs, by, &a, sizeof(a))) {
                quant::matmul_int4_g64(y, x, packed, scales, S, I, O);
                return;
            }
            std::memcpy(y, [by contents], sizeof(float) * (size_t)S * O);
        }
    }

    void gemm_mxfp4(float *y, const float *x, const uint8_t *packed, const uint8_t *scales, int S,
                    int I, int O, bool) override {
        if (!y || !x || !packed || !scales || S <= 0 || I <= 0 || O <= 0)
            return;
        @autoreleasepool {
            const int stride = (I + 1) / 2;
            const int ng = (I + 31) / 32;
            id<MTLBuffer> bx = buf_bytes(device_, x, sizeof(float) * (size_t)S * I);
            id<MTLBuffer> bp = buf_bytes(device_, packed, (size_t)O * stride);
            id<MTLBuffer> bs = buf_bytes(device_, scales, (size_t)O * ng);
            id<MTLBuffer> by = buf_empty(device_, sizeof(float) * (size_t)S * O);
            GemmArgs a{S, I, O};
            if (!dispatch2(p_gemm_mx_, O, S, bx, bp, bs, by, &a, sizeof(a))) {
                quant::matmul_mxfp4(y, x, packed, scales, S, I, O);
                return;
            }
            std::memcpy(y, [by contents], sizeof(float) * (size_t)S * O);
        }
    }

    void dit_block(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                   int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                   int tokens, float eps) override {
        if (!blob || !x || hidden <= 0 || inner <= 0 || ffn <= 0 || tokens <= 0) {
            h3_dit_block_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                             head_dim, x, tokens, eps);
            return;
        }
        if (tokens > 256) {
            h3_dit_block_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                             head_dim, x, tokens, eps);
            return;
        }
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
        id<MTLBuffer> bqkv_bf = buf_bytes(device_, blob, (size_t)qkv_bytes);
        id<MTLBuffer> bout_bf = buf_bytes(device_, blob + qkv_bytes, (size_t)out_bytes);
        id<MTLBuffer> bfc1_bf =
            buf_bytes(device_, blob + qkv_bytes + out_bytes, (size_t)fc1_bytes);
        id<MTLBuffer> bfc2_bf =
            buf_bytes(device_, blob + qkv_bytes + out_bytes + fc1_bytes, (size_t)fc2_bytes);
        id<MTLBuffer> bx = buf_bytes(device_, x, sizeof(float) * (size_t)T * H);
        id<MTLBuffer> bxn = buf_empty(device_, sizeof(float) * (size_t)T * H);
        id<MTLBuffer> wqkv = buf_empty(device_, sizeof(float) * (size_t)qkv_n);
        id<MTLBuffer> wout = buf_empty(device_, sizeof(float) * (size_t)out_n);
        id<MTLBuffer> wfc1 = buf_empty(device_, sizeof(float) * (size_t)fc1_n);
        id<MTLBuffer> wfc2 = buf_empty(device_, sizeof(float) * (size_t)fc2_n);
        id<MTLBuffer> bqkv = buf_empty(device_, sizeof(float) * (size_t)T * 3 * I);
        id<MTLBuffer> bctx = buf_empty(device_, sizeof(float) * (size_t)T * I);
        id<MTLBuffer> battn = buf_empty(device_, sizeof(float) * (size_t)T * H);
        id<MTLBuffer> bh1 = buf_empty(device_, sizeof(float) * (size_t)T * 2 * ffn);
        id<MTLBuffer> bgated = buf_empty(device_, sizeof(float) * (size_t)T * ffn);
        id<MTLBuffer> bdown = buf_empty(device_, sizeof(float) * (size_t)T * H);

        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        if (!cb) {
            h3_dit_block_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                             head_dim, x, tokens, eps);
            return;
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

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
            [enc setComputePipelineState:p_bf16_];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            [enc setBytes:&e length:sizeof(e) atIndex:2];
            MTLSize tg = MTLSizeMake(64, 1, 1);
            MTLSize grid = MTLSizeMake((NSUInteger)(((n + 63) / 64) * 64), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        };
        launch_bf16(bqkv_bf, wqkv, (int)qkv_n);
        launch_bf16(bout_bf, wout, (int)out_n);
        launch_bf16(bfc1_bf, wfc1, (int)fc1_n);
        launch_bf16(bfc2_bf, wfc2, (int)fc2_n);

        NormArgs na{T, H, eps};
        enc1(p_rms_, (uint)T, @[ bx, bxn ], &na, sizeof(na));
        GemmArgs gq{T, H, 3 * I};
        enc2(p_gemm_f32_, (uint)(3 * I), (uint)T, @[ bxn, wqkv, bqkv ], &gq, sizeof(gq));
        AttnArgs aa{T, I, hd, heads};
        enc2(p_attn_, (uint)heads, (uint)T, @[ bqkv, bctx ], &aa, sizeof(aa));
        GemmArgs go{T, I, H};
        enc2(p_gemm_f32_, (uint)H, (uint)T, @[ bctx, wout, battn ], &go, sizeof(go));
        ElemArgs en{T * H};
        enc1(p_add_, (uint)(T * H), @[ bx, battn ], &en, sizeof(en));
        enc1(p_rms_, (uint)T, @[ bx, bxn ], &na, sizeof(na));
        GemmArgs g1{T, H, 2 * ffn};
        enc2(p_gemm_f32_, (uint)(2 * ffn), (uint)T, @[ bxn, wfc1, bh1 ], &g1, sizeof(g1));
        GemmArgs gs{T, 0, ffn};
        enc2(p_swiglu_, (uint)ffn, (uint)T, @[ bh1, bgated ], &gs, sizeof(gs));
        GemmArgs g2{T, ffn, H};
        enc2(p_gemm_f32_, (uint)H, (uint)T, @[ bgated, wfc2, bdown ], &g2, sizeof(g2));
        enc1(p_add_, (uint)(T * H), @[ bx, bdown ], &en, sizeof(en));

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            h3_dit_block_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                             head_dim, x, tokens, eps);
            return;
        }
        std::memcpy(x, [bx contents], sizeof(float) * (size_t)T * H);
        }
    }

private:
    bool dispatch2(id<MTLComputePipelineState> pso, uint gx, uint gy, id<MTLBuffer> b0,
                   id<MTLBuffer> b1, id<MTLBuffer> b2, const void *uni, size_t uni_n) {
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        if (!cb || !pso)
            return false;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:b0 offset:0 atIndex:0];
        [enc setBuffer:b1 offset:0 atIndex:1];
        [enc setBuffer:b2 offset:0 atIndex:2];
        [enc setBytes:uni length:uni_n atIndex:3];
        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(((gx + 15) / 16) * 16, ((gy + 15) / 16) * 16, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return cb.error == nil;
    }

    bool dispatch2(id<MTLComputePipelineState> pso, uint gx, uint gy, id<MTLBuffer> b0,
                   id<MTLBuffer> b1, id<MTLBuffer> b2, id<MTLBuffer> b3, const void *uni,
                   size_t uni_n) {
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        if (!cb || !pso)
            return false;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:b0 offset:0 atIndex:0];
        [enc setBuffer:b1 offset:0 atIndex:1];
        [enc setBuffer:b2 offset:0 atIndex:2];
        [enc setBuffer:b3 offset:0 atIndex:3];
        [enc setBytes:uni length:uni_n atIndex:4];
        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(((gx + 15) / 16) * 16, ((gy + 15) / 16) * 16, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return cb.error == nil;
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> p_gemm_f32_ = nil;
    id<MTLComputePipelineState> p_gemm_i4_ = nil;
    id<MTLComputePipelineState> p_gemm_mx_ = nil;
    id<MTLComputePipelineState> p_bf16_ = nil;
    id<MTLComputePipelineState> p_rms_ = nil;
    id<MTLComputePipelineState> p_attn_ = nil;
    id<MTLComputePipelineState> p_add_ = nil;
    id<MTLComputePipelineState> p_swiglu_ = nil;
};

} // namespace

Backend *make_metal_backend() {
    static std::once_flag once;
    static MetalBackend *inst = nullptr;
    std::call_once(once, []() {
        @autoreleasepool {
            auto *b = new MetalBackend();
            if (b->init())
                inst = b;
            else
                delete b;
        }
    });
    return inst;
}

} // namespace gpu
} // namespace mvllm

#endif // MVLLM_WITH_METAL
