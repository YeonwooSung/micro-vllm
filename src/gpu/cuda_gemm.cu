#if defined(MVLLM_WITH_CUDA_GEMM)

#include "backend.hpp"

#include "../model/family.hpp"
#include "../quant/quant.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <mutex>
#include <vector>

namespace mvllm {
namespace gpu {
namespace {

__device__ __forceinline__ float mx4_lut_dev(unsigned nib) {
    const float lut[16] = {0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
                           -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return lut[nib & 15u];
}

__global__ void k_gemm_f32(const float *x, const float *w, float *y, int S, int I, int O) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= O || s >= S)
        return;
    const float *xs = x + (size_t)s * I;
    const float *wo = w + (size_t)o * I;
    float acc = 0.f;
    for (int i = 0; i < I; ++i)
        acc += xs[i] * wo[i];
    y[(size_t)s * O + o] = acc;
}

__global__ void k_gemm_int4(const float *x, const uint8_t *packed, const float *scales, float *y,
                            int S, int I, int O) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= O || s >= S)
        return;
    const int stride = (I + 1) / 2;
    const int ng = (I + 63) / 64;
    const float *xs = x + (size_t)s * I;
    const uint8_t *prow = packed + (size_t)o * stride;
    const float *srow = scales + (size_t)o * ng;
    float acc = 0.f;
    for (int i = 0; i < I; ++i) {
        uint8_t b = prow[i >> 1];
        unsigned nib = (i & 1) ? (b >> 4) : (b & 0x0f);
        int q = (int)nib - 8;
        acc += xs[i] * ((float)q * srow[i / 64]);
    }
    y[(size_t)s * O + o] = acc;
}

__global__ void k_gemm_mxfp4(const float *x, const uint8_t *packed, const uint8_t *scales, float *y,
                             int S, int I, int O) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= O || s >= S)
        return;
    const int stride = (I + 1) / 2;
    const int ng = (I + 31) / 32;
    const float *xs = x + (size_t)s * I;
    const uint8_t *prow = packed + (size_t)o * stride;
    const uint8_t *srow = scales + (size_t)o * ng;
    float acc = 0.f;
    for (int i = 0; i < I; ++i) {
        uint8_t b = prow[i >> 1];
        unsigned nib = (i & 1) ? (b >> 4) : (b & 0x0f);
        unsigned e = srow[i / 32];
        float scale = __uint_as_float(e << 23);
        acc += xs[i] * (mx4_lut_dev(nib) * scale);
    }
    y[(size_t)s * O + o] = acc;
}

__global__ void k_bf16_to_f32(const uint16_t *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    dst[i] = __uint_as_float((unsigned)src[i] << 16);
}

__global__ void k_rmsnorm_ones(const float *x, float *y, int T, int H, float eps) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T)
        return;
    const float *xs = x + (size_t)t * H;
    float *ys = y + (size_t)t * H;
    float ss = 0.f;
    for (int i = 0; i < H; ++i)
        ss += xs[i] * xs[i];
    float inv = rsqrtf(ss / (float)H + eps);
    for (int i = 0; i < H; ++i)
        ys[i] = xs[i] * inv;
}

__global__ void k_dit_attn(const float *qkv, float *ctx, int T, int I, int hd, int heads) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int qi = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= heads || qi >= T)
        return;
    const float scale = rsqrtf((float)hd);
    float scores[256];
    if (T > 256)
        return;
    const float *q = qkv + (size_t)qi * (3 * I) + h * hd;
    float m = -1e30f;
    for (int ki = 0; ki < T; ++ki) {
        const float *k = qkv + (size_t)ki * (3 * I) + I + h * hd;
        float acc = 0.f;
        for (int d = 0; d < hd; ++d)
            acc += q[d] * k[d];
        scores[ki] = acc * scale;
        if (scores[ki] > m)
            m = scores[ki];
    }
    float sum = 0.f;
    for (int ki = 0; ki < T; ++ki) {
        scores[ki] = expf(scores[ki] - m);
        sum += scores[ki];
    }
    float inv = (sum == 0.f) ? 0.f : 1.f / sum;
    float *o = ctx + (size_t)qi * I + h * hd;
    for (int d = 0; d < hd; ++d)
        o[d] = 0.f;
    for (int vi = 0; vi < T; ++vi) {
        const float *v = qkv + (size_t)vi * (3 * I) + 2 * I + h * hd;
        float w = scores[vi] * inv;
        for (int d = 0; d < hd; ++d)
            o[d] += w * v[d];
    }
}

__global__ void k_add(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += x[i];
}

__global__ void k_swiglu_pack(const float *h1, float *gated, int T, int ffn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= T || i >= ffn)
        return;
    float g = h1[(size_t)t * (2 * ffn) + i];
    float u = h1[(size_t)t * (2 * ffn) + ffn + i];
    float sig = 1.f / (1.f + expf(-g));
    gated[(size_t)t * ffn + i] = g * sig * u;
}

bool ck(cudaError_t e) { return e == cudaSuccess; }

template <typename T>
T *dalloc(size_t n) {
    T *p = nullptr;
    if (n == 0)
        n = 1;
    if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess)
        return nullptr;
    return p;
}

void dfree(void *p) {
    if (p)
        cudaFree(p);
}

dim3 grid2(int gx, int gy) {
    return dim3((gx + 15) / 16, (gy + 15) / 16, 1);
}

class CudaBackend final : public Backend {
public:
    bool init() {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0)
            return false;
        return cudaSetDevice(0) == cudaSuccess;
    }

    Device device() const override { return Device::Cuda; }
    const char *name() const override { return "cuda"; }

    void gemm_f32(float *y, const float *x, const float *w, int S, int I, int O) override {
        if (!y || !x || !w || S <= 0 || I <= 0 || O <= 0)
            return;
        float *dx = dalloc<float>((size_t)S * I);
        float *dw = dalloc<float>((size_t)O * I);
        float *dy = dalloc<float>((size_t)S * O);
        if (!dx || !dw || !dy) {
            dfree(dx);
            dfree(dw);
            dfree(dy);
            quant::matmul_f32(y, x, w, S, I, O);
            return;
        }
        bool ok = ck(cudaMemcpy(dx, x, sizeof(float) * (size_t)S * I, cudaMemcpyHostToDevice)) &&
                  ck(cudaMemcpy(dw, w, sizeof(float) * (size_t)O * I, cudaMemcpyHostToDevice));
        if (ok) {
            k_gemm_f32<<<grid2(O, S), dim3(16, 16)>>>(dx, dw, dy, S, I, O);
            ok = ck(cudaDeviceSynchronize()) &&
                 ck(cudaMemcpy(y, dy, sizeof(float) * (size_t)S * O, cudaMemcpyDeviceToHost));
        }
        dfree(dx);
        dfree(dw);
        dfree(dy);
        if (!ok)
            quant::matmul_f32(y, x, w, S, I, O);
    }

    void gemm_int4_g64(float *y, const float *x, const uint8_t *packed, const float *scales, int S,
                       int I, int O) override {
        if (!y || !x || !packed || !scales || S <= 0 || I <= 0 || O <= 0)
            return;
        const int stride = (I + 1) / 2;
        const int ng = (I + 63) / 64;
        float *dx = dalloc<float>((size_t)S * I);
        uint8_t *dp = dalloc<uint8_t>((size_t)O * stride);
        float *ds = dalloc<float>((size_t)O * ng);
        float *dy = dalloc<float>((size_t)S * O);
        if (!dx || !dp || !ds || !dy) {
            dfree(dx);
            dfree(dp);
            dfree(ds);
            dfree(dy);
            quant::matmul_int4_g64(y, x, packed, scales, S, I, O);
            return;
        }
        bool ok = ck(cudaMemcpy(dx, x, sizeof(float) * (size_t)S * I, cudaMemcpyHostToDevice)) &&
                  ck(cudaMemcpy(dp, packed, (size_t)O * stride, cudaMemcpyHostToDevice)) &&
                  ck(cudaMemcpy(ds, scales, sizeof(float) * (size_t)O * ng, cudaMemcpyHostToDevice));
        if (ok) {
            k_gemm_int4<<<grid2(O, S), dim3(16, 16)>>>(dx, dp, ds, dy, S, I, O);
            ok = ck(cudaDeviceSynchronize()) &&
                 ck(cudaMemcpy(y, dy, sizeof(float) * (size_t)S * O, cudaMemcpyDeviceToHost));
        }
        dfree(dx);
        dfree(dp);
        dfree(ds);
        dfree(dy);
        if (!ok)
            quant::matmul_int4_g64(y, x, packed, scales, S, I, O);
    }

    void gemm_mxfp4(float *y, const float *x, const uint8_t *packed, const uint8_t *scales, int S,
                    int I, int O, bool) override {
        if (!y || !x || !packed || !scales || S <= 0 || I <= 0 || O <= 0)
            return;
        const int stride = (I + 1) / 2;
        const int ng = (I + 31) / 32;
        float *dx = dalloc<float>((size_t)S * I);
        uint8_t *dp = dalloc<uint8_t>((size_t)O * stride);
        uint8_t *ds = dalloc<uint8_t>((size_t)O * ng);
        float *dy = dalloc<float>((size_t)S * O);
        if (!dx || !dp || !ds || !dy) {
            dfree(dx);
            dfree(dp);
            dfree(ds);
            dfree(dy);
            quant::matmul_mxfp4(y, x, packed, scales, S, I, O);
            return;
        }
        bool ok = ck(cudaMemcpy(dx, x, sizeof(float) * (size_t)S * I, cudaMemcpyHostToDevice)) &&
                  ck(cudaMemcpy(dp, packed, (size_t)O * stride, cudaMemcpyHostToDevice)) &&
                  ck(cudaMemcpy(ds, scales, (size_t)O * ng, cudaMemcpyHostToDevice));
        if (ok) {
            k_gemm_mxfp4<<<grid2(O, S), dim3(16, 16)>>>(dx, dp, ds, dy, S, I, O);
            ok = ck(cudaDeviceSynchronize()) &&
                 ck(cudaMemcpy(y, dy, sizeof(float) * (size_t)S * O, cudaMemcpyDeviceToHost));
        }
        dfree(dx);
        dfree(dp);
        dfree(ds);
        dfree(dy);
        if (!ok)
            quant::matmul_mxfp4(y, x, packed, scales, S, I, O);
    }

    void dit_block(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                   int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                   int tokens, float eps) override {
        if (!blob || !x || hidden <= 0 || inner <= 0 || ffn <= 0 || tokens <= 0 || tokens > 256) {
            h3_dit_block_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                             head_dim, x, tokens, eps);
            return;
        }
        const int T = tokens, H = hidden, I = inner;
        int hd = head_dim > 0 ? head_dim : I;
        int heads = I / hd;
        if (heads < 1) {
            heads = 1;
            hd = I;
        }
        const int qkv_n = (int)(qkv_bytes / 2);
        const int out_n = (int)(out_bytes / 2);
        const int fc1_n = (int)(fc1_bytes / 2);
        const int fc2_n = (int)(fc2_bytes / 2);
        const size_t blob_n = (size_t)(qkv_bytes + out_bytes + fc1_bytes + fc2_bytes);
        uint8_t *dblob = dalloc<uint8_t>(blob_n);
        float *dx = dalloc<float>((size_t)T * H);
        float *dxn = dalloc<float>((size_t)T * H);
        float *wqkv = dalloc<float>((size_t)qkv_n);
        float *wout = dalloc<float>((size_t)out_n);
        float *wfc1 = dalloc<float>((size_t)fc1_n);
        float *wfc2 = dalloc<float>((size_t)fc2_n);
        float *dqkv = dalloc<float>((size_t)T * 3 * I);
        float *dctx = dalloc<float>((size_t)T * I);
        float *dattn = dalloc<float>((size_t)T * H);
        float *dh1 = dalloc<float>((size_t)T * 2 * ffn);
        float *dgated = dalloc<float>((size_t)T * ffn);
        float *ddown = dalloc<float>((size_t)T * H);
        bool ok = dblob && dx && dxn && wqkv && wout && wfc1 && wfc2 && dqkv && dctx && dattn &&
                  dh1 && dgated && ddown;
        if (ok)
            ok = ck(cudaMemcpy(dblob, blob, blob_n, cudaMemcpyHostToDevice)) &&
                 ck(cudaMemcpy(dx, x, sizeof(float) * (size_t)T * H, cudaMemcpyHostToDevice));
        auto launch1 = [](int n) {
            return dim3((n + 63) / 64);
        };
        if (ok) {
            k_bf16_to_f32<<<launch1(qkv_n), 64>>>((const uint16_t *)dblob, wqkv, qkv_n);
            k_bf16_to_f32<<<launch1(out_n), 64>>>(
                (const uint16_t *)(dblob + qkv_bytes), wout, out_n);
            k_bf16_to_f32<<<launch1(fc1_n), 64>>>(
                (const uint16_t *)(dblob + qkv_bytes + out_bytes), wfc1, fc1_n);
            k_bf16_to_f32<<<launch1(fc2_n), 64>>>(
                (const uint16_t *)(dblob + qkv_bytes + out_bytes + fc1_bytes), wfc2, fc2_n);
            k_rmsnorm_ones<<<launch1(T), 64>>>(dx, dxn, T, H, eps);
            k_gemm_f32<<<grid2(3 * I, T), dim3(16, 16)>>>(dxn, wqkv, dqkv, T, H, 3 * I);
            k_dit_attn<<<grid2(heads, T), dim3(16, 16)>>>(dqkv, dctx, T, I, hd, heads);
            k_gemm_f32<<<grid2(H, T), dim3(16, 16)>>>(dctx, wout, dattn, T, I, H);
            k_add<<<launch1(T * H), 64>>>(dx, dattn, T * H);
            k_rmsnorm_ones<<<launch1(T), 64>>>(dx, dxn, T, H, eps);
            k_gemm_f32<<<grid2(2 * ffn, T), dim3(16, 16)>>>(dxn, wfc1, dh1, T, H, 2 * ffn);
            k_swiglu_pack<<<grid2(ffn, T), dim3(16, 16)>>>(dh1, dgated, T, ffn);
            k_gemm_f32<<<grid2(H, T), dim3(16, 16)>>>(dgated, wfc2, ddown, T, ffn, H);
            k_add<<<launch1(T * H), 64>>>(dx, ddown, T * H);
            ok = ck(cudaDeviceSynchronize()) &&
                 ck(cudaMemcpy(x, dx, sizeof(float) * (size_t)T * H, cudaMemcpyDeviceToHost));
        }
        dfree(dblob);
        dfree(dx);
        dfree(dxn);
        dfree(wqkv);
        dfree(wout);
        dfree(wfc1);
        dfree(wfc2);
        dfree(dqkv);
        dfree(dctx);
        dfree(dattn);
        dfree(dh1);
        dfree(dgated);
        dfree(ddown);
        if (!ok)
            h3_dit_block_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                             head_dim, x, tokens, eps);
    }
};

} // namespace

Backend *make_cuda_backend() {
    static std::once_flag once;
    static CudaBackend *inst = nullptr;
    std::call_once(once, []() {
        auto *b = new CudaBackend();
        if (b->init())
            inst = b;
        else
            delete b;
    });
    return inst;
}

} // namespace gpu
} // namespace mvllm

#endif // MVLLM_WITH_CUDA_GEMM
