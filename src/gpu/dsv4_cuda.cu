#include "dsv4_cuda_device.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <vector>

namespace mvllm {
namespace dsv4_cuda {
namespace device {
namespace {

// Bit rules match src/quant/native_act.cpp (rewritten for device).
__device__ __forceinline__ float dec_e8m0(uint8_t value) {
    if (value == 0xff)
        return nanf("");
    return ldexpf(1.f, static_cast<int>(value) - 127);
}

__device__ __forceinline__ float dec_e2m1(uint8_t nibble) {
    const float codes[16] = {
        0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
        0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
    };
    return codes[nibble & 15];
}

__device__ __forceinline__ float dec_e4m3fn(uint8_t value) {
    const int sign = value >> 7;
    const int exp = (value >> 3) & 15;
    const int man = value & 7;
    if (exp == 15 && man == 7)
        return nanf("");
    const float mag = (exp == 0) ? ldexpf(static_cast<float>(man), -9)
                                 : ldexpf(1.f + static_cast<float>(man) / 8.f, exp - 7);
    return sign ? -mag : mag;
}

__global__ void k_fp4_matvec(float *y, const float *x, const uint8_t *packed, const uint8_t *scales,
                             int O, int I) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O)
        return;
    const int packed_i = (I + 1) / 2;
    const int sg = (I + 31) / 32;
    const uint8_t *prow = packed + static_cast<size_t>(o) * static_cast<size_t>(packed_i);
    const uint8_t *srow = scales + static_cast<size_t>(o) * static_cast<size_t>(sg);
    float acc = 0.f;
    for (int i = 0; i < I; ++i) {
        const uint8_t byte = prow[i >> 1];
        const uint8_t nib = (i & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 15);
        acc += x[i] * (dec_e2m1(nib) * dec_e8m0(srow[i >> 5]));
    }
    y[o] = acc;
}

__global__ void k_fp8_matvec(float *y, const float *x, const uint8_t *w, const uint8_t *scales,
                             int O, int I) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O)
        return;
    const int si = (I + 127) / 128;
    const uint8_t *wrow = w + static_cast<size_t>(o) * static_cast<size_t>(I);
    float acc = 0.f;
    for (int i = 0; i < I; ++i) {
        const size_t tile = static_cast<size_t>(o / 128) * static_cast<size_t>(si) +
                            static_cast<size_t>(i / 128);
        acc += x[i] * (dec_e4m3fn(wrow[i]) * dec_e8m0(scales[tile]));
    }
    y[o] = acc;
}

__global__ void k_f32_matvec(float *y, const float *x, const float *w, int O, int I) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O)
        return;
    const float *wrow = w + static_cast<size_t>(o) * static_cast<size_t>(I);
    float acc = 0.f;
    for (int i = 0; i < I; ++i)
        acc += x[i] * wrow[i];
    y[o] = acc;
}

struct DevPtr {
    void *p = nullptr;
    ~DevPtr() {
        if (p)
            cudaFree(p);
    }
    bool alloc(size_t bytes) {
        if (bytes == 0)
            return true;
        return cudaMalloc(&p, bytes) == cudaSuccess && p != nullptr;
    }
    template <class T>
    T *as() const {
        return static_cast<T *>(p);
    }
};

bool g_tried = false;
bool g_ok = false;
cudaStream_t g_stream = nullptr;

bool h2d(void *dst, const void *src, size_t bytes) {
    if (bytes == 0)
        return true;
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, g_stream) == cudaSuccess;
}

bool finish_y(float *y, const float *d_y, int O) {
    if (cudaGetLastError() != cudaSuccess)
        return false;
    std::vector<float> tmp(static_cast<size_t>(O));
    const size_t bytes = sizeof(float) * static_cast<size_t>(O);
    if (cudaMemcpyAsync(tmp.data(), d_y, bytes, cudaMemcpyDeviceToHost, g_stream) != cudaSuccess)
        return false;
    if (cudaStreamSynchronize(g_stream) != cudaSuccess)
        return false;
    std::memcpy(y, tmp.data(), bytes);
    return true;
}

bool ready() {
    if (!probe())
        return false;
    return cudaSetDevice(0) == cudaSuccess && g_stream != nullptr;
}

dim3 row_grid(int O) {
    const int threads = 256;
    return dim3(static_cast<unsigned>((O + threads - 1) / threads));
}

constexpr int kThreads = 256;

} // namespace

bool probe() {
    if (g_tried)
        return g_ok;
    g_tried = true;
    g_ok = false;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0)
        return false;
    if (cudaSetDevice(0) != cudaSuccess)
        return false;
    if (cudaStreamCreate(&g_stream) != cudaSuccess) {
        g_stream = nullptr;
        return false;
    }
    g_ok = true;
    return true;
}

void shutdown() {
    if (g_stream) {
        cudaStreamDestroy(g_stream);
        g_stream = nullptr;
    }
    g_ok = false;
    g_tried = false;
}

long long mem_free_mb(int device) {
    if (!probe())
        return 0;
    if (device < 0 || cudaSetDevice(device) != cudaSuccess)
        return 0;
    size_t free_b = 0;
    size_t total_b = 0;
    const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    cudaSetDevice(0);
    if (err != cudaSuccess)
        return 0;
    return static_cast<long long>(free_b / (1024ull * 1024ull));
}

bool try_fp4_matvec(float *y, const float *x, const uint8_t *packed, const uint8_t *scales, int O,
                    int I) {
    if (!y || !x || !packed || !scales || O <= 0 || I <= 0 || !ready())
        return false;
    const size_t packed_n = static_cast<size_t>(O) * static_cast<size_t>((I + 1) / 2);
    const size_t scale_n = static_cast<size_t>(O) * static_cast<size_t>((I + 31) / 32);
    DevPtr dx, dw, ds, dy;
    if (!dx.alloc(sizeof(float) * static_cast<size_t>(I)) || !dw.alloc(packed_n) ||
        !ds.alloc(scale_n) || !dy.alloc(sizeof(float) * static_cast<size_t>(O)))
        return false;
    if (!h2d(dx.p, x, sizeof(float) * static_cast<size_t>(I)) || !h2d(dw.p, packed, packed_n) ||
        !h2d(ds.p, scales, scale_n))
        return false;
    k_fp4_matvec<<<row_grid(O), kThreads, 0, g_stream>>>(dy.as<float>(), dx.as<const float>(),
                                                         dw.as<const uint8_t>(),
                                                         ds.as<const uint8_t>(), O, I);
    return finish_y(y, dy.as<const float>(), O);
}

bool try_fp8_matvec(float *y, const float *x, const uint8_t *w, const uint8_t *scales, int O,
                    int I) {
    if (!y || !x || !w || !scales || O <= 0 || I <= 0 || !ready())
        return false;
    const size_t w_n = static_cast<size_t>(O) * static_cast<size_t>(I);
    const size_t scale_n =
        static_cast<size_t>((O + 127) / 128) * static_cast<size_t>((I + 127) / 128);
    DevPtr dx, dw, ds, dy;
    if (!dx.alloc(sizeof(float) * static_cast<size_t>(I)) || !dw.alloc(w_n) || !ds.alloc(scale_n) ||
        !dy.alloc(sizeof(float) * static_cast<size_t>(O)))
        return false;
    if (!h2d(dx.p, x, sizeof(float) * static_cast<size_t>(I)) || !h2d(dw.p, w, w_n) ||
        !h2d(ds.p, scales, scale_n))
        return false;
    k_fp8_matvec<<<row_grid(O), kThreads, 0, g_stream>>>(
        dy.as<float>(), dx.as<const float>(), dw.as<const uint8_t>(), ds.as<const uint8_t>(), O, I);
    return finish_y(y, dy.as<const float>(), O);
}

bool try_f32_matvec(float *y, const float *x, const float *w, int O, int I) {
    if (!y || !x || !w || O <= 0 || I <= 0 || !ready())
        return false;
    const size_t w_n = sizeof(float) * static_cast<size_t>(O) * static_cast<size_t>(I);
    DevPtr dx, dw, dy;
    if (!dx.alloc(sizeof(float) * static_cast<size_t>(I)) || !dw.alloc(w_n) ||
        !dy.alloc(sizeof(float) * static_cast<size_t>(O)))
        return false;
    if (!h2d(dx.p, x, sizeof(float) * static_cast<size_t>(I)) || !h2d(dw.p, w, w_n))
        return false;
    k_f32_matvec<<<row_grid(O), kThreads, 0, g_stream>>>(dy.as<float>(), dx.as<const float>(),
                                                         dw.as<const float>(), O, I);
    return finish_y(y, dy.as<const float>(), O);
}

} // namespace device
} // namespace dsv4_cuda
} // namespace mvllm
