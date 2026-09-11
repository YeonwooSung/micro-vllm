#include "coli_cuda.hpp"

#include <cuda_runtime.h>

// y[S,O] = x[S,I] @ W[O,I]^T. One thread writes one output element.

__global__ void gemm_f32_kernel(float *y, const float *x, const float *w, int S, int I, int O) {
    const int o = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int s = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (s >= S || o >= O)
        return;

    const float *xrow = x + static_cast<size_t>(s) * static_cast<size_t>(I);
    const float *wrow = w + static_cast<size_t>(o) * static_cast<size_t>(I);
    float acc = 0.f;
    for (int i = 0; i < I; ++i)
        acc += xrow[i] * wrow[i];
    y[static_cast<size_t>(s) * static_cast<size_t>(O) + static_cast<size_t>(o)] = acc;
}

// Device-pointer launch hook for coli_cuda.cpp. 0 = ok, nonzero = reject / launch fail.
extern "C" int coli_cuda_gemm_f32_dev(float *y, const float *x, const float *w, int S, int I,
                                      int O) {
    if (!y || !x || !w || S <= 0 || I <= 0 || O <= 0)
        return 1;

    constexpr unsigned kTx = 16u;
    constexpr unsigned kTy = 16u;
    const dim3 block(kTx, kTy);
    const dim3 grid(static_cast<unsigned>((O + static_cast<int>(kTx) - 1) / static_cast<int>(kTx)),
                    static_cast<unsigned>((S + static_cast<int>(kTy) - 1) / static_cast<int>(kTy)));
    gemm_f32_kernel<<<grid, block>>>(y, x, w, S, I, O);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
