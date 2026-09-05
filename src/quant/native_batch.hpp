#pragma once

#include <cstdint>

namespace mvllm {

// Batch-major y[b*O+o] = W x[b*I+i]. Per-row QDQ; same tiles as fp8_matvec.
int fp8_matmul_batch(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
                     const float *x, int batch);
// xhat already QDQ'd. No activation QDQ.
int fp8_matmul_batch_pre(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
                         const float *xhat, int batch);

// Packed E2M1 W, UE8M0 per 32 cols. Per-row QDQ; same tiles as fp4_matvec.
int fp4_matmul_batch(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
                     const float *x, int batch);
int fp4_matmul_batch_pre(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
                         const float *xhat, int batch);

} // namespace mvllm
