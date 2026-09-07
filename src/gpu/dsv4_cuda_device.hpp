#pragma once

#include <cstdint>

namespace mvllm {
namespace dsv4_cuda {
namespace device {

bool probe(); // true if a CUDA device is usable
void shutdown(); // safe no-op if never probed
long long mem_free_mb(int device);

// All try_* return false to mean "use host". Never throw.
bool try_fp4_matvec(float *y, const float *x, const uint8_t *packed, const uint8_t *scales, int O,
                    int I);
bool try_fp8_matvec(float *y, const float *x, const uint8_t *w, const uint8_t *scales, int O,
                    int I);
bool try_f32_matvec(float *y, const float *x, const float *w, int O, int I);

} // namespace device
} // namespace dsv4_cuda
} // namespace mvllm
