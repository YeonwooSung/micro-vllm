#pragma once

#include <cstddef>

namespace mvllm {
namespace vk_ops {

// Host Vulkan surface. Always compiled: ops run on CPU until a device
// backend is linked. available() is true after init. False return = bad args.

bool init();
void shutdown();
bool available();
const char *backend_name(); // "cpu" until a Vulkan device path exists

bool rmsnorm(float *y, const float *x, const float *w, int nrows, int D, float eps);
bool add(float *y, const float *a, size_t n);
// y[S,O] = x[S,I] @ W[O,I]^T
bool gemm_f32(float *y, const float *x, const float *w, int S, int I, int O);

} // namespace vk_ops
} // namespace mvllm
