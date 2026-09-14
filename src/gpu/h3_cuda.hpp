#pragma once

#include <cstddef>
#include <cstdint>

namespace mvllm {
namespace h3_cuda {

// H3 CUDA (or CPU fallback) DiT residual. available() is true after a
// successful init (CPU always; device kernels when MVLLM_GPU_CUDA and a
// CUDA device exist). dit_residual uses the same blob layout as
// h3_dit_block_cpu. Device path includes AdaLN / QK-norm / RoPE and has
// no tokens<=256 cap.

bool init();
void shutdown();
bool available();
const char *backend_name(); // "cuda" or "cpu"
size_t workspace_bytes();   // 0 if nothing allocated

// AdaLN / QK-norm / RoPE DiT residual. CPU fallback if CUDA is missing or
// the device call fails. adaln_mod is [6, hidden] (scale0, shift0, scale1,
// shift1, scale2, shift2). q_norm/k_norm are [head_dim] or null.
// rope_cos/sin are [tokens, 48] or null (applied when head_dim >= 96).
bool dit_residual(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                  int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                  int tokens, float eps, const float *adaln_mod, const float *q_norm,
                  const float *k_norm, const float *rope_cos, const float *rope_sin);

} // namespace h3_cuda
} // namespace mvllm
