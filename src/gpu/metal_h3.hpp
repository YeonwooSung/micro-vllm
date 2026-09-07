#pragma once

#include <cstdint>

namespace mvllm {
namespace metal_h3 {

// H3 DiT residual with optional AdaLN / QK-norm / RoPE. CPU fallback if Metal
// is missing or the call fails. Same blob layout as h3_dit_block_cpu.

bool init();
void shutdown();
bool available();
const char *backend_name(); // "metal" or "cpu"

// adaln_mod is [6, hidden] (scale0, shift0, scale1, shift1, scale2, shift2).
// q_norm/k_norm are [head_dim] or null. rope_cos/sin are [tokens, 48] or null
// (applied when head_dim >= 96, same as the CPU path).
bool dit_residual(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                  int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                  int tokens, float eps, const float *adaln_mod, const float *q_norm,
                  const float *k_norm, const float *rope_cos, const float *rope_sin);

} // namespace metal_h3
} // namespace mvllm
