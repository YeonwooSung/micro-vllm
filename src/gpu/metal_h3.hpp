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

// y[S,O] = x[S,I] @ W[O,I]^T with int8-row weights and per-row scales.
bool gemm_int8(float *y, const float *x, const int8_t *w, const float *scale, int S, int I, int O);

// Neighborhood-style gated MLP used by H3 NAX: y += down(silu(up(x))).
// w_up is F32 [I, D], w_down is F32 [D, I].
bool nax_mlp(float *y, const float *x, const float *w_up, const float *w_down, int S, int D, int I);

// VAE residual: x += skip; y = rmsnorm(x, w). y may alias x when w is applied in place
// via a separate buffer; here y is nrm_out and x is updated.
bool vae_rms_add(float *x, const float *skip, const float *w, float *y, int n, float eps);

} // namespace metal_h3
} // namespace mvllm
