#pragma once

#include <cstdint>

namespace mvllm {

// Host FP8 e4m3 (OCP fn) for K3/GLM MLA latent KV rows.
// No infinities; finite max ±448; NaN code S.1111.111 is never written.

void kv_fp8_lut_init();
float kv_fp8_lut(uint8_t b);
uint8_t kv_fp8_enc(float f);

// Per-row amax/448 scale. Returns the scale used by dequant (lut[b] * scale).
float kv_fp8_quant_row(const float *src, uint8_t *dst, int n);
void kv_fp8_dequant_row(const uint8_t *src, float scale, float *dst, int n);

// gs <= 0 → one scale for the whole row; else one scale per group of gs.
int kv_fp8_nscale(int n, int gs);
void kv_fp8_quant_row_gs(const float *src, uint8_t *dst, float *scales, int n, int gs);
void kv_fp8_dequant_row_gs(const uint8_t *src, const float *scales, float *dst, int n, int gs);

} // namespace mvllm
