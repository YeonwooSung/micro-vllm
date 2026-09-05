#pragma once

#include <cstdint>

namespace mvllm {

// PolarQuant + rotated int4 for K3/GLM MLA latent KV rows (host, data-oblivious).
// n must be a power of two in [2, kKvTqMaxN].

constexpr uint32_t kKvTqSeed = 0x9E3779B9u;
constexpr int kKvTqMaxN = 2048;

// Normalized FWHT; applying twice is the identity.
void kv_tq_fwht(float *a, int n);

// PolarQuant (codec 0). Packed bytes hold n-1 angles; radius is the row L2.
int kv_tq_row_bytes(int n, int bits);
float kv_tq_quant_row(const float *src, uint8_t *dst, int n, int bits);
void kv_tq_dequant_row(const uint8_t *src, float radius, float *dst, int n, int bits);

// Rotated int4 (codec 1). Lloyd-Max 16-level codebook, 2 nibbles/byte.
int kv_q4_row_bytes(int n);
float kv_q4_quant_row(const float *src, uint8_t *dst, int n);
void kv_q4_dequant_row(const uint8_t *src, float radius, float *dst, int n);

} // namespace mvllm
