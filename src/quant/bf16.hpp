#pragma once

#include <cstdint>

namespace mvllm {

// BF16 is the high 16 bits of IEEE-754 binary32.
// Finite values round nearest-even on the discarded half; Inf/NaN are truncated.

float bf16_round(float value);
uint16_t bf16_encode(float value);
float bf16_decode(uint16_t value);
void bf16_round_array(float *values, int count);

// In-place unnormalized FWHT, then v[i] = bf16_round(v[i] / sqrt(n)).
// n must be a power of two and >= 1. Returns -1 if values is null or n is invalid.
int hadamard_bf16(float *values, int n);

} // namespace mvllm
