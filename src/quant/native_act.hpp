#pragma once

#include <cstdint>

namespace mvllm {

// Native activation QDQ: UE8M0 block scales with E4M3fn (FP8) or E2M1 (FP4).
// E4M3fn NaN encodes as 0x7f.

float e8m0_decode(uint8_t value);
const float *e8m0_table();
float e2m1_decode(uint8_t nibble);
float e4m3fn_decode(uint8_t value);
uint8_t e4m3fn_encode(float value);

// Per-block dynamic QDQ. Last block may be short. Returns -1 on bad args.
int fp8_activation_qdq(float *output, uint8_t *scales, const float *input,
                       int length, int block_size);
int fp4_activation_qdq(float *output, uint8_t *scales, const float *input,
                       int length, int block_size);

} // namespace mvllm
