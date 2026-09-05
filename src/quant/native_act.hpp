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

// y = W x with dynamic E4M3 x (block 128) and E4M3 W * UE8M0 tile scales.
// Returns 0 or -1.
int fp8_matvec(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
               const float *x);

// Shared-qdq dual (official dual_matvec): one x QDQ, two matvecs.
int fp8_dual_matvec(float *ya, float *yb, const uint8_t *wa, const uint8_t *sa,
                    const uint8_t *wb, const uint8_t *sb, int O, int I, const float *x);

} // namespace mvllm
