#pragma once

#include <cstdint>

namespace mvllm {

// x is already qdq'd. w packed [O, I/2] low-nibble-even. scales [O, I/32] UE8M0.
// Returns 0 or -1.
int fp4_matvec_rows16(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
                      const float *x);

} // namespace mvllm
