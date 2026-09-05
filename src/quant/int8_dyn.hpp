#pragma once

#include <cstdint>

namespace mvllm {

// Symmetric int8: one F32 scale per row (activations) or output channel (weights).
// scale = amax(|row|) / 127; a zero row keeps scale 0 and writes all-zero q.

void int8_quant_row(const float *x, int n, int8_t *q, float *scale);
void int8_quant_rows(const float *x, int rows, int n, int8_t *q, float *scales);

// y[S,O] = x[S,I] @ W[O,I]^T via int8. w_q/w_sc come from int8_quant_rows on W.
// Quantizes X each call. bias may be null; added after dequant.
void int8_dyn_gemm(float *y, const float *x, const int8_t *w_q, const float *w_sc,
                   const float *bias, int S, int I, int O);

} // namespace mvllm
