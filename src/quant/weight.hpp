#pragma once

#include "quant.hpp"

#include <cstdint>
#include <vector>

namespace mvllm {
namespace quant {

// Resident dense matrix. fmt 0 = f32, 8 = int8 per-row, 4 = int4-g64.
// from_f32 follows colibri: bits<=4 and I%64==0 → int4, else bits<32 → int8.
struct QuantMat {
    int fmt = 0;
    int O = 0;
    int I = 0;
    std::vector<float> f;
    std::vector<uint8_t> q4;
    std::vector<int8_t> q8;
    std::vector<float> scales;

    bool empty() const { return O <= 0 || I <= 0; }
    void clear();
    void from_f32(const float *w, int rows, int cols, int bits);
    // y[S,O] = x[S,I] @ W[O,I]^T
    void gemm(float *y, const float *x, int S) const;
    // Same as gemm but only W[row0 : row0+rows, :]. y is [S, rows].
    void gemm_rows(float *y, const float *x, int S, int row0, int rows) const;
    int64_t bytes() const;
};

inline int sanitize_bits(int bits) {
    if (bits <= 4)
        return 4;
    if (bits <= 8)
        return 8;
    return 32;
}

} // namespace quant
} // namespace mvllm
