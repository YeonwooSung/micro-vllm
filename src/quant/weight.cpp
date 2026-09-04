#include "weight.hpp"

#include <cstring>

namespace mvllm {
namespace quant {

void QuantMat::clear() {
    fmt = 0;
    O = 0;
    I = 0;
    f.clear();
    q4.clear();
    q8.clear();
    scales.clear();
}

void QuantMat::from_f32(const float *w, int rows, int cols, int bits) {
    clear();
    if (!w || rows <= 0 || cols <= 0)
        return;
    O = rows;
    I = cols;
    bits = sanitize_bits(bits);
    if (bits >= 32) {
        fmt = 0;
        f.assign(w, w + static_cast<size_t>(O) * I);
        return;
    }
    if (bits <= 4 && I % 64 == 0) {
        fmt = 4;
        q4.assign(static_cast<size_t>(O) * (I / 2), 0);
        scales.assign(static_cast<size_t>(O) * (I / 64), 0.f);
        quantize_int4_g64(w, O, I, q4.data(), scales.data());
        return;
    }
    fmt = 8;
    q8.assign(static_cast<size_t>(O) * I, 0);
    scales.assign(static_cast<size_t>(O), 0.f);
    quantize_int8_row(w, O, I, q8.data(), scales.data());
}

void QuantMat::gemm(float *y, const float *x, int S) const {
    gemm_rows(y, x, S, 0, O);
}

void QuantMat::gemm_rows(float *y, const float *x, int S, int row0, int rows) const {
    if (!y || !x || empty() || S <= 0 || rows <= 0)
        return;
    if (row0 < 0 || row0 + rows > O)
        return;
    if (fmt == 4 && !q4.empty() && I % 64 == 0) {
        matmul_int4_g64(y, x, q4.data() + static_cast<size_t>(row0) * (I / 2),
                        scales.data() + static_cast<size_t>(row0) * (I / 64), S, I, rows);
        return;
    }
    if (fmt == 8 && !q8.empty()) {
        matmul_int8_row(y, x, q8.data() + static_cast<size_t>(row0) * I,
                        scales.data() + static_cast<size_t>(row0), S, I, rows);
        return;
    }
    if (!f.empty())
        matmul_f32(y, x, f.data() + static_cast<size_t>(row0) * I, S, I, rows);
}

int64_t QuantMat::bytes() const {
    int64_t n = 0;
    n += static_cast<int64_t>(f.size() * sizeof(float));
    n += static_cast<int64_t>(q4.size());
    n += static_cast<int64_t>(q8.size());
    n += static_cast<int64_t>(scales.size() * sizeof(float));
    return n;
}

} // namespace quant
} // namespace mvllm
