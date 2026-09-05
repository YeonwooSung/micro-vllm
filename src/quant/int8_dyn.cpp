#include "int8_dyn.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mvllm {
namespace {

inline int clampi(int v, int lo, int hi) {
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}

} // namespace

void int8_quant_row(const float *x, int n, int8_t *q, float *scale) {
    if (!x || !q || !scale || n < 1)
        return;

    float amax = 0.f;
    for (int i = 0; i < n; ++i) {
        float a = std::fabs(x[i]);
        if (a > amax)
            amax = a;
    }

    const float sc = amax / 127.f;
    *scale = sc;
    if (sc == 0.f) {
        for (int i = 0; i < n; ++i)
            q[i] = 0;
        return;
    }

    const float inv = 1.f / sc;
    for (int i = 0; i < n; ++i) {
        const int qi = static_cast<int>(std::nearbyintf(x[i] * inv));
        q[i] = static_cast<int8_t>(clampi(qi, -127, 127));
    }
}

void int8_quant_rows(const float *x, int rows, int n, int8_t *q, float *scales) {
    if (!x || !q || !scales || rows < 1 || n < 1)
        return;
    for (int r = 0; r < rows; ++r) {
        const size_t off = static_cast<size_t>(r) * static_cast<size_t>(n);
        int8_quant_row(x + off, n, q + off, scales + r);
    }
}

void int8_dyn_gemm(float *y, const float *x, const int8_t *w_q, const float *w_sc,
                   const float *bias, int S, int I, int O) {
    if (!y || !x || !w_q || !w_sc || S < 1 || I < 1 || O < 1)
        return;

    std::vector<int8_t> xq(static_cast<size_t>(S) * static_cast<size_t>(I));
    std::vector<float> xsc(static_cast<size_t>(S));
    int8_quant_rows(x, S, I, xq.data(), xsc.data());

    for (int s = 0; s < S; ++s) {
        const int8_t *xs = xq.data() + static_cast<size_t>(s) * static_cast<size_t>(I);
        float *ys = y + static_cast<size_t>(s) * static_cast<size_t>(O);
        const float xs_sc = xsc[static_cast<size_t>(s)];
        for (int o = 0; o < O; ++o) {
            const int8_t *wo = w_q + static_cast<size_t>(o) * static_cast<size_t>(I);
            int32_t acc = 0;
            for (int i = 0; i < I; ++i)
                acc += static_cast<int32_t>(xs[i]) * static_cast<int32_t>(wo[i]);
            float v = (w_sc[o] * xs_sc) * static_cast<float>(acc);
            if (bias)
                v += bias[o];
            ys[o] = v;
        }
    }
}

} // namespace mvllm
