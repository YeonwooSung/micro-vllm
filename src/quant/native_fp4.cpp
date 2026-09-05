#include "native_fp4.hpp"

#include "native_act.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mvllm {

int fp4_matvec_rows16(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
                      const float *x) {
    if (!y || !w || !scales || !x || O <= 0 || I <= 0 || (I % 32) != 0 || (O % 16) != 0)
        return -1;

    const int packed_stride = I / 2;
    const int scale_stride = I / 32;

    // Tile 16 rows; fold (x*w)*scale into each row acc column by column.
    for (int o0 = 0; o0 < O; o0 += 16) {
        float acc[16] = {};
        for (int i = 0; i < I; ++i) {
            const int w_col = i / 2;
            const int sc_col = i / 32;
            const float xv = x[i];
            for (int r = 0; r < 16; ++r) {
                const int o = o0 + r;
                const uint8_t byte =
                    w[static_cast<size_t>(o) * static_cast<size_t>(packed_stride) +
                      static_cast<size_t>(w_col)];
                const uint8_t nib = static_cast<uint8_t>((i & 1) ? (byte >> 4) : (byte & 0xF));
                float sc = e8m0_decode(
                    scales[static_cast<size_t>(o) * static_cast<size_t>(scale_stride) +
                           static_cast<size_t>(sc_col)]);
                if (!std::isfinite(sc))
                    sc = 0.f;
                acc[r] += (xv * e2m1_decode(nib)) * sc;
            }
        }
        for (int r = 0; r < 16; ++r)
            y[o0 + r] = acc[r];
    }
    return 0;
}

} // namespace mvllm
