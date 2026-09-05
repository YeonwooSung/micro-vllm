#include "native_batch.hpp"

#include "native_act.hpp"

#include <cstddef>
#include <cstdint>

namespace mvllm {
namespace {

constexpr int kTile = 128;

bool bad_batch(int O, int I, int batch) {
    return batch < 1 || O <= 0 || I <= 0 || (I % kTile) != 0;
}

} // namespace

int fp8_matmul_batch(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
                     const float *x, int batch) {
    if (!y || !w || !scales || !x || bad_batch(O, I, batch))
        return -1;
    for (int b = 0; b < batch; ++b) {
        const int rc =
            fp8_matvec(y + static_cast<size_t>(b) * static_cast<size_t>(O), w, scales, O, I,
                       x + static_cast<size_t>(b) * static_cast<size_t>(I));
        if (rc != 0)
            return rc;
    }
    return 0;
}

int fp8_matmul_batch_pre(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
                         const float *xhat, int batch) {
    if (!y || !w || !scales || !xhat || bad_batch(O, I, batch))
        return -1;
    for (int b = 0; b < batch; ++b) {
        const int rc = fp8_matvec_pre(y + static_cast<size_t>(b) * static_cast<size_t>(O),
                                      w, scales, O, I,
                                      xhat + static_cast<size_t>(b) * static_cast<size_t>(I));
        if (rc != 0)
            return rc;
    }
    return 0;
}

int fp4_matmul_batch(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
                     const float *x, int batch) {
    if (!y || !w || !scales || !x || bad_batch(O, I, batch))
        return -1;
    for (int b = 0; b < batch; ++b) {
        const int rc =
            fp4_matvec(y + static_cast<size_t>(b) * static_cast<size_t>(O), w, scales, O, I,
                       x + static_cast<size_t>(b) * static_cast<size_t>(I));
        if (rc != 0)
            return rc;
    }
    return 0;
}

int fp4_matmul_batch_pre(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
                         const float *xhat, int batch) {
    if (!y || !w || !scales || !xhat || bad_batch(O, I, batch))
        return -1;
    for (int b = 0; b < batch; ++b) {
        const int rc = fp4_matvec_pre(y + static_cast<size_t>(b) * static_cast<size_t>(O),
                                      w, scales, O, I,
                                      xhat + static_cast<size_t>(b) * static_cast<size_t>(I));
        if (rc != 0)
            return rc;
    }
    return 0;
}

} // namespace mvllm
