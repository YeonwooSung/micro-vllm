#include "vk_ops.hpp"

#include "../quant/quant.hpp"

namespace mvllm {
namespace vk_ops {
namespace {

bool g_inited = false;

} // namespace

bool init() {
    g_inited = true;
    return true;
}

void shutdown() { g_inited = false; }

bool available() { return g_inited; }

const char *backend_name() { return "cpu"; }

bool rmsnorm(float *y, const float *x, const float *w, int nrows, int D, float eps) {
    if (!y || !x || nrows < 1 || D < 1)
        return false;
    for (int r = 0; r < nrows; ++r)
        quant::rmsnorm(x + static_cast<size_t>(r) * D, w, y + static_cast<size_t>(r) * D, D, eps);
    return true;
}

bool add(float *y, const float *a, size_t n) {
    if (!y || !a)
        return false;
    for (size_t i = 0; i < n; ++i)
        y[i] += a[i];
    return true;
}

bool gemm_f32(float *y, const float *x, const float *w, int S, int I, int O) {
    if (!y || !x || !w || S < 1 || I < 1 || O < 1)
        return false;
    quant::matmul_f32(y, x, w, S, I, O);
    return true;
}

} // namespace vk_ops
} // namespace mvllm
