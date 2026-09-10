#include "metal_h3.hpp"

#if !defined(MVLLM_WITH_METAL)

#include "../model/family.hpp"
#include "../quant/quant.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace mvllm {
namespace metal_h3 {

bool init() { return true; }

void shutdown() {}

bool available() { return true; }

const char *backend_name() { return "cpu"; }

bool dit_residual(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                  int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                  int tokens, float eps, const float *adaln_mod, const float *q_norm,
                  const float *k_norm, const float *rope_cos, const float *rope_sin) {
    h3_dit_block_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn, head_dim,
                     x, tokens, eps, adaln_mod, q_norm, k_norm, rope_cos, rope_sin);
    return true;
}

bool gemm_int8(float *y, const float *x, const int8_t *w, const float *scale, int S, int I, int O) {
    if (!y || !x || !w || !scale || S < 1 || I < 1 || O < 1)
        return false;
    for (int s = 0; s < S; ++s) {
        const float *xs = x + static_cast<size_t>(s) * I;
        float *ys = y + static_cast<size_t>(s) * O;
        for (int o = 0; o < O; ++o) {
            const int8_t *wo = w + static_cast<size_t>(o) * I;
            float acc = 0.f;
            for (int i = 0; i < I; ++i)
                acc += xs[i] * static_cast<float>(wo[i]);
            ys[o] = acc * scale[o];
        }
    }
    return true;
}

bool nax_mlp(float *y, const float *x, const float *w_up, const float *w_down, int S, int D, int I) {
    if (!y || !x || !w_up || !w_down || S < 1 || D < 1 || I < 1)
        return false;
    std::vector<float> mid(static_cast<size_t>(S) * static_cast<size_t>(I), 0.f);
    quant::matmul_f32(mid.data(), x, w_up, S, D, I);
    for (int i = 0; i < S * I; ++i)
        mid[static_cast<size_t>(i)] =
            mid[static_cast<size_t>(i)] * quant::sigmoid(mid[static_cast<size_t>(i)]);
    std::vector<float> down(static_cast<size_t>(S) * static_cast<size_t>(D), 0.f);
    quant::matmul_f32(down.data(), mid.data(), w_down, S, I, D);
    for (int i = 0; i < S * D; ++i)
        y[i] += down[static_cast<size_t>(i)];
    return true;
}

bool vae_rms_add(float *x, const float *skip, const float *w, float *y, int n, float eps) {
    if (!x || !y || n < 1)
        return false;
    if (skip) {
        for (int i = 0; i < n; ++i)
            x[i] += skip[i];
    }
    quant::rmsnorm(x, w, y, n, eps);
    return true;
}

} // namespace metal_h3
} // namespace mvllm

#endif // !MVLLM_WITH_METAL
