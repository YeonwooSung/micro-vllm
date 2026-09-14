#include "h3_cuda.hpp"

#include "../model/family.hpp"
#include "../quant/quant.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(MVLLM_WITH_CUDA_GEMM)
extern "C" int h3_cuda_dit_residual_dev(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes,
                                        int64_t fc1_bytes, int64_t fc2_bytes, int hidden, int inner,
                                        int ffn, int head_dim, float *x, int tokens, float eps,
                                        const float *adaln_mod, const float *q_norm,
                                        const float *k_norm, const float *rope_cos,
                                        const float *rope_sin, const uint32_t *row_map,
                                        int adaln_groups);
extern "C" int h3_cuda_probe(void);
extern "C" void h3_cuda_ws_free(void);
extern "C" size_t h3_cuda_workspace_bytes(void);
extern "C" int h3_cuda_gemm_bf16_dev(const float *x, const uint16_t *w, float *y, int S, int I,
                                     int O);
#endif

namespace mvllm {
namespace h3_cuda {
namespace {

bool g_inited = false;
bool g_last_dev = false;

void run_cpu(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
             int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x, int tokens,
             float eps, const float *adaln_mod, const float *q_norm, const float *k_norm,
             const float *rope_cos, const float *rope_sin, const uint32_t *row_map,
             int adaln_groups) {
    h3_dit_block_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn, head_dim,
                     x, tokens, eps, adaln_mod, q_norm, k_norm, rope_cos, rope_sin, row_map,
                     adaln_groups);
}

} // namespace

bool init() {
    g_inited = true;
    return true;
}

void shutdown() {
#if defined(MVLLM_WITH_CUDA_GEMM)
    h3_cuda_ws_free();
#endif
    g_inited = false;
}

bool available() { return g_inited; }

size_t workspace_bytes() {
#if defined(MVLLM_WITH_CUDA_GEMM)
    return h3_cuda_workspace_bytes();
#else
    return 0;
#endif
}

const char *backend_name() {
#if defined(MVLLM_WITH_CUDA_GEMM)
    return h3_cuda_probe() == 0 ? "cuda" : "cpu";
#else
    return "cpu";
#endif
}

bool last_on_device() { return g_last_dev; }

bool dit_residual(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                  int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                  int tokens, float eps, const float *adaln_mod, const float *q_norm,
                  const float *k_norm, const float *rope_cos, const float *rope_sin,
                  const uint32_t *row_map, int adaln_groups) {
    g_last_dev = false;
#if defined(MVLLM_WITH_CUDA_GEMM)
    if (h3_cuda_probe() == 0 &&
        h3_cuda_dit_residual_dev(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner,
                                 ffn, head_dim, x, tokens, eps, adaln_mod, q_norm, k_norm, rope_cos,
                                 rope_sin, row_map, adaln_groups) == 0) {
        g_last_dev = true;
        return true;
    }
#endif
    run_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn, head_dim, x,
            tokens, eps, adaln_mod, q_norm, k_norm, rope_cos, rope_sin, row_map, adaln_groups);
    return true;
}

bool gemm_bf16(float *y, const float *x, const uint16_t *w, int S, int I, int O) {
    if (!y || !x || !w || S <= 0 || I <= 0 || O <= 0)
        return false;
#if defined(MVLLM_WITH_CUDA_GEMM)
    if (h3_cuda_probe() == 0 && h3_cuda_gemm_bf16_dev(x, w, y, S, I, O) == 0)
        return true;
#endif
    std::vector<float> wf(static_cast<size_t>(O) * static_cast<size_t>(I));
    for (size_t i = 0; i < wf.size(); ++i) {
        uint32_t bits = static_cast<uint32_t>(w[i]) << 16;
        std::memcpy(&wf[i], &bits, sizeof(float));
    }
    quant::matmul_f32(y, x, wf.data(), S, I, O);
    return true;
}

} // namespace h3_cuda
} // namespace mvllm
