#include "h3_cuda.hpp"

#include "../model/family.hpp"

#include <cstdint>

#if defined(MVLLM_WITH_CUDA_GEMM)
extern "C" int h3_cuda_dit_residual_dev(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes,
                                        int64_t fc1_bytes, int64_t fc2_bytes, int hidden, int inner,
                                        int ffn, int head_dim, float *x, int tokens, float eps,
                                        const float *adaln_mod, const float *q_norm,
                                        const float *k_norm, const float *rope_cos,
                                        const float *rope_sin);
extern "C" int h3_cuda_probe(void);
#endif

namespace mvllm {
namespace h3_cuda {
namespace {

bool g_inited = false;

void run_cpu(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
             int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x, int tokens,
             float eps, const float *adaln_mod, const float *q_norm, const float *k_norm,
             const float *rope_cos, const float *rope_sin) {
    h3_dit_block_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn, head_dim,
                     x, tokens, eps, adaln_mod, q_norm, k_norm, rope_cos, rope_sin);
}

} // namespace

bool init() {
    g_inited = true;
    return true;
}

void shutdown() { g_inited = false; }

bool available() { return g_inited; }

const char *backend_name() {
#if defined(MVLLM_WITH_CUDA_GEMM)
    return h3_cuda_probe() == 0 ? "cuda" : "cpu";
#else
    return "cpu";
#endif
}

bool dit_residual(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                  int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                  int tokens, float eps, const float *adaln_mod, const float *q_norm,
                  const float *k_norm, const float *rope_cos, const float *rope_sin) {
#if defined(MVLLM_WITH_CUDA_GEMM)
    if (h3_cuda_probe() == 0 &&
        h3_cuda_dit_residual_dev(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner,
                                 ffn, head_dim, x, tokens, eps, adaln_mod, q_norm, k_norm, rope_cos,
                                 rope_sin) == 0)
        return true;
#endif
    run_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn, head_dim, x,
            tokens, eps, adaln_mod, q_norm, k_norm, rope_cos, rope_sin);
    return true;
}

} // namespace h3_cuda
} // namespace mvllm
