#include "metal_h3.hpp"

#if !defined(MVLLM_WITH_METAL)

#include "../model/family.hpp"

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

} // namespace metal_h3
} // namespace mvllm

#endif // !MVLLM_WITH_METAL
