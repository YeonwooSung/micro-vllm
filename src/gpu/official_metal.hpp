#pragma once

#include <cstddef>
#include <cstdint>

// Optional wrap of vendored colibri backend_metal + h3_gpu hosts.
// Built only with -DMVLLM_OFFICIAL_METAL_HOST=ON (Apple + Metal).

namespace mvllm {
namespace official_metal {

bool init();
void shutdown();
bool coli_available();
bool h3_available();
const char *status(); // "coli+h3" | "coli" | "h3" | "off" | "err: ..."

// All return false when MVLLM_OFFICIAL_METAL_HOST is off, init failed, or bad args.

bool rmsnorm(float *y, const float *x, const float *w, int nrows, int D, float eps);
bool add(float *y, const float *a, size_t n);
bool silu_mul(float *g, const float *u, size_t n);
bool gemm_f32(float *y, const float *x, const float *w, int S, int I, int O);
// Decode residual: x += attn; nrm = rmsnorm(x, post_ln).
bool layer_residual(float *x, const float *attn, const float *post_ln, float *nrm, int D, float eps);

// Batched expert SwiGLU F32 (fmt 0). Same contract as metal_ops::moe_block fmt 0 + Act::Silu.
bool moe_block_f32(int nb, int D, int Iinter, const float *const *g, const float *const *u,
                   const float *const *d, const float *xg, const int *xoff, const int *nr,
                   const int *rows, const float *rw, float *out, int S);

// Same signature as metal_h3::dit_residual. Use h3_gpu_* when h3_available().
bool dit_residual(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                  int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                  int tokens, float eps, const float *adaln_mod, const float *q_norm,
                  const float *k_norm, const float *rope_cos, const float *rope_sin,
                  const uint32_t *row_map, int adaln_groups, const float *norm1 = nullptr,
                  const float *norm2 = nullptr);

} // namespace official_metal
} // namespace mvllm
