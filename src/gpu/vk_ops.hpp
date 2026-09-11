#pragma once

#include <cstddef>

namespace mvllm {
namespace vk_ops {

// Host Vulkan surface. Always compiled: ops run on CPU until a device
// backend is linked. available() is true after init. False return = bad args.

bool init();
void shutdown();
bool available();
const char *backend_name(); // "vulkan" after a live instance, else "cpu"

bool rmsnorm(float *y, const float *x, const float *w, int nrows, int D, float eps);
bool add(float *y, const float *a, size_t n);
bool silu_mul(float *g, const float *u, size_t n);
// y[S,O] = x[S,I] @ W[O,I]^T
bool gemm_f32(float *y, const float *x, const float *w, int S, int I, int O);

// Decode residual: x += attn; nrm = rmsnorm(x, post_ln). False on bad args.
bool layer_residual(float *x, const float *attn, const float *post_ln, float *nrm, int D,
                    float eps);

// Batched expert SwiGLU (F32, SiLU). Matches metal_ops moe_block fmt 0 + Act::Silu.
// Expert e: nr[e] rows of xg at xoff[e]; hh = down(silu(gate)*up);
// scatter-add rw[row]*hh into out[S,D]. False on bad args.
bool moe_block_f32(int nb, int D, int Iinter, const float *const *g, const float *const *u,
                   const float *const *d, const float *xg, const int *xoff, const int *nr,
                   const int *rows, const float *rw, float *out, int S);

} // namespace vk_ops
} // namespace mvllm
