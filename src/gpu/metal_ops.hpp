#pragma once

#include <cstddef>

namespace mvllm {
namespace metal_ops {

// Standalone Metal (or CPU fallback) ops for K3/GLM decode.
// available() is true after a successful init (CPU always; Metal when compiled).
// Returning false means the caller should use the existing host kernel.

bool init();
void shutdown();
bool available();
const char *backend_name(); // "metal" or "cpu"

bool rmsnorm(float *y, const float *x, const float *w, int nrows, int D, float eps);
bool add(float *y, const float *a, size_t n);
bool silu_mul(float *g, const float *u, size_t n);

// Fused KDA token: depthwise conv+SiLU on q/k/v, L2-norm q/k, state recurrence.
// win_* [P*K] oldest-first; qt/kt/tv [P] in-place; taps_* [P*K];
// S [H*hd*hd] row-major [h][kk][vv]; alpha [H*hd]; beta [H]; oh [H*hd] (caller zeros).
// P must equal H*hd. Returns false on bad args.
bool kda_fused_token(float *win_q, float *qt, float *win_k, float *kt, float *win_v, float *tv,
                     const float *taps_q, const float *taps_k, const float *taps_v, float *S,
                     const float *alpha, const float *beta, float *oh, int P, int K, int H,
                     int hd);

// Decode tail after host attention: x += attn; nrm_out = rmsnorm(x, post_ln).
// If shg/shu/shd are non-null F32 [Iinter,D] / [Iinter,D] / [D,Iinter],
// sh_out = down(silu(gate(nrm))*up(nrm)). One command buffer on Metal.
bool layer_decode(float *x, const float *attn, const float *post_ln, const float *shg,
                  const float *shu, const float *shd, int D, int Iinter, float eps, float *nrm_out,
                  float *sh_out);

// Batched routed-expert SwiGLU. fmt 0 = F32 [O,I], fmt 4 = int4-g64 (qgs along I).
// For expert e: nr[e] rows of xg starting at xoff[e]; hh = down(silu(gate)*up);
// scatter-add rw[row]*hh into out[S,D]. One command buffer on Metal.
bool moe_block(int nb, int D, int Iinter, int fmt, int qgs, const void *const *g,
               const void *const *u, const void *const *d, const float *const *gs,
               const float *const *us, const float *const *ds, const float *xg, const int *xoff,
               const int *nr, const int *rows, const float *rw, float *out, int S);

} // namespace metal_ops
} // namespace mvllm
