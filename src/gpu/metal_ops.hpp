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

// Family activation fused into shared-expert / moe_block.
// Silu = silu(g)*u. ClampSwiGLU = silu(min(g,a))*clamp(u,-a,a).
// Situ = a*tanh(g/a)*σ(g) * b*tanh(u/b) (K3 SiTU-GLU).
enum class Act : int { Silu = 0, ClampSwiGLU = 1, Situ = 2 };

// Decode tail after host attention: x += attn; nrm_out = rmsnorm(x, post_ln).
// If shg/shu/shd are non-null F32 [Iinter,D] / [Iinter,D] / [D,Iinter],
// sh_out = down(silu(gate(nrm))*up(nrm)). One command buffer on Metal.
bool layer_decode(float *x, const float *attn, const float *post_ln, const float *shg,
                  const float *shu, const float *shd, int D, int Iinter, float eps, float *nrm_out,
                  float *sh_out);

// Full decode tail in one CB (S=1): optional in_ln on pre-add x, residual add,
// post_ln, optional F32 shared expert with Act, optional router GEMM + host top-k.
// attn is a precomputed attention vector. KDA fused token can share the decode CB
// via layer_decode_kda; MLA stays on its own kernels.
// router_w is F32 [E,D]; rscale multiplies the normalized mix weights.
// Returns false on bad args. CPU fallback still returns true.
bool layer_decode_full(float *x, const float *attn, const float *in_ln, const float *post_ln,
                       const float *shg, const float *shu, const float *shd, int D, int Iinter,
                       float eps, Act act, float act_a, float act_b, const float *router_w,
                       const float *router_bias, int E, int K, float rscale, float *inrm_out,
                       float *nrm_out, float *sh_out, int *idx_out, float *w_out);

// KDA token buffers (same layout/sizes as kda_fused_token).
struct KdaToken {
    float *win_q, *qt, *win_k, *kt, *win_v, *tv;
    const float *taps_q, *taps_k, *taps_v;
    float *S;
    const float *alpha, *beta;
    float *oh;
    int P, K, H, hd;
};

// KDA fused token + decode tail in one CB: op_kda_fused, then x += oh, post-LN,
// optional F32 shared Act, optional router GEMM + host top-k.
// KDA fused token can share the decode CB; MLA stays on its own kernels.
// oh is the residual (attn); P must be >= D. False on bad args; CPU fallback still true.
bool layer_decode_kda(const KdaToken &kda, float *x, const float *in_ln, const float *post_ln,
                      const float *shg, const float *shu, const float *shd, int D, int Iinter,
                      float eps, Act act, float act_a, float act_b, const float *router_w,
                      const float *router_bias, int E, int K, float rscale, float *inrm_out,
                      float *nrm_out, float *sh_out, int *idx_out, float *w_out);

// Batched routed-expert SwiGLU. fmt 0 = F32 [O,I], fmt 4 = int4-g64 (qgs along I).
// For expert e: nr[e] rows of xg starting at xoff[e]; hh = down(act(gate,up));
// scatter-add rw[row]*hh into out[S,D]. One command buffer on Metal.
bool moe_block(int nb, int D, int Iinter, int fmt, int qgs, const void *const *g,
               const void *const *u, const void *const *d, const float *const *gs,
               const float *const *us, const float *const *ds, const float *xg, const int *xoff,
               const int *nr, const int *rows, const float *rw, float *out, int S,
               Act act = Act::Silu, float act_a = 0.f, float act_b = 0.f);

} // namespace metal_ops
} // namespace mvllm
