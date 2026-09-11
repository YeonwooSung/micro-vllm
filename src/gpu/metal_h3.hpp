#pragma once

#include <cstdint>

namespace mvllm {
namespace metal_h3 {

// H3 Metal (or CPU fallback) ops. available() is true after a successful
// init (CPU always; Metal when compiled). dit_residual uses the same blob
// layout as h3_dit_block_cpu. gemm_int8, nax_mlp, vae_transformer_block,
// vision_block, and audio_pre_block are host ports of h3.c GPU ops.
// Official h3_gpu.m / h3_shaders.metal are not vendored.

bool init();
void shutdown();
bool available();
const char *backend_name(); // "metal" or "cpu"

// AdaLN / QK-norm / RoPE DiT residual. CPU fallback if Metal is missing or
// the call fails. adaln_mod is [6, hidden] (scale0, shift0, scale1, shift1,
// scale2, shift2). q_norm/k_norm are [head_dim] or null. rope_cos/sin are
// [tokens, 48] or null (applied when head_dim >= 96, same as the CPU path).
bool dit_residual(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                  int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                  int tokens, float eps, const float *adaln_mod, const float *q_norm,
                  const float *k_norm, const float *rope_cos, const float *rope_sin);

// Host port of the h3.c int8 GEMM. y[S,O] = x[S,I] @ W[O,I]^T with
// int8-row weights and per-row scales. False on bad args.
bool gemm_int8(float *y, const float *x, const int8_t *w, const float *scale, int S, int I, int O);

// Host port of the h3.c NAX gated MLP: y += down(silu(up(x))).
// w_up is F32 [I, D], w_down is F32 [D, I]. False on bad args.
bool nax_mlp(float *y, const float *x, const float *w_up, const float *w_down, int S, int D, int I);

// VAE residual: x += skip; y = rmsnorm(x, w). y may alias x when w is applied in place
// via a separate buffer; here y is nrm_out and x is updated.
bool vae_rms_add(float *x, const float *skip, const float *w, float *y, int n, float eps);

// Host port of the h3.c VAE transformer block (matches host apply_block):
// RMS → QKV SDPA → out → residual RMS → SwiGLU FFN. x is [tokens, hidden].
// QKV/out/w1/w2 are F32 [O,I] row-major (quant::matmul_f32). QKV is packed
// [q|k|v] per row, size [3*hidden, hidden]. Official [seq,heads,3,hd]+rope
// is not handled (returns false). Optional biases / scale1 / scale2 may be
// null. w1 is [2*ffn, hidden], w2 is [hidden, ffn]. False on bad args / size
// mismatch.
bool vae_transformer_block(float *x, int tokens, int hidden, int heads, int hd,
                           const float *norm1, const float *qkv_w, const float *qkv_b,
                           const float *out_w, const float *out_b, const float *scale1,
                           const float *norm2, const float *w1, const float *b1,
                           const float *w2, const float *b2, const float *scale2, int ffn,
                           float eps);

// Host port of the h3.c vision block (matches host run_block): LN → QKV →
// optional RoPE → SDPA → proj → residual → LN → GELU-tanh MLP → residual.
// x is [rows, hidden]. QKV/proj/fc are F32 [O,I]. LN weights/biases and
// linear biases may be null. rope_cos/sin are [rows, rope_half] or null.
// False on bad args / heads*hd != hidden / missing qkv.
bool vision_block(float *x, int rows, int hidden, int heads, int hd, int intermediate,
                  const float *norm1_w, const float *norm1_b, const float *qkv_w,
                  const float *qkv_b, const float *proj_w, const float *proj_b,
                  const float *norm2_w, const float *norm2_b, const float *fc1_w,
                  const float *fc1_b, const float *fc2_w, const float *fc2_b,
                  const float *rope_cos, const float *rope_sin, int rope_half, float eps);

// Host port of the h3.c audio VAE pre_block: LN(seq) → QKV (+q/k/v bias) →
// causal SDPA over B batches of length L → pool first `ch` of each row →
// proj → base += attn → LN → LN → GELU-gate MLP → base += mlp. seq is
// [B*L, C], base is [B*L, ch]. QKV is [3C,C], proj [ch,ch], w0/w1 [2ch,ch],
// w2 [ch,2ch]. False on bad args / missing qkv / heads do not divide C.
bool audio_pre_block(float *base, const float *seq, int B, int L, int C, int ch, int heads,
                     const float *norm1_w, const float *norm1_b, const float *qkv_w,
                     const float *q_bias, const float *k_bias, const float *v_bias,
                     const float *proj_w, const float *proj_b, const float *norm2_w,
                     const float *norm2_b, const float *mlp_norm_w, const float *mlp_norm_b,
                     const float *w0, const float *b0, const float *w1, const float *b1,
                     const float *w2, const float *b2, float eps);

} // namespace metal_h3
} // namespace mvllm
