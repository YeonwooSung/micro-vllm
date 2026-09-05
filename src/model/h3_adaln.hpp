#pragma once

namespace mvllm {

// Official AdaLN: 3 modalities * 6 slots * hidden.
constexpr int kH3AdalnSlots = 6;
constexpr int kH3AdalnModalities = 3;

// Official time MLP widths are 256 -> 5376 -> 2688; callers may use any positive dims.
int h3_adaln_out(int hidden);

// SiLU: x * sigmoid(x) = x / (1 + exp(-x)).
void h3_silu(float *x, int n);

// y[rows, out] = x[rows, in] @ W[out, in]^T + b[out]. b may be null (zero).
void h3_linear(float *y, const float *x, const float *w, const float *b, int rows, int in,
               int out);

// temb[rows, time_dim] = SiLU(W_out @ SiLU(W_in @ features + b_in) + b_out).
bool h3_time_embed(const float *features, int rows, int time_input, const float *w_in,
                   const float *b_in, int time_hidden, const float *w_out, const float *b_out,
                   int time_dim, float *temb);

// mod[rows, adaln_out] = W_adaln @ temb + b. Official adaln_out = 3 * 6 * hidden.
bool h3_adaln_mod(const float *temb, int rows, int time_dim, const float *w_adaln,
                  const float *b_adaln, int adaln_out, float *mod);

} // namespace mvllm
