#pragma once

#include <cstdint>
#include <cstddef>

namespace mvllm {
namespace quant {

// MXFP4 e2m1 nibble LUT: bits 0..2 index {0,.5,1,1.5,2,3,4,6}, bit3 = sign.
extern const float mx4_lut[16];

// ue8m0 scale: 2^(s-127). s=0 -> +0, s=255 -> +inf (bit-identical to IEEE).
inline float mx4_scale(uint8_t s) {
    union {
        uint32_t u;
        float f;
    } b;
    b.u = static_cast<uint32_t>(s) << 23;
    return b.f;
}

inline float sigmoid(float x) {
    if (x >= 0.f) {
        float z = 1.f / (1.f + __builtin_expf(-x));
        return z;
    }
    float z = __builtin_expf(x);
    return z / (1.f + z);
}

// SiTU-GLU (Kimi K3):  b1*tanh(g/b1)*σ(g) * b2*tanh(u/b2)
inline float situ_glu(float g, float u, float b1, float b2) {
    return b1 * __builtin_tanhf(g / b1) * sigmoid(g) * b2 * __builtin_tanhf(u / b2);
}

// GLM-5.3 clamped SwiGLU: silu(min(gate, limit)) * clamp(up, -limit, limit)
inline float clamped_swiglu(float gate, float up, float limit) {
    float g = gate;
    if (limit > 0.f && g > limit)
        g = limit;
    float s = g * sigmoid(g);
    float u = up;
    if (limit > 0.f) {
        if (u > limit)
            u = limit;
        if (u < -limit)
            u = -limit;
    }
    return s * u;
}

// y[S,O] = x[S,I] @ W[O,I]^T   W is row-major F32
void matmul_f32(float *y, const float *x, const float *w, int S, int I, int O);

// int4-g64: packed nibbles [O, I/2], scales [O, ceil(I/64)] F32
// nibble value = (n - 8) * scale[group]
void quantize_int4_g64(const float *w, int O, int I, uint8_t *packed, float *scales);
void matmul_int4_g64(float *y, const float *x, const uint8_t *packed, const float *scales,
                     int S, int I, int O);

// int8 per-row: q[O,I] + scale[O]
void quantize_int8_row(const float *w, int O, int I, int8_t *q, float *scales);
void matmul_int8_row(float *y, const float *x, const int8_t *q, const float *scales, int S, int I,
                     int O);

// MXFP4: packed [O, I/2] e2m1, scales [O, ceil(I/32)] ue8m0
void pack_mxfp4(const float *w, int O, int I, uint8_t *packed, uint8_t *scales);
void matmul_mxfp4(float *y, const float *x, const uint8_t *packed, const uint8_t *scales, int S,
                  int I, int O);

// Per-32-group int8 activation quant: xq[i] in [-127,127], xsc[g] = max(|x group|)/127
void mxfp4_quant_acts(const float *x, int I, int8_t *xq, float *xsc);

// y[S,O] = x @ W^T using int8 activations * doubled-e2m1 weights.
// If I % 32 != 0, fall back to matmul_mxfp4.
void matmul_mxfp4_i8(float *y, const float *x, const uint8_t *packed, const uint8_t *scales,
                     int S, int I, int O);

void rmsnorm(const float *x, const float *w, float *y, int n, float eps);
void silu_mul(float *gate, const float *up, int n);
void softmax_inplace(float *x, int n);

} // namespace quant
} // namespace mvllm
