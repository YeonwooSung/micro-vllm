#include "quant.hpp"

#include <cmath>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace mvllm {
namespace quant {

const float mx4_lut[16] = {0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
                           -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

namespace {

inline int packed_stride(int I) { return (I + 1) / 2; }

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline int clampi(int v, int lo, int hi) {
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}

inline uint8_t nibble_at(const uint8_t *row, int i) {
    uint8_t b = row[i >> 1];
    return (i & 1) ? static_cast<uint8_t>(b >> 4) : static_cast<uint8_t>(b & 0x0f);
}

inline void pack_nibble(uint8_t *row, int i, uint8_t n) {
    int b = i >> 1;
    if ((i & 1) == 0)
        row[b] = static_cast<uint8_t>((row[b] & 0xf0) | (n & 0x0f));
    else
        row[b] = static_cast<uint8_t>((row[b] & 0x0f) | ((n & 0x0f) << 4));
}

// Smallest ue8m0 code e with 2^(e-127) >= need. e=0 is +0 (need==0 only).
uint8_t ue8m0_cover(float need) {
    if (!(need > 0.f))
        return 0;
    if (!std::isfinite(need))
        return 255;

    union {
        uint32_t u;
        float f;
    } b;
    b.f = need;
    int exp_field = static_cast<int>((b.u >> 23) & 0xffu);
    uint32_t frac = b.u & 0x7fffffu;
    int e;
    if (exp_field == 0)
        e = 1; // any positive subnormal is covered by 2^-126
    else if (exp_field == 255)
        e = 255;
    else
        e = (frac == 0) ? exp_field : exp_field + 1;
    if (e < 0)
        e = 0;
    if (e > 255)
        e = 255;

    while (e < 255 && mx4_scale(static_cast<uint8_t>(e)) < need)
        ++e;
    while (e > 0 && mx4_scale(static_cast<uint8_t>(e - 1)) >= need)
        --e;
    return static_cast<uint8_t>(e);
}

uint8_t closest_mx4(float w, float scale) {
    float target = 0.f;
    if (scale > 0.f && std::isfinite(scale))
        target = w / scale;
    float a = std::fabs(target);
    int best = 0;
    float best_d = std::fabs(mx4_lut[0] - a);
    for (int k = 1; k < 8; ++k) {
        float d = std::fabs(mx4_lut[k] - a);
        if (d < best_d) {
            best_d = d;
            best = k;
        }
    }
    if (std::signbit(w))
        best |= 8;
    return static_cast<uint8_t>(best);
}

// Doubled e2m1: {0,.5,1,1.5,2,3,4,6} * 2, exact int8. Sign in bit 3.
alignas(16) const int8_t mx4_i8_lut[16] = {0,  1,  2,  3,  4,  6,  8,  12,
                                           0, -1, -2, -3, -4, -6, -8, -12};

inline void unpack_mx4_i8(const uint8_t *packed16, int8_t *w32) {
    for (int b = 0; b < 16; ++b) {
        uint8_t v = packed16[b];
        w32[2 * b] = mx4_i8_lut[v & 0x0f];
        w32[2 * b + 1] = mx4_i8_lut[v >> 4];
    }
}

inline int32_t idot32(const int8_t *a, const int8_t *b) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int8x16_t a0 = vld1q_s8(a);
    int8x16_t b0 = vld1q_s8(b);
    int8x16_t a1 = vld1q_s8(a + 16);
    int8x16_t b1 = vld1q_s8(b + 16);
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdotq_s32(vdupq_n_s32(0), a0, b0);
    acc = vdotq_s32(acc, a1, b1);
#else
    int16x8_t p0 = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
    int16x8_t p1 = vmull_s8(vget_high_s8(a0), vget_high_s8(b0));
    int16x8_t p2 = vmull_s8(vget_low_s8(a1), vget_low_s8(b1));
    int16x8_t p3 = vmull_s8(vget_high_s8(a1), vget_high_s8(b1));
    int32x4_t acc = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
    acc = vaddq_s32(acc, vpaddlq_s16(p2));
    acc = vaddq_s32(acc, vpaddlq_s16(p3));
#endif
#if defined(__aarch64__)
    return vaddvq_s32(acc);
#else
    int32x2_t p = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    return vget_lane_s32(vpadd_s32(p, p), 0);
#endif
#elif defined(__AVX2__)
    __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
    __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b));
    __m256i alo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(av));
    __m256i blo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(bv));
    __m256i ahi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 1));
    __m256i bhi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv, 1));
    __m256i p = _mm256_add_epi32(_mm256_madd_epi16(alo, blo), _mm256_madd_epi16(ahi, bhi));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    return _mm_cvtsi128_si32(s);
#else
    int32_t acc = 0;
    for (int i = 0; i < 32; ++i)
        acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    return acc;
#endif
}

} // namespace

void matmul_f32(float *y, const float *x, const float *w, int S, int I, int O) {
    for (int s = 0; s < S; ++s) {
        const float *xs = x + static_cast<size_t>(s) * I;
        float *ys = y + static_cast<size_t>(s) * O;
        for (int o = 0; o < O; ++o) {
            const float *wo = w + static_cast<size_t>(o) * I;
            float acc = 0.f;
            for (int i = 0; i < I; ++i)
                acc += xs[i] * wo[i];
            ys[o] = acc;
        }
    }
}

void quantize_int4_g64(const float *w, int O, int I, uint8_t *packed, float *scales) {
    const int stride = packed_stride(I);
    const int ng = ceil_div(I, 64);
    for (int o = 0; o < O; ++o) {
        const float *row = w + static_cast<size_t>(o) * I;
        uint8_t *prow = packed + static_cast<size_t>(o) * stride;
        for (int b = 0; b < stride; ++b)
            prow[b] = 0;
        for (int g = 0; g < ng; ++g) {
            int i0 = g * 64;
            int i1 = i0 + 64;
            if (i1 > I)
                i1 = I;
            float amax = 0.f;
            for (int i = i0; i < i1; ++i) {
                float a = std::fabs(row[i]);
                if (a > amax)
                    amax = a;
            }
            float scale = amax / 7.f;
            if (scale < 1e-8f)
                scale = 1e-8f;
            scales[static_cast<size_t>(o) * ng + g] = scale;
            float inv = 1.f / scale;
            for (int i = i0; i < i1; ++i) {
                int q = static_cast<int>(std::round(row[i] * inv));
                q = clampi(q, -8, 7);
                pack_nibble(prow, i, static_cast<uint8_t>(q + 8));
            }
        }
    }
}

void matmul_int4_g64(float *y, const float *x, const uint8_t *packed, const float *scales, int S,
                     int I, int O) {
    const int stride = packed_stride(I);
    const int ng = ceil_div(I, 64);
    for (int s = 0; s < S; ++s) {
        const float *xs = x + static_cast<size_t>(s) * I;
        float *ys = y + static_cast<size_t>(s) * O;
        for (int o = 0; o < O; ++o) {
            const uint8_t *prow = packed + static_cast<size_t>(o) * stride;
            const float *srow = scales + static_cast<size_t>(o) * ng;
            float acc = 0.f;
            for (int g = 0; g < ng; ++g) {
                int i0 = g * 64;
                int i1 = i0 + 64;
                if (i1 > I)
                    i1 = I;
                float scale = srow[g];
                for (int i = i0; i < i1; ++i) {
                    int q = static_cast<int>(nibble_at(prow, i)) - 8;
                    acc += xs[i] * (static_cast<float>(q) * scale);
                }
            }
            ys[o] = acc;
        }
    }
}

void quantize_int8_row(const float *w, int O, int I, int8_t *q, float *scales) {
    for (int o = 0; o < O; ++o) {
        const float *row = w + static_cast<size_t>(o) * I;
        int8_t *qrow = q + static_cast<size_t>(o) * I;
        float amax = 0.f;
        for (int i = 0; i < I; ++i) {
            float a = std::fabs(row[i]);
            if (a > amax)
                amax = a;
        }
        float scale = amax / 127.f;
        scales[o] = scale;
        if (scale == 0.f) {
            for (int i = 0; i < I; ++i)
                qrow[i] = 0;
            continue;
        }
        float inv = 1.f / scale;
        for (int i = 0; i < I; ++i) {
            int qi = static_cast<int>(std::round(row[i] * inv));
            qrow[i] = static_cast<int8_t>(clampi(qi, -127, 127));
        }
    }
}

void matmul_int8_row(float *y, const float *x, const int8_t *q, const float *scales, int S, int I,
                     int O) {
    for (int s = 0; s < S; ++s) {
        const float *xs = x + static_cast<size_t>(s) * I;
        float *ys = y + static_cast<size_t>(s) * O;
        for (int o = 0; o < O; ++o) {
            const int8_t *qrow = q + static_cast<size_t>(o) * I;
            float acc = 0.f;
            for (int i = 0; i < I; ++i)
                acc += xs[i] * static_cast<float>(qrow[i]);
            ys[o] = acc * scales[o];
        }
    }
}

void pack_mxfp4(const float *w, int O, int I, uint8_t *packed, uint8_t *scales) {
    const int stride = packed_stride(I);
    const int ng = ceil_div(I, 32);
    for (int o = 0; o < O; ++o) {
        const float *row = w + static_cast<size_t>(o) * I;
        uint8_t *prow = packed + static_cast<size_t>(o) * stride;
        for (int b = 0; b < stride; ++b)
            prow[b] = 0;
        for (int g = 0; g < ng; ++g) {
            int i0 = g * 32;
            int i1 = i0 + 32;
            if (i1 > I)
                i1 = I;
            float amax = 0.f;
            for (int i = i0; i < i1; ++i) {
                float a = std::fabs(row[i]);
                if (a > amax)
                    amax = a;
            }
            uint8_t e = ue8m0_cover(amax / 6.f);
            scales[static_cast<size_t>(o) * ng + g] = e;
            float scale = mx4_scale(e);
            for (int i = i0; i < i1; ++i)
                pack_nibble(prow, i, closest_mx4(row[i], scale));
        }
    }
}

void matmul_mxfp4(float *y, const float *x, const uint8_t *packed, const uint8_t *scales, int S,
                  int I, int O) {
    const int stride = packed_stride(I);
    const int ng = ceil_div(I, 32);
    for (int s = 0; s < S; ++s) {
        const float *xs = x + static_cast<size_t>(s) * I;
        float *ys = y + static_cast<size_t>(s) * O;
        for (int o = 0; o < O; ++o) {
            const uint8_t *prow = packed + static_cast<size_t>(o) * stride;
            const uint8_t *srow = scales + static_cast<size_t>(o) * ng;
            float acc = 0.f;
            for (int g = 0; g < ng; ++g) {
                int i0 = g * 32;
                int i1 = i0 + 32;
                if (i1 > I)
                    i1 = I;
                float scale = mx4_scale(srow[g]);
                for (int i = i0; i < i1; ++i)
                    acc += xs[i] * (mx4_lut[nibble_at(prow, i)] * scale);
            }
            ys[o] = acc;
        }
    }
}

void mxfp4_quant_acts(const float *x, int I, int8_t *xq, float *xsc) {
    const int ng = ceil_div(I, 32);
    for (int g = 0; g < ng; ++g) {
        int i0 = g * 32;
        int i1 = i0 + 32;
        if (i1 > I)
            i1 = I;
        float amax = 0.f;
        for (int i = i0; i < i1; ++i) {
            float a = std::fabs(x[i]);
            if (a > amax)
                amax = a;
        }
        float scale = amax / 127.f;
        xsc[g] = scale;
        if (scale == 0.f) {
            for (int i = i0; i < i1; ++i)
                xq[i] = 0;
            continue;
        }
        float inv = 1.f / scale;
        for (int i = i0; i < i1; ++i) {
            int q = static_cast<int>(std::round(x[i] * inv));
            xq[i] = static_cast<int8_t>(clampi(q, -127, 127));
        }
    }
}

void matmul_mxfp4_i8(float *y, const float *x, const uint8_t *packed, const uint8_t *scales, int S,
                     int I, int O) {
    if (I % 32 != 0) {
        matmul_mxfp4(y, x, packed, scales, S, I, O);
        return;
    }
    const int stride = packed_stride(I);
    const int ng = I / 32;
    std::vector<int8_t> xq(static_cast<size_t>(S) * static_cast<size_t>(I));
    std::vector<float> xsc(static_cast<size_t>(S) * static_cast<size_t>(ng));
    for (int s = 0; s < S; ++s)
        mxfp4_quant_acts(x + static_cast<size_t>(s) * I, I, xq.data() + static_cast<size_t>(s) * I,
                         xsc.data() + static_cast<size_t>(s) * ng);

    int8_t w32[32];
    for (int s = 0; s < S; ++s) {
        const int8_t *xs = xq.data() + static_cast<size_t>(s) * I;
        const float *ssc = xsc.data() + static_cast<size_t>(s) * ng;
        float *ys = y + static_cast<size_t>(s) * O;
        for (int o = 0; o < O; ++o) {
            const uint8_t *prow = packed + static_cast<size_t>(o) * stride;
            const uint8_t *srow = scales + static_cast<size_t>(o) * ng;
            float acc = 0.f;
            for (int g = 0; g < ng; ++g) {
                unpack_mx4_i8(prow + static_cast<size_t>(g) * 16, w32);
                float gscale = mx4_scale(srow[g]) * 0.5f * ssc[g];
                acc += static_cast<float>(idot32(xs + g * 32, w32)) * gscale;
            }
            ys[o] = acc;
        }
    }
}

void rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
    if (n <= 0)
        return;
    float ss = 0.f;
    for (int i = 0; i < n; ++i)
        ss += x[i] * x[i];
    float inv = 1.f / std::sqrt(ss / static_cast<float>(n) + eps);
    if (w) {
        for (int i = 0; i < n; ++i)
            y[i] = x[i] * inv * w[i];
    } else {
        for (int i = 0; i < n; ++i)
            y[i] = x[i] * inv;
    }
}

void silu_mul(float *gate, const float *up, int n) {
    for (int i = 0; i < n; ++i)
        gate[i] = gate[i] * sigmoid(gate[i]) * up[i];
}

void softmax_inplace(float *x, int n) {
    if (n <= 0)
        return;
    float m = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > m)
            m = x[i];
    }
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - m);
        sum += x[i];
    }
    float inv = (sum == 0.f) ? 0.f : 1.f / sum;
    for (int i = 0; i < n; ++i)
        x[i] *= inv;
}

} // namespace quant
} // namespace mvllm
