#include "kv_fp8.hpp"

#include <cmath>
#include <cstring>

namespace mvllm {
namespace {

constexpr float kFp8Max = 448.f;
constexpr float kDenormHi = 0x1p-6f; // 2^-6: first normal
constexpr float kTinyAmax = 1e-35f;
constexpr float kHugeAmax = 3.4e38f;

float g_lut[256];
bool g_lut_ready = false;

void fill_lut() {
    if (g_lut_ready)
        return;
    for (int b = 0; b < 256; ++b) {
        const int E = (b >> 3) & 0xF;
        const int M = b & 7;
        float v;
        if (E == 15 && M == 7)
            v = 0.f; // NaN payload: inert on read
        else if (E != 0)
            v = std::ldexp(1.f + static_cast<float>(M) / 8.f, E - 7);
        else
            v = std::ldexp(static_cast<float>(M), -9);
        g_lut[b] = (b & 0x80) ? -v : v;
    }
    g_lut_ready = true;
}

uint8_t pack_signed(uint8_t sign, uint8_t mag) {
    return static_cast<uint8_t>(sign | mag);
}

} // namespace

void kv_fp8_lut_init() { fill_lut(); }

float kv_fp8_lut(uint8_t b) {
    fill_lut();
    return g_lut[b];
}

uint8_t kv_fp8_enc(float f) {
    fill_lut();

    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = f;
    const uint8_t sign = static_cast<uint8_t>((bits.u >> 24) & 0x80u);
    const float a = std::fabs(f);

    // NaN → 0; |x| > 448 or ±inf → signed 448 (code 0x7e).
    if (!(a <= kFp8Max))
        return (a != a) ? static_cast<uint8_t>(0) : pack_signed(sign, 0x7e);

    if (a < kDenormHi) {
        // Denorm grid: k / 2^9, k in [0, 8]; k == 8 is the first normal.
        const int k = static_cast<int>(std::rintf(a * 0x1p9f));
        return pack_signed(sign, static_cast<uint8_t>(k));
    }

    int e = 0;
    (void)std::frexp(a, &e);
    int E = e + 6;
    int k = static_cast<int>(std::rintf(std::ldexp(a, 3 - (e - 1))));
    if (k == 16) {
        k = 8;
        ++E;
    }
    if (E > 15 || (E == 15 && k > 14))
        return pack_signed(sign, 0x7e);
    return pack_signed(sign, static_cast<uint8_t>((E << 3) | (k - 8)));
}

float kv_fp8_quant_row(const float *src, uint8_t *dst, int n) {
    float amax = 0.f;
    for (int i = 0; i < n; ++i) {
        const float a = std::fabs(src[i]);
        if (a > amax)
            amax = a;
    }
    // Degenerate row (all ~0, all NaN, or ±inf): store zeros, identity scale.
    if (!(amax > kTinyAmax) || amax > kHugeAmax) {
        if (n > 0)
            std::memset(dst, 0, static_cast<size_t>(n));
        return 1.f;
    }
    const float inv = kFp8Max / amax;
    for (int i = 0; i < n; ++i)
        dst[i] = kv_fp8_enc(src[i] * inv);
    return amax / kFp8Max;
}

void kv_fp8_dequant_row(const uint8_t *src, float scale, float *dst, int n) {
    fill_lut();
    for (int i = 0; i < n; ++i)
        dst[i] = g_lut[src[i]] * scale;
}

int kv_fp8_nscale(int n, int gs) { return gs > 0 ? (n + gs - 1) / gs : 1; }

void kv_fp8_quant_row_gs(const float *src, uint8_t *dst, float *scales, int n, int gs) {
    if (gs <= 0) {
        scales[0] = kv_fp8_quant_row(src, dst, n);
        return;
    }
    int k = 0;
    for (int g = 0; g < n; g += gs, ++k) {
        const int m = (n - g < gs) ? (n - g) : gs;
        scales[k] = kv_fp8_quant_row(src + g, dst + g, m);
    }
}

void kv_fp8_dequant_row_gs(const uint8_t *src, const float *scales, float *dst, int n, int gs) {
    if (gs <= 0) {
        kv_fp8_dequant_row(src, scales[0], dst, n);
        return;
    }
    int k = 0;
    for (int g = 0; g < n; g += gs, ++k) {
        const int m = (n - g < gs) ? (n - g) : gs;
        kv_fp8_dequant_row(src + g, scales[k], dst + g, m);
    }
}

} // namespace mvllm
