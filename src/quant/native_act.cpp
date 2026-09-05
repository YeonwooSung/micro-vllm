#include "native_act.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mvllm {
namespace {

constexpr float kE4m3Max = 448.f;
constexpr float kE4m3Denorm = 0.015625f; // 2^-6
constexpr float kFp8AmaxFloor = 1e-4f;
constexpr float kE2m1Max = 6.f;

int clamp_scale_exp(int exp) {
    if (exp < -127)
        return -127;
    if (exp > 127)
        return 127;
    return exp;
}

// ceil(log2(v)) for v > 0 via frexp.
int ceil_log2_positive(float v) {
    int exp = 0;
    const float frac = std::frexp(v, &exp);
    return frac == 0.5f ? exp - 1 : exp;
}

int short_block(int length, int base, int block_size) {
    const int remain = length - base;
    return remain < block_size ? remain : block_size;
}

float absmax_n(const float *x, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) {
        const float a = std::fabs(x[i]);
        if (a > m)
            m = a;
    }
    return m;
}

uint8_t pack_ue8m0(int exp) {
    return static_cast<uint8_t>(clamp_scale_exp(exp) + 127);
}

int nearest_e2m1_code(float value) {
    int best = 0;
    float dist = std::fabs(value - e2m1_decode(0));
    for (int code = 1; code < 16; ++code) {
        const float d = std::fabs(value - e2m1_decode(static_cast<uint8_t>(code)));
        if (d < dist) {
            dist = d;
            best = code;
        }
    }
    return best;
}

} // namespace

float e8m0_decode(uint8_t value) {
    if (value == 0xff)
        return NAN;
    return std::ldexp(1.f, static_cast<int>(value) - 127);
}

const float *e8m0_table() {
    static float table[256];
    static const bool filled = []() {
        for (int i = 0; i < 256; ++i)
            table[i] = e8m0_decode(static_cast<uint8_t>(i));
        return true;
    }();
    (void)filled;
    return table;
}

float e2m1_decode(uint8_t nibble) {
    static const float kCodes[16] = {
        0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
        0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
    };
    return kCodes[nibble & 15];
}

float e4m3fn_decode(uint8_t value) {
    const int sign = value >> 7;
    const int exp = (value >> 3) & 15;
    const int man = value & 7;
    if (exp == 15 && man == 7)
        return NAN;
    const float mag = (exp == 0)
                          ? std::ldexp(static_cast<float>(man), -9)
                          : std::ldexp(1.f + static_cast<float>(man) / 8.f, exp - 7);
    return sign ? -mag : mag;
}

uint8_t e4m3fn_encode(float value) {
    if (std::isnan(value))
        return 0x7f;

    const uint8_t sign = std::signbit(value) ? 0x80 : 0;
    const float mag = std::fabs(value);
    if (mag == 0.f)
        return sign;
    if (mag >= kE4m3Max)
        return static_cast<uint8_t>(sign | 0x7e);

    uint8_t mag_code = 0;
    if (mag < kE4m3Denorm) {
        const float scaled = mag * 512.f;
        auto rounded = static_cast<uint8_t>(scaled);
        const float frac = scaled - static_cast<float>(rounded);
        if (frac > 0.5f || (frac == 0.5f && (rounded & 1)))
            ++rounded;
        mag_code = rounded;
    } else {
        uint32_t bits = 0;
        std::memcpy(&bits, &mag, sizeof(bits));
        int exp = static_cast<int>((bits >> 23) & 0xffu) - 127;
        const uint32_t sig = 0x800000u | (bits & 0x7fffffu);
        uint32_t rounded = sig >> 20;
        const uint32_t rem = sig & 0xfffffu;
        if (rem > 0x80000u || (rem == 0x80000u && (rounded & 1u)))
            ++rounded;
        if (rounded == 16u) {
            rounded = 8u;
            ++exp;
        }
        mag_code = static_cast<uint8_t>((exp + 7) * 8 + static_cast<int>(rounded) - 8);
    }
    return static_cast<uint8_t>(mag_code | sign);
}

int fp8_activation_qdq(float *output, uint8_t *scales, const float *input,
                       int length, int block_size) {
    if (!output || !scales || !input || length <= 0 || block_size <= 0)
        return -1;

    for (int base = 0; base < length; base += block_size) {
        const int n = short_block(length, base, block_size);
        const float amax = std::fmax(absmax_n(input + base, n), kFp8AmaxFloor);
        const uint8_t sc = pack_ue8m0(ceil_log2_positive(amax / kE4m3Max));
        scales[base / block_size] = sc;
        const float scale = e8m0_decode(sc);
        for (int i = 0; i < n; ++i) {
            const float q =
                std::fmax(-kE4m3Max, std::fmin(kE4m3Max, input[base + i] / scale));
            output[base + i] = e4m3fn_decode(e4m3fn_encode(q)) * scale;
        }
    }
    return 0;
}

int fp4_activation_qdq(float *output, uint8_t *scales, const float *input,
                       int length, int block_size) {
    if (!output || !scales || !input || length <= 0 || block_size <= 0)
        return -1;

    const float amax_floor = kE2m1Max * std::ldexp(1.f, -126);
    for (int base = 0; base < length; base += block_size) {
        const int n = short_block(length, base, block_size);
        const float amax = std::fmax(absmax_n(input + base, n), amax_floor);
        const uint8_t sc = pack_ue8m0(ceil_log2_positive(amax / kE2m1Max));
        scales[base / block_size] = sc;
        const float scale = e8m0_decode(sc);
        for (int i = 0; i < n; ++i) {
            const float q =
                std::fmax(-kE2m1Max, std::fmin(kE2m1Max, input[base + i] / scale));
            output[base + i] =
                e2m1_decode(static_cast<uint8_t>(nearest_e2m1_code(q))) * scale;
        }
    }
    return 0;
}

namespace {

constexpr int kFp8Tile = 128;

bool bad_fp8_matvec(int O, int I) {
    return O <= 0 || I <= 0 || (I % kFp8Tile) != 0;
}

void fp8_matvec_xhat(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
                     const float *xhat) {
    const int ntiles = I / kFp8Tile;
    for (int o = 0; o < O; ++o) {
        float acc = 0.f;
        const uint8_t *row = w + static_cast<size_t>(o) * static_cast<size_t>(I);
        const uint8_t *row_sc =
            scales + static_cast<size_t>(o) * static_cast<size_t>(ntiles);
        for (int t = 0; t < ntiles; ++t) {
            float sc = e8m0_decode(row_sc[t]);
            if (!std::isfinite(sc))
                sc = 0.f;
            const int base = t * kFp8Tile;
            for (int k = 0; k < kFp8Tile; ++k)
                acc += e4m3fn_decode(row[base + k]) * xhat[base + k] * sc;
        }
        y[o] = acc;
    }
}

int qdq_xhat(std::vector<float> &xhat, const float *x, int I) {
    xhat.assign(static_cast<size_t>(I), 0.f);
    std::vector<uint8_t> act_scales(static_cast<size_t>(I / kFp8Tile));
    return fp8_activation_qdq(xhat.data(), act_scales.data(), x, I, kFp8Tile);
}

} // namespace

int fp8_matvec(float *y, const uint8_t *w, const uint8_t *scales, int O, int I,
               const float *x) {
    if (!y || !w || !scales || !x || bad_fp8_matvec(O, I))
        return -1;

    std::vector<float> xhat;
    if (qdq_xhat(xhat, x, I) != 0)
        return -1;
    fp8_matvec_xhat(y, w, scales, O, I, xhat.data());
    return 0;
}

int fp8_dual_matvec(float *ya, float *yb, const uint8_t *wa, const uint8_t *sa,
                    const uint8_t *wb, const uint8_t *sb, int O, int I, const float *x) {
    if (!ya || !yb || !wa || !sa || !wb || !sb || !x || bad_fp8_matvec(O, I))
        return -1;

    std::vector<float> xhat;
    if (qdq_xhat(xhat, x, I) != 0)
        return -1;
    fp8_matvec_xhat(ya, wa, sa, O, I, xhat.data());
    fp8_matvec_xhat(yb, wb, sb, O, I, xhat.data());
    return 0;
}

} // namespace mvllm
