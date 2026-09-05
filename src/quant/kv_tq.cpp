#include "kv_tq.hpp"

#include <cmath>
#include <cstring>

namespace mvllm {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTinyNorm = 1e-35f;
constexpr float kHugeNorm = 3.4e38f;

constexpr float kQ4Lev[16] = {-2.733f, -2.069f, -1.618f, -1.256f, -0.942f, -0.657f,
                              -0.388f, -0.128f, 0.128f,  0.388f,  0.657f,  0.942f,
                              1.256f,  1.618f,  2.069f,  2.733f};

bool power_of_two_n(int n) {
    return n >= 2 && n <= kKvTqMaxN && (n & (n - 1)) == 0;
}

int bits_l1(int bits) { return bits; }
int bits_l2(int bits) { return bits > 1 ? bits - 1 : 1; }

uint32_t splitmix32(uint32_t x) {
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

float hadamard_sign(int i, uint32_t seed) {
    return (splitmix32(seed ^ static_cast<uint32_t>(i)) & 1u) ? -1.f : 1.f;
}

void rotate_row(const float *x, float *y, int n, uint32_t seed) {
    for (int i = 0; i < n; ++i)
        y[i] = x[i] * hadamard_sign(i, seed);
    kv_tq_fwht(y, n);
}

void unrotate_row(float *y, int n, uint32_t seed) {
    kv_tq_fwht(y, n);
    for (int i = 0; i < n; ++i)
        y[i] *= hadamard_sign(i, seed);
}

float row_l2(const float *src, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = static_cast<double>(src[i]);
        ss += v * v;
    }
    return static_cast<float>(std::sqrt(ss));
}

bool finite_pos_norm(float norm) { return (norm > kTinyNorm) && !(norm > kHugeNorm); }

bool finite_pos_radius(float radius) { return (radius > 0.f) && !(radius > kHugeNorm); }

void zero_dst(float *dst, int n) {
    if (n > 0)
        std::memset(dst, 0, static_cast<std::size_t>(n) * sizeof(float));
}

void write_bits(uint8_t *buf, int *pos, uint32_t val, int w) {
    for (int k = 0; k < w; ++k) {
        const int bit = (*pos) + k;
        if ((val >> k) & 1u)
            buf[bit >> 3] = static_cast<uint8_t>(buf[bit >> 3] | (1u << (bit & 7)));
    }
    *pos += w;
}

uint32_t read_bits(const uint8_t *buf, int *pos, int w) {
    uint32_t v = 0;
    for (int k = 0; k < w; ++k) {
        const int bit = (*pos) + k;
        if ((buf[bit >> 3] >> (bit & 7)) & 1u)
            v |= (1u << k);
    }
    *pos += w;
    return v;
}

uint32_t enc_uniform(float ang, int b) {
    const float u = (ang + kPi) * (0.5f / kPi);
    const int m = (1 << b) - 1;
    int q = static_cast<int>(std::floor(u * static_cast<float>(1 << b)));
    if (q < 0)
        q = 0;
    if (q > m)
        q = m;
    return static_cast<uint32_t>(q);
}

float dec_uniform(uint32_t q, int b) {
    return -kPi + (static_cast<float>(q) + 0.5f) * (2.f * kPi) / static_cast<float>(1 << b);
}

float half_width(int level) {
    const float s = static_cast<float>(1u << (level - 1));
    const float w = 1.5f / std::sqrt(s);
    const float cap = 0.25f * kPi;
    return w < cap ? w : cap;
}

uint32_t enc_centered(float ang, int b, int level) {
    const float w = half_width(level);
    const float lo = 0.25f * kPi - w;
    const float u = (ang - lo) / (2.f * w);
    const int m = (1 << b) - 1;
    int q = static_cast<int>(std::floor(u * static_cast<float>(1 << b)));
    if (q < 0)
        q = 0;
    if (q > m)
        q = m;
    return static_cast<uint32_t>(q);
}

float dec_centered(uint32_t q, int b, int level) {
    const float w = half_width(level);
    const float lo = 0.25f * kPi - w;
    return lo + (static_cast<float>(q) + 0.5f) * (2.f * w) / static_cast<float>(1 << b);
}

int nearest_q4(float v) {
    int best = 0;
    float bd = 1e30f;
    for (int k = 0; k < 16; ++k) {
        const float d = std::fabs(v - kQ4Lev[k]);
        if (d < bd) {
            bd = d;
            best = k;
        }
    }
    return best;
}

} // namespace

void kv_tq_fwht(float *a, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
            for (int j = i; j < i + len; ++j) {
                const float u = a[j];
                const float v = a[j + len];
                a[j] = u + v;
                a[j + len] = u - v;
            }
        }
    }
    const float inv = 1.f / std::sqrt(static_cast<float>(n));
    for (int i = 0; i < n; ++i)
        a[i] *= inv;
}

int kv_tq_row_bytes(int n, int bits) {
    const long total =
        static_cast<long>(n / 2) * bits_l1(bits) + static_cast<long>(n / 2 - 1) * bits_l2(bits);
    return static_cast<int>((total + 7) / 8);
}

float kv_tq_quant_row(const float *src, uint8_t *dst, int n, int bits) {
    const int rb = kv_tq_row_bytes(n, bits);
    if (rb > 0)
        std::memset(dst, 0, static_cast<std::size_t>(rb));
    if (!power_of_two_n(n))
        return 0.f;

    const float norm = row_l2(src, n);
    if (!finite_pos_norm(norm))
        return 0.f;

    float r[kKvTqMaxN];
    rotate_row(src, r, n, kKvTqSeed);

    const int b1 = bits_l1(bits);
    const int b2 = bits_l2(bits);
    int pos = 0;
    int rlen = n;
    for (int level = 1; rlen > 1; ++level) {
        const int half = rlen / 2;
        const int b = (level == 1) ? b1 : b2;
        for (int j = 0; j < half; ++j) {
            const float a = r[2 * j];
            const float c = r[2 * j + 1];
            const float ang = std::atan2(c, a);
            const uint32_t code =
                (level == 1) ? enc_uniform(ang, b) : enc_centered(ang, b, level);
            write_bits(dst, &pos, code, b);
            r[j] = std::hypot(a, c);
        }
        rlen = half;
    }
    return norm;
}

void kv_tq_dequant_row(const uint8_t *src, float radius, float *dst, int n, int bits) {
    if (!power_of_two_n(n)) {
        zero_dst(dst, n);
        return;
    }
    if (!finite_pos_radius(radius)) {
        zero_dst(dst, n);
        return;
    }

    const int b1 = bits_l1(bits);
    const int b2 = bits_l2(bits);
    int pos = 0;
    float ang[kKvTqMaxN];
    int off = 0;
    for (int level = 1, len = n / 2; len >= 1; ++level) {
        const int b = (level == 1) ? b1 : b2;
        for (int j = 0; j < len; ++j) {
            const uint32_t code = read_bits(src, &pos, b);
            ang[off + j] = (level == 1) ? dec_uniform(code, b) : dec_centered(code, b, level);
        }
        off += len;
        if (len == 1)
            break;
        len >>= 1;
    }

    float cur[kKvTqMaxN];
    float nxt[kKvTqMaxN];
    cur[0] = radius;
    for (int clen = 1; clen < n; clen *= 2) {
        const int aoff = n - 2 * clen;
        for (int j = 0; j < clen; ++j) {
            const float aa = ang[aoff + j];
            nxt[2 * j] = cur[j] * std::cos(aa);
            nxt[2 * j + 1] = cur[j] * std::sin(aa);
        }
        std::memcpy(cur, nxt, static_cast<std::size_t>(2 * clen) * sizeof(float));
    }
    std::memcpy(dst, cur, static_cast<std::size_t>(n) * sizeof(float));
    unrotate_row(dst, n, kKvTqSeed);
}

int kv_q4_row_bytes(int n) { return n / 2; }

float kv_q4_quant_row(const float *src, uint8_t *dst, int n) {
    const int rb = kv_q4_row_bytes(n);
    if (rb > 0)
        std::memset(dst, 0, static_cast<std::size_t>(rb));
    if (!power_of_two_n(n))
        return 0.f;

    const float norm = row_l2(src, n);
    if (!finite_pos_norm(norm))
        return 0.f;

    float y[kKvTqMaxN];
    rotate_row(src, y, n, kKvTqSeed);
    const float invstd = std::sqrt(static_cast<float>(n)) / norm;
    for (int i = 0; i < n; ++i) {
        const int c = nearest_q4(y[i] * invstd);
        dst[i >> 1] = static_cast<uint8_t>(dst[i >> 1] | (c << ((i & 1) * 4)));
    }
    return norm;
}

void kv_q4_dequant_row(const uint8_t *src, float radius, float *dst, int n) {
    if (!power_of_two_n(n)) {
        zero_dst(dst, n);
        return;
    }
    if (!finite_pos_radius(radius)) {
        zero_dst(dst, n);
        return;
    }
    const float stdv = radius / std::sqrt(static_cast<float>(n));
    for (int i = 0; i < n; ++i) {
        const int c = (src[i >> 1] >> ((i & 1) * 4)) & 0xF;
        dst[i] = kQ4Lev[c] * stdv;
    }
    unrotate_row(dst, n, kKvTqSeed);
}

} // namespace mvllm
