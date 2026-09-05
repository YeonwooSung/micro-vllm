#include "bf16.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace mvllm {

float bf16_round(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Leave Inf/NaN payloads alone aside from zeroing the low 16 bits.
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        const uint32_t tie = (bits >> 16) & 1u;
        bits += 0x7fffu + tie;
    }
    bits &= 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t bf16_encode(float value) {
    const float rounded = bf16_round(value);
    uint32_t bits = 0;
    std::memcpy(&bits, &rounded, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_decode(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out = 0.f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

void bf16_round_array(float *values, int count) {
    if (!values || count < 1)
        return;
    for (int i = 0; i < count; ++i)
        values[i] = bf16_round(values[i]);
}

int hadamard_bf16(float *values, int n) {
    if (!values || n < 1 || (n & (n - 1)) != 0)
        return -1;

    for (int width = 1; width < n; width *= 2) {
        for (int base = 0; base < n; base += 2 * width) {
            for (int i = 0; i < width; ++i) {
                const float left = values[base + i];
                const float right = values[base + width + i];
                values[base + i] = left + right;
                values[base + width + i] = left - right;
            }
        }
    }

    const float scale = 1.f / std::sqrt(static_cast<float>(n));
    for (int i = 0; i < n; ++i)
        values[i] = bf16_round(values[i] * scale);
    return 0;
}

} // namespace mvllm
