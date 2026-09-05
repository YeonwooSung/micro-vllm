#include "utf8.hpp"

namespace mvllm {

int utf8_next(const unsigned char *s, int len, int i, uint32_t *cp) {
    if (!s || !cp) {
        if (cp)
            *cp = 0;
        return 0;
    }
    if (i < 0 || i >= len) {
        *cp = 0;
        return 0;
    }

    const unsigned char lead = s[i];
    const int left = len - i;

    if (lead < 0x80) {
        *cp = lead;
        return 1;
    }
    if ((lead >> 5) == 0x06 && left >= 2) {
        *cp = (static_cast<uint32_t>(lead & 0x1f) << 6) | static_cast<uint32_t>(s[i + 1] & 0x3f);
        return 2;
    }
    if ((lead >> 4) == 0x0e && left >= 3) {
        *cp = (static_cast<uint32_t>(lead & 0x0f) << 12) |
              (static_cast<uint32_t>(s[i + 1] & 0x3f) << 6) | static_cast<uint32_t>(s[i + 2] & 0x3f);
        return 3;
    }
    if ((lead >> 3) == 0x1e && left >= 4) {
        *cp = (static_cast<uint32_t>(lead & 0x07) << 18) |
              (static_cast<uint32_t>(s[i + 1] & 0x3f) << 12) |
              (static_cast<uint32_t>(s[i + 2] & 0x3f) << 6) | static_cast<uint32_t>(s[i + 3] & 0x3f);
        return 4;
    }

    *cp = lead;
    return 1;
}

int utf8_put(char *o, uint32_t cp) {
    if (!o)
        return 0;

    if (cp < 0x80u) {
        o[0] = static_cast<char>(cp);
        return 1;
    }
    if (cp < 0x800u) {
        o[0] = static_cast<char>(0xc0u | (cp >> 6));
        o[1] = static_cast<char>(0x80u | (cp & 0x3fu));
        return 2;
    }
    if (cp < 0x10000u) {
        o[0] = static_cast<char>(0xe0u | (cp >> 12));
        o[1] = static_cast<char>(0x80u | ((cp >> 6) & 0x3fu));
        o[2] = static_cast<char>(0x80u | (cp & 0x3fu));
        return 3;
    }
    o[0] = static_cast<char>(0xf0u | (cp >> 18));
    o[1] = static_cast<char>(0x80u | ((cp >> 12) & 0x3fu));
    o[2] = static_cast<char>(0x80u | ((cp >> 6) & 0x3fu));
    o[3] = static_cast<char>(0x80u | (cp & 0x3fu));
    return 4;
}

int utf8_decode_all(const unsigned char *s, int len, uint32_t *out, int cap) {
    if (!s || !out || len <= 0 || cap <= 0)
        return 0;

    int n = 0;
    int i = 0;
    while (i < len && n < cap) {
        uint32_t cp = 0;
        const int k = utf8_next(s, len, i, &cp);
        if (k <= 0)
            break;
        out[n++] = cp;
        i += k;
    }
    return n;
}

std::string utf8_encode_cp(uint32_t cp) {
    char buf[4];
    const int n = utf8_put(buf, cp);
    if (n <= 0)
        return {};
    return std::string(buf, buf + n);
}

} // namespace mvllm
