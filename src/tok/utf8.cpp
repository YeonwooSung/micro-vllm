#include "utf8.hpp"
#include "tok_unicode.h"

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

namespace {

struct UniRange {
    uint32_t lo;
    uint32_t hi;
};

// Inclusive [lo, hi], sorted, non-overlapping.
bool uni_in(uint32_t cp, const UniRange *t, int n) {
    int a = 0;
    int b = n - 1;
    while (a <= b) {
        const int m = a + (b - a) / 2;
        if (cp < t[m].lo)
            b = m - 1;
        else if (cp > t[m].hi)
            a = m + 1;
        else
            return true;
    }
    return false;
}

template <int N>
bool uni_in(uint32_t cp, const UniRange (&t)[N]) {
    return uni_in(cp, t, N);
}

bool uni_scalar(uint32_t cp) {
    return cp <= 0x10FFFFu && (cp < 0xD800u || cp > 0xDFFFu);
}

// Uppercase letters used by pretok case runs: ASCII, Latin-1, Greek, Cyrillic, fullwidth.
constexpr UniRange k_Lu[] = {
    {0x0041, 0x005A}, {0x00C0, 0x00D6}, {0x00D8, 0x00DE}, {0x0391, 0x03A1},
    {0x03A3, 0x03A9}, {0x0410, 0x042F}, {0xFF21, 0xFF3A},
};

// Lowercase letters used by pretok case runs: ASCII, Latin-1, Greek, Cyrillic, fullwidth.
constexpr UniRange k_Ll[] = {
    {0x0061, 0x007A}, {0x00DF, 0x00F6}, {0x00F8, 0x00FF}, {0x03B1, 0x03C9},
    {0x0430, 0x044F}, {0xFF41, 0xFF5A},
};

// Combining marks (Mn/Mc/Me): dedicated blocks + compact Hebrew/Arabic/Indic.
constexpr UniRange k_M[] = {
    {0x0300, 0x036F}, {0x0483, 0x0489}, {0x0591, 0x05BD}, {0x05BF, 0x05BF},
    {0x05C1, 0x05C2}, {0x05C4, 0x05C5}, {0x05C7, 0x05C7}, {0x0610, 0x061A},
    {0x064B, 0x065F}, {0x0670, 0x0670}, {0x06D6, 0x06DC}, {0x06DF, 0x06E4},
    {0x06E7, 0x06E8}, {0x06EA, 0x06ED}, {0x0730, 0x074A}, {0x07A6, 0x07B0},
    {0x07EB, 0x07F3}, {0x0900, 0x0903}, {0x093A, 0x093C}, {0x093E, 0x094F},
    {0x0951, 0x0957}, {0x0962, 0x0963}, {0x1AB0, 0x1AFF}, {0x1DC0, 0x1DFF},
    {0x20D0, 0x20FF}, {0x2DE0, 0x2DFF}, {0x302A, 0x302F}, {0x3099, 0x309A},
    {0xFE20, 0xFE2F},
};

// Punctuation (P*): ASCII, a few Latin-1, General Punctuation, CJK (skip U+3000),
// fullwidth. Compact inclusive ranges — not the full Unicode P set.
constexpr UniRange k_P[] = {
    {0x0021, 0x002F}, {0x003A, 0x0040}, {0x005B, 0x0060}, {0x007B, 0x007E},
    {0x00A1, 0x00A1}, {0x00B7, 0x00B7}, {0x00BF, 0x00BF}, {0x2010, 0x2027},
    {0x2030, 0x205E}, {0x3001, 0x303F}, {0xFF01, 0xFF0F}, {0xFF1A, 0xFF20},
    {0xFF3B, 0xFF40}, {0xFF5B, 0xFF65},
};

} // namespace

bool uni_is_L(uint32_t cp) {
    return uni_scalar(cp) && is_L(cp);
}

bool uni_is_N(uint32_t cp) {
    return uni_scalar(cp) && is_N(cp);
}

bool uni_is_S(uint32_t cp) {
    return uni_scalar(cp) && is_S(cp);
}

bool uni_is_Lu(uint32_t cp) {
    return uni_scalar(cp) && uni_in(cp, k_Lu);
}

uint32_t uni_to_lower(uint32_t cp) {
    if (uni_is_Ll(cp))
        return cp;
    // Simple 1:1 fold (not full Unicode casefold).
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 0xFF21u && cp <= 0xFF3Au) ||
        (cp >= 0x00C0u && cp <= 0x00D6u) || (cp >= 0x00D8u && cp <= 0x00DEu) ||
        (cp >= 0x0391u && cp <= 0x03A9u && cp != 0x03A2u) ||
        (cp >= 0x0410u && cp <= 0x042Fu))
        return cp + 0x20u;
    return cp;
}

uint32_t uni_to_upper(uint32_t cp) {
    if (uni_is_Lu(cp))
        return cp;
    // Simple 1:1 fold (not full Unicode casefold). Inverse of uni_to_lower.
    if ((cp >= 'a' && cp <= 'z') || (cp >= 0xFF41u && cp <= 0xFF5Au) ||
        (cp >= 0x00E0u && cp <= 0x00F6u) || (cp >= 0x00F8u && cp <= 0x00FEu) ||
        (cp >= 0x03B1u && cp <= 0x03C9u && cp != 0x03C2u) ||
        (cp >= 0x0430u && cp <= 0x044Fu))
        return cp - 0x20u;
    return cp;
}

bool uni_is_Ll(uint32_t cp) {
    return uni_scalar(cp) && uni_in(cp, k_Ll);
}

bool uni_is_M(uint32_t cp) {
    return uni_scalar(cp) && uni_in(cp, k_M);
}

bool uni_is_P(uint32_t cp) {
    return uni_scalar(cp) && uni_in(cp, k_P);
}

} // namespace mvllm
