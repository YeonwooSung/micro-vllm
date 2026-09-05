#include "mux_submit.hpp"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace mvllm {
namespace {

bool is_ht(char c) { return c == ' ' || c == '\t'; }

bool at_eol(char c) { return c == '\0' || c == '\n' || c == '\r'; }

void skip_ht(const char *&p) {
    while (is_ht(*p))
        ++p;
}

bool take_token(const char *&p, const char *&beg, size_t &n) {
    skip_ht(p);
    if (at_eol(*p))
        return false;
    beg = p;
    while (*p && !is_ht(*p) && *p != '\n' && *p != '\r')
        ++p;
    n = static_cast<size_t>(p - beg);
    return n > 0;
}

bool parse_u64(const char *b, size_t n, uint64_t &out) {
    if (n == 0 || n >= 32 || b[0] == '-')
        return false;
    char tmp[32];
    std::memcpy(tmp, b, n);
    tmp[n] = 0;
    char *end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(tmp, &end, 10);
    if (errno || !end || *end)
        return false;
    out = static_cast<uint64_t>(v);
    return true;
}

bool parse_i32(const char *b, size_t n, int &out) {
    if (n == 0 || n >= 32)
        return false;
    char tmp[32];
    std::memcpy(tmp, b, n);
    tmp[n] = 0;
    char *end = nullptr;
    errno = 0;
    const long v = std::strtol(tmp, &end, 10);
    if (errno || !end || *end || v < INT32_MIN || v > INT32_MAX)
        return false;
    out = static_cast<int>(v);
    return true;
}

bool parse_f32(const char *b, size_t n, float &out) {
    if (n == 0)
        return false;
    char *end = nullptr;
    float v = 0.f;
    if (n < 128) {
        char tmp[128];
        std::memcpy(tmp, b, n);
        tmp[n] = 0;
        v = std::strtof(tmp, &end);
        if (!end || end == tmp || *end)
            return false;
    } else {
        std::string copy(b, n);
        v = std::strtof(copy.c_str(), &end);
        if (!end || end == copy.c_str() || *end)
            return false;
    }
    out = v;
    return true;
}

bool take_u64(const char *&p, uint64_t &out) {
    const char *b = nullptr;
    size_t n = 0;
    return take_token(p, b, n) && parse_u64(b, n, out);
}

bool take_i32(const char *&p, int &out) {
    const char *b = nullptr;
    size_t n = 0;
    return take_token(p, b, n) && parse_i32(b, n, out);
}

bool take_f32(const char *&p, float &out) {
    const char *b = nullptr;
    size_t n = 0;
    return take_token(p, b, n) && parse_f32(b, n, out);
}

bool submit_fields_ok(const MuxSubmit &s) {
    return s.id > 0 && s.bytes <= (16u << 20) && s.gbytes <= (1u << 20) && s.slot >= 0 &&
           s.max_tokens >= 1 && std::isfinite(s.temperature) && std::isfinite(s.top_p) &&
           s.temperature >= 0.f && s.temperature <= 2.f && s.top_p > 0.f && s.top_p <= 1.f;
}

} // namespace

int mux_submit_ext(const char *p, MuxSubmitExt &ext) {
    if (!p)
        return -1;
    int seen = 0;
    int seen_logprobs = 0;
    int seen_ids = 0;
    skip_ht(p);
    if (at_eol(*p))
        return 0;
    while (*p && !at_eol(*p)) {
        char key[16];
        int kn = 0;
        int val = 0;
        while (kn < 15 && ((*p >= 'a' && *p <= 'z') || *p == '_'))
            key[kn++] = *p++;
        key[kn] = 0;
        if (kn == 0 || *p != '=')
            return -1;
        ++p;
        const char *digits = p;
        while (*p >= '0' && *p <= '9' && val <= kMuxSubmitTopkMax)
            val = val * 10 + (*p++ - '0');
        if (p == digits)
            return -1;
        if (*p >= '0' && *p <= '9')
            return -1;
        if (*p && !is_ht(*p) && !at_eol(*p))
            return -1;
        if (kn == 8 && std::memcmp(key, "logprobs", 8) == 0) {
            if (seen_logprobs || val > kMuxSubmitTopkMax)
                return -1;
            seen_logprobs = 1;
            ext.logprobs = val;
        } else if (kn == 3 && std::memcmp(key, "ids", 3) == 0) {
            if (seen_ids || val > 1)
                return -1;
            seen_ids = 1;
            ext.tok_ids = val;
        } else {
            return -1;
        }
        seen = 1;
        skip_ht(p);
    }
    return seen;
}

bool mux_submit_parse(const char *line, MuxSubmit &out) {
    out = MuxSubmit{};
    if (!line || std::strncmp(line, "SUBMIT", 6) != 0)
        return false;
    const char *p = line + 6;
    if (!is_ht(*p))
        return false;

    if (!take_u64(p, out.id) || !take_i32(p, out.slot) || !take_u64(p, out.bytes) ||
        !take_i32(p, out.max_tokens) || !take_f32(p, out.temperature) || !take_f32(p, out.top_p))
        return false;

    skip_ht(p);
    if (at_eol(*p))
        return submit_fields_ok(out);

    if (!take_u64(p, out.gbytes))
        return false;

    skip_ht(p);
    if (at_eol(*p))
        return submit_fields_ok(out);

    MuxSubmitExt ext;
    if (mux_submit_ext(p, ext) <= 0)
        return false;
    out.logprobs = ext.logprobs;
    out.tok_ids = ext.tok_ids;
    return submit_fields_ok(out);
}

int mux_ids_parse(const char *buf, size_t len, int *out, int cap, int vocab) {
    if (!buf || !out || cap < 1 || vocab < 1)
        return -1;
    size_t i = 0;
    int n = 0;
    while (i < len) {
        while (i < len && static_cast<unsigned char>(buf[i]) <= ' ')
            ++i;
        if (i >= len)
            break;
        if (n >= cap)
            return cap;
        long v = 0;
        const size_t d = i;
        while (i < len && buf[i] >= '0' && buf[i] <= '9' && v < vocab)
            v = v * 10 + (buf[i++] - '0');
        if (i == d || v >= vocab || (i < len && static_cast<unsigned char>(buf[i]) > ' '))
            return -1;
        out[n++] = static_cast<int>(v);
    }
    return n;
}

} // namespace mvllm
