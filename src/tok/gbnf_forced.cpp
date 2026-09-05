#include "gbnf_forced.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {
namespace {

int unique_ok(const uint8_t *ok, int vocab) {
    int t = -1;
    int n = 0;
    for (int i = 0; i < vocab; ++i) {
        if (ok[i] != 1)
            continue;
        ++n;
        t = i;
        if (n > 1)
            return -1;
    }
    return n == 1 ? t : -1;
}

std::string byte_decode(int id) {
    if (id <= 0 || id > 255)
        return {};
    return std::string(1, static_cast<char>(id));
}

} // namespace

int gbnf_forced_tokens(Gbnf &g, const std::function<std::string(int)> &decode, int vocab, int *out,
                       int max_tokens) {
    if (!g.ready() || !decode || !out || vocab <= 0 || max_tokens <= 0)
        return 0;
    std::vector<uint8_t> ok(static_cast<size_t>(vocab));
    int n = 0;
    while (n < max_tokens) {
        g.allow_mask(decode, ok.data(), vocab);
        const int t = unique_ok(ok.data(), vocab);
        if (t < 0)
            break;
        const std::string text = decode(t);
        if (text.empty())
            break;
        if (!g.accept_bytes(text))
            break;
        out[n++] = t;
    }
    return n;
}

int gbnf_forced_bytes(Gbnf &g, char *out, int max_bytes) {
    if (!g.ready() || !out || max_bytes <= 0)
        return 0;
    uint8_t ok[256];
    int n = 0;
    while (n < max_bytes) {
        g.allow_mask(byte_decode, ok, 256);
        const int t = unique_ok(ok, 256);
        if (t <= 0 || t > 255)
            break;
        if (!g.accept_byte(static_cast<unsigned char>(t)))
            break;
        out[n++] = static_cast<char>(t);
    }
    return n;
}

} // namespace mvllm
