#include "legacy_stdio.hpp"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

namespace mvllm {
namespace {

constexpr uint64_t kMaxPromptBytes = 1ull << 24; // 16 MiB
constexpr char kReset[] = "\x02RESET";
constexpr char kMore[] = "\x02MORE";
constexpr char kPrompt[] = "\x02PROMPT";
constexpr size_t kResetN = 6;
constexpr size_t kMoreN = 5;
constexpr size_t kPromptN = 7;

bool parse_i32(std::string_view text, int *value) {
    if (text.empty() || text.size() >= 32)
        return false;
    char tmp[32];
    std::memcpy(tmp, text.data(), text.size());
    tmp[text.size()] = 0;
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(tmp, &end, 10);
    if (errno || !end || *end || parsed < INT32_MIN || parsed > INT32_MAX)
        return false;
    *value = static_cast<int>(parsed);
    return true;
}

bool parse_u64(std::string_view text, uint64_t *value) {
    if (text.empty() || text[0] == '-' || text.size() >= 32)
        return false;
    char tmp[32];
    std::memcpy(tmp, text.data(), text.size());
    tmp[text.size()] = 0;
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(tmp, &end, 10);
    if (errno || !end || *end)
        return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
}

bool parse_f32(std::string_view text, float *value) {
    if (text.empty())
        return false;
    char *end = nullptr;
    float parsed = 0.f;
    if (text.size() < 128) {
        char tmp[128];
        std::memcpy(tmp, text.data(), text.size());
        tmp[text.size()] = 0;
        parsed = std::strtof(tmp, &end);
        if (!end || end == tmp || *end)
            return false;
    } else {
        std::string copy(text);
        parsed = std::strtof(copy.c_str(), &end);
        if (!end || end == copy.c_str() || *end)
            return false;
    }
    *value = parsed;
    return true;
}

int split_ws(std::string_view line, std::string_view *fields, int cap) {
    int nfields = 0;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t'))
            ++i;
        if (i >= line.size())
            break;
        size_t j = i;
        while (j < line.size() && line[j] != ' ' && line[j] != '\t')
            ++j;
        if (nfields < cap)
            fields[nfields] = line.substr(i, j - i);
        ++nfields;
        i = j;
    }
    return nfields;
}

bool parse_prompt(std::string_view rest, LegacyCommand &out) {
    std::string_view fields[5];
    const int nfields = split_ws(rest, fields, 5);
    if (nfields < 4)
        return false;

    uint64_t bytes = 0;
    int max_tokens = 0;
    float temperature = 0.f;
    float top_p = 0.f;
    int slot = 0;
    if (!parse_u64(fields[0], &bytes) || !parse_i32(fields[1], &max_tokens) ||
        !parse_f32(fields[2], &temperature) || !parse_f32(fields[3], &top_p))
        return false;
    const bool have_slot = nfields >= 5;
    if (have_slot && !parse_i32(fields[4], &slot))
        return false;
    if (bytes > kMaxPromptBytes || max_tokens < 1 || temperature < 0.f || temperature > 2.f ||
        top_p <= 0.f || top_p > 1.f || (have_slot && slot < 0))
        return false;

    out.kind = LegacyCmd::Prompt;
    out.payload_bytes = bytes;
    out.max_tokens = max_tokens;
    out.temperature = temperature;
    out.top_p = top_p;
    out.slot = have_slot ? slot : 0;
    out.have_slot = have_slot;
    return true;
}

} // namespace

bool legacy_parse_line(const char *line, size_t n, LegacyCommand &out, std::string &err) {
    out = LegacyCommand{};
    err.clear();
    if (!line || n == 0) {
        err = "empty";
        return false;
    }

    if (n == kResetN && std::memcmp(line, kReset, kResetN) == 0) {
        out.kind = LegacyCmd::Reset;
        return true;
    }
    if (n == kMoreN && std::memcmp(line, kMore, kMoreN) == 0) {
        out.kind = LegacyCmd::More;
        return true;
    }

    // "\x02PROMPT" is 7 bytes; require a following space, then the fields.
    if (n >= kPromptN && std::memcmp(line, kPrompt, kPromptN) == 0) {
        if (n < kPromptN + 1 || line[kPromptN] != ' ' ||
            !parse_prompt(std::string_view(line + kPromptN + 1, n - kPromptN - 1), out)) {
            out = LegacyCommand{};
            err = "bad prompt";
            return false;
        }
        return true;
    }

    out.kind = LegacyCmd::Line;
    return true;
}

std::string legacy_format_ready() { return "\x01\x01READY\x01\x01\n"; }

std::string legacy_format_end() { return "\x01\x01END\x01\x01\n"; }

std::string legacy_format_stat(int emitted, double tok_s, double hit_pct, double rss_gb) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "STAT %d %.2f %.1f %.2f\n", emitted, tok_s, hit_pct, rss_gb);
    return buf;
}

} // namespace mvllm
