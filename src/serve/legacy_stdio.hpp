#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mvllm {

enum class LegacyCmd : int { None = 0, Reset, More, Prompt, Line };

struct LegacyCommand {
    LegacyCmd kind = LegacyCmd::None;
    uint64_t payload_bytes = 0; // PROMPT byte count (payload not consumed here)
    int max_tokens = 0;
    float temperature = 0.f;
    float top_p = 1.f;
    int slot = 0;
    bool have_slot = false;
};

// Parse one stdin header line with trailing LF/CRLF already stripped.
// "\x02RESET" → Reset
// "\x02MORE" → More
// "\x02PROMPT <bytes> <max_tokens> <temperature> <top_p> [kv_slot]" → Prompt
// any other non-empty line → Line (interactive prompt; payload_bytes unused)
// empty line → false, err="empty"
// Bad PROMPT header (not enough fields, non-numeric, bytes > 16MiB, max_tokens < 1,
// temperature not in [0,2], top_p not in (0,1], slot < 0) → false, err="bad prompt"
bool legacy_parse_line(const char *line, size_t n, LegacyCommand &out, std::string &err);
inline bool legacy_parse_line(const std::string &line, LegacyCommand &out, std::string &err) {
    return legacy_parse_line(line.data(), line.size(), out, err);
}

// Official sentinels, always LF-terminated, no CR.
std::string legacy_format_ready(); // "\x01\x01READY\x01\x01\n"
std::string legacy_format_end();   // "\x01\x01END\x01\x01\n"
// STAT emitted tok_s hit_pct rss_gb   — same field widths as mux STAT extras:
// "STAT %d %.2f %.1f %.2f\n"
std::string legacy_format_stat(int emitted, double tok_s = 0, double hit_pct = 0,
                               double rss_gb = 0);

} // namespace mvllm
