#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mvllm {

enum class MuxCmd : int { Unknown = 0, Submit, Stop, Cancel, Image };

enum class MuxRead : int {
    Ok = 1,
    Ignored = 0,
    NeedMore = -5,
    Eof = -1,
    BadFrame = -2,
    Nomem = -3,
    BadRequest = -4,
};

struct MuxWireProfile {
    size_t max_header_bytes = 4096;
    uint64_t max_payload_bytes = 1u << 24;
    uint64_t max_extension_bytes = 1u << 20;
    int max_tokens = 0; // 0 = no cap
    bool require_exact_lf = true;
    bool require_finite_sampling = false;
    bool allow_extension_bytes = true;
    bool allow_prefix_hint = true;
};

struct MuxCommand {
    MuxCmd kind = MuxCmd::Unknown;
    uint64_t id = 0;
    int slot = 0;
    uint64_t payload_bytes = 0;
    int max_tokens = 0;
    float temperature = 0.f;
    float top_p = 0.f;
    uint64_t extension_bytes = 0;
    int prefix_bytes = 0;
    int grid_h = 0, grid_w = 0;
    std::vector<uint8_t> payload;    // payload_bytes
    std::vector<uint8_t> extension;  // extension_bytes
};

// Parse one command from buf[0..n). On Ok/Ignored/BadRequest/BadFrame set *consumed.
// NeedMore: *consumed = 0. Eof only if n==0.
// SUBMIT body is payload_bytes + extension_bytes + 1 terminator (no hole on the wire).
// IMAGE body is payload_bytes + LF, or CRLF.
MuxRead mux_parse_command(const uint8_t *buf, size_t n, const MuxWireProfile &profile,
                          MuxCommand &out, size_t *consumed);

} // namespace mvllm
