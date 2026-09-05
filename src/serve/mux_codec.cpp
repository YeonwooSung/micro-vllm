#include "mux_codec.hpp"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mvllm {
namespace {

constexpr int kMaxFields = 10;

static MuxRead set_consumed(MuxRead rc, size_t n, size_t *consumed) {
    if (consumed) {
        if (rc == MuxRead::NeedMore || rc == MuxRead::Eof)
            *consumed = 0;
        else
            *consumed = n;
    }
    return rc;
}

static bool parse_i32(std::string_view text, int *value) {
    if (text.empty())
        return false;
    char tmp[32];
    if (text.size() >= sizeof(tmp))
        return false;
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

static bool parse_u64(std::string_view text, uint64_t *value) {
    if (text.empty() || text[0] == '-')
        return false;
    char tmp[32];
    if (text.size() >= sizeof(tmp))
        return false;
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

static bool parse_f32(std::string_view text, float *value) {
    if (text.empty())
        return false;
    char tmp[128];
    char *end = nullptr;
    float parsed = 0.f;
    if (text.size() < sizeof(tmp)) {
        std::memcpy(tmp, text.data(), text.size());
        tmp[text.size()] = 0;
        parsed = std::strtof(tmp, &end);
        if (!end || end == tmp || *end)
            return false;
    } else {
        // Rare: a single header token longer than the stack buffer.
        std::string copy(text);
        parsed = std::strtof(copy.c_str(), &end);
        if (!end || end == copy.c_str() || *end)
            return false;
    }
    *value = parsed;
    return true;
}

static MuxCmd classify_name(std::string_view name) {
    if (name.size() == 6 && name == "SUBMIT")
        return MuxCmd::Submit;
    if (name.size() == 4 && name == "STOP")
        return MuxCmd::Stop;
    if (name.size() == 6 && name == "CANCEL")
        return MuxCmd::Cancel;
    if (name.size() == 5 && name == "IMAGE")
        return MuxCmd::Image;
    return MuxCmd::Unknown;
}

static void classify_line(std::string_view line, MuxCommand &command) {
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t'))
        ++i;
    const size_t name_b = i;
    while (i < line.size() && line[i] != ' ' && line[i] != '\t')
        ++i;
    command.kind = classify_name(line.substr(name_b, i - name_b));
    if (command.kind == MuxCmd::Unknown)
        return;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t'))
        ++i;
    const size_t id_b = i;
    while (i < line.size() && line[i] != ' ' && line[i] != '\t')
        ++i;
    if (i > id_b) {
        uint64_t id = 0;
        if (parse_u64(line.substr(id_b, i - id_b), &id))
            command.id = id;
    }
}

static int split_fields(std::string_view line, std::string_view *fields, int cap) {
    int nfields = 0;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t'))
            ++i;
        if (i >= line.size())
            break;
        if (nfields == cap)
            return -1;
        size_t j = i;
        while (j < line.size() && line[j] != ' ' && line[j] != '\t')
            ++j;
        fields[nfields++] = line.substr(i, j - i);
        i = j;
    }
    return nfields;
}

static bool copy_bytes(const uint8_t *src, size_t n, std::vector<uint8_t> &dst) {
    try {
        dst.assign(src, src + n);
    } catch (const std::bad_alloc &) {
        return false;
    } catch (const std::length_error &) {
        return false;
    }
    return true;
}

} // namespace

MuxRead mux_parse_command(const uint8_t *buf, size_t n, const MuxWireProfile &profile,
                          MuxCommand &out, size_t *consumed) {
    if (consumed)
        *consumed = 0;
    out = MuxCommand{};
    if (n == 0)
        return MuxRead::Eof;
    if (!buf)
        return MuxRead::BadFrame;

    const size_t maximum = profile.max_header_bytes ? profile.max_header_bytes : 4096;
    if (maximum == std::numeric_limits<size_t>::max())
        return MuxRead::Nomem;

    const void *nl = std::memchr(buf, '\n', n);
    if (!nl)
        return MuxRead::NeedMore;

    const size_t nl_pos = static_cast<size_t>(static_cast<const uint8_t *>(nl) - buf);
    const size_t header_span = nl_pos + 1;
    if (nl_pos > maximum) {
        classify_line(std::string_view(reinterpret_cast<const char *>(buf), maximum), out);
        return set_consumed(MuxRead::BadRequest, header_span, consumed);
    }

    size_t hlen = nl_pos;
    if (hlen && buf[hlen - 1] == '\r')
        --hlen;
    const std::string_view line(reinterpret_cast<const char *>(buf), hlen);

    MuxCommand cmd;
    classify_line(line, cmd);

    std::string_view fields[kMaxFields];
    const int nfields = split_fields(line, fields, kMaxFields);
    if (nfields < 0)
        return set_consumed(MuxRead::BadRequest, header_span, consumed);
    if (nfields == 0)
        return set_consumed(MuxRead::Ignored, header_span, consumed);

    cmd.kind = classify_name(fields[0]);
    if (cmd.kind == MuxCmd::Unknown)
        return set_consumed(MuxRead::Ignored, header_span, consumed);

    if (nfields < 2 || !parse_u64(fields[1], &cmd.id)) {
        out = cmd;
        return set_consumed(MuxRead::BadRequest, header_span, consumed);
    }

    if (cmd.kind == MuxCmd::Stop || cmd.kind == MuxCmd::Cancel) {
        if (nfields != 2) {
            out = cmd;
            return set_consumed(MuxRead::BadRequest, header_span, consumed);
        }
        out = cmd;
        return set_consumed(MuxRead::Ok, header_span, consumed);
    }

    const size_t body = header_span;
    const size_t avail = n - body;

    if (cmd.kind == MuxCmd::Image) {
        if (nfields != 5 || !parse_u64(fields[2], &cmd.payload_bytes) ||
            !parse_i32(fields[3], &cmd.grid_h) || !parse_i32(fields[4], &cmd.grid_w) ||
            cmd.payload_bytes > profile.max_payload_bytes || cmd.grid_h < 1 || cmd.grid_w < 1) {
            out = cmd;
            return set_consumed(MuxRead::BadRequest, header_span, consumed);
        }
        if (cmd.payload_bytes > static_cast<uint64_t>(avail))
            return MuxRead::NeedMore;
        const size_t pay = static_cast<size_t>(cmd.payload_bytes);
        if (avail - pay < 1)
            return MuxRead::NeedMore;
        const uint8_t term = buf[body + pay];
        size_t term_n = 1;
        if (term == '\r') {
            if (avail - pay < 2)
                return MuxRead::NeedMore;
            if (buf[body + pay + 1] != '\n')
                return set_consumed(MuxRead::BadFrame, body + pay + 2, consumed);
            term_n = 2;
        } else if (term != '\n') {
            return set_consumed(MuxRead::BadFrame, body + pay + 1, consumed);
        }
        if (pay && !copy_bytes(buf + body, pay, cmd.payload))
            return MuxRead::Nomem;
        out = std::move(cmd);
        return set_consumed(MuxRead::Ok, body + pay + term_n, consumed);
    }

    const int min_fields = 7;
    const int max_fields = min_fields + (profile.allow_extension_bytes ? 1 : 0) +
                           (profile.allow_prefix_hint ? 1 : 0);
    if (nfields < min_fields || nfields > max_fields ||
        (nfields >= 8 && !profile.allow_extension_bytes) ||
        (nfields >= 9 && (!profile.allow_prefix_hint || !profile.allow_extension_bytes))) {
        out = cmd;
        return set_consumed(MuxRead::BadRequest, header_span, consumed);
    }

    if (!parse_i32(fields[2], &cmd.slot) || !parse_u64(fields[3], &cmd.payload_bytes) ||
        !parse_i32(fields[4], &cmd.max_tokens) || !parse_f32(fields[5], &cmd.temperature) ||
        !parse_f32(fields[6], &cmd.top_p) ||
        (nfields >= 8 && !parse_u64(fields[7], &cmd.extension_bytes)) ||
        (nfields >= 9 && !parse_i32(fields[8], &cmd.prefix_bytes)) ||
        cmd.payload_bytes > profile.max_payload_bytes ||
        cmd.extension_bytes > profile.max_extension_bytes || cmd.max_tokens < 1 ||
        (profile.max_tokens && cmd.max_tokens > profile.max_tokens) ||
        (profile.require_finite_sampling &&
         (!std::isfinite(cmd.temperature) || !std::isfinite(cmd.top_p)))) {
        out = cmd;
        return set_consumed(MuxRead::BadRequest, header_span, consumed);
    }

    if (cmd.payload_bytes > std::numeric_limits<size_t>::max() - 1 ||
        cmd.extension_bytes > std::numeric_limits<size_t>::max() - 1 -
                                  static_cast<size_t>(cmd.payload_bytes))
        return MuxRead::Nomem;

    const uint64_t need = cmd.payload_bytes + cmd.extension_bytes;
    if (need < cmd.payload_bytes)
        return MuxRead::Nomem;
    if (need > static_cast<uint64_t>(avail))
        return MuxRead::NeedMore;
    const size_t pay = static_cast<size_t>(cmd.payload_bytes);
    const size_t ext = static_cast<size_t>(cmd.extension_bytes);
    if (avail - pay - ext < 1)
        return MuxRead::NeedMore;

    const uint8_t term = buf[body + pay + ext];
    if (profile.require_exact_lf && term != '\n')
        return set_consumed(MuxRead::BadFrame, body + pay + ext + 1, consumed);

    if (pay && !copy_bytes(buf + body, pay, cmd.payload))
        return MuxRead::Nomem;
    if (ext && !copy_bytes(buf + body + pay, ext, cmd.extension))
        return MuxRead::Nomem;
    out = std::move(cmd);
    return set_consumed(MuxRead::Ok, body + pay + ext + 1, consumed);
}

} // namespace mvllm
