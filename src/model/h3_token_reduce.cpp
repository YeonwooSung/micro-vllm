#include "h3_token_reduce.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace mvllm {
namespace {

bool nonempty(const char *text) {
    return text != nullptr && text[0] != '\0';
}

bool parse_pair(const char *text, unsigned long *left, unsigned long *right) {
    if (!text || !left || !right)
        return false;
    char *colon = nullptr;
    const unsigned long first = std::strtoul(text, &colon, 10);
    if (colon == text || *colon != ':')
        return false;
    char *tail = nullptr;
    const unsigned long second = std::strtoul(colon + 1, &tail, 10);
    if (tail == colon + 1 || *tail != '\0')
        return false;
    *left = first;
    *right = second;
    return true;
}

bool parse_scale(const char *text, float *value) {
    if (!text || !value)
        return false;
    char *tail = nullptr;
    const float parsed = std::strtof(text, &tail);
    if (tail == text || *tail != '\0' || !std::isfinite(parsed) || parsed < 0.f || parsed > 2.f)
        return false;
    *value = parsed;
    return true;
}

} // namespace

bool h3_token_reduce_configure(H3TokenReduce &cfg, bool enabled, int latent_t, int latent_h,
                               int latent_w, uint32_t video_target_start, uint32_t video_rows,
                               uint32_t sequence, const char *blocks_range, const char *early,
                               const char *scale_text, std::string &err) {
    if (!enabled) {
        cfg.enabled = false;
        return true;
    }

    unsigned begin = 4;
    unsigned end = 30;
    if (nonempty(blocks_range)) {
        unsigned long parsed_begin = 0;
        unsigned long parsed_end = 0;
        if (!parse_pair(blocks_range, &parsed_begin, &parsed_end)) {
            err = "blocks range must be BEGIN:END";
            return false;
        }
        if (parsed_begin >= parsed_end || parsed_end > static_cast<unsigned long>(kH3DitBlocks)) {
            err = "block range requires 0 <= BEGIN < END <= 50";
            return false;
        }
        begin = static_cast<unsigned>(parsed_begin);
        end = static_cast<unsigned>(parsed_end);
    }

    unsigned early_steps = end < 40u ? 10u : 0u;
    unsigned early_end = end < 40u ? 40u : end;
    if (nonempty(early)) {
        if (std::strcmp(early, "0") == 0) {
            early_steps = 0;
            early_end = end;
        } else {
            unsigned long parsed_steps = 0;
            unsigned long parsed_end = 0;
            if (!parse_pair(early, &parsed_steps, &parsed_end)) {
                err = "early range must be STEPS:END";
                return false;
            }
            if (parsed_steps == 0 || parsed_steps > 1000ul ||
                parsed_end <= static_cast<unsigned long>(end) ||
                parsed_end > static_cast<unsigned long>(kH3DitBlocks)) {
                err = "early range requires STEPS > 0 and base END < END <= 50";
                return false;
            }
            early_steps = static_cast<unsigned>(parsed_steps);
            early_end = static_cast<unsigned>(parsed_end);
        }
    }

    float scale = 1.f;
    if (nonempty(scale_text) && !parse_scale(scale_text, &scale)) {
        err = "scale must be finite in [0, 2]";
        return false;
    }

    if (latent_t < 0 || latent_h < 0 || latent_w < 0) {
        err = "token reduction requires video to end the packed layout";
        return false;
    }

    const uint32_t spatial_h = static_cast<uint32_t>(latent_h) / 2u;
    const uint32_t spatial_w = static_cast<uint32_t>(latent_w) / 2u;
    const uint32_t reduced_w = (spatial_w + 1u) / 2u;
    const uint64_t reduced_video =
        static_cast<uint64_t>(static_cast<uint32_t>(latent_t)) * spatial_h * reduced_w;
    const uint64_t full_video =
        static_cast<uint64_t>(static_cast<uint32_t>(latent_t)) * spatial_h * spatial_w;
    const uint32_t max_u32 = std::numeric_limits<uint32_t>::max();
    if (spatial_h == 0 || spatial_w == 0 || full_video != video_rows ||
        static_cast<uint64_t>(video_target_start) + video_rows != sequence ||
        reduced_video > max_u32 || reduced_video > static_cast<uint64_t>(max_u32) - video_target_start) {
        err = "token reduction requires video to end the packed layout";
        return false;
    }

    const uint32_t reduced_video_rows = static_cast<uint32_t>(reduced_video);
    cfg.enabled = true;
    cfg.begin = begin;
    cfg.end = end;
    cfg.early_steps = early_steps;
    cfg.early_end = early_end;
    cfg.scale = scale;
    cfg.latent_t = latent_t;
    cfg.latent_h = latent_h;
    cfg.latent_w = latent_w;
    cfg.spatial_h = spatial_h;
    cfg.spatial_w = spatial_w;
    cfg.reduced_w = reduced_w;
    cfg.video_target_start = video_target_start;
    cfg.video_rows = video_rows;
    cfg.sequence = sequence;
    cfg.reduced_video_rows = reduced_video_rows;
    cfg.reduced_sequence = video_target_start + reduced_video_rows;
    cfg.baseline_rows = video_rows - reduced_video_rows;
    err.clear();
    return true;
}

void h3_token_pool_sources(const H3TokenReduce &cfg, uint32_t reduced_row, uint32_t *first,
                           uint32_t *second) {
    if (!first || !second)
        return;
    if (reduced_row < cfg.video_target_start || cfg.spatial_w == 0 || cfg.reduced_w == 0) {
        *first = reduced_row;
        *second = reduced_row;
        return;
    }
    const uint32_t local = reduced_row - cfg.video_target_start;
    const uint32_t source = cfg.video_target_start + (local / cfg.reduced_w) * cfg.spatial_w +
                            (local % cfg.reduced_w) * 2u;
    *first = source;
    const uint32_t col = (source - cfg.video_target_start) % cfg.spatial_w;
    *second = source + (col + 1u < cfg.spatial_w ? 1u : 0u);
}

uint32_t h3_token_reduced_parent(const H3TokenReduce &cfg, uint32_t full_row) {
    if (full_row < cfg.video_target_start || cfg.spatial_w == 0)
        return full_row;
    const uint32_t local = full_row - cfg.video_target_start;
    return cfg.video_target_start + (local / cfg.spatial_w) * cfg.reduced_w +
           (local % cfg.spatial_w) / 2u;
}

void h3_token_pool_mean(const float *a, const float *b, float *out, int hidden) {
    if (!a || !b || !out || hidden < 1)
        return;
    if (a == b) {
        if (out != a)
            std::memcpy(out, a, static_cast<size_t>(hidden) * sizeof(float));
        return;
    }
    for (int i = 0; i < hidden; ++i)
        out[i] = 0.5f * (a[i] + b[i]);
}

uint32_t h3_token_reduce_baseline_index(const H3TokenReduce &cfg, uint32_t reduced_row) {
    if (reduced_row < cfg.video_target_start || reduced_row >= cfg.reduced_sequence)
        return UINT32_MAX;
    uint32_t first = 0, second = 0;
    h3_token_pool_sources(cfg, reduced_row, &first, &second);
    if (first == second)
        return UINT32_MAX;
    uint32_t index = 0;
    for (uint32_t row = cfg.video_target_start; row < reduced_row; ++row) {
        uint32_t a = 0, b = 0;
        h3_token_pool_sources(cfg, row, &a, &b);
        if (a != b)
            ++index;
    }
    return index;
}

void h3_token_reduce_enter(const H3TokenReduce &cfg, const float *full, float *reduced,
                           float *original, float *baseline, int hidden) {
    if (!cfg.enabled || !full || !reduced || !original || hidden < 1)
        return;
    const size_t row_bytes = static_cast<size_t>(hidden) * sizeof(float);
    std::memcpy(original, full, static_cast<size_t>(cfg.sequence) * row_bytes);
    uint32_t baseline_row = 0;
    for (uint32_t row = 0; row < cfg.reduced_sequence; ++row) {
        uint32_t first = 0, second = 0;
        h3_token_pool_sources(cfg, row, &first, &second);
        const float *a = full + static_cast<size_t>(first) * hidden;
        const float *b = full + static_cast<size_t>(second) * hidden;
        float *out = reduced + static_cast<size_t>(row) * hidden;
        h3_token_pool_mean(a, b, out, hidden);
        if (baseline && row >= cfg.video_target_start && first != second) {
            std::memcpy(baseline + static_cast<size_t>(baseline_row) * hidden, out, row_bytes);
            ++baseline_row;
        }
    }
}

void h3_token_reduce_leave(const H3TokenReduce &cfg, const float *reduced, const float *original,
                           const float *baseline, float *full, int hidden) {
    if (!cfg.enabled || !reduced || !original || !full || hidden < 1)
        return;
    const size_t row_bytes = static_cast<size_t>(hidden) * sizeof(float);
    for (uint32_t row = 0; row < cfg.sequence; ++row) {
        const uint32_t parent = h3_token_reduced_parent(cfg, row);
        float *dst = full + static_cast<size_t>(row) * hidden;
        const float *src = reduced + static_cast<size_t>(parent) * hidden;
        if (row < cfg.video_target_start) {
            std::memcpy(dst, src, row_bytes);
            continue;
        }
        const uint32_t bidx = h3_token_reduce_baseline_index(cfg, parent);
        if (bidx == UINT32_MAX || !baseline) {
            std::memcpy(dst, src, row_bytes);
            continue;
        }
        const float *base = baseline + static_cast<size_t>(bidx) * hidden;
        const float *orig = original + static_cast<size_t>(row) * hidden;
        for (int i = 0; i < hidden; ++i)
            dst[i] = orig[i] + cfg.scale * (src[i] - base[i]);
    }
}

void h3_token_reduce_row_map(const H3TokenReduce &cfg, const uint32_t *full_map,
                             uint32_t *reduced_map) {
    if (!cfg.enabled || !full_map || !reduced_map)
        return;
    for (uint32_t row = 0; row < cfg.reduced_sequence; ++row) {
        uint32_t first = 0, second = 0;
        h3_token_pool_sources(cfg, row, &first, &second);
        (void)second;
        reduced_map[row] = full_map[first];
    }
}

void h3_token_reduce_rope_tables(const H3TokenReduce &cfg, const H3Layout &layout,
                                 const float *inv_freq, float spatial_scale,
                                 std::vector<float> &cos, std::vector<float> &sin) {
    const int seq = static_cast<int>(cfg.reduced_sequence);
    cos.assign(static_cast<size_t>(seq) * kH3RopeHalf, 1.f);
    sin.assign(static_cast<size_t>(seq) * kH3RopeHalf, 0.f);
    if (!cfg.enabled || !inv_freq || seq < 1 ||
        static_cast<uint32_t>(layout.positions.size()) < cfg.sequence)
        return;
    if (!(spatial_scale > 0.f))
        spatial_scale = 1.f;
    for (uint32_t row = 0; row < cfg.reduced_sequence; ++row) {
        uint32_t first = 0, second = 0;
        h3_token_pool_sources(cfg, row, &first, &second);
        const H3Position &pa = layout.positions[static_cast<size_t>(first)];
        const H3Position &pb = layout.positions[static_cast<size_t>(second)];
        const float axes[3] = {0.5f * (pa.t + pb.t), 0.5f * (pa.h + pb.h) * spatial_scale,
                               0.5f * (pa.w + pb.w) * spatial_scale};
        for (int axis = 0; axis < 3; ++axis) {
            for (int f = 0; f < kH3RopeFreqs; ++f) {
                const int idx = static_cast<int>(row) * kH3RopeHalf + axis * kH3RopeFreqs + f;
                const float ang = axes[axis] * inv_freq[f];
                cos[static_cast<size_t>(idx)] = std::cos(ang);
                sin[static_cast<size_t>(idx)] = std::sin(ang);
            }
        }
    }
}

} // namespace mvllm
