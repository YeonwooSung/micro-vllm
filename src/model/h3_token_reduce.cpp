#include "h3_token_reduce.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

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

} // namespace mvllm
