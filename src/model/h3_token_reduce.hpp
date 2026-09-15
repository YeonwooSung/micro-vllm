#pragma once

#include "h3_layout.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {

constexpr int kH3DitBlocks = 50;

// Horizontal pair-pool of target video tokens (spatial W halved, odd last column kept).
struct H3TokenReduce {
    bool enabled = false;
    unsigned begin = 4, end = 30;
    unsigned early_steps = 10, early_end = 40;
    float scale = 1.f;
    int latent_t = 0, latent_h = 0, latent_w = 0;
    uint32_t spatial_h = 0, spatial_w = 0, reduced_w = 0;
    uint32_t video_target_start = 0, video_rows = 0, sequence = 0;
    uint32_t reduced_video_rows = 0, reduced_sequence = 0, baseline_rows = 0;
};

// enabled=false → cfg.enabled=false and true (no-op success).
// blocks_range / early / scale_text may be null or empty (defaults).
bool h3_token_reduce_configure(H3TokenReduce &cfg, bool enabled, int latent_t, int latent_h,
                               int latent_w, uint32_t video_target_start, uint32_t video_rows,
                               uint32_t sequence, const char *blocks_range, const char *early,
                               const char *scale_text, std::string &err);

void h3_token_pool_sources(const H3TokenReduce &cfg, uint32_t reduced_row, uint32_t *first,
                           uint32_t *second);
uint32_t h3_token_reduced_parent(const H3TokenReduce &cfg, uint32_t full_row);

// Mean of two hidden-wide rows. Identical pointers copy; otherwise out = 0.5 * (a + b).
void h3_token_pool_mean(const float *a, const float *b, float *out, int hidden);

// UINT32_MAX when the reduced row is not a paired video token.
uint32_t h3_token_reduce_baseline_index(const H3TokenReduce &cfg, uint32_t reduced_row);

// Snapshot full → original, pair-pool into reduced, and store paired video baselines.
void h3_token_reduce_enter(const H3TokenReduce &cfg, const float *full, float *reduced,
                           float *original, float *baseline, int hidden);

// Restore full = original + scale * (reduced[parent] - baseline). Prefix rows copy reduced.
void h3_token_reduce_leave(const H3TokenReduce &cfg, const float *reduced, const float *original,
                           const float *baseline, float *full, int hidden);

// reduced_map[row] = full_map[first of pair].
void h3_token_reduce_row_map(const H3TokenReduce &cfg, const uint32_t *full_map,
                             uint32_t *reduced_map);

// RoPE tables for the reduced sequence (pair-averaged positions).
void h3_token_reduce_rope_tables(const H3TokenReduce &cfg, const H3Layout &layout,
                                 const float *inv_freq, float spatial_scale,
                                 std::vector<float> &cos, std::vector<float> &sin);

} // namespace mvllm
