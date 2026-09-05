#pragma once

#include "h3_layout.hpp"

#include <cstdint>
#include <vector>

namespace mvllm {

constexpr int kH3MaxSteps = 1000;
constexpr int kH3DitModalities = 3;
constexpr int kH3TimeInput = 256;
constexpr float kH3VideoSigmaShift = 12.f;
// Audio default is kH3AudioSigmaShift (3) in h3_audio_vae.hpp.

// Independent video/audio sigma grids. video/audio have size steps+1 (terminal 0).
struct H3SigmaSchedule {
    int steps = 0;
    std::vector<float> video;
    std::vector<float> audio;
};

bool h3_schedule_build(int steps, H3SigmaSchedule &out,
                       float video_shift = kH3VideoSigmaShift,
                       float audio_shift = 3.f);
bool h3_serving_schedule_build(int evaluations, H3SigmaSchedule &out,
                               float video_shift = kH3VideoSigmaShift,
                               float audio_shift = 3.f);

double h3_time_shift_sigma(double sigma, double from_shift, double to_shift);
double h3_time_shift_slope(double sigma, double from_shift, double to_shift);

// Host timestep row map + sinusoidal time features [time_rows, 256].
class H3DitSchedule {
public:
    bool prepare(const H3SigmaSchedule &sigmas, bool visual_condition, bool audio_condition);
    int steps() const;
    uint32_t time_rows() const;
    uint32_t video_row(int step) const;
    uint32_t audio_row(int step) const;
    uint32_t visual_condition_row(int step) const;
    uint32_t audio_condition_row(int step) const;
    const std::vector<float> &time_features() const;
    bool row_map(int step, const H3Layout &layout, const uint8_t *text_tags, int text_tag_count,
                 uint32_t *rows, int row_count) const;

private:
    int steps_ = 0;
    uint32_t time_rows_ = 0;
    std::vector<uint32_t> video_rows_;
    std::vector<uint32_t> audio_rows_;
    std::vector<uint32_t> visual_condition_rows_;
    std::vector<uint32_t> audio_condition_rows_;
    std::vector<float> time_features_;
};

// RES multistep. n is the element count. Returns 1 on success, 0 on failure.
int h3_res_step(float *output, const float *sample, const float *denoised,
                const float *old_denoised, int n, const float *sigmas, int step, int total_steps);

// mod is [time_rows, 3*6*hidden] row-major F32 AdaLN (same as h3_adaln_mod).
// Mean |mod| of slots 2 and 5 over modalities and hidden. -1 on bad args.
double h3_adaln_gate_score(const float *mod, int time_rows, int hidden);

// active[i]=0 when scores[i] < min_score or scores[i] < 0. n_blocks must be >0.
// Returns how many stay active, or -1 on bad args.
int h3_dit_prune_blocks(uint8_t *active, const double *scores, int n_blocks, double min_score);

} // namespace mvllm
