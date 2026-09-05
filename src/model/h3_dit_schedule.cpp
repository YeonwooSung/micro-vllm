#include "h3_dit_schedule.hpp"
#include "h3_adaln.hpp"

#include <cmath>
#include <cstdint>
#include <cstddef>

namespace mvllm {
namespace {

float shifted_sigma(int index, int steps, float shift) {
    const int base_index = (index * 1000) / steps;
    const float base = static_cast<float>(1000 - base_index) / 1000.f;
    return shift * base / (1.f + (shift - 1.f) * base);
}

float apply_shift(float base, float shift) {
    return shift * base / (1.f + (shift - 1.f) * base);
}

double phi1(double value) {
    return std::expm1(value) / value;
}

uint32_t row_or_max(const std::vector<uint32_t> &rows, int step, int steps) {
    if (rows.empty() || step < 0 || step >= steps)
        return UINT32_MAX;
    const size_t i = static_cast<size_t>(step);
    return i < rows.size() ? rows[i] : UINT32_MAX;
}

} // namespace

bool h3_schedule_build(int steps, H3SigmaSchedule &out, float video_shift, float audio_shift) {
    if (steps < 1 || steps > kH3MaxSteps)
        return false;
    out.steps = steps;
    out.video.resize(static_cast<size_t>(steps) + 1);
    out.audio.resize(static_cast<size_t>(steps) + 1);
    for (int i = 0; i < steps; ++i) {
        out.video[static_cast<size_t>(i)] = shifted_sigma(i, steps, video_shift);
        out.audio[static_cast<size_t>(i)] = shifted_sigma(i, steps, audio_shift);
    }
    out.video[static_cast<size_t>(steps)] = 0.f;
    out.audio[static_cast<size_t>(steps)] = 0.f;
    return true;
}

bool h3_serving_schedule_build(int evaluations, H3SigmaSchedule &out, float video_shift,
                               float audio_shift) {
    if (evaluations < 2 || evaluations > kH3MaxSteps)
        return false;
    out.steps = evaluations;
    out.video.resize(static_cast<size_t>(evaluations) + 1);
    out.audio.resize(static_cast<size_t>(evaluations) + 1);
    const float denominator = static_cast<float>(evaluations);
    for (int i = 0; i <= evaluations; ++i) {
        const float base = 1.f - static_cast<float>(i) / denominator;
        out.video[static_cast<size_t>(i)] = apply_shift(base, video_shift);
        out.audio[static_cast<size_t>(i)] = apply_shift(base, audio_shift);
    }
    out.video[static_cast<size_t>(evaluations)] = 0.f;
    out.audio[static_cast<size_t>(evaluations)] = 0.f;
    return true;
}

double h3_time_shift_sigma(double sigma, double from_shift, double to_shift) {
    const double base = sigma / (from_shift + sigma * (1.0 - from_shift));
    return to_shift * base / (1.0 + (to_shift - 1.0) * base);
}

double h3_time_shift_slope(double sigma, double from_shift, double to_shift) {
    const double base = sigma / (from_shift + sigma * (1.0 - from_shift));
    const double a = 1.0 + (from_shift - 1.0) * base;
    const double b = 1.0 + (to_shift - 1.0) * base;
    return to_shift * a * a / (from_shift * b * b);
}

bool H3DitSchedule::prepare(const H3SigmaSchedule &sigmas, bool visual_condition,
                            bool audio_condition) {
    *this = H3DitSchedule{};
    if (sigmas.steps < 1 || sigmas.steps > kH3MaxSteps)
        return false;
    const int n = sigmas.steps;
    if (sigmas.video.size() < static_cast<size_t>(n) ||
        sigmas.audio.size() < static_cast<size_t>(n))
        return false;

    steps_ = n;
    video_rows_.assign(static_cast<size_t>(n), 0);
    audio_rows_.assign(static_cast<size_t>(n), 0);
    if (visual_condition)
        visual_condition_rows_.assign(static_cast<size_t>(n), 0);
    if (audio_condition)
        audio_condition_rows_.assign(static_cast<size_t>(n), 0);

    uint32_t count = 0;
    for (int step = 0; step < n; ++step) {
        const float video_t = 1.f - sigmas.video[static_cast<size_t>(step)];
        const float audio_t = 1.f - sigmas.audio[static_cast<size_t>(step)];
        if (video_t == audio_t) {
            video_rows_[static_cast<size_t>(step)] = count;
            audio_rows_[static_cast<size_t>(step)] = count++;
        } else if (video_t < audio_t) {
            video_rows_[static_cast<size_t>(step)] = count++;
            audio_rows_[static_cast<size_t>(step)] = count++;
        } else {
            audio_rows_[static_cast<size_t>(step)] = count++;
            video_rows_[static_cast<size_t>(step)] = count++;
        }
    }

    uint32_t visual_row = UINT32_MAX;
    uint32_t audio_row = UINT32_MAX;
    if (visual_condition)
        visual_row = count++;
    if (audio_condition)
        audio_row = count++;

    for (int step = 0; step < n; ++step) {
        const float video_t = 1.f - sigmas.video[static_cast<size_t>(step)];
        const float audio_t = 1.f - sigmas.audio[static_cast<size_t>(step)];
        if (visual_condition)
            visual_condition_rows_[static_cast<size_t>(step)] =
                video_t >= 0.999f ? video_rows_[static_cast<size_t>(step)] : visual_row;
        if (audio_condition)
            audio_condition_rows_[static_cast<size_t>(step)] =
                audio_t >= 1.f ? audio_rows_[static_cast<size_t>(step)] : audio_row;
    }

    if (count == 0 || count > UINT32_MAX / static_cast<uint32_t>(kH3TimeInput)) {
        *this = H3DitSchedule{};
        return false;
    }

    time_rows_ = count;
    std::vector<float> times(static_cast<size_t>(count), 0.f);
    for (int step = 0; step < n; ++step) {
        times[video_rows_[static_cast<size_t>(step)]] =
            1.f - sigmas.video[static_cast<size_t>(step)];
        times[audio_rows_[static_cast<size_t>(step)]] =
            1.f - sigmas.audio[static_cast<size_t>(step)];
    }
    if (visual_condition)
        times[visual_row] = 0.999f;
    if (audio_condition)
        times[audio_row] = 1.f;

    time_features_.assign(static_cast<size_t>(count) * kH3TimeInput, 0.f);
    constexpr int kHalf = kH3TimeInput / 2;
    for (uint32_t row = 0; row < count; ++row) {
        for (int index = 0; index < kHalf; ++index) {
            const float frequency =
                std::exp(-std::log(10000.f) * static_cast<float>(index) / static_cast<float>(kHalf));
            const float angle = times[row] * frequency;
            const size_t base = static_cast<size_t>(row) * kH3TimeInput;
            time_features_[base + static_cast<size_t>(index)] = std::cos(angle);
            time_features_[base + static_cast<size_t>(kHalf + index)] = std::sin(angle);
        }
    }
    return true;
}

int H3DitSchedule::steps() const {
    return steps_;
}

uint32_t H3DitSchedule::time_rows() const {
    return time_rows_;
}

uint32_t H3DitSchedule::video_row(int step) const {
    return row_or_max(video_rows_, step, steps_);
}

uint32_t H3DitSchedule::audio_row(int step) const {
    return row_or_max(audio_rows_, step, steps_);
}

uint32_t H3DitSchedule::visual_condition_row(int step) const {
    return row_or_max(visual_condition_rows_, step, steps_);
}

uint32_t H3DitSchedule::audio_condition_row(int step) const {
    return row_or_max(audio_condition_rows_, step, steps_);
}

const std::vector<float> &H3DitSchedule::time_features() const {
    return time_features_;
}

bool H3DitSchedule::row_map(int step, const H3Layout &layout, const uint8_t *text_tags,
                            int text_tag_count, uint32_t *rows, int row_count) const {
    if (step < 0 || step >= steps_ || !rows || row_count != layout.seq_len)
        return false;
    if (text_tags && text_tag_count != layout.text_rows)
        return false;
    const size_t si = static_cast<size_t>(step);
    if (si >= video_rows_.size() || si >= audio_rows_.size())
        return false;

    size_t text_index = 0;
    for (const H3Segment &segment : layout.segments) {
        if (segment.start < 0 || segment.start > segment.stop || segment.stop > row_count)
            return false;
        uint32_t time_row = 0;
        uint32_t tag = 0;
        switch (segment.kind) {
        case H3SegKind::Text:
            time_row = video_rows_[si];
            for (int row = segment.start; row < segment.stop; ++row) {
                uint32_t text_tag = 1u;
                if (text_tags) {
                    if (text_index >= static_cast<size_t>(text_tag_count))
                        return false;
                    text_tag = text_tags[text_index];
                }
                if (text_tag >= static_cast<uint32_t>(kH3DitModalities))
                    return false;
                rows[row] = time_row * static_cast<uint32_t>(kH3DitModalities) + text_tag;
                ++text_index;
            }
            continue;
        case H3SegKind::Cond:
        case H3SegKind::RefImage:
            if (si >= visual_condition_rows_.size())
                return false;
            time_row = visual_condition_rows_[si];
            tag = 0;
            break;
        case H3SegKind::RefAudio:
            if (si >= audio_condition_rows_.size())
                return false;
            time_row = audio_condition_rows_[si];
            tag = 2;
            break;
        case H3SegKind::Audio:
            time_row = audio_rows_[si];
            tag = 2;
            break;
        case H3SegKind::Video:
            time_row = video_rows_[si];
            tag = 0;
            break;
        default:
            return false;
        }
        const uint32_t modulation = time_row * static_cast<uint32_t>(kH3DitModalities) + tag;
        for (int row = segment.start; row < segment.stop; ++row)
            rows[row] = modulation;
    }
    return text_index == static_cast<size_t>(layout.text_rows);
}

int h3_res_step(float *output, const float *sample, const float *denoised,
                const float *old_denoised, int n, const float *sigmas, int step,
                int total_steps) {
    if (!output || !sample || !denoised || !sigmas || n < 0 || step < 0 || total_steps < 1 ||
        step >= total_steps)
        return 0;
    const double sigma = sigmas[step];
    const double next = sigmas[step + 1];
    if (!(sigma > next && next >= 0.0))
        return 0;
    if (!old_denoised || next == 0.0) {
        const double delta = next - sigma;
        for (int i = 0; i < n; ++i) {
            const double derivative = (static_cast<double>(sample[i]) - denoised[i]) / sigma;
            output[i] = static_cast<float>(static_cast<double>(sample[i]) + derivative * delta);
        }
        return 1;
    }
    if (step == 0)
        return 0;
    const double t = -std::log(sigma);
    const double t_next = -std::log(next);
    const double t_prev = -std::log(sigmas[step - 1]);
    const double h = t_next - t;
    const double c2 = (t_prev - t) / h;
    const double p1 = phi1(-h);
    const double p2 = (p1 - 1.0) / -h;
    const double b1 = p1 - p2 / c2;
    const double b2 = p2 / c2;
    const double decay = std::exp(-h);
    for (int i = 0; i < n; ++i) {
        output[i] = static_cast<float>(decay * sample[i] +
                                       h * (b1 * denoised[i] + b2 * old_denoised[i]));
    }
    return 1;
}

double h3_adaln_gate_score(const float *mod, int time_rows, int hidden) {
    if (!mod || time_rows < 1 || hidden < 1)
        return -1.0;
    double total = 0.0;
    size_t samples = 0;
    const size_t hidden_n = static_cast<size_t>(hidden);
    for (int row = 0; row < time_rows; ++row) {
        for (int modality = 0; modality < kH3AdalnModalities; ++modality) {
            for (int slot = 2; slot <= 5; slot += 3) {
                const size_t base =
                    ((static_cast<size_t>(row) * kH3AdalnModalities * kH3AdalnSlots +
                      static_cast<size_t>(modality) * kH3AdalnSlots + static_cast<size_t>(slot)) *
                     hidden_n);
                for (int column = 0; column < hidden; ++column)
                    total += std::fabs(static_cast<double>(mod[base + static_cast<size_t>(column)]));
                samples += hidden_n;
            }
        }
    }
    return samples ? total / static_cast<double>(samples) : -1.0;
}

int h3_dit_prune_blocks(uint8_t *active, const double *scores, int n_blocks, double min_score) {
    if (!active || !scores || n_blocks < 1)
        return -1;
    int stay = 0;
    for (int i = 0; i < n_blocks; ++i) {
        if (scores[i] < min_score || scores[i] < 0.0)
            active[i] = 0;
        if (active[i])
            ++stay;
    }
    return stay;
}

} // namespace mvllm
