#include "h3_layout.hpp"

#include <algorithm>
#include <cmath>

namespace mvllm {
namespace {

constexpr int kFramePerToken[5] = {1, 4, 4, 4, 4};
constexpr float kFrameRescale = 5.f / 3.f;

float video_span_sum(int latent_t) {
    float sum = 0.f;
    for (int i = 0; i < latent_t; ++i)
        sum += kFrameRescale * static_cast<float>(kFramePerToken[i % 5]);
    return sum;
}

void fill_audio_grid(float cursor, int audio_t, float w0, float w1, std::vector<H3Position> &audio) {
    audio.resize(static_cast<size_t>(std::max(audio_t, 0)) * 2);
    for (int i = 0; i < audio_t; ++i) {
        audio[static_cast<size_t>(i)] = H3Position{cursor + static_cast<float>(i), 0.f, w0};
        audio[static_cast<size_t>(audio_t + i)] =
            H3Position{cursor + static_cast<float>(i), 0.f, w1};
    }
}

void fill_video_grid(float cursor, int latent_t, const std::vector<H3Position> &cell,
                     std::vector<H3Position> &video) {
    video.clear();
    if (latent_t < 1 || cell.empty())
        return;
    video.reserve(static_cast<size_t>(latent_t) * cell.size());
    float time = cursor;
    for (int ti = 0; ti < latent_t; ++ti) {
        for (const auto &c : cell)
            video.push_back(H3Position{time, c.h, c.w});
        time += kFrameRescale * static_cast<float>(kFramePerToken[ti % 5]);
    }
}

bool frame_grid(int latent_h, int latent_w, std::vector<H3Position> &grid,
                std::vector<float> &widths) {
    if (latent_h < 2 || latent_w < 2 || (latent_h % 2) || (latent_w % 2))
        return false;
    const int nh = latent_h / 2;
    const int nw = latent_w / 2;
    grid.resize(static_cast<size_t>(nh) * nw);
    widths.resize(static_cast<size_t>(nw));
    const float area = std::sqrt(static_cast<float>(latent_h) * static_cast<float>(latent_w));
    const float ratio_h = static_cast<float>(latent_h) / area;
    const float ratio_w = static_cast<float>(latent_w) / area;
    const float step_h = ratio_h / static_cast<float>(nh);
    const float step_w = ratio_w / static_cast<float>(nw);
    const float base_h = (1.f - ratio_h) / 2.f;
    const float base_w = (1.f - ratio_w) / 2.f;
    for (int c = 0; c < nw; ++c)
        widths[static_cast<size_t>(c)] = (static_cast<float>(c) * step_w + base_w) * 32.f;
    int o = 0;
    for (int r = 0; r < nh; ++r) {
        float hh = (static_cast<float>(r) * step_h + base_h) * 32.f;
        for (int c = 0; c < nw; ++c)
            grid[static_cast<size_t>(o++)] = H3Position{0.f, hh, widths[static_cast<size_t>(c)]};
    }
    return true;
}

void emit(H3Layout &out, H3SegKind kind, const H3Position *pos, int n) {
    H3Segment s;
    s.start = out.seq_len;
    s.stop = out.seq_len + n;
    s.kind = kind;
    out.segments.push_back(s);
    if (pos && n > 0)
        out.positions.insert(out.positions.end(), pos, pos + n);
    else
        out.positions.resize(static_cast<size_t>(s.stop));
    out.seq_len = s.stop;
}

void rotate_pair(float *x, const float *cos, const float *sin, int half) {
    for (int d = 0; d < half; ++d) {
        float a = x[d];
        float b = x[d + half];
        x[d] = a * cos[d] - b * sin[d];
        x[d + half] = a * sin[d] + b * cos[d];
    }
}

} // namespace

bool h3_layout_build(int text_len, int latent_t, int latent_h, int latent_w, int audio_t,
                     int frames, H3Layout &out, const int *keyframes, int n_keyframes,
                     const H3LayoutRef *refs, int n_refs) {
    out = {};
    if (text_len < 0 || latent_t < 1 || audio_t < 0)
        return false;
    if (n_keyframes < 0 || n_refs < 0)
        return false;
    if (n_keyframes > 0 && n_refs > 0)
        return false;
    if (n_keyframes > 0 && !keyframes)
        return false;
    if (n_refs > 0 && !refs)
        return false;
    if (frames < 5)
        frames = 5;
    std::vector<H3Position> cell;
    std::vector<float> widths;
    if (!frame_grid(latent_h, latent_w, cell, widths))
        return false;
    std::vector<H3Position> text(static_cast<size_t>(text_len));
    for (int i = 0; i < text_len; ++i)
        text[static_cast<size_t>(i)] = H3Position{static_cast<float>(i), 0.f, 0.f};
    emit(out, H3SegKind::Text, text.data(), text_len);
    out.text_rows = text_len;

    const float w0 = widths.empty() ? 0.f : widths.front();
    const float w1 = widths.empty() ? 0.f : widths.back();
    float cursor = static_cast<float>(text_len);

    for (int i = 0; i < n_keyframes; ++i) {
        const int kf = keyframes[i];
        float condition_time = 0.f;
        if (kf == 0)
            condition_time = static_cast<float>(text_len);
        else if (kf == frames - 1)
            condition_time = static_cast<float>(text_len) + video_span_sum(latent_t) - kFrameRescale;
        else {
            out = {};
            return false;
        }
        for (auto &c : cell)
            c.t = condition_time;
        emit(out, H3SegKind::Cond, cell.data(), static_cast<int>(cell.size()));
        out.img_cond_rows += static_cast<int>(cell.size());
    }

    for (int i = 0; i < n_refs; ++i) {
        const H3LayoutRef &ref = refs[i];
        if (ref.kind == H3SegKind::RefImage) {
            std::vector<H3Position> ref_cell;
            std::vector<float> ref_w;
            if (!frame_grid(ref.latent_h, ref.latent_w, ref_cell, ref_w)) {
                out = {};
                return false;
            }
            for (auto &c : ref_cell)
                c.t = cursor;
            emit(out, H3SegKind::RefImage, ref_cell.data(), static_cast<int>(ref_cell.size()));
            out.img_cond_rows += static_cast<int>(ref_cell.size());
            cursor += 1.f;
        } else if (ref.kind == H3SegKind::RefAudio) {
            if (ref.audio_t < 0) {
                out = {};
                return false;
            }
            std::vector<H3Position> audio;
            fill_audio_grid(cursor, ref.audio_t, w0, w1, audio);
            const int rows = ref.audio_t * 2;
            if (rows > 0)
                emit(out, H3SegKind::RefAudio, audio.data(), rows);
            out.audio_cond_rows += rows;
            cursor += static_cast<float>(ref.audio_t);
        } else if (ref.kind == H3SegKind::Video) {
            if (ref.latent_t < 0 || ref.audio_t < 0) {
                out = {};
                return false;
            }
            std::vector<H3Position> ref_cell;
            std::vector<float> ref_w;
            if (!frame_grid(ref.latent_h, ref.latent_w, ref_cell, ref_w)) {
                out = {};
                return false;
            }
            const float rw0 = ref_w.empty() ? 0.f : ref_w.front();
            const float rw1 = ref_w.empty() ? 0.f : ref_w.back();
            if (ref.audio_t > 0) {
                std::vector<H3Position> audio;
                fill_audio_grid(cursor, ref.audio_t, rw0, rw1, audio);
                const int rows = ref.audio_t * 2;
                emit(out, H3SegKind::RefAudio, audio.data(), rows);
                out.audio_cond_rows += rows;
            }
            std::vector<H3Position> video;
            fill_video_grid(cursor, ref.latent_t, ref_cell, video);
            emit(out, H3SegKind::RefImage, video.data(), static_cast<int>(video.size()));
            out.img_cond_rows += static_cast<int>(video.size());
            cursor += std::max(static_cast<float>(ref.audio_t), video_span_sum(ref.latent_t));
        } else {
            out = {};
            return false;
        }
    }

    std::vector<H3Position> audio;
    fill_audio_grid(cursor, audio_t, w0, w1, audio);
    emit(out, H3SegKind::Audio, audio.data(), audio_t * 2);
    out.audio_rows = audio_t * 2;

    std::vector<H3Position> video;
    fill_video_grid(cursor, latent_t, cell, video);
    emit(out, H3SegKind::Video, video.data(), static_cast<int>(video.size()));
    out.video_rows = static_cast<int>(video.size());
    return out.seq_len == static_cast<int>(out.positions.size());
}

void h3_dit_default_inv_freq(float *inv, int n) {
    if (!inv || n < 1)
        return;
    for (int i = 0; i < n; ++i)
        inv[i] = 1.f / std::pow(10000.f, static_cast<float>(i) / static_cast<float>(n));
}

void h3_dit_rope_tables(const H3Layout &layout, const float *inv_freq, float spatial_scale,
                        std::vector<float> &cos, std::vector<float> &sin) {
    const int seq = layout.seq_len;
    cos.assign(static_cast<size_t>(seq) * kH3RopeHalf, 1.f);
    sin.assign(static_cast<size_t>(seq) * kH3RopeHalf, 0.f);
    if (!inv_freq || seq < 1 || static_cast<int>(layout.positions.size()) < seq)
        return;
    if (!(spatial_scale > 0.f))
        spatial_scale = 1.f;
    for (int row = 0; row < seq; ++row) {
        const H3Position &p = layout.positions[static_cast<size_t>(row)];
        const float axes[3] = {p.t, p.h * spatial_scale, p.w * spatial_scale};
        for (int axis = 0; axis < 3; ++axis) {
            for (int f = 0; f < kH3RopeFreqs; ++f) {
                const int idx = row * kH3RopeHalf + axis * kH3RopeFreqs + f;
                const float ang = axes[axis] * inv_freq[f];
                cos[static_cast<size_t>(idx)] = std::cos(ang);
                sin[static_cast<size_t>(idx)] = std::sin(ang);
            }
        }
    }
}

void h3_dit_apply_rope(float *q, float *k, const float *cos, const float *sin, int tokens,
                       int heads, int hd) {
    if (!q || !k || !cos || !sin || tokens < 1 || heads < 1 || hd < 96)
        return;
    const int half = kH3RopeHalf;
    for (int t = 0; t < tokens; ++t) {
        const float *c = cos + static_cast<size_t>(t) * half;
        const float *s = sin + static_cast<size_t>(t) * half;
        for (int h = 0; h < heads; ++h) {
            rotate_pair(q + (static_cast<size_t>(t) * heads + h) * hd, c, s, half);
            rotate_pair(k + (static_cast<size_t>(t) * heads + h) * hd, c, s, half);
        }
    }
}

} // namespace mvllm
