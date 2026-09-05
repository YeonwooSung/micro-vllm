#pragma once

#include <cstddef>
#include <vector>

namespace mvllm {

enum class H3SegKind { Text, Cond, RefImage, RefAudio, Audio, Video };

struct H3Position {
    float t = 0.f;
    float h = 0.f;
    float w = 0.f;
};

struct H3Segment {
    int start = 0;
    int stop = 0;
    H3SegKind kind = H3SegKind::Text;
};

struct H3Layout {
    int seq_len = 0;
    int text_rows = 0;
    int audio_rows = 0;
    int video_rows = 0;
    int img_cond_rows = 0;
    int audio_cond_rows = 0;
    std::vector<H3Segment> segments;
    std::vector<H3Position> positions;
};

struct H3LayoutRef {
    H3SegKind kind = H3SegKind::RefImage; // RefImage, RefAudio, or Video (ref video)
    int latent_t = 0;
    int latent_h = 0;
    int latent_w = 0;
    int audio_t = 0;
};

// Official pack: TEXT, optional first/last keyframe COND or ordered Ref2VA
// refs, then AUDIO (2 * audio_t), then VIDEO (latent_t * (lh/2) * (lw/2)).
// keyframes and refs are mutually exclusive.
bool h3_layout_build(int text_len, int latent_t, int latent_h, int latent_w, int audio_t,
                     int frames, H3Layout &out, const int *keyframes = nullptr,
                     int n_keyframes = 0, const H3LayoutRef *refs = nullptr, int n_refs = 0);

constexpr int kH3RopeFreqs = 16;
constexpr int kH3RopeHalf = 48; // 3 axes * 16

// cos/sin [seq, 48]. spatial_scale is 0.5 only for 256x256 T2VA.
void h3_dit_rope_tables(const H3Layout &layout, const float *inv_freq, float spatial_scale,
                        std::vector<float> &cos, std::vector<float> &sin);
void h3_dit_default_inv_freq(float *inv, int n = kH3RopeFreqs);

// Split-half RoPE: rotate (x[d], x[d+48]) with (cos[d], sin[d]) for d in [0,48).
// No-op when hd < 96 or tables are null.
void h3_dit_apply_rope(float *q, float *k, const float *cos, const float *sin, int tokens,
                       int heads, int hd);

} // namespace mvllm
