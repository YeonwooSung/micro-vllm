#pragma once

#include <cstdint>

namespace mvllm {

constexpr int kH3CanvasMultiple = 32;
constexpr int kH3MaxPixels = 768 * 1344;

// Snap a source size onto the official H3 pixel canvas (multiple of 32).
bool h3_adapt_canvas(int width, int height, int *adapted_w, int *adapted_h);

// Ref2VA image sizing is down-only and aspect-preserving.
// max_short_edge == 0 matches the target pixel area; a positive value caps the short edge.
bool h3_reference_image_canvas(int width, int height, int target_width, int target_height,
                               int max_short_edge, int *adapted_w, int *adapted_h);

// Ref2VA video uses the target-style canvas, but never enlarges a smaller source.
bool h3_reference_video_canvas(int width, int height, int *adapted_w, int *adapted_h);

struct H3Rng {
    uint64_t state = 0;
    uint64_t increment = 0;
    float spare = 0.f;
    int has_spare = 0;
};

void h3_rng_seed(H3Rng &rng, uint64_t seed);
uint32_t h3_rng_u32(H3Rng &rng);
float h3_rng_normal(H3Rng &rng);
void h3_rng_fill_normal(H3Rng &rng, float *values, int count);

} // namespace mvllm
