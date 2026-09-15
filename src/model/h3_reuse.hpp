#pragma once

#include <cstdint>

namespace mvllm {

// Serving velocity-evaluation mask. Returns the evaluation count, or -1.
int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected, int selected_count);

// Parse a strictly increasing "0,3,6,19" mask. Empty text is not an override (0).
int h3_parse_reuse_steps(int steps, const char *text, uint8_t *selected);

// Official linear velocity extrapolation along a sigma grid. Ratio is 0 without
// a previous evaluation and is clamped to [-2, 2].
float h3_dit_extrapolation_ratio(float current_sigma, float last_sigma, float previous_sigma,
                                 bool have_previous);

// output = last when there is no previous sample; otherwise
// last + ratio * (last - previous). count is the element count.
void h3_dit_extrapolate_velocity(float *output, const float *last, const float *previous, int count,
                                 float current_sigma, float last_sigma, float previous_sigma,
                                 bool have_previous);

} // namespace mvllm
