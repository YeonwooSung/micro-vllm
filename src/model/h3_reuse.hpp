#pragma once

#include <cstdint>

namespace mvllm {

// Serving velocity-evaluation mask. Returns the evaluation count, or -1.
int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected, int selected_count);

// Parse a strictly increasing "0,3,6,19" mask. Empty text is not an override (0).
int h3_parse_reuse_steps(int steps, const char *text, uint8_t *selected);

} // namespace mvllm
