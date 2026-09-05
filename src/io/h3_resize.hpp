#pragma once

#include "../core/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {

// RGB24 uint8, frame-major. On success `out` is frames*out_h*out_w*3 bytes.
// Identity still copies. Returns InvalidArgument on bad sizes.
Status h3_resize_rgb24(const uint8_t *input, int frames, int in_w, int in_h, int out_w, int out_h,
                       std::vector<uint8_t> &out, std::string &err);

// Float HWC [0,1], frame-major. Same geometry rules.
Status h3_resize_rgb_f32(const float *input, int frames, int in_w, int in_h, int out_w, int out_h,
                         std::vector<float> &out, std::string &err);

} // namespace mvllm
