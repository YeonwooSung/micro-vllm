#pragma once

#include "../core/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {

// Official GLM-5.3 smart_resize: align to factor (patch*merge), token budget.
void glm_smart_resize(int height, int width, int *out_h, int *out_w, int patch = 14,
                      int merge = 2, int temporal = 2, int min_tokens = 16,
                      int max_tokens = 8000);

Status decode_base64(const std::string &s, std::vector<uint8_t> &out, std::string &err);

// data: URI, file://, or local path. Remote http(s) is rejected.
Status decode_image_url(const std::string &url, std::vector<float> &rgb, int &width, int &height,
                        std::string &err);
// PPM P6, 24-bit BMP, PNG, or baseline JPEG → RGB HWC [0,1].
// On Apple, ImageIO is tried first for JPEG/PNG (progressive / 16-bit / interlaced).
Status decode_image_bytes(const uint8_t *data, size_t n, std::vector<float> &rgb, int &width,
                          int &height, std::string &err);

} // namespace mvllm
