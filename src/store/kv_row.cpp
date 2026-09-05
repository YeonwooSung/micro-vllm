#include "kv_row.hpp"

#include <cstddef>
#include <limits>

namespace mvllm {
namespace {

// Official decode-batch stride: position * width elements.
bool elem_offset(int position, int width, size_t *off) {
    if (position < 0)
        return false;
    *off = static_cast<size_t>(position) * static_cast<size_t>(width);
    return true;
}

} // namespace

const float *kv_row(const float *base, int position, int width) {
    size_t off = 0;
    if (!base || !elem_offset(position, width, &off))
        return nullptr;
    return base + off;
}

float *kv_row(float *base, int position, int width) {
    return const_cast<float *>(kv_row(static_cast<const float *>(base), position, width));
}

const uint8_t *kv_row8(const uint8_t *base, int position, int width) {
    size_t off = 0;
    if (!base || !elem_offset(position, width, &off))
        return nullptr;
    return base + off;
}

uint8_t *kv_row8(uint8_t *base, int position, int width) {
    return const_cast<uint8_t *>(kv_row8(static_cast<const uint8_t *>(base), position, width));
}

int64_t kv_row_bytes(int position, int width, int elem_size) {
    if (position < 0 || width < 0 || elem_size < 0)
        return -1;
    const int64_t p = position;
    const int64_t w = width;
    const int64_t e = elem_size;
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    if (w != 0 && p > kMax / w)
        return -1;
    const int64_t n = p * w;
    if (e != 0 && n > kMax / e)
        return -1;
    return n * e;
}

bool kv_row_ok(int position, int width, int cap_elems) {
    if (position < 0 || width <= 0)
        return false;
    const int64_t need = (static_cast<int64_t>(position) + 1) * static_cast<int64_t>(width);
    return need <= static_cast<int64_t>(cap_elems);
}

} // namespace mvllm
