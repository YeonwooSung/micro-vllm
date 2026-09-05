#pragma once

#include <cstdint>

namespace mvllm {

// `base` is one sequence's KV state. Ragged decode rows must not alias across sequences.
float *kv_row(float *base, int position, int width);
const float *kv_row(const float *base, int position, int width);

// Same arithmetic on the fp8 (e4m3) byte cache.
uint8_t *kv_row8(uint8_t *base, int position, int width);
const uint8_t *kv_row8(const uint8_t *base, int position, int width);

int64_t kv_row_bytes(int position, int width, int elem_size);
bool kv_row_ok(int position, int width, int cap_elems);

} // namespace mvllm
