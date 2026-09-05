#pragma once

#include <cstdint>

namespace mvllm {

// xorshift64*. Zero *state is replaced by 0x9E3779B97F4A7C15.
uint64_t nuc_rng_next(uint64_t *state);
double nuc_rng_u01(uint64_t *state);

// Softmax + optional nucleus. p[0..vocab) stays indexed by token id.
// Returns 0 on bad args, 1 on success.
int nuc_dist_build(const float *logits, int vocab, float temperature, float top_p, float *p);

// Sample from p. ban<0 means no ban. rng state in/out.
int nuc_dist_sample(const float *p, int vocab, int ban, uint64_t *rng);

// temperature<=0: argmax among finite logits (ban ignored).
// Otherwise: build + sample.
int nuc_pick(const float *logits, int vocab, float temperature, float top_p, int ban,
             uint64_t *rng);

} // namespace mvllm
