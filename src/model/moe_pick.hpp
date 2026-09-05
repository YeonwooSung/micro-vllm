#pragma once

#include <string>

namespace mvllm {

// Valid id if best >= 0; else slot index (or 0). Optional once-only warning.
int moe_router_pick(int best, int kk, int n_experts, int layer, std::string *warn);

// Index of the first finite maximum, or -1 if none is finite.
int moe_argmax(const float *scores, int n);

// Official unused-scan top-k. NaN never wins. Returns k, or 0 on bad args.
int moe_topk_pick(const float *scores, int n_experts, int k, int *idx, float *w);

} // namespace mvllm
