#pragma once

namespace mvllm {

// Returns log p(target). If am != null, *am = 1 iff argmax == target.
double logprob_target(const float *logits, int vocab, int target, int *am);

// Case-insensitive scan for the letters g,l,m in sequence.
bool model_type_is_glm(const char *s);

} // namespace mvllm
