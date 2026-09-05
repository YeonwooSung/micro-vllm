#include "logprob.hpp"

#include <cmath>

namespace mvllm {

double logprob_target(const float *logits, int vocab, int target, int *am) {
    if (!logits || vocab < 1 || target < 0 || target >= vocab) {
        if (am)
            *am = 0;
        return -INFINITY;
    }

    float mx = logits[0];
    int best = 0;
    for (int i = 1; i < vocab; ++i) {
        if (logits[i] > mx) {
            mx = logits[i];
            best = i;
        }
    }

    double se = 0.0;
    for (int i = 0; i < vocab; ++i)
        se += std::exp(static_cast<double>(logits[i]) - mx);

    if (am)
        *am = (best == target) ? 1 : 0;

    return (static_cast<double>(logits[target]) - mx) - std::log(se);
}

bool model_type_is_glm(const char *s) {
    if (!s)
        return false;
    // Require three remaining bytes so s[1]/s[2] are never past the terminator.
    for (; s[0] && s[1] && s[2]; ++s) {
        if ((s[0] | 32) == 'g' && (s[1] | 32) == 'l' && (s[2] | 32) == 'm')
            return true;
    }
    return false;
}

} // namespace mvllm
