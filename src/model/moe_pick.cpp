#include "moe_pick.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace mvllm {

int moe_router_pick(int best, int kk, int n_experts, int layer, std::string *warn) {
    if (best >= 0)
        return best;

    static bool warned = false;
    if (warn && !warned) {
        warned = true;
        *warn = "[router] non-finite logits at layer " + std::to_string(layer) +
                ": selection degraded (corrupt expert tile, or overflow at an eviction boundary)";
    }
    return kk < n_experts ? kk : 0;
}

int moe_argmax(const float *scores, int n) {
    if (!scores || n <= 0)
        return -1;

    int best = -1;
    float bv = 0.f;
    for (int e = 0; e < n; ++e) {
        const float s = scores[e];
        if (!std::isfinite(s))
            continue;
        if (best < 0 || s > bv) {
            bv = s;
            best = e;
        }
    }
    return best;
}

int moe_topk_pick(const float *scores, int n_experts, int k, int *idx, float *w) {
    if (!scores || !idx || !w || n_experts < 0)
        return 0;
    if (k > n_experts)
        k = n_experts;
    if (k < 0)
        k = 0;
    if (k == 0)
        return 0;

    std::vector<char> taken(static_cast<size_t>(n_experts), 0);
    for (int kk = 0; kk < k; ++kk) {
        int best = -1;
        float bv = -1e30f;
        for (int e = 0; e < n_experts; ++e) {
            if (!taken[static_cast<size_t>(e)] && scores[e] > bv) {
                bv = scores[e];
                best = e;
            }
        }
        const int pick = moe_router_pick(best, kk, n_experts, -1, nullptr);
        idx[kk] = pick;
        if (pick >= 0 && pick < n_experts)
            taken[static_cast<size_t>(pick)] = 1;
    }

    float sum = 0.f;
    for (int kk = 0; kk < k; ++kk) {
        const int e = idx[kk];
        float v = (e >= 0 && e < n_experts) ? scores[e] : 0.f;
        if (!std::isfinite(v) || v < 0.f)
            v = 0.f;
        w[kk] = v;
        sum += v;
    }
    if (sum > 0.f) {
        for (int kk = 0; kk < k; ++kk)
            w[kk] /= sum;
    } else {
        const float u = 1.f / static_cast<float>(k);
        for (int kk = 0; kk < k; ++kk)
            w[kk] = u;
    }
    return k;
}

} // namespace mvllm
