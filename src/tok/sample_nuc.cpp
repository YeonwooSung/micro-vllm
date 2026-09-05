#include "sample_nuc.hpp"

#include <cmath>
#include <vector>

namespace mvllm {
namespace {

constexpr uint64_t kNucRngSeed = 0x9E3779B97F4A7C15ULL;
constexpr double kNucU01Scale = 1.0 / 9007199254740992.0;

uint64_t load_rng(uint64_t *state) {
    const uint64_t s = state ? *state : 0;
    return s ? s : kNucRngSeed;
}

// Max-heap siftdown on ids in h[0..n), keyed by p[id]. Hole variant.
void heap_siftdown(int *h, int n, int i, const float *p) {
    const int iv = h[i];
    const float kv = p[iv];
    for (;;) {
        const int l = 2 * i + 1;
        if (l >= n)
            break;
        int b = l;
        if (l + 1 < n && p[h[l + 1]] > p[h[l]])
            b = l + 1;
        if (p[h[b]] <= kv)
            break;
        h[i] = h[b];
        i = b;
    }
    h[i] = iv;
}

int argmax_finite(const float *lo, int vocab) {
    int best = -1;
    float bv = 0.f;
    for (int i = 0; i < vocab; ++i) {
        const float x = lo[i];
        if (!std::isfinite(x))
            continue;
        if (best < 0 || x > bv) {
            bv = x;
            best = i;
        }
    }
    return best < 0 ? 0 : best;
}

} // namespace

uint64_t nuc_rng_next(uint64_t *state) {
    uint64_t s = load_rng(state);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    if (state)
        *state = s;
    return s;
}

double nuc_rng_u01(uint64_t *state) {
    return static_cast<double>(nuc_rng_next(state) >> 11) * kNucU01Scale;
}

int nuc_dist_build(const float *logits, int vocab, float temperature, float top_p, float *p) {
    if (!logits || !p || vocab <= 0)
        return 0;

    int mxi = -1;
    float mx = 0.f;
    for (int i = 0; i < vocab; ++i) {
        const float x = logits[i];
        if (std::isfinite(x) && (mxi < 0 || x > mx)) {
            mx = x;
            mxi = i;
        }
    }

    const float invt = 1.f / (temperature > 1e-4f ? temperature : 1e-4f);
    double sum = 0.0;
    if (mxi >= 0) {
        for (int i = 0; i < vocab; ++i) {
            p[i] = std::isfinite(logits[i]) ? std::exp((logits[i] - mx) * invt) : 0.f;
            sum += p[i];
        }
    }

    if (mxi < 0 || !std::isfinite(sum) || sum <= 0.0) {
        const int a = mxi >= 0 ? mxi : 0;
        for (int i = 0; i < vocab; ++i)
            p[i] = 0.f;
        p[a] = 1.f;
        return 1;
    }

    const float sf = static_cast<float>(sum);
    for (int i = 0; i < vocab; ++i)
        p[i] /= sf;

    if (!(top_p > 0.f && top_p < 1.f))
        return 1;

    std::vector<int> h(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i)
        h[static_cast<size_t>(i)] = i;
    for (int i = vocab / 2 - 1; i >= 0; --i)
        heap_siftdown(h.data(), vocab, i, p);

    double kept = 0.0;
    double cum = 0.0;
    int n = vocab;
    do {
        const int root = h[0];
        h[0] = h[static_cast<size_t>(--n)];
        h[static_cast<size_t>(n)] = root;
        kept += p[root];
        cum += p[root];
        if (n > 0)
            heap_siftdown(h.data(), n, 0, p);
    } while (cum < static_cast<double>(top_p) && n > 0);

    for (int i = 0; i < n; ++i)
        p[h[static_cast<size_t>(i)]] = 0.f;
    const float kept_f = static_cast<float>(kept);
    for (int i = n; i < vocab; ++i)
        p[h[static_cast<size_t>(i)]] /= kept_f;
    return 1;
}

int nuc_dist_sample(const float *p, int vocab, int ban, uint64_t *rng) {
    if (!p || vocab <= 0)
        return 0;

    const int skip = (ban >= 0 && ban < vocab) ? ban : -1;
    double z = 1.0 - (skip >= 0 ? static_cast<double>(p[skip]) : 0.0);
    if (z <= 1e-12)
        z = 1e-12;
    const double u = nuc_rng_u01(rng) * z;
    double cum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        if (i == skip)
            continue;
        cum += p[i];
        if (cum >= u)
            return i;
    }
    for (int i = vocab - 1; i >= 0; --i)
        if (i != skip && p[i] > 0.f)
            return i;
    return 0;
}

int nuc_pick(const float *logits, int vocab, float temperature, float top_p, int ban,
             uint64_t *rng) {
    if (!logits || vocab <= 0)
        return 0;
    if (temperature <= 0.f)
        return argmax_finite(logits, vocab);

    std::vector<float> p(static_cast<size_t>(vocab));
    if (!nuc_dist_build(logits, vocab, temperature, top_p, p.data()))
        return 0;
    return nuc_dist_sample(p.data(), vocab, ban, rng);
}

} // namespace mvllm
