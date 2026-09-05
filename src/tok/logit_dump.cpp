#include "logit_dump.hpp"

#include <cmath>
#include <cstdio>
#include <string>

namespace mvllm {
namespace {

constexpr int kDumpSlots = 5;
constexpr float kEmptyLogit = -3e38f;

void reset_slots(LogitTop *out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i].id = -1;
        out[i].logit = kEmptyLogit;
    }
}

// First finite-or-Inf max. NaN skipped. Returns 0 if none.
int argmax_id(const float *logits, int vocab) {
    int best = -1;
    float bv = -INFINITY;
    for (int i = 0; i < vocab; ++i) {
        const float x = logits[i];
        if (!std::isnan(x) && x > bv) {
            bv = x;
            best = i;
        }
    }
    return best < 0 ? 0 : best;
}

} // namespace

int logit_topn(const float *logits, int vocab, LogitTop *out, int n) {
    if (!out || n < 1)
        return 0;
    reset_slots(out, n);
    if (!logits || vocab < 1)
        return 0;

    for (int t = 0; t < vocab; ++t) {
        const float x = logits[t];
        for (int k = 0; k < n; ++k) {
            if (!(x > out[k].logit))
                continue;
            for (int j = n - 1; j > k; --j)
                out[j] = out[j - 1];
            out[k].id = t;
            out[k].logit = x;
            break;
        }
    }

    int valid = 0;
    for (int k = 0; k < n; ++k)
        if (out[k].id >= 0)
            ++valid;
    return valid;
}

std::string logit_dump_line(const float *logits, int vocab) {
    if (!logits || vocab < 1)
        return "[LOGITS]\n";

    LogitTop top[kDumpSlots];
    logit_topn(logits, vocab, top, kDumpSlots);

    std::string line = "[LOGITS]";
    char pair[96];
    for (int k = 0; k < kDumpSlots; ++k) {
        std::snprintf(pair, sizeof(pair), " %d:%.6f", top[k].id, static_cast<double>(top[k].logit));
        line += pair;
    }
    line += '\n';
    return line;
}

std::string logit_gap_line(int pos, const float *logits, int vocab) {
    int top1 = 0;
    int top2 = -1;
    double t1 = 0.0;
    double t2 = 0.0;

    if (logits && vocab >= 1) {
        top1 = argmax_id(logits, vocab);
        t1 = static_cast<double>(logits[top1]);

        float bv = -INFINITY;
        for (int i = 0; i < vocab; ++i) {
            const float x = logits[i];
            if (i == top1 || std::isnan(x) || !(x > bv))
                continue;
            bv = x;
            top2 = i;
        }
        if (top2 >= 0)
            t2 = static_cast<double>(bv);
    }

    const double gap = top2 >= 0 ? t1 - t2 : 0.0;
    char buf[384];
    std::snprintf(buf, sizeof(buf), "LOGITGAP pos=%d top1=%d:%.6f top2=%d:%.6f gap=%.9f\n", pos,
                  top1, t1, top2, t2, gap);
    return std::string(buf);
}

} // namespace mvllm
