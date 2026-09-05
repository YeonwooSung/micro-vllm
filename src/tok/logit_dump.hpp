#pragma once

#include <string>

namespace mvllm {

struct LogitTop {
    int id = -1;
    float logit = -3e38f;
};

// Fill out[0..n) descending. Unused slots stay id=-1, logit=-3e38.
// Returns how many valid ids (>= 0).
int logit_topn(const float *logits, int vocab, LogitTop *out, int n);

// "[LOGITS] id:logit ..." — always five pairs. Empty/null -> "[LOGITS]\n".
std::string logit_dump_line(const float *logits, int vocab);

// "LOGITGAP pos=... top1=... top2=... gap=...". Empty/null -> top1=0 logit=0 gap=0.
std::string logit_gap_line(int pos, const float *logits, int vocab);

// 1 if COLI_LOGIT_DUMP or MVLLM_LOGIT_DUMP is set and not empty/"0". Cached.
int logit_dump_enabled();
// 1 if COLI_LOGIT_GAP or MVLLM_LOGIT_GAP is set and not empty/"0". Cached.
int logit_gap_enabled();

// If dump enabled, write logit_dump_line to stderr. Returns 1 if printed.
int logit_dump_maybe(const float *logits, int vocab);
// If gap enabled, write logit_gap_line(pos, ...) to stderr. Returns 1 if printed.
int logit_gap_maybe(int pos, const float *logits, int vocab);

} // namespace mvllm
