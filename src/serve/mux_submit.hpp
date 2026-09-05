#pragma once

#include <cstddef>
#include <cstdint>

namespace mvllm {

constexpr int kMuxSubmitTopkMax = 32;

struct MuxSubmitExt {
    int logprobs = 0;
    int tok_ids = 0; // 0/1
};

// Extras from the first key=value token. 1 = keys ok, 0 = empty, -1 = reject.
int mux_submit_ext(const char *p, MuxSubmitExt &ext);

struct MuxSubmit {
    uint64_t id = 0, bytes = 0, gbytes = 0;
    int slot = 0, max_tokens = 0;
    float temperature = 0.f, top_p = 0.f;
    int logprobs = 0;
    int tok_ids = 0;
};

// Header only (no payload). 6-field, 7-field (gbytes), or 7-field + key=value.
bool mux_submit_parse(const char *line, MuxSubmit &out);

// ASCII decimal token ids separated by bytes <= ' '. Count, cap on overflow, or -1.
int mux_ids_parse(const char *buf, size_t len, int *out, int cap, int vocab);

} // namespace mvllm
