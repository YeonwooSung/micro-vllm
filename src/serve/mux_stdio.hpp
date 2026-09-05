#pragma once

#include "../core/types.hpp"
#include "../model/family.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {

class Engine;

// Official mux wire (docs/serve_protocol.md): stdin SUBMIT/STOP/CANCEL,
// stdout READY/ACCEPT/DATA/DONE/ERROR/STAT. Prefill serial via Engine scheduler.
Status mux_stdio_run(Engine &engine, std::string &err);

// SUBMIT extra bytes: JSON object overlay onto GenParams when a key is present.
// Honored: stop, grammar, seed, frequency/presence/repetition_penalty, min_p,
// temperature, top_p, top_k, logprobs, max_tokens/max_new_tokens (>0),
// logit_bias, cache_slot, think/enable_thinking. Unknown keys ignored.
bool mux_apply_extra_json(const std::string &extra, GenParams &gp, std::string &err);

// IMAGE payload: encoded PNG/JPEG/PPM/BMP, or raw RGB24 when n==h*w*3.
// Wire: IMAGE id nbytes h w [slot] + payload + newline → ACCEPT id 0.
// Stashed by id for the next SUBMIT with that id (consumed on SUBMIT).
// In-flight id → ERROR DUPLICATE_ID (pending for a later SUBMIT is replaced).
Status mux_decode_image(const uint8_t *data, size_t n, int hint_h, int hint_w,
                        std::vector<float> &rgb, int &width, int &height, std::string &err);

// DONE line. stop_kind: 0 = eos, 1 = length, 2 = stop string.
// Prefix: DONE id STAT emitted 0.00 0.0 0.00 prompt_tokens length_limited
std::string mux_format_done(uint64_t id, int emitted, int prompt_tokens, int length_limited,
                            int stop_kind = 0);
// STAT n_live tps tpot load. After READY (0), after SUBMIT ACCEPT, and after
// DONE/ERROR that forgets a flight. IMAGE stash is not counted.
std::string mux_format_stat(int n_live, double tps = 0, double tpot = 0, double load = 0);

} // namespace mvllm
