#pragma once

#include "gbnf.hpp"

#include <functional>
#include <string>

namespace mvllm {

// Appends forced token ids while exactly one token is legal. Returns count added.
int gbnf_forced_tokens(Gbnf &g, const std::function<std::string(int)> &decode, int vocab, int *out,
                       int max_tokens);

// Unique next-byte walk for tests / ASCII grammars. Returns bytes written.
int gbnf_forced_bytes(Gbnf &g, char *out, int max_bytes);

} // namespace mvllm
