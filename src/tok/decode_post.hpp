#pragma once

#include "../core/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace mvllm {

// If any stop string occurs in text, return the cut index (first match).
// npos if none.
std::size_t stop_cut(const std::string &text, const std::vector<std::string> &stops);

// Trim text at the first stop occurrence. Returns true if a stop was found.
bool trim_stop(std::string &text, const std::vector<std::string> &stops);

// Split generated assistant text into reasoning + visible content.
// GLM / Llama: before/after </think> (also strips leftover <think>).
// K3: think channel before <|close|>think / response body after.
// Other: reasoning empty, content = text.
void split_assistant_text(Family family, const std::string &text, std::string &reasoning,
                          std::string &content);

// Strip trailing whitespace and leftover chat/eom tags (incl. <|endoftext|>, <|end_of_text|>, <|end|>, <|endofprompt|>, <|end_of_turn|>, <|eom_id|>).
void trim_assistant_tail(std::string &text);

} // namespace mvllm
