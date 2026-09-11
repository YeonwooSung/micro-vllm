#pragma once

#include <string>

namespace mvllm {

class Tokenizer;

// Loads HF tokenizer.json via Tokenizer. Backend is "json" after success, else "off".
bool h3_tok_load(const std::string &model_dir, Tokenizer &tok, std::string &err);
const char *h3_tok_backend(); // "json" after successful load, else "off"

} // namespace mvllm
