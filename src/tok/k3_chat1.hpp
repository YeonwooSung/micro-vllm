#pragma once

#include "k3_tools.hpp"
#include "tokenizer.hpp"

#include <string>
#include <vector>

namespace mvllm {

// Official K3CHAT1 length-framed chat wire (kimi_k3.c chat_build_wire).
struct K3Chat1 {
    std::vector<ChatMessage> msgs;
    bool think = false;
    std::vector<K3ToolDecl> tools;
};

// Parse "K3CHAT1\n" records: M/A/Y/O/B+F/V/J then G. Returns false on malformed.
bool k3_chat1_parse(const char *wire, int n, K3Chat1 &out, std::string &err);
inline bool k3_chat1_parse(const std::string &wire, K3Chat1 &out, std::string &err) {
    return k3_chat1_parse(wire.data(), static_cast<int>(wire.size()), out, err);
}

} // namespace mvllm
