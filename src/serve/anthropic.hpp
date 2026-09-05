#pragma once

#include "../core/types.hpp"

#include <string>
#include <vector>

namespace mvllm {

class Engine;
struct ChatMessage;
struct GenParams;
struct GenResult;

// Translate Anthropic /v1/messages JSON into OpenAI-shaped chat messages.
bool anthropic_to_chat(const std::string &body, std::vector<ChatMessage> &msgs, GenParams &gp,
                       std::string &err);

// Anthropic Messages API JSON (non-stream).
std::string anthropic_messages_response(const std::string &id, const std::string &model,
                                        const GenResult &out);

// SSE event stream for Anthropic messages.
std::string anthropic_sse_start(const std::string &id, const std::string &model);
std::string anthropic_sse_delta(const std::string &text, int index = 0);
std::string anthropic_sse_stop(const char *stop_reason, const char *stop_sequence = nullptr,
                               int output_tokens = 0, int input_tokens = 0);
std::string anthropic_sse_block_start(int index = 0, const char *block_type = "text");
std::string anthropic_sse_block_stop(int index = 0);
std::string anthropic_sse_thinking_delta(const std::string &text);
std::string anthropic_sse_ping();

// Full POST /v1/messages handler (writes HTTP response on fd).
void handle_anthropic_messages(int fd, const std::string &body, Engine *engine);

} // namespace mvllm
