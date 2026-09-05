#pragma once

#include "../core/config.hpp"
#include "../model/family.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace mvllm {

class Engine;

struct ServeStats {
    uint64_t requests = 0;
    uint64_t tokens_out = 0;
};

class HttpServer {
public:
    Status bind(const std::string &host, int port, std::string &err);
    void set_engine(Engine *engine);
    Status serve_forever(std::string &err);
    void stop();
    ServeStats stats() const;
    int port() const { return port_; }

    ~HttpServer();
    HttpServer();

private:
    void handle_client(int cfd);

    int listen_fd_ = -1;
    int port_ = 0;
    Engine *engine_ = nullptr;
    std::atomic<bool> stop_{false};
    std::atomic<int> inflight_{0};
    std::mutex engine_mu_;
    mutable std::mutex stats_mu_;
    ServeStats stats_{};
};

// Pure helpers, unit-tested without a socket.
std::string json_escape(const std::string &s);
std::string openai_chat_response(const std::string &id, const std::string &model,
                                 const std::string &content, int prompt_tokens,
                                 int completion_tokens,
                                 const std::string &reasoning = {},
                                 const char *finish_reason = "stop",
                                 int cached_tokens = 0, int reasoning_tokens = 0);
bool extract_json_string_array(const std::string &body, const char *key,
                               std::vector<std::string> &out);
bool extract_tool_choice(const std::string &body, std::string &out);
std::string openai_sse_chunk(const std::string &id, const std::string &model,
                             const std::string &delta_json, const char *finish_reason);
std::string openai_sse_done();
std::string openai_models_response(const std::string &id);
bool extract_json_string(const std::string &body, const char *key, std::string &out);
bool extract_json_int(const std::string &body, const char *key, int &out);
bool extract_json_number(const std::string &body, const char *key, float &out);
bool extract_json_bool(const std::string &body, const char *key, bool &out);
bool extract_json_logit_bias(const std::string &body, std::vector<std::pair<int, float>> &out);
bool extract_response_format(const std::string &body, std::string &grammar, std::string &err);
bool extract_chat_messages(const std::string &body, std::vector<ChatMessage> &out);
bool extract_image_url_from_part(const std::string &part, std::string &url);
bool api_key_ok(const std::string &authorization, const std::string &x_api_key = {});
std::string health_json(Engine *engine);
std::string openai_model_object(const std::string &id);
std::string metrics_json(uint64_t requests, uint64_t tokens_out, int kv_slots, int queue,
                         int running = 0, int queued = 0, int max_queue = 0);

} // namespace mvllm
