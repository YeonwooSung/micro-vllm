#pragma once

#include "../core/config.hpp"
#include "../model/family.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
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
    ServeStats stats() const { return stats_; }
    int port() const { return port_; }

    ~HttpServer();
    HttpServer();

private:
    int listen_fd_ = -1;
    int port_ = 0;
    Engine *engine_ = nullptr;
    std::atomic<bool> stop_{false};
    ServeStats stats_{};
};

// Pure helpers, unit-tested without a socket.
std::string json_escape(const std::string &s);
std::string openai_chat_response(const std::string &id, const std::string &model,
                                 const std::string &content, int prompt_tokens,
                                 int completion_tokens);
std::string openai_models_response(const std::string &id);
bool extract_json_string(const std::string &body, const char *key, std::string &out);
bool extract_json_int(const std::string &body, const char *key, int &out);
bool extract_json_number(const std::string &body, const char *key, float &out);
bool extract_json_bool(const std::string &body, const char *key, bool &out);
bool extract_chat_messages(const std::string &body, std::vector<ChatMessage> &out);

} // namespace mvllm
