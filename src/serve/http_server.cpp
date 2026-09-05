#include "http_server.hpp"
#include "anthropic.hpp"
#include "../engine.hpp"
#include "../io/image.hpp"
#include "../tok/glm_tools.hpp"
#include "../tok/json_schema.hpp"
#include "../tok/k3_tools.hpp"
#include "../tok/tokenizer.hpp"

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mvllm {
namespace {

size_t skip_ws(const std::string &s, size_t i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
        ++i;
    return i;
}

bool parse_json_string(const std::string &s, size_t &i, std::string &out) {
    if (i >= s.size() || s[i] != '"')
        return false;
    ++i;
    std::string v;
    while (i < s.size()) {
        char c = s[i++];
        if (c == '"') {
            out = std::move(v);
            return true;
        }
        if (c != '\\' || i >= s.size()) {
            v += c;
            continue;
        }
        char e = s[i++];
        switch (e) {
        case '"':
        case '\\':
        case '/':
            v += e;
            break;
        case 'b':
            v += '\b';
            break;
        case 'f':
            v += '\f';
            break;
        case 'n':
            v += '\n';
            break;
        case 'r':
            v += '\r';
            break;
        case 't':
            v += '\t';
            break;
        case 'u': {
            if (i + 4 > s.size())
                return false;
            int cp = 0;
            for (int k = 0; k < 4; ++k) {
                char h = s[i++];
                cp <<= 4;
                if (h >= '0' && h <= '9')
                    cp += h - '0';
                else if (h >= 'a' && h <= 'f')
                    cp += h - 'a' + 10;
                else if (h >= 'A' && h <= 'F')
                    cp += h - 'A' + 10;
                else
                    return false;
            }
            if (cp < 0x80) {
                v += static_cast<char>(cp);
            } else if (cp < 0x800) {
                v += static_cast<char>(0xc0 | (cp >> 6));
                v += static_cast<char>(0x80 | (cp & 0x3f));
            } else {
                v += static_cast<char>(0xe0 | (cp >> 12));
                v += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
                v += static_cast<char>(0x80 | (cp & 0x3f));
            }
            break;
        }
        default:
            v += e;
            break;
        }
    }
    return false;
}

bool extract_json_string_from(const std::string &body, const char *key, size_t start, std::string &out,
                              size_t *next) {
    if (!key)
        return false;
    std::string pat = std::string("\"") + key + "\"";
    size_t pos = start;
    while ((pos = body.find(pat, pos)) != std::string::npos) {
        size_t i = skip_ws(body, pos + pat.size());
        if (i < body.size() && body[i] == ':') {
            i = skip_ws(body, i + 1);
            if (parse_json_string(body, i, out)) {
                if (next)
                    *next = i;
                return true;
            }
        }
        pos += 1;
    }
    return false;
}

bool extract_last_json_string(const std::string &body, const char *key, std::string &out) {
    size_t pos = 0;
    std::string last;
    bool found = false;
    while (extract_json_string_from(body, key, pos, last, &pos)) {
        out = last;
        found = true;
    }
    return found;
}

std::string to_lower(std::string s) {
    for (char &c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string trim(const std::string &s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a])))
        ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])))
        --b;
    return s.substr(a, b - a);
}

bool send_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t w = ::send(fd, p, n, 0);
        if (w < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (w == 0)
            return false;
        p += w;
        n -= static_cast<size_t>(w);
    }
    return true;
}

void append_request_id(std::ostringstream &os, const std::string &request_id) {
    if (!request_id.empty())
        os << "X-Request-Id: " << request_id << "\r\n";
}

void http_reply(int fd, int code, const char *reason, const std::string &body,
                const std::string &request_id = {}) {
    std::ostringstream os;
    os << "HTTP/1.1 " << code << ' ' << reason << "\r\n"
       << "Content-Type: application/json; charset=utf-8\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Access-Control-Allow-Origin: *\r\n";
    append_request_id(os, request_id);
    os << "Connection: close\r\n"
       << "\r\n"
       << body;
    std::string resp = os.str();
    send_all(fd, resp.data(), resp.size());
}

void http_sse_headers(int fd, const std::string &request_id = {}) {
    std::ostringstream os;
    os << "HTTP/1.1 200 OK\r\n"
       << "Content-Type: text/event-stream\r\n"
       << "Cache-Control: no-cache\r\n"
       << "Access-Control-Allow-Origin: *\r\n";
    append_request_id(os, request_id);
    os << "Connection: close\r\n"
       << "\r\n";
    std::string h = os.str();
    send_all(fd, h.data(), h.size());
}

void http_options(int fd) {
    const char *h = "HTTP/1.1 204 No Content\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Access-Control-Allow-Headers: authorization, content-type, x-api-key, "
                    "anthropic-version\r\n"
                    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                    "Content-Length: 0\r\n"
                    "Connection: close\r\n"
                    "\r\n";
    send_all(fd, h, std::strlen(h));
}

std::string openai_sse_text_chunk(const std::string &id, const std::string &model,
                                  const std::string &text, const char *finish_reason) {
    std::ostringstream os;
    os << "data: {\"id\":\"" << json_escape(id) << "\",\"object\":\"text_completion\",\"created\":"
       << static_cast<long long>(std::time(nullptr)) << ",\"model\":\"" << json_escape(model)
       << "\",\"system_fingerprint\":\"fp_mvllm\",\"service_tier\":\"default\","
          "\"choices\":[{\"index\":0,\"text\":\""
       << json_escape(text) << "\",\"finish_reason\":";
    if (finish_reason)
        os << '"' << json_escape(finish_reason) << '"';
    else
        os << "null";
    os << "}]}\n\n";
    return os.str();
}

void append_usage_token_details(std::ostringstream &os, int cached_tokens, int reasoning_tokens) {
    if (cached_tokens > 0)
        os << ",\"prompt_tokens_details\":{\"cached_tokens\":" << cached_tokens << "}";
    if (reasoning_tokens > 0)
        os << ",\"completion_tokens_details\":{\"reasoning_tokens\":" << reasoning_tokens << "}";
}

// OpenAI stream_options.include_usage: empty choices + usage, then [DONE].
std::string openai_sse_usage_chunk(const std::string &id, const std::string &model,
                                   int prompt_tokens, int completion_tokens, bool chat,
                                   int reasoning_tokens = 0) {
    const int total = prompt_tokens + completion_tokens;
    std::ostringstream os;
    os << "data: {\"id\":\"" << json_escape(id) << "\",\"object\":\""
       << (chat ? "chat.completion.chunk" : "text_completion") << "\",\"created\":"
       << static_cast<long long>(std::time(nullptr)) << ",\"model\":\"" << json_escape(model)
       << "\",\"system_fingerprint\":\"fp_mvllm\",\"service_tier\":\"default\","
          "\"choices\":[],\"usage\":{\"prompt_tokens\":"
       << prompt_tokens << ",\"completion_tokens\":" << completion_tokens
       << ",\"total_tokens\":" << total;
    append_usage_token_details(os, 0, reasoning_tokens);
    os << "}}\n\n";
    return os.str();
}

std::string openai_tool_calls_array(const std::vector<K3ParsedCall> &calls, bool with_index) {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < calls.size(); ++i) {
        if (i)
            os << ',';
        std::string cid = calls[i].id.empty() ? ("call_" + std::to_string(i)) : calls[i].id;
        os << '{';
        if (with_index)
            os << "\"index\":" << i << ',';
        os << "\"id\":\"" << json_escape(cid)
           << "\",\"type\":\"function\",\"function\":{\"name\":\"" << json_escape(calls[i].name)
           << "\",\"arguments\":\"" << json_escape(calls[i].arguments) << "\"}}";
    }
    os << ']';
    return os.str();
}

std::string openai_chat_tools_response(const std::string &id, const std::string &model,
                                       const std::string &content,
                                       const std::vector<K3ParsedCall> &calls, int prompt_tokens,
                                       int completion_tokens, int reasoning_tokens = 0) {
    int total = prompt_tokens + completion_tokens;
    std::ostringstream os;
    os << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion\",\"created\":"
       << static_cast<long long>(std::time(nullptr)) << ",\"model\":\"" << json_escape(model)
       << "\",\"system_fingerprint\":\"fp_mvllm\",\"service_tier\":\"default\","
          "\"choices\":[{\"index\":0,\"message\":{\"role\":"
          "\"assistant\",\"content\":\""
       << json_escape(content) << "\",\"tool_calls\":" << openai_tool_calls_array(calls, false)
       << "},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":" << prompt_tokens
       << ",\"completion_tokens\":" << completion_tokens << ",\"total_tokens\":" << total;
    append_usage_token_details(os, 0, reasoning_tokens);
    os << "}}";
    return os.str();
}

const char *sse_finish_reason(const GenResult &out, const GenParams &gp, const ModelConfig &cfg) {
    if (out.stopped_by_stop)
        return "stop";
    if (out.completion_tokens >= gp.max_new_tokens && !out.tokens.empty() &&
        !is_stop_token(out.tokens.back(), cfg, gp.eos))
        return "length";
    return "stop";
}

bool parse_family_tool_calls(Family family, const std::string &text, std::string &stripped,
                             std::vector<K3ParsedCall> &calls) {
    if (family == Family::Glm53 && glm_parse_tool_calls(text, stripped, calls) && !calls.empty())
        return true;
    if (family == Family::KimiK3 || family == Family::Glm53)
        return k3_parse_tool_calls(text, stripped, calls) && !calls.empty();
    return false;
}

void apply_tool_choice(GenParams &gp) {
    if (gp.tool_choice == "none") {
        gp.tools.clear();
        return;
    }
    if (gp.tool_choice.empty() || gp.tool_choice == "auto" || gp.tool_choice == "required")
        return;
    std::vector<K3ToolDecl> keep;
    for (const K3ToolDecl &t : gp.tools) {
        if (t.name == gp.tool_choice)
            keep.push_back(t);
    }
    gp.tools = std::move(keep);
}

std::string model_name(Engine *engine) {
    if (engine && !engine->model_id().empty())
        return engine->model_id();
    return "micro-vllm";
}

bool client_hungup(int fd) {
    pollfd p{};
    p.fd = fd;
    p.events = POLLIN;
    int r = ::poll(&p, 1, 0);
    if (r <= 0)
        return false;
    if (p.revents & (POLLERR | POLLHUP | POLLNVAL))
        return true;
    if (p.revents & POLLIN) {
        char b = 0;
        ssize_t n = ::recv(fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
        if (n == 0)
            return true;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
            return true;
    }
    return false;
}

bool tokenize_request(Engine *engine, const std::vector<ChatMessage> &msgs,
                      const std::string &prompt, bool chat, GenParams &gp, std::vector<int> &ids,
                      std::string &err) {
    Tokenizer tok;
    std::string terr;
    tok.load(engine->runtime().model_dir, terr);
    const ModelConfig &cfg = engine->config();
    const Family family = engine->family();
    if (chat && !msgs.empty()) {
        Status st = tok.encode_chat(family, msgs, gp.think, ids, gp.reasoning_effort,
                                    gp.tools.empty() ? nullptr : &gp.tools);
        if (st != Status::Ok) {
            err = "tokenize failed";
            return false;
        }
        if (family == Family::KimiK3 && cfg.bos >= 0 &&
            (ids.empty() || ids.front() != cfg.bos))
            ids.insert(ids.begin(), cfg.bos);
        if (ids.empty())
            ids.push_back(cfg.bos);
        if (gp.image_token < 0) {
            int img = tok.id_of("<image>");
            if (img < 0)
                img = tok.id_of("<|image|>");
            if (img < 0)
                img = cfg.vision.image_token;
            if (img >= 0)
                gp.image_token = img;
        }
        if (gp.eos < 0 && family == Family::KimiK3) {
            int eom = tok.id_of("<|end_of_msg|>");
            if (eom >= 0)
                gp.eos = eom;
        }
    } else {
        Status st = tok.encode(prompt, ids);
        if (st != Status::Ok) {
            err = "tokenize failed";
            return false;
        }
        if (ids.empty())
            ids.push_back(cfg.bos);
    }
    return true;
}

int resolve_http_slot(Engine *engine, const std::vector<ChatMessage> &msgs, int cache_slot) {
    if (!msgs.empty() && engine->sessions().n_slots() > 1)
        return engine->sessions().assign(msgs, cache_slot);
    return cache_slot >= 0 ? cache_slot : 0;
}

std::string busy_error_body(const std::string &err) {
    if (err.find("queue") != std::string::npos || err.find("QUEUE") != std::string::npos ||
        err.find("full") != std::string::npos)
        return "{\"error\":\"queue full\"}";
    return "{\"error\":\"SLOT_BUSY\"}";
}

bool read_http(int fd, std::string &method, std::string &path, std::string &body,
               std::string &authorization, std::string &x_api_key, std::string &request_id) {
    authorization.clear();
    x_api_key.clear();
    request_id.clear();
    std::string req;
    char buf[4096];
    size_t hdr_end = std::string::npos;
    while (req.size() < (1u << 20)) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (n == 0)
            break;
        req.append(buf, static_cast<size_t>(n));
        hdr_end = req.find("\r\n\r\n");
        if (hdr_end != std::string::npos)
            break;
        hdr_end = req.find("\n\n");
        if (hdr_end != std::string::npos)
            break;
    }
    if (hdr_end == std::string::npos)
        return false;

    bool crlf = req.compare(hdr_end, 4, "\r\n\r\n") == 0;
    size_t hdr_bytes = crlf ? 4 : 2;
    std::string headers = req.substr(0, hdr_end);
    body = req.substr(hdr_end + hdr_bytes);

    size_t line_end = headers.find("\r\n");
    if (line_end == std::string::npos)
        line_end = headers.find('\n');
    if (line_end == std::string::npos)
        return false;
    std::string line = headers.substr(0, line_end);
    std::istringstream ls(line);
    std::string ver;
    if (!(ls >> method >> path >> ver))
        return false;
    size_t q = path.find('?');
    if (q != std::string::npos)
        path.resize(q);

    int content_length = 0;
    size_t i = line_end;
    while (i < headers.size()) {
        if (headers[i] == '\r')
            ++i;
        if (i < headers.size() && headers[i] == '\n')
            ++i;
        size_t nl = headers.find('\n', i);
        if (nl == std::string::npos)
            nl = headers.size();
        std::string hline = headers.substr(i, nl - i);
        if (!hline.empty() && hline.back() == '\r')
            hline.pop_back();
        size_t colon = hline.find(':');
        if (colon != std::string::npos) {
            std::string name = to_lower(trim(hline.substr(0, colon)));
            std::string val = trim(hline.substr(colon + 1));
            if (name == "content-length")
                content_length = std::atoi(val.c_str());
            else if (name == "authorization")
                authorization = val;
            else if (name == "x-api-key")
                x_api_key = val;
            else if (name == "x-request-id")
                request_id = val;
            else if (name == "request-id" && request_id.empty())
                request_id = val;
        }
        i = nl;
    }
    if (content_length < 0)
        content_length = 0;
    if (content_length > 8 * 1024 * 1024)
        return false;
    while (static_cast<int>(body.size()) < content_length && body.size() < (8u << 20)) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (n == 0)
            break;
        body.append(buf, static_cast<size_t>(n));
    }
    if (static_cast<int>(body.size()) > content_length)
        body.resize(static_cast<size_t>(content_length));
    return true;
}

void apply_sampling_extras(const std::string &body, GenParams &gp) {
    extract_json_number(body, "frequency_penalty", gp.frequency_penalty);
    extract_json_number(body, "presence_penalty", gp.presence_penalty);
    int top_k = 0;
    if (extract_json_int(body, "top_k", top_k)) {
        if (top_k < 0)
            top_k = 0;
        gp.top_k = top_k;
    }
    extract_json_number(body, "min_p", gp.min_p);
    extract_json_number(body, "repetition_penalty", gp.repetition_penalty);
    int top_lp = 0;
    extract_json_int(body, "top_logprobs", top_lp);
    if (top_lp < 0)
        top_lp = 0;
    if (top_lp > 20)
        top_lp = 20;
    bool lp_flag = false;
    int lp_n = 0;
    if (extract_json_bool(body, "logprobs", lp_flag)) {
        if (lp_flag)
            gp.logprobs = top_lp > 0 ? top_lp : 1;
    } else if (extract_json_int(body, "logprobs", lp_n)) {
        if (lp_n < 0)
            lp_n = 0;
        gp.logprobs = lp_n;
    }
    extract_json_logit_bias(body, gp.logit_bias);
}

std::string openai_logprobs_content(Engine *engine, const GenResult &out) {
    if (out.token_logprobs.empty())
        return {};
    std::ostringstream os;
    os << "{\"content\":[";
    for (size_t i = 0; i < out.token_logprobs.size(); ++i) {
        if (i)
            os << ',';
        const int tid = (i < out.tokens.size()) ? out.tokens[i] : -1;
        const std::string tok = (engine && tid >= 0) ? engine->decode_token(tid) : std::string();
        os << "{\"token\":\"" << json_escape(tok) << "\",\"logprob\":" << out.token_logprobs[i]
           << ",\"top_logprobs\":[";
        if (i < out.top_logprobs.size()) {
            const std::vector<GenLogprob> &tops = out.top_logprobs[i];
            for (size_t j = 0; j < tops.size(); ++j) {
                if (j)
                    os << ',';
                const std::string ttok =
                    engine ? engine->decode_token(tops[j].token) : std::string();
                os << "{\"token\":\"" << json_escape(ttok) << "\",\"logprob\":" << tops[j].logprob
                   << "}";
            }
        }
        os << "]}";
    }
    os << "]}";
    return os.str();
}

std::string with_choice_logprobs(std::string body, const std::string &lp) {
    if (lp.empty())
        return body;
    const std::string needle = "},\"finish_reason\"";
    size_t p = body.find(needle);
    if (p == std::string::npos)
        return body;
    body.insert(p + 1, "\"logprobs\":" + lp + ",");
    return body;
}

bool models_id_allowed(const std::string &id, const std::string &model) {
    if (id.empty())
        return false;
    if (id == model)
        return true;
    // Any non-empty remainder after /v1/models/ is a valid path suffix.
    return true;
}

} // namespace

bool api_key_ok(const std::string &authorization, const std::string &x_api_key) {
    const char *key = std::getenv("MVLLM_API_KEY");
    if (!key || !key[0])
        return true;
    if (authorization == std::string("Bearer ") + key)
        return true;
    return x_api_key == key;
}

std::string health_json(Engine *engine) {
    int kv = 1;
    int q = 0;
    if (engine) {
        kv = engine->runtime().kv_slots;
        if (kv < 1)
            kv = 1;
        q = engine->scheduler().queue_depth();
    }
    std::ostringstream os;
    os << "{\"ok\":true,\"kv_slots\":" << kv << ",\"queue\":" << q;
    if (engine) {
        os << ",\"running\":" << engine->scheduler().running_count()
           << ",\"queued\":" << engine->scheduler().queued_count()
           << ",\"max_queue\":" << engine->scheduler().max_queue() << ",\"model\":\""
           << json_escape(engine->model_id()) << "\",\"family\":\""
           << family_name(engine->family()) << "\",\"device\":\""
           << device_name(engine->runtime().device) << '"';
    }
    os << "}";
    return os.str();
}

std::string openai_model_object(const std::string &id) {
    return "{\"id\":\"" + json_escape(id) +
           "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"micro-vllm\"}";
}

std::string metrics_json(uint64_t requests, uint64_t tokens_out, int kv_slots, int queue,
                         int running, int queued, int max_queue) {
    std::ostringstream os;
    os << "{\"requests\":" << requests << ",\"tokens_out\":" << tokens_out
       << ",\"kv_slots\":" << kv_slots << ",\"queue\":" << queue << ",\"running\":" << running
       << ",\"queued\":" << queued << ",\"max_queue\":" << max_queue << "}";
    return os.str();
}

HttpServer::HttpServer() = default;

HttpServer::~HttpServer() {
    stop();
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
}

void HttpServer::set_engine(Engine *engine) { engine_ = engine; }

ServeStats HttpServer::stats() const {
    std::lock_guard<std::mutex> lock(stats_mu_);
    return stats_;
}

void HttpServer::stop() {
    stop_.store(true);
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        // Darwin: shutdown() on a listening socket is ENOTCONN and does not
        // wake accept(); a self-connect unblocks the waiter.
        int poke = ::socket(AF_INET, SOCK_STREAM, 0);
        if (poke >= 0) {
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_port = htons(static_cast<uint16_t>(port_));
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            ::connect(poke, reinterpret_cast<sockaddr *>(&a), sizeof(a));
            ::close(poke);
        }
    }
}

Status HttpServer::bind(const std::string &host, int port, std::string &err) {
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }

    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        err = std::string("socket: ") + std::strerror(errno);
        return Status::IoError;
    }

    int yes = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#if defined(SO_NOSIGPIPE)
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (host.empty() || host == "0.0.0.0" || host == "*") {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (host == "localhost") {
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    } else if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        err = "invalid host";
        ::close(fd);
        return Status::InvalidArgument;
    }

    if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        err = std::string("bind: ") + std::strerror(errno);
        ::close(fd);
        return Status::IoError;
    }
    if (::listen(fd, 128) < 0) {
        err = std::string("listen: ") + std::strerror(errno);
        ::close(fd);
        return Status::IoError;
    }

    sockaddr_in actual{};
    socklen_t alen = sizeof(actual);
    if (::getsockname(fd, reinterpret_cast<sockaddr *>(&actual), &alen) == 0)
        port_ = ntohs(actual.sin_port);
    else
        port_ = port;

    listen_fd_ = fd;
    stop_.store(false);
    return Status::Ok;
}

Status HttpServer::serve_forever(std::string &err) {
    if (listen_fd_ < 0) {
        err = "not bound";
        return Status::InvalidArgument;
    }

    while (!stop_.load()) {
        pollfd pfd{};
        pfd.fd = listen_fd_;
        pfd.events = POLLIN;
        int pr = ::poll(&pfd, 1, 200);
        if (stop_.load())
            break;
        if (pr < 0) {
            if (errno == EINTR)
                continue;
            if (errno == EBADF || errno == EINVAL)
                break;
            continue;
        }
        if (pr == 0)
            continue;

        sockaddr_in cli{};
        socklen_t clen = sizeof(cli);
        int cfd = ::accept(listen_fd_, reinterpret_cast<sockaddr *>(&cli), &clen);
        if (cfd < 0) {
            if (stop_.load())
                break;
            if (errno == EINTR)
                continue;
            if (errno == EBADF || errno == EINVAL || errno == ECONNABORTED)
                break;
            continue;
        }
        if (stop_.load()) {
            ::close(cfd);
            break;
        }
#if defined(SO_NOSIGPIPE)
        int nosig = 1;
        ::setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, &nosig, sizeof(nosig));
#endif

        inflight_.fetch_add(1);
        std::thread([this, cfd]() {
            handle_client(cfd);
            inflight_.fetch_sub(1);
        }).detach();
    }

    while (inflight_.load() > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return Status::Ok;
}

void HttpServer::handle_client(int cfd) {
        {
            std::lock_guard<std::mutex> lock(stats_mu_);
            ++stats_.requests;
        }

        std::string method, path, body, authorization, x_api_key, request_id;
        if (!read_http(cfd, method, path, body, authorization, x_api_key, request_id)) {
            http_reply(cfd, 400, "Bad Request", "{\"error\":\"bad request\"}", request_id);
            ::close(cfd);
            return;
        }

        const bool v1 = path.size() >= 4 && path.compare(0, 4, "/v1/") == 0;
        if (method == "OPTIONS" && (path == "/health" || path == "/metrics" || v1)) {
            http_options(cfd);
            ::close(cfd);
            return;
        }

        if (method == "POST" && v1 && !api_key_ok(authorization, x_api_key)) {
            http_reply(cfd, 401, "Unauthorized", "{\"error\":\"unauthorized\"}", request_id);
            ::close(cfd);
            return;
        }

        if (method == "GET" && path == "/health") {
            http_reply(cfd, 200, "OK", health_json(engine_), request_id);
        } else if (method == "GET" && (path == "/metrics" || path == "/v1/metrics")) {
            uint64_t requests = 0;
            uint64_t tokens_out = 0;
            {
                std::lock_guard<std::mutex> lock(stats_mu_);
                requests = stats_.requests;
                tokens_out = stats_.tokens_out;
            }
            int kv = 1;
            int q = 0;
            int running = 0;
            int queued = 0;
            int max_queue = 0;
            if (engine_) {
                kv = engine_->runtime().kv_slots;
                if (kv < 1)
                    kv = 1;
                q = engine_->scheduler().queue_depth();
                running = engine_->scheduler().running_count();
                queued = engine_->scheduler().queued_count();
                max_queue = engine_->scheduler().max_queue();
            }
            http_reply(cfd, 200, "OK",
                       metrics_json(requests, tokens_out, kv, q, running, queued, max_queue),
                       request_id);
        } else if (method == "GET" && path == "/v1/models") {
            http_reply(cfd, 200, "OK", openai_models_response(model_name(engine_)), request_id);
        } else if (method == "GET" && path.size() > 11 &&
                   path.compare(0, 11, "/v1/models/") == 0) {
            const std::string id = path.substr(11);
            if (models_id_allowed(id, model_name(engine_)))
                http_reply(cfd, 200, "OK", openai_model_object(id), request_id);
            else
                http_reply(cfd, 404, "Not Found", "{\"error\":\"not found\"}", request_id);
        } else if (method == "POST" && path == "/v1/messages") {
            if (!engine_) {
                http_reply(cfd, 503, "Service Unavailable", "{\"error\":\"no engine\"}",
                           request_id);
                ::close(cfd);
                return;
            }
            handle_anthropic_messages(cfd, body, engine_);
        } else if (method == "POST" &&
                   (path == "/v1/chat/completions" || path == "/v1/completions" ||
                    path == "/v1/videos/generations")) {
            if (!engine_) {
                http_reply(cfd, 503, "Service Unavailable", "{\"error\":\"no engine\"}",
                           request_id);
                ::close(cfd);
                return;
            }
            if (path == "/v1/videos/generations") {
                H3GenParams hp;
                extract_json_string(body, "prompt", hp.prompt);
                extract_json_int(body, "width", hp.width);
                extract_json_int(body, "height", hp.height);
                extract_json_int(body, "frames", hp.frames);
                extract_json_int(body, "steps", hp.steps);
                extract_json_string(body, "audio_path", hp.audio_path);
                if (hp.audio_path.empty())
                    extract_json_string(body, "audio", hp.audio_path);
                extract_json_string(body, "first_frame", hp.first_frame);
                extract_json_string(body, "last_frame", hp.last_frame);
                extract_json_string_array(body, "ref_images", hp.ref_images);
                int h3_seed = 0;
                if (extract_json_int(body, "seed", h3_seed))
                    hp.seed = static_cast<uint64_t>(h3_seed);
                if (!extract_json_string(body, "output_path", hp.output_path))
                    extract_json_string(body, "output", hp.output_path);
                if (!extract_json_int(body, "dit_layers", hp.dit_layers))
                    extract_json_int(body, "layers", hp.dit_layers);
                extract_json_int(body, "denoise_reuse", hp.denoise_reuse);
                H3GenResult out;
                std::string gerr;
                Status st = Status::Ok;
                {
                    std::lock_guard<std::mutex> lock(engine_mu_);
                    st = engine_->generate_video(hp, out, gerr);
                }
                if (st != Status::Ok) {
                    http_reply(cfd, 500, "Internal Server Error",
                               "{\"error\":\"" + json_escape(gerr) + "\"}", request_id);
                } else {
                    std::string jb = "{\"output_path\":\"" + json_escape(out.output_path) +
                                     "\",\"note\":\"" + json_escape(out.note) +
                                     "\",\"blocks_streamed\":" +
                                     std::to_string(out.blocks_streamed) +
                                     ",\"steps_run\":" + std::to_string(out.steps_run) + "}";
                    http_reply(cfd, 200, "OK", jb, request_id);
                }
            } else {
                int n_choices = 1;
                if (extract_json_int(body, "n", n_choices) && n_choices != 1) {
                    http_reply(cfd, 400, "Bad Request", "{\"error\":\"n must be 1\"}", request_id);
                    ::close(cfd);
                    return;
                }
                std::string prompt;
                std::vector<ChatMessage> msgs;
                if (path == "/v1/chat/completions") {
                    if (!extract_chat_messages(body, msgs))
                        extract_last_json_string(body, "content", prompt);
                } else
                    extract_json_string(body, "prompt", prompt);
                int max_tokens = 32;
                extract_json_int(body, "max_tokens", max_tokens);
                int max_completion_tokens = 0;
                if (extract_json_int(body, "max_completion_tokens", max_completion_tokens))
                    max_tokens = max_completion_tokens;
                if (max_tokens < 0)
                    max_tokens = 0;
                GenParams gp;
                gp.max_new_tokens = max_tokens;
                extract_json_number(body, "temperature", gp.temperature);
                extract_json_number(body, "top_p", gp.top_p);
                apply_sampling_extras(body, gp);
                int seed = 0;
                if (extract_json_int(body, "seed", seed) && seed >= 0)
                    gp.seed = static_cast<uint64_t>(seed);
                extract_json_string(body, "reasoning_effort", gp.reasoning_effort);
                bool enable_thinking = false;
                if (path == "/v1/chat/completions" && engine_->family() == Family::Glm53)
                    enable_thinking = true;
                bool parsed_think = false;
                if (extract_json_bool(body, "enable_thinking", enable_thinking))
                    parsed_think = true;
                if (gp.reasoning_effort == "none")
                    enable_thinking = false;
                else if (!parsed_think && !gp.reasoning_effort.empty() &&
                         gp.reasoning_effort != "none")
                    enable_thinking = true;
                gp.think = enable_thinking;
                if (path == "/v1/completions")
                    gp.apply_template = false;
                std::vector<float> image_rgb;
                int iw = 0, ih = 0;
                std::string ierr;
                for (const auto &m : msgs) {
                    for (const std::string &u : m.image_urls) {
                        if (decode_image_url(u, image_rgb, iw, ih, ierr) == Status::Ok && iw > 0 &&
                            ih > 0) {
                            gp.image_rgb = image_rgb.data();
                            gp.image_w = iw;
                            gp.image_h = ih;
                            break;
                        }
                    }
                    if (gp.image_rgb)
                        break;
                }
                if (!gp.image_rgb) {
                    std::string raw_url;
                    if (extract_json_string(body, "image_url", raw_url) &&
                        decode_image_url(raw_url, image_rgb, iw, ih, ierr) == Status::Ok) {
                        gp.image_rgb = image_rgb.data();
                        gp.image_w = iw;
                        gp.image_h = ih;
                    }
                }
                extract_json_string(body, "grammar", gp.grammar);
                {
                    std::string rferr;
                    extract_response_format(body, gp.grammar, rferr);
                }
                if (!extract_json_string_array(body, "stop", gp.stop)) {
                    std::string stop_one;
                    if (extract_json_string(body, "stop", stop_one))
                        gp.stop.push_back(std::move(stop_one));
                }
                if (path == "/v1/completions") {
                    std::string suffix;
                    if (extract_json_string(body, "suffix", suffix) && !suffix.empty())
                        gp.stop.push_back(std::move(suffix));
                }
                int cache_slot = -1;
                if (extract_json_int(body, "cache_slot", cache_slot))
                    gp.cache_slot = cache_slot;
                bool stream = false;
                extract_json_bool(body, "stream", stream);
                bool echo = false;
                if (path == "/v1/completions")
                    extract_json_bool(body, "echo", echo);
                bool include_usage = false;
                extract_json_bool(body, "include_usage", include_usage);
                if (gp.tools.empty())
                    k3_extract_tools_json(body, gp.tools);
                if (extract_tool_choice(body, gp.tool_choice))
                    apply_tool_choice(gp);
                // Accepted; engine already emits at most one call.
                bool parallel_tool_calls = true;
                extract_json_bool(body, "parallel_tool_calls", parallel_tool_calls);
                (void)parallel_tool_calls;
                std::string user;
                extract_json_string(body, "user", user);
                (void)user;
                const bool chat = (path == "/v1/chat/completions");
                uint64_t reqn = 0;
                {
                    std::lock_guard<std::mutex> lock(stats_mu_);
                    reqn = stats_.requests;
                }
                std::string id = (chat ? "chatcmpl-" : "cmpl-") + std::to_string(reqn);
                std::string model = model_name(engine_);
                const int kv_slots = engine_->runtime().kv_slots;
                if (stream) {
                    http_sse_headers(cfd, request_id);
                    if (chat) {
                        std::string role = openai_sse_chunk(id, model, "{\"role\":\"assistant\"}",
                                                            nullptr);
                        send_all(cfd, role.data(), role.size());
                    } else if (echo && !prompt.empty()) {
                        std::string ev = openai_sse_text_chunk(id, model, prompt, nullptr);
                        send_all(cfd, ev.data(), ev.size());
                    }
                    Engine *eng = engine_;
                    gp.token_text = [eng](int tid) { return eng->decode_token(tid); };
                    auto decode = gp.token_text;
                    gp.on_token = [cfd, id, model, chat, decode](int tok) {
                        std::string piece = decode ? decode(tok) : std::string();
                        std::string ev;
                        if (chat)
                            ev = openai_sse_chunk(id, model,
                                                  "{\"content\":\"" + json_escape(piece) + "\"}",
                                                  nullptr);
                        else
                            ev = openai_sse_text_chunk(id, model, piece, nullptr);
                        send_all(cfd, ev.data(), ev.size());
                    };
                }
                GenResult out;
                std::string gerr;
                Status st = Status::Ok;
                if (kv_slots > 1) {
                    std::vector<int> ids;
                    if (!tokenize_request(engine_, msgs, prompt, chat, gp, ids, gerr)) {
                        st = Status::InvalidArgument;
                    } else {
                        const int slot = resolve_http_slot(engine_, msgs, gp.cache_slot);
                        gp.cache_slot = slot;
                        gp.prefix_reuse = engine_->sessions().match(slot, ids);
                        if (!gp.token_text) {
                            Engine *eng = engine_;
                            gp.token_text = [eng](int tid) { return eng->decode_token(tid); };
                        }
                        uint64_t jid = 0;
                        {
                            std::lock_guard<std::mutex> lock(engine_mu_);
                            jid = engine_->scheduler().submit(slot, ids, gp, gerr);
                        }
                        if (jid == 0) {
                            std::string ebody = busy_error_body(gerr);
                            if (stream) {
                                std::string ev = "data: " + ebody + "\n\n";
                                send_all(cfd, ev.data(), ev.size());
                                std::string done = openai_sse_done();
                                send_all(cfd, done.data(), done.size());
                            } else {
                                http_reply(cfd, 429, "Too Many Requests", ebody, request_id);
                            }
                            ::close(cfd);
                            return;
                        }
                        bool hungup = false;
                        auto sched_finished = [&]() {
                            std::lock_guard<std::mutex> lock(engine_mu_);
                            return engine_->scheduler().finished(jid);
                        };
                        while (!sched_finished() && !stop_.load()) {
                            if (client_hungup(cfd)) {
                                std::lock_guard<std::mutex> lock(engine_mu_);
                                engine_->scheduler().cancel(jid);
                                hungup = true;
                                break;
                            }
                            int left = 0;
                            {
                                std::lock_guard<std::mutex> lock(engine_mu_);
                                left = engine_->scheduler().pump();
                            }
                            if (left == 0 && !sched_finished())
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                        bool have_job = false;
                        {
                            std::lock_guard<std::mutex> lock(engine_mu_);
                            const BatchJob *job = engine_->scheduler().job(jid);
                            if (job) {
                                have_job = true;
                                out = job->out;
                                st = job->status;
                                gerr = job->err;
                            }
                        }
                        if (have_job) {
                            if (out.text.empty() && !out.tokens.empty()) {
                                for (int t : out.tokens)
                                    out.text += engine_->decode_token(t);
                            }
                        } else if (st == Status::Ok) {
                            st = Status::InvalidArgument;
                            if (gerr.empty())
                                gerr = "job lost";
                        }
                        if (hungup) {
                            if (stream) {
                                std::string done = openai_sse_done();
                                send_all(cfd, done.data(), done.size());
                            }
                            ::close(cfd);
                            return;
                        }
                    }
                } else {
                    std::lock_guard<std::mutex> lock(engine_mu_);
                    if (!msgs.empty())
                        st = engine_->generate_chat(msgs, gp, out, gerr);
                    else
                        st = engine_->generate(prompt, gp, out, gerr);
                }
                if (st != Status::Ok) {
                    if (stream) {
                        std::string ev = "data: {\"error\":\"" + json_escape(gerr) + "\"}\n\n";
                        send_all(cfd, ev.data(), ev.size());
                        std::string done = openai_sse_done();
                        send_all(cfd, done.data(), done.size());
                    } else {
                        http_reply(cfd, 500, "Internal Server Error",
                                   "{\"error\":\"" + json_escape(gerr) + "\"}", request_id);
                    }
                } else {
                    {
                        std::lock_guard<std::mutex> lock(stats_mu_);
                        stats_.tokens_out += static_cast<uint64_t>(out.completion_tokens);
                    }
                    if (stream) {
                        if (chat && !out.reasoning.empty()) {
                            std::string rev = openai_sse_chunk(
                                id, model,
                                "{\"reasoning_content\":\"" + json_escape(out.reasoning) + "\"}",
                                nullptr);
                            send_all(cfd, rev.data(), rev.size());
                        }
                        const char *fr = sse_finish_reason(out, gp, engine_->config());
                        bool tools_delta = false;
                        if (chat) {
                            std::string stripped;
                            std::vector<K3ParsedCall> calls;
                            if (parse_family_tool_calls(engine_->family(), out.text, stripped,
                                                        calls)) {
                                std::string delta =
                                    "{\"tool_calls\":" + openai_tool_calls_array(calls, true) + "}";
                                std::string ev = openai_sse_chunk(id, model, delta, "tool_calls");
                                send_all(cfd, ev.data(), ev.size());
                                tools_delta = true;
                            }
                        }
                        if (!tools_delta) {
                            std::string ev;
                            if (chat)
                                ev = openai_sse_chunk(id, model, "{}", fr);
                            else
                                ev = openai_sse_text_chunk(id, model, "", fr);
                            send_all(cfd, ev.data(), ev.size());
                        }
                        if (include_usage) {
                            std::string uev = openai_sse_usage_chunk(
                                id, model, out.prompt_tokens, out.completion_tokens, chat,
                                out.reasoning_tokens);
                            send_all(cfd, uev.data(), uev.size());
                        }
                        std::string done = openai_sse_done();
                        send_all(cfd, done.data(), done.size());
                    } else if (chat) {
                        const std::string lp = openai_logprobs_content(engine_, out);
                        std::string stripped;
                        std::vector<K3ParsedCall> calls;
                        if (parse_family_tool_calls(engine_->family(), out.text, stripped, calls)) {
                            http_reply(cfd, 200, "OK",
                                       with_choice_logprobs(
                                           openai_chat_tools_response(id, model, stripped, calls,
                                                                      out.prompt_tokens,
                                                                      out.completion_tokens,
                                                                      out.reasoning_tokens),
                                           lp),
                                       request_id);
                        } else {
                            const char *fr = sse_finish_reason(out, gp, engine_->config());
                            http_reply(cfd, 200, "OK",
                                       with_choice_logprobs(
                                           openai_chat_response(id, model, out.text,
                                                                out.prompt_tokens,
                                                                out.completion_tokens, out.reasoning,
                                                                fr, gp.prefix_reuse,
                                                                out.reasoning_tokens),
                                           lp),
                                       request_id);
                        }
                    } else {
                        const char *fr = sse_finish_reason(out, gp, engine_->config());
                        const std::string text = echo ? prompt + out.text : out.text;
                        http_reply(cfd, 200, "OK",
                                   with_choice_logprobs(
                                       openai_chat_response(id, model, text, out.prompt_tokens,
                                                            out.completion_tokens, out.reasoning,
                                                            fr, gp.prefix_reuse,
                                                            out.reasoning_tokens),
                                       openai_logprobs_content(engine_, out)),
                                   request_id);
                    }
                }
            }
        } else {
            http_reply(cfd, 404, "Not Found", "{\"error\":\"not found\"}", request_id);
        }
        ::close(cfd);
}

std::string json_escape(const std::string &s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            o += "\\\"";
            break;
        case '\\':
            o += "\\\\";
            break;
        case '\b':
            o += "\\b";
            break;
        case '\f':
            o += "\\f";
            break;
        case '\n':
            o += "\\n";
            break;
        case '\r':
            o += "\\r";
            break;
        case '\t':
            o += "\\t";
            break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                o += buf;
            } else {
                o += static_cast<char>(c);
            }
            break;
        }
    }
    return o;
}

std::string openai_chat_response(const std::string &id, const std::string &model,
                                 const std::string &content, int prompt_tokens,
                                 int completion_tokens, const std::string &reasoning,
                                 const char *finish_reason, int cached_tokens,
                                 int reasoning_tokens) {
    int total = prompt_tokens + completion_tokens;
    if (!finish_reason || !finish_reason[0])
        finish_reason = "stop";
    std::ostringstream os;
    os << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion\",\"created\":"
       << static_cast<long long>(std::time(nullptr)) << ",\"model\":\"" << json_escape(model)
       << "\",\"system_fingerprint\":\"fp_mvllm\",\"service_tier\":\"default\","
          "\"choices\":[{\"index\":0,\"message\":{\"role\":"
          "\"assistant\",\"content\":\""
       << json_escape(content) << "\"";
    if (!reasoning.empty())
        os << ",\"reasoning_content\":\"" << json_escape(reasoning) << "\"";
    os << "},\"finish_reason\":\"" << json_escape(finish_reason)
       << "\"}],\"usage\":{\"prompt_tokens\":" << prompt_tokens
       << ",\"completion_tokens\":" << completion_tokens << ",\"total_tokens\":" << total;
    append_usage_token_details(os, cached_tokens, reasoning_tokens);
    os << "}}";
    return os.str();
}

namespace {

bool extract_choice_object(const std::string &body, const char *key, std::string &out) {
    if (!key)
        return false;
    const std::string pat = std::string("\"") + key + "\"";
    size_t pos = 0;
    while ((pos = body.find(pat, pos)) != std::string::npos) {
        size_t i = skip_ws(body, pos + pat.size());
        if (i >= body.size() || body[i] != ':') {
            pos += 1;
            continue;
        }
        i = skip_ws(body, i + 1);
        if (i >= body.size() || body[i] != '{') {
            pos += 1;
            continue;
        }
        int depth = 0;
        bool in_str = false;
        size_t end = std::string::npos;
        for (size_t j = i; j < body.size(); ++j) {
            char c = body[j];
            if (in_str) {
                if (c == '\\') {
                    ++j;
                    continue;
                }
                if (c == '"')
                    in_str = false;
                continue;
            }
            if (c == '"')
                in_str = true;
            else if (c == '{')
                ++depth;
            else if (c == '}') {
                --depth;
                if (depth == 0) {
                    end = j;
                    break;
                }
            }
        }
        if (end == std::string::npos)
            return false;
        const std::string obj = body.substr(i, end - i + 1);
        std::string name;
        size_t fpos = obj.find("\"function\"");
        if (fpos != std::string::npos) {
            size_t fi = skip_ws(obj, fpos + 10);
            if (fi < obj.size() && obj[fi] == ':') {
                fi = skip_ws(obj, fi + 1);
                if (fi < obj.size() && obj[fi] == '{') {
                    std::string fn_name;
                    if (extract_json_string(obj.substr(fi), "name", fn_name) && !fn_name.empty()) {
                        out = std::move(fn_name);
                        return true;
                    }
                }
            }
        }
        if (extract_json_string(obj, "name", name) && !name.empty()) {
            out = std::move(name);
            return true;
        }
        return false;
    }
    return false;
}

} // namespace

bool extract_tool_choice(const std::string &body, std::string &out) {
    out.clear();
    if (extract_json_string(body, "tool_choice", out) && !out.empty())
        return true;
    if (extract_choice_object(body, "tool_choice", out))
        return true;
    // Legacy OpenAI function_call → tool_choice (only if tool_choice was absent).
    if (extract_json_string(body, "function_call", out) && !out.empty())
        return true;
    return extract_choice_object(body, "function_call", out);
}

std::string openai_sse_chunk(const std::string &id, const std::string &model,
                             const std::string &delta_json, const char *finish_reason) {
    // delta_json is inserted raw as the "delta" object (e.g. {"content":"..."}).
    std::ostringstream os;
    os << "data: {\"id\":\"" << json_escape(id)
       << "\",\"object\":\"chat.completion.chunk\",\"created\":"
       << static_cast<long long>(std::time(nullptr)) << ",\"model\":\"" << json_escape(model)
       << "\",\"system_fingerprint\":\"fp_mvllm\",\"service_tier\":\"default\","
          "\"choices\":[{\"index\":0,\"delta\":"
       << (delta_json.empty() ? "{}" : delta_json) << ",\"finish_reason\":";
    if (finish_reason)
        os << '"' << json_escape(finish_reason) << '"';
    else
        os << "null";
    os << "}]}\n\n";
    return os.str();
}

std::string openai_sse_done() { return "data: [DONE]\n\n"; }

std::string openai_models_response(const std::string &id) {
    return "{\"object\":\"list\",\"data\":[" + openai_model_object(id) + "]}";
}

bool extract_json_string(const std::string &body, const char *key, std::string &out) {
    return extract_json_string_from(body, key, 0, out, nullptr);
}

bool extract_json_logit_bias(const std::string &body, std::vector<std::pair<int, float>> &out) {
    const std::string pat = "\"logit_bias\"";
    size_t pos = 0;
    while ((pos = body.find(pat, pos)) != std::string::npos) {
        size_t i = skip_ws(body, pos + pat.size());
        if (i >= body.size() || body[i] != ':') {
            pos += 1;
            continue;
        }
        i = skip_ws(body, i + 1);
        if (i >= body.size() || body[i] != '{') {
            pos += 1;
            continue;
        }
        ++i;
        std::vector<std::pair<int, float>> vals;
        while (i < body.size()) {
            i = skip_ws(body, i);
            if (i < body.size() && body[i] == '}') {
                out = std::move(vals);
                return true;
            }
            std::string key;
            if (!parse_json_string(body, i, key))
                return false;
            i = skip_ws(body, i);
            if (i >= body.size() || body[i] != ':')
                return false;
            i = skip_ws(body, i + 1);
            char *end = nullptr;
            float v = std::strtof(body.c_str() + i, &end);
            if (!end || end == body.c_str() + i)
                return false;
            i = static_cast<size_t>(end - body.c_str());
            int tid = 0;
            try {
                tid = std::stoi(key);
            } catch (...) {
                return false;
            }
            vals.emplace_back(tid, v);
            i = skip_ws(body, i);
            if (i < body.size() && body[i] == ',') {
                ++i;
                continue;
            }
            if (i < body.size() && body[i] == '}') {
                out = std::move(vals);
                return true;
            }
            return false;
        }
        return false;
    }
    return false;
}

bool extract_json_string_array(const std::string &body, const char *key,
                               std::vector<std::string> &out) {
    if (!key)
        return false;
    std::string pat = std::string("\"") + key + "\"";
    size_t pos = 0;
    while ((pos = body.find(pat, pos)) != std::string::npos) {
        size_t i = skip_ws(body, pos + pat.size());
        if (i >= body.size() || body[i] != ':') {
            pos += 1;
            continue;
        }
        i = skip_ws(body, i + 1);
        if (i >= body.size() || body[i] != '[') {
            pos += 1;
            continue;
        }
        ++i;
        std::vector<std::string> vals;
        while (i < body.size()) {
            i = skip_ws(body, i);
            if (i < body.size() && body[i] == ']') {
                out = std::move(vals);
                return true;
            }
            std::string item;
            if (!parse_json_string(body, i, item))
                return false;
            vals.push_back(std::move(item));
            i = skip_ws(body, i);
            if (i < body.size() && body[i] == ',') {
                ++i;
                continue;
            }
            if (i < body.size() && body[i] == ']') {
                out = std::move(vals);
                return true;
            }
            return false;
        }
        return false;
    }
    return false;
}

bool extract_json_bool(const std::string &body, const char *key, bool &out) {
    if (!key)
        return false;
    std::string pat = std::string("\"") + key + "\"";
    size_t pos = body.find(pat);
    if (pos == std::string::npos)
        return false;
    size_t i = skip_ws(body, pos + pat.size());
    if (i >= body.size() || body[i] != ':')
        return false;
    i = skip_ws(body, i + 1);
    if (body.compare(i, 4, "true") == 0) {
        out = true;
        return true;
    }
    if (body.compare(i, 5, "false") == 0) {
        out = false;
        return true;
    }
    return false;
}

bool extract_image_url_from_part(const std::string &part, std::string &url) {
    url.clear();
    std::string typ;
    extract_json_string(part, "type", typ);
    if (typ == "image_url" || typ == "input_image" || typ.empty()) {
        if (extract_json_string(part, "url", url) && !url.empty())
            return true;
    }
    return false;
}

namespace {

using json = nlohmann::json;

size_t match_brace(const std::string &s, size_t open) {
    if (open >= s.size() || s[open] != '{')
        return std::string::npos;
    int depth = 0;
    bool in_str = false;
    for (size_t i = open; i < s.size(); ++i) {
        char c = s[i];
        if (in_str) {
            if (c == '\\') {
                ++i;
                continue;
            }
            if (c == '"')
                in_str = false;
            continue;
        }
        if (c == '"')
            in_str = true;
        else if (c == '{')
            ++depth;
        else if (c == '}') {
            --depth;
            if (depth == 0)
                return i;
        }
    }
    return std::string::npos;
}

void parse_content_array(const std::string &slice, size_t arr, ChatMessage &msg) {
    size_t i = arr + 1;
    while (i < slice.size()) {
        i = skip_ws(slice, i);
        if (i >= slice.size() || slice[i] == ']')
            break;
        if (slice[i] == ',') {
            ++i;
            continue;
        }
        if (slice[i] != '{')
            break;
        size_t end = match_brace(slice, i);
        if (end == std::string::npos)
            break;
        std::string part = slice.substr(i, end - i + 1);
        std::string typ, text, url;
        extract_json_string(part, "type", typ);
        if (typ == "text" || typ.empty()) {
            if (extract_json_string(part, "text", text) && !text.empty())
                msg.content += text;
        }
        if (typ == "image_url" || typ == "input_image" || extract_image_url_from_part(part, url)) {
            if (url.empty())
                extract_json_string(part, "url", url);
            if (!url.empty())
                msg.image_urls.push_back(url);
        }
        i = end + 1;
    }
}

bool apply_response_format(const json &rf, std::string &grammar, std::string &err) {
    std::string type;
    json schema_src;
    if (rf.is_string()) {
        type = rf.get<std::string>();
    } else if (rf.is_object()) {
        if (rf.contains("type") && rf["type"].is_string())
            type = rf["type"].get<std::string>();
        if (rf.contains("json_schema"))
            schema_src = rf["json_schema"];
        else
            schema_src = rf;
    } else {
        return false;
    }
    if (type == "text")
        return false;
    if (type == "json_object") {
        grammar = json_object_gbnf();
        return !grammar.empty();
    }
    if (type == "json_schema") {
        grammar = json_schema_to_gbnf(schema_src.dump(), err);
        if (grammar.empty())
            return false;
        return true;
    }
    return false;
}

} // namespace

bool extract_response_format(const std::string &body, std::string &grammar, std::string &err) {
    err.clear();
    if (!grammar.empty())
        return true;

    try {
        nlohmann::json root = nlohmann::json::parse(body);
        if (root.is_object() && root.contains("response_format"))
            return apply_response_format(root["response_format"], grammar, err);
        return false;
    } catch (...) {
    }

    const std::string pat = "\"response_format\"";
    size_t pos = body.find(pat);
    if (pos == std::string::npos)
        return false;
    size_t i = skip_ws(body, pos + pat.size());
    if (i >= body.size() || body[i] != ':')
        return false;
    i = skip_ws(body, i + 1);
    if (i >= body.size())
        return false;
    if (body[i] == '"') {
        std::string typ;
        if (!parse_json_string(body, i, typ))
            return false;
        if (typ == "json_object") {
            grammar = json_object_gbnf();
            return !grammar.empty();
        }
        return false;
    }
    if (body[i] != '{')
        return false;
    size_t end = match_brace(body, i);
    if (end == std::string::npos)
        return false;
    try {
        nlohmann::json rf = nlohmann::json::parse(body.substr(i, end - i + 1));
        return apply_response_format(rf, grammar, err);
    } catch (...) {
        return false;
    }
}

bool extract_chat_messages(const std::string &body, std::vector<ChatMessage> &out) {
    out.clear();
    size_t mpos = body.find("\"messages\"");
    if (mpos == std::string::npos)
        return false;
    size_t arr = body.find('[', mpos);
    if (arr == std::string::npos)
        return false;
    size_t i = arr + 1;
    while (i < body.size()) {
        i = skip_ws(body, i);
        if (i >= body.size() || body[i] == ']')
            break;
        if (body[i] == ',') {
            ++i;
            continue;
        }
        size_t obj = body.find('{', i);
        size_t close = body.find(']', i);
        if (obj == std::string::npos || (close != std::string::npos && close < obj))
            break;
        size_t end = match_brace(body, obj);
        if (end == std::string::npos)
            break;
        std::string slice = body.substr(obj, end - obj + 1);
        ChatMessage msg;
        extract_json_string(slice, "role", msg.role);
        size_t cpos = slice.find("\"content\"");
        bool got_array = false;
        if (cpos != std::string::npos) {
            size_t after = skip_ws(slice, cpos + 9);
            if (after < slice.size() && slice[after] == ':') {
                after = skip_ws(slice, after + 1);
                if (after < slice.size() && slice[after] == '[') {
                    parse_content_array(slice, after, msg);
                    got_array = true;
                }
            }
        }
        if (!got_array)
            extract_json_string(slice, "content", msg.content);
        if (!extract_json_string(slice, "tool_name", msg.tool_name))
            extract_json_string(slice, "name", msg.tool_name);
        extract_json_string(slice, "reasoning", msg.reasoning);
        k3_extract_tool_calls_json(slice, msg.tool_calls);
        if (!msg.role.empty())
            out.push_back(msg);
        i = end + 1;
    }
    return !out.empty();
}

bool extract_json_int(const std::string &body, const char *key, int &out) {
    if (!key)
        return false;
    std::string pat = std::string("\"") + key + "\"";
    size_t pos = 0;
    while ((pos = body.find(pat, pos)) != std::string::npos) {
        size_t i = skip_ws(body, pos + pat.size());
        if (i < body.size() && body[i] == ':') {
            i = skip_ws(body, i + 1);
            if (i < body.size() &&
                (body[i] == '-' || std::isdigit(static_cast<unsigned char>(body[i])))) {
                size_t end = i;
                if (body[end] == '-')
                    ++end;
                if (end < body.size() && std::isdigit(static_cast<unsigned char>(body[end]))) {
                    while (end < body.size() && std::isdigit(static_cast<unsigned char>(body[end])))
                        ++end;
                    try {
                        out = std::stoi(body.substr(i, end - i));
                        return true;
                    } catch (...) {
                        return false;
                    }
                }
            }
        }
        pos += 1;
    }
    return false;
}

bool extract_json_number(const std::string &body, const char *key, float &out) {
    if (!key)
        return false;
    std::string pat = std::string("\"") + key + "\"";
    size_t pos = 0;
    while ((pos = body.find(pat, pos)) != std::string::npos) {
        size_t i = skip_ws(body, pos + pat.size());
        if (i < body.size() && body[i] == ':') {
            i = skip_ws(body, i + 1);
            if (i < body.size()) {
                char *end = nullptr;
                float v = std::strtof(body.c_str() + i, &end);
                if (end && end != body.c_str() + i) {
                    out = v;
                    return true;
                }
            }
        }
        pos += 1;
    }
    return false;
}

} // namespace mvllm
