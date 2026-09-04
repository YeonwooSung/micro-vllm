#include "http_server.hpp"
#include "../engine.hpp"

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

#include <sstream>
#include <string>

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

void http_reply(int fd, int code, const char *reason, const std::string &body) {
    std::ostringstream os;
    os << "HTTP/1.1 " << code << ' ' << reason << "\r\n"
       << "Content-Type: application/json\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Connection: close\r\n"
       << "\r\n"
       << body;
    std::string resp = os.str();
    send_all(fd, resp.data(), resp.size());
}

std::string model_name(Engine *engine) {
    if (engine && !engine->model_id().empty())
        return engine->model_id();
    return "micro-vllm";
}

bool read_http(int fd, std::string &method, std::string &path, std::string &body) {
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

} // namespace

HttpServer::HttpServer() = default;

HttpServer::~HttpServer() {
    stop();
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
}

void HttpServer::set_engine(Engine *engine) { engine_ = engine; }

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

        ++stats_.requests;

        std::string method, path, body;
        if (!read_http(cfd, method, path, body)) {
            http_reply(cfd, 400, "Bad Request", "{\"error\":\"bad request\"}");
            ::close(cfd);
            continue;
        }

        if (method == "GET" && path == "/health") {
            http_reply(cfd, 200, "OK", "{\"ok\":true}");
        } else if (method == "GET" && path == "/v1/models") {
            http_reply(cfd, 200, "OK", openai_models_response(model_name(engine_)));
        } else if (method == "POST" &&
                   (path == "/v1/chat/completions" || path == "/v1/completions" ||
                    path == "/v1/videos/generations")) {
            if (!engine_) {
                http_reply(cfd, 503, "Service Unavailable", "{\"error\":\"no engine\"}");
                ::close(cfd);
                continue;
            }
            if (path == "/v1/videos/generations") {
                H3GenParams hp;
                extract_json_string(body, "prompt", hp.prompt);
                extract_json_int(body, "width", hp.width);
                extract_json_int(body, "height", hp.height);
                extract_json_int(body, "frames", hp.frames);
                extract_json_int(body, "steps", hp.steps);
                H3GenResult out;
                std::string gerr;
                Status st = engine_->generate_video(hp, out, gerr);
                if (st != Status::Ok) {
                    http_reply(cfd, 500, "Internal Server Error",
                               "{\"error\":\"" + json_escape(gerr) + "\"}");
                } else {
                    std::string jb = "{\"output_path\":\"" + json_escape(out.output_path) +
                                     "\",\"note\":\"" + json_escape(out.note) + "\"}";
                    http_reply(cfd, 200, "OK", jb);
                }
            } else {
                std::string prompt;
                std::vector<ChatMessage> msgs;
                if (path == "/v1/chat/completions") {
                    if (!extract_chat_messages(body, msgs))
                        extract_last_json_string(body, "content", prompt);
                } else
                    extract_json_string(body, "prompt", prompt);
                int max_tokens = 32;
                extract_json_int(body, "max_tokens", max_tokens);
                if (max_tokens < 0)
                    max_tokens = 0;
                GenParams gp;
                gp.max_new_tokens = max_tokens;
                extract_json_number(body, "temperature", gp.temperature);
                extract_json_number(body, "top_p", gp.top_p);
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
                GenResult out;
                std::string gerr;
                Status st = Status::Ok;
                if (!msgs.empty())
                    st = engine_->generate_chat(msgs, gp, out, gerr);
                else
                    st = engine_->generate(prompt, gp, out, gerr);
                if (st != Status::Ok) {
                    http_reply(cfd, 500, "Internal Server Error",
                               "{\"error\":\"" + json_escape(gerr) + "\"}");
                } else {
                    stats_.tokens_out += static_cast<uint64_t>(out.completion_tokens);
                    std::string id = (path == "/v1/chat/completions" ? "chatcmpl-" : "cmpl-") +
                                     std::to_string(stats_.requests);
                    http_reply(cfd, 200, "OK",
                               openai_chat_response(id, model_name(engine_), out.text,
                                                    out.prompt_tokens, out.completion_tokens));
                }
            }
        } else {
            http_reply(cfd, 404, "Not Found", "{\"error\":\"not found\"}");
        }
        ::close(cfd);
    }
    return Status::Ok;
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
                                 int completion_tokens) {
    int total = prompt_tokens + completion_tokens;
    std::ostringstream os;
    os << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion\",\"created\":"
       << static_cast<long long>(std::time(nullptr)) << ",\"model\":\"" << json_escape(model)
       << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
       << json_escape(content) << "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":"
       << prompt_tokens << ",\"completion_tokens\":" << completion_tokens
       << ",\"total_tokens\":" << total << "}}";
    return os.str();
}

std::string openai_models_response(const std::string &id) {
    return "{\"object\":\"list\",\"data\":[{\"id\":\"" + json_escape(id) +
           "\",\"object\":\"model\",\"owned_by\":\"micro-vllm\"}]}";
}

bool extract_json_string(const std::string &body, const char *key, std::string &out) {
    return extract_json_string_from(body, key, 0, out, nullptr);
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
        size_t obj = body.find('{', i);
        size_t close = body.find(']', i);
        if (obj == std::string::npos || (close != std::string::npos && close < obj))
            break;
        size_t end = body.find('}', obj);
        if (end == std::string::npos)
            break;
        std::string slice = body.substr(obj, end - obj + 1);
        ChatMessage msg;
        extract_json_string(slice, "role", msg.role);
        extract_json_string(slice, "content", msg.content);
        if (!extract_json_string(slice, "tool_name", msg.tool_name))
            extract_json_string(slice, "name", msg.tool_name);
        extract_json_string(slice, "reasoning", msg.reasoning);
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
