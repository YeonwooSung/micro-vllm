#include "anthropic.hpp"
#include "http_server.hpp"
#include "../engine.hpp"
#include "../model/family.hpp"
#include "../tok/glm_tools.hpp"
#include "../tok/k3_tools.hpp"

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include <cerrno>
#include <cstring>
#include <ctime>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/socket.h>

namespace mvllm {
namespace {

using json = nlohmann::json;

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

void write_http(int fd, int code, const char *reason, const std::string &body) {
    std::ostringstream os;
    os << "HTTP/1.1 " << code << ' ' << reason << "\r\n"
       << "Content-Type: application/json\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Connection: close\r\n"
       << "\r\n"
       << body;
    const std::string resp = os.str();
    send_all(fd, resp.data(), resp.size());
}

void write_sse_headers(int fd) {
    const char *h = "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Connection: close\r\n"
                    "\r\n";
    send_all(fd, h, std::strlen(h));
}

std::string sse_event(const char *name, const json &payload) {
    std::string o = "event: ";
    o += name;
    o += "\ndata: ";
    o += payload.dump(-1, ' ', false);
    o += "\n\n";
    return o;
}

std::string jget_str(const json &o, const char *key) {
    if (!o.is_object() || !o.contains(key) || !o[key].is_string())
        return {};
    return o[key].get<std::string>();
}

bool jget_num(const json &o, const char *key, float &out) {
    if (!o.is_object() || !o.contains(key) || !o[key].is_number())
        return false;
    out = o[key].get<float>();
    return true;
}

bool jget_int(const json &o, const char *key, int &out) {
    if (!o.is_object() || !o.contains(key) || !o[key].is_number())
        return false;
    if (o[key].is_number_integer())
        out = o[key].get<int>();
    else
        out = static_cast<int>(o[key].get<double>());
    return true;
}

bool cache_related_key(const std::string &k) {
    return k == "cache_control" || k == "cache" || k.rfind("cache", 0) == 0;
}

// Unknown block whose only extras are cache_control / cache-* — skip, do not fail.
bool skippable_cache_block(const json &block) {
    if (!block.is_object())
        return false;
    const std::string typ = jget_str(block, "type");
    if (typ == "text" || typ == "image" || typ == "thinking" || typ == "tool_use" ||
        typ == "tool_result")
        return false;
    if (cache_related_key(typ))
        return true;
    bool any_cache = false;
    for (auto it = block.begin(); it != block.end(); ++it) {
        const std::string k = it.key();
        if (k == "type")
            continue;
        if (cache_related_key(k)) {
            any_cache = true;
            continue;
        }
        return false;
    }
    return any_cache;
}

// Anthropic image block → OpenAI-style URL / data URI for ChatMessage.image_urls.
bool parse_image_url(const json &block, std::string &url) {
    url.clear();
    if (!block.is_object())
        return false;
    const json *src = &block;
    if (block.contains("source") && block["source"].is_object())
        src = &block["source"];
    const json &s = *src;
    const std::string typ = jget_str(s, "type");
    if (typ == "url" || (typ.empty() && s.contains("url"))) {
        std::string u = jget_str(s, "url");
        if (u.empty())
            return false;
        url = std::move(u);
        return true;
    }
    if (typ == "base64" || s.contains("data")) {
        std::string data = jget_str(s, "data");
        if (data.empty())
            return false;
        std::string mt = jget_str(s, "media_type");
        if (mt.empty())
            mt = "image/png";
        url = "data:" + mt + ";base64," + data;
        return true;
    }
    return false;
}

bool collect_text(const json &blocks, std::string &out, std::string &err) {
    if (blocks.is_string()) {
        out = blocks.get<std::string>();
        return true;
    }
    if (!blocks.is_array()) {
        err = "Content must be a string or an array of blocks.";
        return false;
    }
    out.clear();
    for (const auto &block : blocks) {
        if (!block.is_object()) {
            err = "text blocks only are supported here";
            return false;
        }
        if (skippable_cache_block(block))
            continue;
        if (jget_str(block, "type") != "text") {
            err = "text blocks only are supported here";
            return false;
        }
        if (!block.contains("text") || !block["text"].is_string()) {
            err = "Text blocks require a string `text` field.";
            return false;
        }
        out += block["text"].get<std::string>();
    }
    return true;
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

bool parse_output_tools(const std::string &text, std::string &stripped,
                        std::vector<K3ParsedCall> &calls) {
    stripped.clear();
    calls.clear();
    if (glm_parse_tool_calls(text, stripped, calls) && !calls.empty())
        return true;
    if (k3_parse_tool_calls(text, stripped, calls) && !calls.empty())
        return true;
    stripped = text;
    calls.clear();
    return false;
}

json tool_input(const std::string &arguments) {
    json v = json::parse(arguments, nullptr, false);
    if (v.is_object())
        return v;
    return json::object();
}

json content_blocks(const GenResult &out, std::string &stripped, std::vector<K3ParsedCall> &calls) {
    parse_output_tools(out.text, stripped, calls);
    json content = json::array();
    if (!out.reasoning.empty()) {
        json b = json::object();
        b["type"] = "text";
        b["text"] = out.reasoning;
        content.push_back(std::move(b));
    }
    if (!stripped.empty() || calls.empty()) {
        json b = json::object();
        b["type"] = "text";
        b["text"] = stripped;
        content.push_back(std::move(b));
    }
    for (size_t i = 0; i < calls.size(); ++i) {
        json b = json::object();
        b["type"] = "tool_use";
        b["id"] = calls[i].id.empty() ? ("toolu_" + std::to_string(i + 1)) : calls[i].id;
        b["name"] = calls[i].name;
        b["input"] = tool_input(calls[i].arguments);
        content.push_back(std::move(b));
    }
    return content;
}

const char *infer_stop_reason(const GenResult &out, const std::vector<K3ParsedCall> &calls,
                              const GenParams *gp, Engine *engine) {
    if (!calls.empty())
        return "tool_use";
    if (out.stopped_by_stop)
        return "end_turn";
    if (gp && gp->max_new_tokens > 0 && out.completion_tokens >= gp->max_new_tokens) {
        if (!out.tokens.empty() && engine &&
            is_stop_token(out.tokens.back(), engine->config(), gp->eos))
            return "end_turn";
        return "max_tokens";
    }
    return "end_turn";
}

const char *echo_stop_sequence(const GenResult &out, const GenParams &gp) {
    if (!out.stopped_by_stop || gp.stop.size() != 1 || gp.stop[0].empty())
        return nullptr;
    return gp.stop[0].c_str();
}

std::string format_messages_response(const std::string &id, const std::string &model,
                                     const GenResult &out, const char *stop_reason,
                                     const char *stop_sequence = nullptr) {
    std::string stripped;
    std::vector<K3ParsedCall> calls;
    json content = content_blocks(out, stripped, calls);
    if (!stop_reason || !stop_reason[0])
        stop_reason = infer_stop_reason(out, calls, nullptr, nullptr);
    json usage = json::object();
    usage["input_tokens"] = out.prompt_tokens;
    usage["output_tokens"] = out.completion_tokens;
    json resp = json::object();
    resp["id"] = id;
    resp["type"] = "message";
    resp["role"] = "assistant";
    resp["model"] = model;
    resp["content"] = std::move(content);
    resp["stop_reason"] = stop_reason;
    if (stop_sequence && stop_sequence[0])
        resp["stop_sequence"] = stop_sequence;
    else
        resp["stop_sequence"] = nullptr;
    resp["usage"] = std::move(usage);
    return resp.dump(-1, ' ', false);
}

std::string make_msg_id() {
    static unsigned seq = 0;
    std::ostringstream os;
    os << "msg_" << std::hex << static_cast<unsigned long long>(std::time(nullptr)) << '_'
       << (++seq);
    return os.str();
}

std::string error_body(const std::string &msg) {
    return std::string("{\"error\":\"") + json_escape(msg) + "\"}";
}

K3ToolCall tool_use_to_call(const json &block) {
    K3ToolCall c;
    c.name = jget_str(block, "name");
    if (block.contains("input") && block["input"].is_object())
        c.json = block["input"].dump(-1, ' ', false);
    else
        c.json = "{}";
    return c;
}

} // namespace

bool anthropic_to_chat(const std::string &body, std::vector<ChatMessage> &msgs, GenParams &gp,
                       std::string &err) {
    msgs.clear();
    err.clear();
    gp.tools.clear();
    gp.tool_choice.clear();
    gp.think = false;
    json j = json::parse(body, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        err = "invalid json";
        return false;
    }

    if (!j.contains("max_tokens") || j["max_tokens"].is_null()) {
        err = "max_tokens is required";
        return false;
    }
    if (j["max_tokens"].is_number_integer())
        gp.max_new_tokens = j["max_tokens"].get<int>();
    else if (j["max_tokens"].is_number())
        gp.max_new_tokens = static_cast<int>(j["max_tokens"].get<double>());
    else {
        err = "max_tokens is required";
        return false;
    }
    if (gp.max_new_tokens < 0)
        gp.max_new_tokens = 0;

    float num = 0.f;
    if (jget_num(j, "temperature", num))
        gp.temperature = num;
    if (jget_num(j, "top_p", num))
        gp.top_p = num;
    if (jget_num(j, "min_p", num))
        gp.min_p = num;
    if (jget_num(j, "frequency_penalty", num))
        gp.frequency_penalty = num;
    if (jget_num(j, "presence_penalty", num))
        gp.presence_penalty = num;
    if (jget_num(j, "repetition_penalty", num))
        gp.repetition_penalty = num;
    int iv = 0;
    if (jget_int(j, "top_k", iv))
        gp.top_k = iv < 0 ? 0 : iv;
    if (j.contains("seed") && j["seed"].is_number_integer()) {
        if (j["seed"].is_number_unsigned()) {
            gp.seed = j["seed"].get<uint64_t>();
        } else {
            const int64_t s = j["seed"].get<int64_t>();
            if (s >= 0)
                gp.seed = static_cast<uint64_t>(s);
        }
    }
    if (j.contains("cache_slot") && j["cache_slot"].is_number_integer())
        gp.cache_slot = j["cache_slot"].get<int>();

    if (j.contains("stop_sequences") && j["stop_sequences"].is_array()) {
        for (const auto &s : j["stop_sequences"]) {
            if (!s.is_string())
                continue;
            std::string v = s.get<std::string>();
            if (!v.empty())
                gp.stop.push_back(std::move(v));
        }
    }

    if (j.contains("thinking") && !j["thinking"].is_null()) {
        if (!j["thinking"].is_object()) {
            err = "`thinking` must be an object.";
            return false;
        }
        const json &th = j["thinking"];
        const std::string typ = jget_str(th, "type");
        const bool disabled = typ == "disabled";
        gp.think = typ == "enabled";
        int budget = 0;
        if (!disabled && jget_int(th, "budget_tokens", budget) && budget > 0) {
            gp.think = true;
            if (budget >= 16000)
                gp.reasoning_effort = "high";
            else if (budget >= 4000)
                gp.reasoning_effort = "medium";
            else
                gp.reasoning_effort = "low";
        }
    }

    if (j.contains("system") && !j["system"].is_null()) {
        const json &system = j["system"];
        std::string text;
        if (system.is_string()) {
            text = system.get<std::string>();
        } else if (system.is_array()) {
            if (!collect_text(system, text, err))
                return false;
        } else {
            err = "`system` must be a string or an array of text blocks.";
            return false;
        }
        if (!text.empty()) {
            ChatMessage sys;
            sys.role = "system";
            sys.content = std::move(text);
            msgs.push_back(std::move(sys));
        }
    }

    std::unordered_map<std::string, std::string> tool_names;
    int tool_index = 0;

    if (j.contains("messages") && !j["messages"].is_null()) {
        if (!j["messages"].is_array()) {
            err = "`messages` must be an array.";
            return false;
        }
        for (const auto &message : j["messages"]) {
            if (!message.is_object()) {
                err = "Each message must be an object.";
                return false;
            }
            const std::string role = jget_str(message, "role");
            if (role != "user" && role != "assistant") {
                err = "Input message role is not supported. Anthropic messages are `user` or "
                      "`assistant`; a system prompt goes in the top-level `system`.";
                return false;
            }
            if (!message.contains("content")) {
                err = "Message content must be a string or an array of blocks.";
                return false;
            }
            const json &content = message["content"];
            if (content.is_string()) {
                ChatMessage m;
                m.role = role;
                m.content = content.get<std::string>();
                msgs.push_back(std::move(m));
                continue;
            }
            if (!content.is_array()) {
                err = "Message content must be a string or an array of blocks.";
                return false;
            }
            std::string texts;
            std::string reasoning;
            std::vector<K3ToolCall> calls;
            std::vector<ChatMessage> results;
            std::vector<std::string> images;
            for (const auto &block : content) {
                if (!block.is_object()) {
                    err = "Each content block must be an object.";
                    return false;
                }
                const std::string kind = jget_str(block, "type");
                if (kind == "text") {
                    if (!block.contains("text") || !block["text"].is_string()) {
                        err = "Text blocks require a string `text` field.";
                        return false;
                    }
                    texts += block["text"].get<std::string>();
                } else if (kind == "image") {
                    std::string url;
                    if (parse_image_url(block, url) && !url.empty())
                        images.push_back(std::move(url));
                } else if (kind == "thinking") {
                    if (role != "assistant") {
                        err = "`thinking` blocks are valid only in assistant messages.";
                        return false;
                    }
                    if (block.contains("thinking") && block["thinking"].is_string())
                        reasoning += block["thinking"].get<std::string>();
                } else if (kind == "tool_use") {
                    const std::string name = jget_str(block, "name");
                    if (name.empty()) {
                        err = "`tool_use` blocks require a string `name`.";
                        return false;
                    }
                    if (block.contains("input") && !block["input"].is_null() &&
                        !block["input"].is_object()) {
                        err = "`tool_use.input` must be an object.";
                        return false;
                    }
                    K3ToolCall c = tool_use_to_call(block);
                    c.name = name;
                    c.index = static_cast<int>(calls.size()) + 1;
                    const std::string id = jget_str(block, "id");
                    if (!id.empty())
                        tool_names[id] = name;
                    calls.push_back(std::move(c));
                } else if (kind == "tool_result") {
                    ChatMessage t;
                    t.role = "tool";
                    if (!collect_text(block.contains("content") ? block["content"] : json(""),
                                      t.content, err))
                        return false;
                    const std::string use_id = jget_str(block, "tool_use_id");
                    auto it = tool_names.find(use_id);
                    if (it != tool_names.end())
                        t.tool_name = it->second;
                    else
                        t.tool_name = jget_str(block, "name");
                    t.tool_index = ++tool_index;
                    results.push_back(std::move(t));
                } else if (skippable_cache_block(block)) {
                    continue;
                } else {
                    err = "unsupported content type";
                    return false;
                }
            }
            for (auto &t : results)
                msgs.push_back(std::move(t));
            if (role == "assistant") {
                if (!texts.empty() || !reasoning.empty() || !calls.empty() || !images.empty()) {
                    ChatMessage m;
                    m.role = "assistant";
                    m.content = std::move(texts);
                    m.reasoning = std::move(reasoning);
                    m.tool_calls = std::move(calls);
                    m.image_urls = std::move(images);
                    msgs.push_back(std::move(m));
                }
            } else if (!texts.empty() || !images.empty() || results.empty()) {
                ChatMessage m;
                m.role = "user";
                m.content = std::move(texts);
                m.image_urls = std::move(images);
                msgs.push_back(std::move(m));
            }
        }
    }

    if (j.contains("tools") && !j["tools"].is_null()) {
        if (!j["tools"].is_array()) {
            err = "`tools` must be an array.";
            return false;
        }
        for (const auto &tool : j["tools"]) {
            if (!tool.is_object()) {
                err = "Each tool must be an object.";
                return false;
            }
            K3ToolDecl d;
            d.name = jget_str(tool, "name");
            if (d.name.empty()) {
                err = "Each tool requires a string `name`.";
                return false;
            }
            d.description = jget_str(tool, "description");
            if (tool.contains("input_schema") && !tool["input_schema"].is_null()) {
                if (!tool["input_schema"].is_object()) {
                    err = "`input_schema` must be an object.";
                    return false;
                }
                d.parameters_json = tool["input_schema"].dump(-1, ' ', false);
            } else {
                d.parameters_json = "{\"type\":\"object\",\"properties\":{}}";
            }
            gp.tools.push_back(std::move(d));
        }
    }

    if (j.contains("tool_choice") && !j["tool_choice"].is_null()) {
        if (!j["tool_choice"].is_object()) {
            err = "`tool_choice` must be an object.";
            return false;
        }
        const std::string kind = jget_str(j["tool_choice"], "type");
        if (kind == "auto") {
            gp.tool_choice = "auto";
        } else if (kind == "any") {
            gp.tool_choice = "required";
        } else if (kind == "none") {
            gp.tool_choice = "none";
        } else if (kind == "tool") {
            const std::string name = jget_str(j["tool_choice"], "name");
            if (name.empty()) {
                err = "`tool_choice.name` is required when type is `tool`.";
                return false;
            }
            gp.tool_choice = name;
        } else {
            err = "`tool_choice.type` must be auto, any, none, or tool.";
            return false;
        }
        apply_tool_choice(gp);
    }

    return true;
}

std::string anthropic_messages_response(const std::string &id, const std::string &model,
                                        const GenResult &out) {
    return format_messages_response(id, model, out, nullptr);
}

std::string anthropic_sse_start(const std::string &id, const std::string &model) {
    json usage = json::object();
    usage["input_tokens"] = 0;
    usage["output_tokens"] = 0;
    json message = json::object();
    message["id"] = id;
    message["type"] = "message";
    message["role"] = "assistant";
    message["model"] = model;
    message["content"] = json::array();
    message["stop_reason"] = nullptr;
    message["usage"] = std::move(usage);
    json payload = json::object();
    payload["type"] = "message_start";
    payload["message"] = std::move(message);
    return sse_event("message_start", payload);
}

std::string anthropic_sse_delta(const std::string &text, int index) {
    json delta = json::object();
    delta["type"] = "text_delta";
    delta["text"] = text;
    json payload = json::object();
    payload["type"] = "content_block_delta";
    payload["index"] = index;
    payload["delta"] = std::move(delta);
    return sse_event("content_block_delta", payload);
}

std::string anthropic_sse_stop(const char *stop_reason, const char *stop_sequence,
                               int output_tokens, int input_tokens) {
    if (!stop_reason || !stop_reason[0])
        stop_reason = "end_turn";
    json delta = json::object();
    delta["stop_reason"] = stop_reason;
    if (stop_sequence && stop_sequence[0])
        delta["stop_sequence"] = stop_sequence;
    else
        delta["stop_sequence"] = nullptr;
    json usage = json::object();
    usage["input_tokens"] = input_tokens;
    usage["output_tokens"] = output_tokens;
    json payload = json::object();
    payload["type"] = "message_delta";
    payload["delta"] = std::move(delta);
    payload["usage"] = std::move(usage);
    std::string o = sse_event("message_delta", payload);
    json stop = json::object();
    stop["type"] = "message_stop";
    o += sse_event("message_stop", stop);
    return o;
}

std::string anthropic_sse_block_start(int index, const char *block_type) {
    if (!block_type)
        block_type = "text";
    json content_block = json::object();
    if (std::strcmp(block_type, "thinking") == 0) {
        content_block["type"] = "thinking";
        content_block["thinking"] = "";
    } else {
        content_block["type"] = "text";
        content_block["text"] = "";
    }
    json payload = json::object();
    payload["type"] = "content_block_start";
    payload["index"] = index;
    payload["content_block"] = std::move(content_block);
    return sse_event("content_block_start", payload);
}

std::string anthropic_sse_thinking_delta(const std::string &text) {
    json delta = json::object();
    delta["type"] = "thinking_delta";
    delta["thinking"] = text;
    json payload = json::object();
    payload["type"] = "content_block_delta";
    payload["index"] = 0;
    payload["delta"] = std::move(delta);
    return sse_event("content_block_delta", payload);
}

std::string anthropic_sse_block_stop(int index) {
    json payload = json::object();
    payload["type"] = "content_block_stop";
    payload["index"] = index;
    return sse_event("content_block_stop", payload);
}

void handle_anthropic_messages(int fd, const std::string &body, Engine *engine) {
    if (!engine) {
        write_http(fd, 503, "Service Unavailable", "{\"error\":\"no engine\"}");
        return;
    }

    std::vector<ChatMessage> msgs;
    GenParams gp;
    std::string err;
    if (!anthropic_to_chat(body, msgs, gp, err)) {
        write_http(fd, 400, "Bad Request", error_body(err));
        return;
    }

    bool stream = false;
    extract_json_bool(body, "stream", stream);

    GenResult out;
    std::string gerr;
    Status st = msgs.empty() ? engine->generate(std::string(), gp, out, gerr)
                             : engine->generate_chat(msgs, gp, out, gerr);
    if (st != Status::Ok) {
        write_http(fd, 500, "Internal Server Error", error_body(gerr.empty() ? "generate failed" : gerr));
        return;
    }

    const std::string id = make_msg_id();
    const std::string model = engine->model_id().empty() ? "micro-vllm" : engine->model_id();
    std::string stripped;
    std::vector<K3ParsedCall> calls;
    parse_output_tools(out.text, stripped, calls);
    const char *reason = infer_stop_reason(out, calls, &gp, engine);
    const char *seq = echo_stop_sequence(out, gp);

    if (!stream) {
        write_http(fd, 200, "OK", format_messages_response(id, model, out, reason, seq));
        return;
    }

    write_sse_headers(fd);
    std::string ev = anthropic_sse_start(id, model);
    send_all(fd, ev.data(), ev.size());
    int text_index = 0;
    int index = 1;
    if (!out.reasoning.empty()) {
        ev = anthropic_sse_block_start(0, "thinking");
        send_all(fd, ev.data(), ev.size());
        ev = anthropic_sse_thinking_delta(out.reasoning);
        send_all(fd, ev.data(), ev.size());
        ev = anthropic_sse_block_stop(0);
        send_all(fd, ev.data(), ev.size());
        text_index = 1;
        index = 2;
    }
    ev = anthropic_sse_block_start(text_index);
    send_all(fd, ev.data(), ev.size());
    if (!stripped.empty() || calls.empty()) {
        ev = anthropic_sse_delta(stripped, text_index);
        send_all(fd, ev.data(), ev.size());
    }
    ev = anthropic_sse_block_stop(text_index);
    send_all(fd, ev.data(), ev.size());
    for (size_t i = 0; i < calls.size(); ++i) {
        const int blk = index + static_cast<int>(i);
        const std::string cid =
            calls[i].id.empty() ? ("toolu_" + std::to_string(i + 1)) : calls[i].id;
        json start_block = json::object();
        start_block["type"] = "tool_use";
        start_block["id"] = cid;
        start_block["name"] = calls[i].name;
        start_block["input"] = json::object();
        json start = json::object();
        start["type"] = "content_block_start";
        start["index"] = blk;
        start["content_block"] = std::move(start_block);
        ev = sse_event("content_block_start", start);
        send_all(fd, ev.data(), ev.size());

        json dlt = json::object();
        dlt["type"] = "input_json_delta";
        dlt["partial_json"] = calls[i].arguments.empty() ? "{}" : calls[i].arguments;
        json delta = json::object();
        delta["type"] = "content_block_delta";
        delta["index"] = blk;
        delta["delta"] = std::move(dlt);
        ev = sse_event("content_block_delta", delta);
        send_all(fd, ev.data(), ev.size());

        json stop = json::object();
        stop["type"] = "content_block_stop";
        stop["index"] = blk;
        ev = sse_event("content_block_stop", stop);
        send_all(fd, ev.data(), ev.size());
    }
    ev = anthropic_sse_stop(reason, seq, out.completion_tokens, out.prompt_tokens);
    send_all(fd, ev.data(), ev.size());
}

} // namespace mvllm
