#include "glm_tools.hpp"

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include <cctype>
#include <utility>

namespace mvllm {
namespace {

using json = nlohmann::ordered_json;

constexpr const char *kBoxStart = "<tool_call>";
constexpr const char *kBoxEnd = "</tool_call>";
constexpr const char *kArgKeyOpen = "<arg_key>";
constexpr const char *kArgKeyClose = "</arg_key>";
constexpr const char *kArgValOpen = "<arg_value>";
constexpr const char *kArgValClose = "</arg_value>";

size_t skip_ws(const std::string &s, size_t i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
        ++i;
    return i;
}

std::string trim_copy(const std::string &s) {
    size_t a = skip_ws(s, 0);
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])))
        --b;
    return s.substr(a, b - a);
}

bool is_ident_char(unsigned char c) {
    return std::isalnum(c) || c == '_' || c == '.' || c == '-';
}

std::string leading_name(const std::string &inner) {
    const size_t i = skip_ws(inner, 0);
    size_t j = i;
    while (j < inner.size() && is_ident_char(static_cast<unsigned char>(inner[j])))
        ++j;
    if (j > i)
        return inner.substr(i, j - i);
    return trim_copy(inner);
}

bool is_just_name(const std::string &inner) {
    const std::string t = trim_copy(inner);
    if (t.empty())
        return false;
    for (unsigned char c : t) {
        if (!is_ident_char(c))
            return false;
    }
    return true;
}

// Drop a trailing incomplete </tool_call> ("</tool_cal") so an unclosed tail can be recovered.
void strip_partial_end(std::string &inner) {
    const size_t lt = inner.rfind('<');
    if (lt == std::string::npos)
        return;
    const std::string tail = inner.substr(lt);
    const std::string end = kBoxEnd;
    if (end.compare(0, tail.size(), tail) == 0)
        inner.resize(lt);
}

void erase_all(std::string &s, const char *pat) {
    const size_t n = std::char_traits<char>::length(pat);
    size_t pos = 0;
    while ((pos = s.find(pat, pos)) != std::string::npos)
        s.erase(pos, n);
}

bool next_arg_pair(const std::string &inner, size_t pos, size_t &key_s, size_t &key_e,
                   size_t &val_s, size_t &val_e, size_t &next) {
    const size_t ko = inner.find(kArgKeyOpen, pos);
    if (ko == std::string::npos)
        return false;
    key_s = ko + std::char_traits<char>::length(kArgKeyOpen);
    const size_t kc = inner.find(kArgKeyClose, key_s);
    if (kc == std::string::npos)
        return false;
    const size_t after_key = kc + std::char_traits<char>::length(kArgKeyClose);
    if (inner.compare(after_key, std::char_traits<char>::length(kArgValOpen), kArgValOpen) != 0)
        return false;
    key_e = kc;
    val_s = after_key + std::char_traits<char>::length(kArgValOpen);
    const size_t vc = inner.find(kArgValClose, val_s);
    if (vc == std::string::npos)
        return false;
    val_e = vc;
    next = vc + std::char_traits<char>::length(kArgValClose);
    return true;
}

bool has_full_arg_pair(const std::string &inner) {
    size_t ks = 0, ke = 0, vs = 0, ve = 0, next = 0;
    return next_arg_pair(inner, 0, ks, ke, vs, ve, next);
}

void args_to_json(const std::string &inner, std::string &out) {
    json o = json::object();
    size_t pos = 0;
    size_t ks = 0, ke = 0, vs = 0, ve = 0, next = 0;
    while (next_arg_pair(inner, pos, ks, ke, vs, ve, next)) {
        o[inner.substr(ks, ke - ks)] = inner.substr(vs, ve - vs);
        pos = next;
    }
    out = o.dump(-1, ' ', false);
}

std::string render_arg_value(const json &v) {
    if (v.is_string())
        return v.get<std::string>();
    return v.dump(-1, ' ', false);
}

void append_arg(std::string &o, const std::string &key, const std::string &value) {
    o += kArgKeyOpen;
    o += key;
    o += kArgKeyClose;
    o += kArgValOpen;
    o += value;
    o += kArgValClose;
}

std::string render_one_call(const K3ToolCall &c) {
    std::string o;
    o += kBoxStart;
    o += c.name;
    if (!c.json.empty()) {
        json parsed = json::parse(c.json, nullptr, false);
        if (parsed.is_object()) {
            for (auto it = parsed.begin(); it != parsed.end(); ++it)
                append_arg(o, it.key(), render_arg_value(it.value()));
        } else {
            o += c.json;
        }
    } else {
        for (const auto &a : c.args) {
            if (a.type == "string") {
                append_arg(o, a.key, a.value);
                continue;
            }
            json v = json::parse(a.value, nullptr, false);
            if (!v.is_discarded() && !v.is_string())
                append_arg(o, a.key, v.dump(-1, ' ', false));
            else
                append_arg(o, a.key, a.value);
        }
    }
    o += kBoxEnd;
    return o;
}

std::string one_tool_json(const K3ToolDecl &t) {
    json fn = json::object();
    fn["name"] = t.name;
    fn["description"] = t.description;
    json params = json::object();
    if (!t.parameters_json.empty()) {
        json p = json::parse(t.parameters_json, nullptr, false);
        if (!p.is_discarded())
            params = std::move(p);
    }
    fn["parameters"] = std::move(params);
    return fn.dump(-1, ' ', false);
}

} // namespace

std::string glm_tool_declare(const std::vector<K3ToolDecl> &tools) {
    std::string o;
    o += "<|system|>\n# Tools\n\n";
    o += "You may call one or more functions to assist with the user query.\n\n";
    o += "You are provided with function signatures within <tools></tools> XML tags:\n";
    o += "<tools>\n";
    for (const auto &t : tools) {
        o += "\n";
        o += one_tool_json(t);
        o += "\n\n";
    }
    o += "\n</tools>\n\n";
    o += "For each function call, output the function name and arguments within the ";
    o += "following XML format:\n";
    o += "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>";
    o += "<arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>";
    o += "<arg_value>{arg-value-2}</arg_value>...</tool_call>";
    return o;
}

std::string glm_render_tool_calls(const std::vector<K3ToolCall> &calls) {
    if (calls.empty())
        return {};
    std::string o = "\n";
    for (const auto &c : calls)
        o += render_one_call(c);
    o += "\n";
    return o;
}

bool glm_parse_tool_calls(const std::string &reply, std::string &content,
                          std::vector<K3ParsedCall> &calls) {
    calls.clear();
    std::vector<std::string> boxes;
    const size_t start_n = std::char_traits<char>::length(kBoxStart);
    const size_t end_n = std::char_traits<char>::length(kBoxEnd);
    size_t pos = 0;
    while ((pos = reply.find(kBoxStart, pos)) != std::string::npos) {
        const size_t inner = pos + start_n;
        const size_t close = reply.find(kBoxEnd, inner);
        if (close == std::string::npos)
            break;
        boxes.push_back(reply.substr(inner, close - inner));
        pos = close + end_n;
    }

    bool recovered = false;
    const size_t last = reply.rfind(kBoxStart);
    if (last != std::string::npos && reply.find(kBoxEnd, last) == std::string::npos) {
        std::string inner = reply.substr(last + start_n);
        strip_partial_end(inner);
        if (has_full_arg_pair(inner) || is_just_name(inner)) {
            boxes.push_back(std::move(inner));
            recovered = true;
        }
    }

    int n = 0;
    for (const auto &inner : boxes) {
        K3ParsedCall pc;
        pc.name = leading_name(inner);
        args_to_json(inner, pc.arguments);
        ++n;
        pc.id = "call_" + std::to_string(n);
        calls.push_back(std::move(pc));
    }

    std::string text = reply;
    pos = 0;
    while ((pos = text.find(kBoxStart, pos)) != std::string::npos) {
        const size_t close = text.find(kBoxEnd, pos + start_n);
        if (close == std::string::npos)
            break;
        text.erase(pos, close + end_n - pos);
    }
    if (recovered) {
        const size_t tail = text.rfind(kBoxStart);
        if (tail != std::string::npos)
            text.resize(tail);
    }
    static const char kThinkClose[] = "</think>";
    static const char kThinkOpen[] = "<think>";
    const size_t tc = text.find(kThinkClose);
    if (tc != std::string::npos)
        text = text.substr(tc + sizeof(kThinkClose) - 1);
    erase_all(text, kThinkOpen);
    erase_all(text, kThinkClose);
    content = trim_copy(text);
    return !calls.empty();
}

} // namespace mvllm
