#include "k3_tools.hpp"

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include <cctype>
#include <utility>

namespace mvllm {
namespace {

using json = nlohmann::json;

constexpr const char *kOpen = "<|open|>";
constexpr const char *kClose = "<|close|>";
constexpr const char *kSep = "<|sep|>";
constexpr const char *kEom = "<|end_of_msg|>";
constexpr const char *kToolsOpen = "<|open|>tools<|sep|>";
constexpr const char *kToolsClose = "<|close|>tools<|sep|>";
constexpr const char *kCallClose = "<|close|>call<|sep|>";
constexpr const char *kArgClose = "<|close|>argument<|sep|>";
constexpr const char *kJsonClose = "<|close|>json<|sep|>";
constexpr const char *kThinkOpen = "<|open|>think<|sep|>";
constexpr const char *kThinkClose = "<|close|>think<|sep|>";
constexpr const char *kRespOpen = "<|open|>response<|sep|>";
constexpr const char *kRespClose = "<|close|>response<|sep|>";

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

void replace_all(std::string &s, const std::string &from, const std::string &to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

void erase_all(std::string &s, const std::string &pat) {
    size_t pos = 0;
    while ((pos = s.find(pat, pos)) != std::string::npos)
        s.erase(pos, pat.size());
}

size_t match_quoted(const std::string &s, size_t q) {
    if (q >= s.size() || s[q] != '"')
        return std::string::npos;
    for (size_t i = q + 1; i < s.size(); ++i) {
        if (s[i] == '\\') {
            ++i;
            continue;
        }
        if (s[i] == '"')
            return i;
    }
    return std::string::npos;
}

size_t match_json_value(const std::string &s, size_t i) {
    i = skip_ws(s, i);
    if (i >= s.size())
        return std::string::npos;
    if (s[i] == '"')
        return match_quoted(s, i);
    if (s[i] == '{' || s[i] == '[') {
        const char open = s[i];
        const char close = open == '{' ? '}' : ']';
        int depth = 0;
        bool in_str = false;
        for (size_t j = i; j < s.size(); ++j) {
            const char c = s[j];
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
            else if (c == open)
                ++depth;
            else if (c == close) {
                --depth;
                if (depth == 0)
                    return j;
            }
        }
        return std::string::npos;
    }
    size_t j = i;
    while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j])) && s[j] != ',' &&
           s[j] != '}' && s[j] != ']')
        ++j;
    return j == i ? std::string::npos : j - 1;
}

int json_depth_at(const std::string &s, size_t at) {
    int depth = 0;
    bool in_str = false;
    for (size_t i = 0; i < at && i < s.size(); ++i) {
        const char c = s[i];
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
        else if (c == '{' || c == '[')
            ++depth;
        else if (c == '}' || c == ']')
            --depth;
    }
    return depth;
}

size_t find_top_level_array(const std::string &s, const char *key) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t pos = 0;
    while ((pos = s.find(pat, pos)) != std::string::npos) {
        if (json_depth_at(s, pos) == 1) {
            size_t i = skip_ws(s, pos + pat.size());
            if (i < s.size() && s[i] == ':') {
                i = skip_ws(s, i + 1);
                if (i < s.size() && s[i] == '[')
                    return i;
            }
        }
        pos += 1;
    }
    return std::string::npos;
}

size_t find_key_value(const std::string &s, const char *key) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t pos = 0;
    while ((pos = s.find(pat, pos)) != std::string::npos) {
        size_t i = skip_ws(s, pos + pat.size());
        if (i < s.size() && s[i] == ':')
            return skip_ws(s, i + 1);
        pos += 1;
    }
    return std::string::npos;
}

std::string attr_value(const std::string &head, const char *key) {
    const std::string pat = std::string(key) + "=\"";
    const size_t p = head.find(pat);
    if (p == std::string::npos)
        return {};
    const size_t v = p + pat.size();
    const size_t e = head.find('"', v);
    if (e == std::string::npos)
        return {};
    return k3_xtml_unescape_attr(head.substr(v, e - v));
}

void parse_args_into(const std::string &inner, K3ToolCall &c) {
    const std::string open = std::string(kOpen) + "argument";
    size_t pos = 0;
    while ((pos = inner.find(open, pos)) != std::string::npos) {
        const size_t sep = inner.find(kSep, pos);
        if (sep == std::string::npos)
            break;
        const std::string head = inner.substr(pos, sep - pos);
        const size_t vs = sep + std::char_traits<char>::length(kSep);
        const size_t ce = inner.find(kArgClose, vs);
        if (ce == std::string::npos)
            break;
        K3ToolArg a;
        a.key = attr_value(head, "key");
        a.type = attr_value(head, "type");
        if (a.type.empty())
            a.type = "string";
        a.value = inner.substr(vs, ce - vs);
        c.args.push_back(std::move(a));
        pos = ce + std::char_traits<char>::length(kArgClose);
    }
}

std::string strip_reply_chrome(std::string text) {
    const size_t tc = text.find(kThinkClose);
    if (tc != std::string::npos)
        text = text.substr(tc + std::char_traits<char>::length(kThinkClose));
    erase_all(text, kThinkOpen);
    erase_all(text, kThinkClose);
    const size_t ro = text.find(kRespOpen);
    if (ro != std::string::npos) {
        const size_t rs = ro + std::char_traits<char>::length(kRespOpen);
        const size_t rc = text.find(kRespClose, rs);
        if (rc != std::string::npos)
            text = text.substr(rs, rc - rs);
        else
            text = text.substr(rs);
    }
    erase_all(text, kRespClose);
    erase_all(text, std::string(kClose) + "message" + kSep);
    erase_all(text, kEom);
    const std::string msg_open = std::string(kOpen) + "message";
    const size_t mo = text.find(msg_open);
    if (mo != std::string::npos) {
        const size_t sep = text.find(kSep, mo);
        if (sep != std::string::npos)
            text.erase(mo, sep + std::char_traits<char>::length(kSep) - mo);
    }
    return trim_copy(text);
}

void calls_from_tools_array(const json &arr, std::vector<K3ToolCall> &calls) {
    int index = 1;
    for (const auto &tc : arr) {
        if (!tc.is_object())
            continue;
        const json &fn = tc.contains("function") && tc["function"].is_object() ? tc["function"] : tc;
        K3ToolCall c;
        if (fn.contains("name") && fn["name"].is_string())
            c.name = fn["name"].get<std::string>();
        else if (tc.contains("name") && tc["name"].is_string())
            c.name = tc["name"].get<std::string>();
        if (c.name.empty())
            continue;
        c.index = index++;
        json args = json();
        if (fn.contains("arguments"))
            args = fn["arguments"];
        else if (tc.contains("arguments"))
            args = tc["arguments"];
        if (args.is_object()) {
            c.json = args.dump(-1, ' ', false);
        } else if (args.is_string()) {
            const std::string raw = args.get<std::string>();
            json parsed = json::parse(raw, nullptr, false);
            if (parsed.is_object()) {
                c.json = raw;
            } else if (parsed.is_discarded()) {
                const std::string t = trim_copy(raw);
                if (t.size() >= 2 && t.front() == '{' && t.back() == '}')
                    c.json = t;
                else if (!raw.empty()) {
                    K3ToolArg a;
                    a.key = "input";
                    a.type = "string";
                    a.value = raw;
                    c.args.push_back(std::move(a));
                }
            } else if (!raw.empty()) {
                K3ToolArg a;
                a.key = "input";
                a.type = "string";
                a.value = raw;
                c.args.push_back(std::move(a));
            }
        }
        calls.push_back(std::move(c));
    }
}

} // namespace

std::string k3_xtml_escape_attr(const std::string &s) {
    std::string o;
    o.reserve(s.size());
    for (char ch : s) {
        if (ch == '&')
            o += "&amp;";
        else if (ch == '"')
            o += "&quot;";
        else
            o += ch;
    }
    return o;
}

std::string k3_xtml_unescape_attr(const std::string &s) {
    std::string o = s;
    replace_all(o, "&quot;", "\"");
    replace_all(o, "&amp;", "&");
    return o;
}

std::string k3_tool_declare_body(const std::vector<K3ToolDecl> &tools) {
    json arr = json::array();
    for (const auto &t : tools) {
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
        json item = json::object();
        item["type"] = "function";
        item["function"] = std::move(fn);
        arr.push_back(std::move(item));
    }
    return "# Tools\nHere are the available tools, described in JSONSchema.\n\n```json\n" +
           arr.dump(-1, ' ', false) + "\n```";
}

std::string k3_render_tools_block(const std::vector<K3ToolCall> &calls) {
    if (calls.empty())
        return {};
    std::string o;
    o += kToolsOpen;
    for (size_t i = 0; i < calls.size(); ++i) {
        const auto &c = calls[i];
        const int idx = c.index >= 1 ? c.index : static_cast<int>(i) + 1;
        o += kOpen;
        o += "call tool=\"";
        o += k3_xtml_escape_attr(c.name);
        o += "\" index=\"";
        o += std::to_string(idx);
        o += "\"";
        o += kSep;
        if (!c.json.empty()) {
            o += kOpen;
            o += "json type=\"object\"";
            o += kSep;
            o += c.json;
            o += kJsonClose;
        } else {
            for (const auto &a : c.args) {
                o += kOpen;
                o += "argument key=\"";
                o += k3_xtml_escape_attr(a.key);
                o += "\" type=\"";
                o += k3_xtml_escape_attr(a.type.empty() ? "string" : a.type);
                o += "\"";
                o += kSep;
                o += a.value;
                o += kArgClose;
            }
        }
        o += kCallClose;
    }
    o += kToolsClose;
    return o;
}

bool k3_parse_tool_calls(const std::string &reply, std::string &content,
                         std::vector<K3ParsedCall> &calls) {
    calls.clear();
    std::vector<std::string> blocks;
    std::string text = reply;
    size_t pos = 0;
    while ((pos = text.find(kToolsOpen, pos)) != std::string::npos) {
        const size_t inner = pos + std::char_traits<char>::length(kToolsOpen);
        const size_t close = text.find(kToolsClose, inner);
        if (close == std::string::npos)
            break;
        blocks.push_back(text.substr(inner, close - inner));
        text.erase(pos, close + std::char_traits<char>::length(kToolsClose) - pos);
    }
    const size_t tail = text.rfind(kToolsOpen);
    if (tail != std::string::npos) {
        blocks.push_back(text.substr(tail + std::char_traits<char>::length(kToolsOpen)));
        text.erase(tail);
    }

    const std::string call_open = std::string(kOpen) + "call";
    int n = 0;
    for (const auto &block : blocks) {
        size_t cp = 0;
        while ((cp = block.find(call_open, cp)) != std::string::npos) {
            const size_t sep = block.find(kSep, cp);
            if (sep == std::string::npos)
                break;
            const std::string head = block.substr(cp, sep - cp);
            const size_t vs = sep + std::char_traits<char>::length(kSep);
            const size_t ce = block.find(kCallClose, vs);
            if (ce == std::string::npos)
                break;
            const std::string inner = block.substr(vs, ce - vs);
            K3ParsedCall pc;
            pc.name = attr_value(head, "tool");
            const std::string json_open = std::string(kOpen) + "json";
            const size_t jo = inner.find(json_open);
            if (jo != std::string::npos) {
                const size_t js = inner.find(kSep, jo);
                if (js != std::string::npos) {
                    const size_t jv = js + std::char_traits<char>::length(kSep);
                    const size_t jc = inner.find(kJsonClose, jv);
                    if (jc != std::string::npos)
                        pc.arguments = inner.substr(jv, jc - jv);
                }
            }
            if (pc.arguments.empty()) {
                K3ToolCall tmp;
                parse_args_into(inner, tmp);
                pc.arguments = k3_args_to_json(tmp);
            }
            ++n;
            pc.id = "call_" + std::to_string(n);
            calls.push_back(std::move(pc));
            cp = ce + std::char_traits<char>::length(kCallClose);
        }
    }

    content = strip_reply_chrome(text);
    return !calls.empty();
}

bool k3_extract_tools_json(const std::string &body, std::vector<K3ToolDecl> &tools) {
    tools.clear();
    auto take_tool = [&](const json &t, const std::string &slice) {
        if (!t.is_object())
            return;
        const json &fn = t.contains("function") && t["function"].is_object() ? t["function"] : t;
        K3ToolDecl d;
        if (fn.contains("name") && fn["name"].is_string())
            d.name = fn["name"].get<std::string>();
        if (fn.contains("description") && fn["description"].is_string())
            d.description = fn["description"].get<std::string>();
        const size_t pv = find_key_value(slice, "parameters");
        if (pv != std::string::npos) {
            const size_t pe = match_json_value(slice, pv);
            if (pe != std::string::npos) {
                std::string raw = trim_copy(slice.substr(pv, pe - pv + 1));
                if (!raw.empty() && raw.front() == '"') {
                    json sv = json::parse(raw, nullptr, false);
                    if (sv.is_string())
                        d.parameters_json = sv.get<std::string>();
                    else
                        d.parameters_json = raw;
                } else {
                    d.parameters_json = raw;
                }
            }
        } else if (fn.contains("parameters")) {
            if (fn["parameters"].is_string())
                d.parameters_json = fn["parameters"].get<std::string>();
            else
                d.parameters_json = fn["parameters"].dump(-1, ' ', false);
        }
        if (!d.name.empty())
            tools.push_back(std::move(d));
    };

    const size_t arr = find_top_level_array(body, "tools");
    if (arr != std::string::npos) {
        const size_t arr_end = match_json_value(body, arr);
        if (arr_end != std::string::npos) {
            size_t i = arr + 1;
            while (i < arr_end) {
                i = skip_ws(body, i);
                if (i >= arr_end || body[i] == ']')
                    break;
                if (body[i] == ',') {
                    ++i;
                    continue;
                }
                if (body[i] != '{')
                    break;
                const size_t end = match_json_value(body, i);
                if (end == std::string::npos)
                    break;
                const std::string slice = body.substr(i, end - i + 1);
                take_tool(json::parse(slice, nullptr, false), slice);
                i = end + 1;
            }
        }
    }
    if (tools.empty()) {
        json j = json::parse(body, nullptr, false);
        if (j.is_object() && j.contains("tools") && j["tools"].is_array()) {
            for (const auto &t : j["tools"])
                take_tool(t, t.is_object() ? t.dump() : std::string());
        }
    }
    return !tools.empty();
}

bool k3_extract_tool_calls_json(const std::string &message_obj, std::vector<K3ToolCall> &calls) {
    calls.clear();
    json j = json::parse(message_obj, nullptr, false);
    if (j.is_object() && j.contains("tool_calls") && j["tool_calls"].is_array()) {
        calls_from_tools_array(j["tool_calls"], calls);
        return !calls.empty();
    }
    const size_t arr = find_top_level_array(message_obj, "tool_calls");
    if (arr == std::string::npos)
        return false;
    const size_t arr_end = match_json_value(message_obj, arr);
    if (arr_end == std::string::npos)
        return false;
    json parsed = json::parse(message_obj.substr(arr, arr_end - arr + 1), nullptr, false);
    if (!parsed.is_array())
        return false;
    calls_from_tools_array(parsed, calls);
    return !calls.empty();
}

std::string k3_args_to_json(const K3ToolCall &c) {
    if (!c.json.empty())
        return c.json;
    json o = json::object();
    for (const auto &a : c.args) {
        if (a.type == "string") {
            o[a.key] = a.value;
            continue;
        }
        json v = json::parse(a.value, nullptr, false);
        if (!v.is_discarded())
            o[a.key] = std::move(v);
        else
            o[a.key] = a.value;
    }
    return o.dump(-1, ' ', false);
}

} // namespace mvllm
