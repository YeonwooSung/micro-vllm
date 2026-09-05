#pragma once

#include <string>
#include <vector>

namespace mvllm {

// Official K3 XTML tools (#1143): tools / call / argument (not tool_call name=).
struct K3ToolDecl {
    std::string name;
    std::string description;
    std::string parameters_json;
};

struct K3ToolArg {
    std::string key;
    std::string type = "string"; // string | number | integer | boolean | object | array
    std::string value;
};

struct K3ToolCall {
    std::string name;
    int index = 1;
    std::vector<K3ToolArg> args;
    std::string json; // if non-empty, render <|open|>json type="object"
};

struct K3ParsedCall {
    std::string id;
    std::string name;
    std::string arguments; // JSON object string
};

std::string k3_xtml_escape_attr(const std::string &s);
std::string k3_xtml_unescape_attr(const std::string &s);

// Official declare body:
// "# Tools\nHere are the available tools, described in JSONSchema.\n\n```json\n...```"
std::string k3_tool_declare_body(const std::vector<K3ToolDecl> &tools);

// Render the inner tools>call>argument (or json) block, no surrounding message.
std::string k3_render_tools_block(const std::vector<K3ToolCall> &calls);

// Strip tools blocks from reply; fill OpenAI-shaped calls. Unclosed tail recovered.
bool k3_parse_tool_calls(const std::string &reply, std::string &content,
                         std::vector<K3ParsedCall> &calls);

// OpenAI request JSON helpers.
bool k3_extract_tools_json(const std::string &body, std::vector<K3ToolDecl> &tools);
bool k3_extract_tool_calls_json(const std::string &message_obj, std::vector<K3ToolCall> &calls);
std::string k3_args_to_json(const K3ToolCall &c);

} // namespace mvllm
