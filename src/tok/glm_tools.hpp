#pragma once

#include "k3_tools.hpp"

#include <string>
#include <vector>

namespace mvllm {

// Official GLM <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
std::string glm_tool_declare(const std::vector<K3ToolDecl> &tools);
std::string glm_render_tool_calls(const std::vector<K3ToolCall> &calls);
bool glm_parse_tool_calls(const std::string &reply, std::string &content,
                          std::vector<K3ParsedCall> &calls);

} // namespace mvllm
