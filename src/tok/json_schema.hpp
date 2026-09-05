#pragma once

#include <string>

namespace mvllm {

// GBNF that accepts any JSON value (OpenAI response_format json_object).
std::string json_object_gbnf();

// Compile a JSON Schema (draft-ish subset) to GBNF. On failure err is set and
// the result is empty. Supported: type object/array/string/number/integer/
// boolean/null, properties, required, enum, const, additionalProperties.
std::string json_schema_to_gbnf(const std::string &schema_json, std::string &err);

} // namespace mvllm
