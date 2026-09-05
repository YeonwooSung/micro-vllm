#include "json_schema.hpp"

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mvllm {
namespace {

using json = nlohmann::ordered_json;

// Generic JSON value grammar. `root` is prefixed by the caller.
// Rule names are identifier-safe; `ws` matches optional whitespace.
constexpr const char *kJsonPrims =
    R"gbnf(value ::= object | array | string | number | boolean | null
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"" ws
number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
integer ::= "-"? [0-9]+ ws
boolean ::= ("true" | "false") ws
null ::= "null" ws
ws ::= [ \t\n\r]*
)gbnf";

constexpr const char *kPrimNames[] = {"value", "object", "array",   "string", "number",
                                      "integer", "boolean", "null", "ws"};

bool is_prim_name(const std::string &n) {
    for (const char *p : kPrimNames)
        if (n == p)
            return true;
    return n == "root";
}

// GBNF string literal for the exact byte sequence `raw`.
std::string gbnf_lit(const std::string &raw) {
    std::string o;
    o.push_back('"');
    for (unsigned char c : raw) {
        switch (c) {
        case '"':
            o += "\\\"";
            break;
        case '\\':
            o += "\\\\";
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
                static const char kHex[] = "0123456789ABCDEF";
                o += "\\x";
                o.push_back(kHex[c >> 4]);
                o.push_back(kHex[c & 15]);
            } else {
                o.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    o.push_back('"');
    return o;
}

std::string sanitize_ident(const std::string &s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (unsigned char c : s) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')
            o.push_back(static_cast<char>(c));
        else if (!o.empty() && o.back() != '_')
            o.push_back('_');
    }
    while (!o.empty() && o.back() == '_')
        o.pop_back();
    if (o.empty())
        o = "r";
    if (o[0] >= '0' && o[0] <= '9')
        o.insert(o.begin(), 'r');
    if (o.size() > 48)
        o.resize(48);
    return o;
}

bool is_json_literal(const json &v) {
    return v.is_string() || v.is_number() || v.is_boolean() || v.is_null();
}

// Structural keywords this subset cannot compile. Annotation / numeric
// bounds (pattern, format, minLength, minimum, ...) are ignored.
bool has_unsupported(const json &schema) {
    static const char *kBad[] = {"$ref",
                                 "$dynamicRef",
                                 "$recursiveRef",
                                 "oneOf",
                                 "anyOf",
                                 "allOf",
                                 "not",
                                 "if",
                                 "then",
                                 "else",
                                 "prefixItems",
                                 "patternProperties",
                                 "propertyNames",
                                 "unevaluatedProperties",
                                 "unevaluatedItems",
                                 "dependentSchemas",
                                 "dependentRequired",
                                 "dependencies",
                                 "contains"};
    for (const char *k : kBad) {
        if (schema.contains(k))
            return true;
    }
    return false;
}

enum TypeBit : unsigned {
    TObject = 1u << 0,
    TArray = 1u << 1,
    TString = 1u << 2,
    TNumber = 1u << 3,
    TInteger = 1u << 4,
    TBoolean = 1u << 5,
    TNull = 1u << 6,
};

bool add_type_name(unsigned &bits, const std::string &s) {
    if (s == "object")
        bits |= TObject;
    else if (s == "array")
        bits |= TArray;
    else if (s == "string")
        bits |= TString;
    else if (s == "number")
        bits |= TNumber;
    else if (s == "integer")
        bits |= TInteger;
    else if (s == "boolean")
        bits |= TBoolean;
    else if (s == "null")
        bits |= TNull;
    else
        return false;
    return true;
}

// Returns false if `type` is present but not a supported string / array.
bool parse_types(const json &schema, unsigned &bits, bool &present) {
    bits = 0;
    present = schema.contains("type");
    if (!present)
        return true;
    const json &t = schema["type"];
    if (t.is_string())
        return add_type_name(bits, t.get<std::string>());
    if (!t.is_array() || t.empty())
        return false;
    for (const auto &el : t) {
        if (!el.is_string() || !add_type_name(bits, el.get<std::string>()))
            return false;
    }
    return true;
}

struct Compiler {
    std::vector<std::pair<std::string, std::string>> rules;
    std::unordered_set<std::string> used;
    std::unordered_set<std::string> prims;
    int next_id = 0;

    Compiler() {
        used.insert("root");
        for (const char *p : kPrimNames)
            used.insert(p);
    }

    std::string prim(const char *name) {
        prims.insert(name);
        return name;
    }

    std::string uniq(const std::string &hint) {
        const std::string base = sanitize_ident(hint);
        if (!is_prim_name(base) && used.insert(base).second)
            return base;
        for (;;) {
            std::string n = base + "_" + std::to_string(next_id++);
            if (used.insert(n).second)
                return n;
        }
    }

    void define(const std::string &name, const std::string &body) {
        rules.emplace_back(name, body);
    }

    std::string never_rule() {
        if (!used.count("never")) {
            used.insert("never");
            define("never", "[]");
        }
        return "never";
    }

    std::string emit_literal(const json &v, const std::string &hint) {
        const std::string name = uniq(hint.empty() ? "lit" : hint);
        define(name, gbnf_lit(v.dump()) + " ws");
        prim("ws");
        return name;
    }

    std::string emit_enum(const json &arr, const std::string &hint) {
        if (!arr.is_array() || arr.empty())
            return prim("value");
        std::ostringstream inner;
        bool first = true;
        for (const auto &v : arr) {
            if (!is_json_literal(v))
                return prim("value");
            if (!first)
                inner << " | ";
            first = false;
            inner << gbnf_lit(v.dump());
        }
        const std::string name = uniq(hint.empty() ? "enum" : hint);
        define(name, "(" + inner.str() + ") ws");
        prim("ws");
        return name;
    }

    static std::string kv(const std::string &key, const std::string &val_rule) {
        return gbnf_lit(json(key).dump()) + " ws \":\" ws " + val_rule;
    }

    std::string emit_object(const json &schema, const std::string &hint);
    std::string emit_array(const json &schema, const std::string &hint);
    std::string visit(const json &schema, const std::string &hint);

    void close_prim_deps() {
        bool changed = true;
        while (changed) {
            changed = false;
            auto need = [&](const char *n) {
                if (prims.insert(n).second)
                    changed = true;
            };
            if (prims.count("value")) {
                need("object");
                need("array");
                need("string");
                need("number");
                need("boolean");
                need("null");
            }
            if (prims.count("object")) {
                need("string");
                need("value");
            }
            if (prims.count("array"))
                need("value");
        }
        prims.insert("ws");
    }

    std::string finish(const std::string &root_ref) {
        close_prim_deps();
        std::ostringstream o;
        o << "root ::= ws " << root_ref << "\n";
        for (const auto &r : rules)
            o << r.first << " ::= " << r.second << "\n";
        // Re-emit only the primitive lines that were referenced.
        std::istringstream in(kJsonPrims);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty())
                continue;
            const auto sp = line.find(" ::=");
            if (sp == std::string::npos)
                continue;
            if (prims.count(line.substr(0, sp)))
                o << line << "\n";
        }
        return o.str();
    }
};

std::string Compiler::emit_object(const json &schema, const std::string &hint) {
    std::vector<std::string> prop_order;
    std::unordered_map<std::string, json> props;
    if (schema.contains("properties")) {
        if (!schema["properties"].is_object())
            return prim("object");
        for (auto it = schema["properties"].begin(); it != schema["properties"].end(); ++it) {
            prop_order.push_back(it.key());
            props.emplace(it.key(), it.value());
        }
    }

    std::unordered_set<std::string> required;
    std::vector<std::string> required_extra;
    if (schema.contains("required")) {
        if (!schema["required"].is_array())
            return prim("object");
        for (const auto &r : schema["required"]) {
            if (!r.is_string())
                continue;
            const std::string k = r.get<std::string>();
            if (required.insert(k).second && !props.count(k))
                required_extra.push_back(k);
        }
    }

    // Required keys first, in properties order, then required-only names.
    std::vector<std::string> req_keys;
    std::vector<std::string> opt_keys;
    for (const auto &k : prop_order) {
        if (required.count(k))
            req_keys.push_back(k);
        else
            opt_keys.push_back(k);
    }
    for (const auto &k : required_extra) {
        req_keys.push_back(k);
        props.emplace(k, json::object());
    }

    bool additional = true;
    json add_schema = json(true);
    if (schema.contains("additionalProperties")) {
        const json &ap = schema["additionalProperties"];
        if (ap.is_boolean()) {
            additional = ap.get<bool>();
        } else if (ap.is_object()) {
            additional = true;
            add_schema = ap;
        } else {
            additional = true;
        }
    }

    const bool add_any = additional && add_schema.is_boolean() && add_schema.get<bool>();
    if (req_keys.empty() && opt_keys.empty() && add_any)
        return prim("object");

    std::unordered_map<std::string, std::string> val_of;
    for (const auto &k : req_keys)
        val_of[k] = visit(props[k], k);
    for (const auto &k : opt_keys)
        val_of[k] = visit(props[k], k);

    std::string extra_kv;
    if (additional) {
        const std::string vr =
            add_any ? prim("value") : visit(add_schema, hint.empty() ? "add" : hint + "_add");
        extra_kv = prim("string") + " \":\" ws " + vr;
    }

    std::ostringstream body;
    body << "\"{\" ws ";
    if (!req_keys.empty()) {
        for (size_t i = 0; i < req_keys.size(); ++i) {
            if (i)
                body << "\",\" ws ";
            body << kv(req_keys[i], val_of[req_keys[i]]) << " ";
        }
        for (const auto &k : opt_keys)
            body << "(\",\" ws " << kv(k, val_of[k]) << ")? ";
        if (additional)
            body << "(\",\" ws " << extra_kv << ")* ";
    } else {
        std::vector<std::string> alts;
        for (size_t i = 0; i < opt_keys.size(); ++i) {
            std::ostringstream alt;
            alt << kv(opt_keys[i], val_of[opt_keys[i]]);
            for (size_t j = i + 1; j < opt_keys.size(); ++j)
                alt << " (\",\" ws " << kv(opt_keys[j], val_of[opt_keys[j]]) << ")?";
            if (additional)
                alt << " (\",\" ws " << extra_kv << ")*";
            alts.push_back(alt.str());
        }
        if (additional)
            alts.push_back(extra_kv + " (\",\" ws " + extra_kv + ")*");
        if (!alts.empty()) {
            body << "(";
            for (size_t i = 0; i < alts.size(); ++i) {
                if (i)
                    body << " | ";
                body << alts[i];
            }
            body << ")? ";
        }
    }
    body << "\"}\" ws";

    const std::string name = uniq(hint.empty() ? "obj" : hint);
    define(name, body.str());
    prim("ws");
    return name;
}

std::string Compiler::emit_array(const json &schema, const std::string &hint) {
    if (!schema.contains("items"))
        return prim("array");
    const json &it = schema["items"];
    if (it.is_array())
        return prim("array");
    const std::string item = visit(it, hint.empty() ? "item" : hint + "_item");
    if (item == "value")
        return prim("array");
    const std::string name = uniq(hint.empty() ? "arr" : hint);
    define(name, "\"[\" ws (" + item + " (\",\" ws " + item + ")*)? \"]\" ws");
    prim("ws");
    return name;
}

std::string Compiler::visit(const json &schema, const std::string &hint) {
    if (schema.is_boolean())
        return schema.get<bool>() ? prim("value") : never_rule();
    if (!schema.is_object())
        return prim("value");
    if (has_unsupported(schema))
        return prim("value");

    if (schema.contains("const"))
        return emit_literal(schema["const"], hint.empty() ? "const" : hint);
    if (schema.contains("enum"))
        return emit_enum(schema["enum"], hint.empty() ? "enum" : hint);

    unsigned bits = 0;
    bool has_type = false;
    if (!parse_types(schema, bits, has_type))
        return prim("value");

    if (schema.contains("nullable") && schema["nullable"].is_boolean() &&
        schema["nullable"].get<bool>())
        bits |= TNull;

    const bool looks_obj = schema.contains("properties") || schema.contains("required") ||
                           schema.contains("additionalProperties");
    const bool looks_arr = schema.contains("items");
    if (!has_type) {
        if (looks_obj && looks_arr)
            bits = TObject | TArray;
        else if (looks_obj)
            bits = TObject;
        else if (looks_arr)
            bits = TArray;
        else
            return prim("value");
    }

    std::vector<std::string> alts;
    if (bits & TObject)
        alts.push_back(emit_object(schema, hint.empty() ? "obj" : hint));
    if (bits & TArray)
        alts.push_back(emit_array(schema, hint.empty() ? "arr" : hint + "_arr"));
    if (bits & TString)
        alts.push_back(prim("string"));
    if (bits & TNumber)
        alts.push_back(prim("number"));
    if (bits & TInteger)
        alts.push_back(prim("integer"));
    if (bits & TBoolean)
        alts.push_back(prim("boolean"));
    if (bits & TNull)
        alts.push_back(prim("null"));

    if (alts.empty())
        return prim("value");
    if (alts.size() == 1)
        return alts[0];

    const std::string name = uniq(hint.empty() ? "union" : hint);
    std::string body = alts[0];
    for (size_t i = 1; i < alts.size(); ++i) {
        body += " | ";
        body += alts[i];
    }
    define(name, body);
    return name;
}

} // namespace

std::string json_object_gbnf() {
    return std::string("root ::= ws value\n") + kJsonPrims;
}

std::string json_schema_to_gbnf(const std::string &schema_json, std::string &err) {
    err.clear();
    if (schema_json.empty()) {
        err = "empty schema";
        return {};
    }
    json j = json::parse(schema_json, nullptr, false);
    if (j.is_discarded()) {
        err = "invalid JSON";
        return {};
    }
    // OpenAI json_schema wrapper: {"name":"...","schema":{...}}
    if (j.is_object() && j.contains("schema") && j["schema"].is_object())
        j = j["schema"];
    if (!j.is_object()) {
        err = "schema must be a JSON object";
        return {};
    }
    Compiler c;
    const std::string top = c.visit(j, "schema");
    return c.finish(top);
}

} // namespace mvllm
