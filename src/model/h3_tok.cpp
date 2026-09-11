#include "h3_tok.hpp"
#include "../tok/tokenizer.hpp"

#include <fstream>

namespace mvllm {
namespace {

const char *g_backend = "off";

std::string with_slash(std::string dir) {
    if (!dir.empty() && dir.back() != '/')
        dir += '/';
    return dir;
}

// Tokenizer::load takes a directory and looks for tokenizer.json inside it.
std::string tokenizer_dir(const std::string &model_dir) {
    const std::string name = "tokenizer.json";
    if (model_dir.size() >= name.size() &&
        model_dir.compare(model_dir.size() - name.size(), name.size(), name) == 0) {
        const auto slash = model_dir.find_last_of('/');
        if (slash == std::string::npos)
            return ".";
        if (slash == 0)
            return "/";
        return model_dir.substr(0, slash);
    }
    return model_dir;
}

bool has_tokenizer_json(const std::string &dir) {
    std::ifstream in(with_slash(dir) + "tokenizer.json");
    return static_cast<bool>(in);
}

} // namespace

bool h3_tok_load(const std::string &model_dir, Tokenizer &tok, std::string &err) {
    g_backend = "off";
    const std::string dir = tokenizer_dir(model_dir);
    const std::string cands[] = {dir, with_slash(dir) + "tokenizer"};
    for (const std::string &cand : cands) {
        if (!has_tokenizer_json(cand))
            continue;
        if (tok.load(cand, err) != Status::Ok)
            return false;
        g_backend = "json";
        return true;
    }
    err = "tokenizer.json not found";
    return false;
}

const char *h3_tok_backend() { return g_backend; }

} // namespace mvllm
