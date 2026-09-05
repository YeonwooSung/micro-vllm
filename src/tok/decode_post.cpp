#include "decode_post.hpp"

namespace mvllm {
namespace {

bool starts_with(const std::string &s, const char *p, std::size_t n) {
    return s.size() >= n && s.compare(0, n, p) == 0;
}

bool ends_with(const std::string &s, const char *p, std::size_t n) {
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

bool is_tail_ws(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

void erase_all(std::string &s, const char *pat, std::size_t n) {
    std::size_t i = 0;
    while ((i = s.find(pat, i)) != std::string::npos)
        s.erase(i, n);
}

void split_think_xml(const std::string &text, std::string &reasoning, std::string &content) {
    static const char kClose[] = "</think>";
    static const char kOpen[] = "<think>";
    const std::size_t clen = sizeof(kClose) - 1;
    const std::size_t olen = sizeof(kOpen) - 1;
    const std::size_t c = text.find(kClose);
    if (c != std::string::npos) {
        reasoning = text.substr(0, c);
        content = text.substr(c + clen);
        if (starts_with(reasoning, kOpen, olen))
            reasoning.erase(0, olen);
        return;
    }
    const std::size_t o = text.find(kOpen);
    if (o != std::string::npos) {
        reasoning = text.substr(o + olen);
        return;
    }
    content = text;
}

} // namespace

std::size_t stop_cut(const std::string &text, const std::vector<std::string> &stops) {
    std::size_t cut = std::string::npos;
    for (const std::string &s : stops) {
        if (s.empty())
            continue;
        std::size_t p = text.find(s);
        if (p != std::string::npos && (cut == std::string::npos || p < cut))
            cut = p;
    }
    return cut;
}

bool trim_stop(std::string &text, const std::vector<std::string> &stops) {
    std::size_t c = stop_cut(text, stops);
    if (c == std::string::npos)
        return false;
    text.resize(c);
    return true;
}

void split_assistant_text(Family family, const std::string &text, std::string &reasoning,
                          std::string &content) {
    reasoning.clear();
    content.clear();
    if (family == Family::Glm53 || family == Family::Llama || family == Family::Dsv4) {
        split_think_xml(text, reasoning, content);
    } else if (family == Family::KimiK3) {
        static const char kClose[] = "<|close|>think";
        static const char kOpenThink[] = "<|open|>think";
        static const char kSep[] = "<|sep|>";
        static const char kOpenResp[] = "<|open|>response<|sep|>";
        const std::size_t clen = sizeof(kClose) - 1;
        const std::size_t p = text.find(kClose);
        if (p == std::string::npos) {
            content = text;
        } else {
            reasoning = text.substr(0, p);
            content = text.substr(p + clen);
            erase_all(reasoning, kOpenThink, sizeof(kOpenThink) - 1);
            erase_all(reasoning, kSep, sizeof(kSep) - 1);
            if (starts_with(content, kSep, sizeof(kSep) - 1))
                content.erase(0, sizeof(kSep) - 1);
            if (starts_with(content, kOpenResp, sizeof(kOpenResp) - 1))
                content.erase(0, sizeof(kOpenResp) - 1);
        }
    } else {
        content = text;
    }
    trim_assistant_tail(reasoning);
    trim_assistant_tail(content);
}

void trim_assistant_tail(std::string &text) {
    static const char kEndOfMsg[] = "<|end_of_msg|>";
    static const char kImEnd[] = "<|im_end|>";
    static const char kAssistant[] = "<|assistant|>";
    static const char kEotId[] = "<|eot_id|>";
    static const char kEos[] = "</s>";
    static const char kEndOfText[] = "<|endoftext|>";
    static const char kEndOfTextHf[] = "<|end_of_text|>";
    static const char kEnd[] = "<|end|>";
    static const char kEndOfPrompt[] = "<|endofprompt|>";
    static const char kEndOfTurn[] = "<|end_of_turn|>";
    static const char kEomId[] = "<|eom_id|>";
    for (;;) {
        bool stripped = false;
        while (!text.empty() && is_tail_ws(text.back())) {
            text.pop_back();
            stripped = true;
        }
        if (ends_with(text, kEndOfMsg, sizeof(kEndOfMsg) - 1)) {
            text.resize(text.size() - (sizeof(kEndOfMsg) - 1));
            stripped = true;
        } else if (ends_with(text, kImEnd, sizeof(kImEnd) - 1)) {
            text.resize(text.size() - (sizeof(kImEnd) - 1));
            stripped = true;
        } else if (ends_with(text, kAssistant, sizeof(kAssistant) - 1)) {
            text.resize(text.size() - (sizeof(kAssistant) - 1));
            stripped = true;
        } else if (ends_with(text, kEotId, sizeof(kEotId) - 1)) {
            text.resize(text.size() - (sizeof(kEotId) - 1));
            stripped = true;
        } else if (ends_with(text, kEos, sizeof(kEos) - 1)) {
            text.resize(text.size() - (sizeof(kEos) - 1));
            stripped = true;
        } else if (ends_with(text, kEndOfText, sizeof(kEndOfText) - 1)) {
            text.resize(text.size() - (sizeof(kEndOfText) - 1));
            stripped = true;
        } else if (ends_with(text, kEndOfTextHf, sizeof(kEndOfTextHf) - 1)) {
            text.resize(text.size() - (sizeof(kEndOfTextHf) - 1));
            stripped = true;
        } else if (ends_with(text, kEnd, sizeof(kEnd) - 1)) {
            text.resize(text.size() - (sizeof(kEnd) - 1));
            stripped = true;
        } else if (ends_with(text, kEndOfPrompt, sizeof(kEndOfPrompt) - 1)) {
            text.resize(text.size() - (sizeof(kEndOfPrompt) - 1));
            stripped = true;
        } else if (ends_with(text, kEndOfTurn, sizeof(kEndOfTurn) - 1)) {
            text.resize(text.size() - (sizeof(kEndOfTurn) - 1));
            stripped = true;
        } else if (ends_with(text, kEomId, sizeof(kEomId) - 1)) {
            text.resize(text.size() - (sizeof(kEomId) - 1));
            stripped = true;
        }
        if (!stripped)
            return;
    }
}

} // namespace mvllm
