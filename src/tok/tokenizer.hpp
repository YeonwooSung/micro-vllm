#pragma once

#include "../core/types.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mvllm {

struct ChatMessage {
    std::string role;
    std::string content;
    std::string reasoning; // past assistant think body; empty = none
};

class Tokenizer {
public:
    Status load(const std::string &model_dir, std::string &err);
    Status encode(const std::string &text, std::vector<int> &ids) const;
    Status decode(const std::vector<int> &ids, std::string &text) const;
    // Prompt string for inspection. GLM: [gMASK]<sop>…; K3 XTML or <|im_start|> fallback.
    std::string apply_chat(Family family, const std::vector<ChatMessage> &msgs, bool think,
                          const std::string &effort = {}) const;
    // K3 XTML encodes tag/attr pieces as separate tok_encode calls (rank-BPE contract).
    Status encode_chat(Family family, const std::vector<ChatMessage> &msgs, bool think,
                       std::vector<int> &ids, const std::string &effort = {}) const;
    int id_of(const std::string &content) const;
    bool has_xtml() const;
    int vocab_size() const { return vocab_size_; }
    bool loaded() const { return loaded_; }
    bool rank_bpe() const { return rank_bpe_; }
    bool kimi() const { return kimi_; }
    bool from_tiktoken() const { return from_tiktoken_; }

private:
    void bpe_piece(const unsigned char *p, int a, int b, std::vector<int> &ids) const;
    void pretok(const unsigned char *p, int a, int b, std::vector<int> &ids) const;
    void pretok_cl100k(const uint32_t *cp, const int *off, int n, const unsigned char *p,
                       std::vector<int> &ids) const;
    void pretok_kimi(const uint32_t *cp, const int *off, int n, const unsigned char *p,
                     std::vector<int> &ids) const;

    bool loaded_ = false;
    bool rank_bpe_ = false;
    bool kimi_ = false;
    bool from_tiktoken_ = false;
    int vocab_size_ = 0;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> merges_;
    std::vector<std::string> id_to_token_;
    std::vector<uint8_t> id_added_;
    std::vector<std::pair<std::string, int>> specials_; // longest first
    std::string byte2str_[256];
    int16_t cp2byte_[1024];
};

} // namespace mvllm
