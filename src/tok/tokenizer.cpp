#include "tokenizer.hpp"

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace mvllm {
namespace {

using json = nlohmann::json;

std::string read_file(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool file_exists(const std::string &path) {
    std::ifstream in(path);
    return static_cast<bool>(in);
}

int json_id(const json &v) {
    if (v.is_number_integer())
        return v.get<int>();
    if (v.is_number_unsigned())
        return static_cast<int>(v.get<uint64_t>());
    if (v.is_number())
        return static_cast<int>(v.get<double>());
    return -1;
}

int u8_next(const unsigned char *s, int len, int i, uint32_t *cp) {
    unsigned char c = s[i];
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    if ((c >> 5) == 0x6 && i + 1 < len) {
        *cp = (static_cast<uint32_t>(c & 0x1f) << 6) | (s[i + 1] & 0x3f);
        return 2;
    }
    if ((c >> 4) == 0xe && i + 2 < len) {
        *cp = (static_cast<uint32_t>(c & 0x0f) << 12) | (static_cast<uint32_t>(s[i + 1] & 0x3f) << 6) |
              (s[i + 2] & 0x3f);
        return 3;
    }
    if ((c >> 3) == 0x1e && i + 3 < len) {
        *cp = (static_cast<uint32_t>(c & 0x07) << 18) | (static_cast<uint32_t>(s[i + 1] & 0x3f) << 12) |
              (static_cast<uint32_t>(s[i + 2] & 0x3f) << 6) | (s[i + 3] & 0x3f);
        return 4;
    }
    *cp = c;
    return 1;
}

int u8_put(char *o, uint32_t cp) {
    if (cp < 0x80) {
        o[0] = static_cast<char>(cp);
        return 1;
    }
    if (cp < 0x800) {
        o[0] = static_cast<char>(0xc0 | (cp >> 6));
        o[1] = static_cast<char>(0x80 | (cp & 0x3f));
        return 2;
    }
    if (cp < 0x10000) {
        o[0] = static_cast<char>(0xe0 | (cp >> 12));
        o[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
        o[2] = static_cast<char>(0x80 | (cp & 0x3f));
        return 3;
    }
    o[0] = static_cast<char>(0xf0 | (cp >> 18));
    o[1] = static_cast<char>(0x80 | ((cp >> 12) & 0x3f));
    o[2] = static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
    o[3] = static_cast<char>(0x80 | (cp & 0x3f));
    return 4;
}

bool is_han(uint32_t c) {
    return (c >= 0x3400 && c <= 0x4dbf) || (c >= 0x4e00 && c <= 0x9fff) ||
           (c >= 0xf900 && c <= 0xfaff) || (c >= 0x20000 && c <= 0x2a6df);
}

bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }

bool is_num(uint32_t c) {
    return (c >= '0' && c <= '9') || (c >= 0xff10 && c <= 0xff19);
}

bool is_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == 0xa0 || c == 0x3000;
}

bool is_letter(uint32_t c, bool exclude_han) {
    if (exclude_han && is_han(c))
        return false;
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
        return true;
    if (c < 0xc0)
        return false;
    if (is_num(c) || is_space(c) || is_nl(c))
        return false;
    if (c == 0xd7 || c == 0xf7)
        return false;
    if (is_han(c))
        return true; // \p{L} includes Han unless excluded
    return c >= 0xc0;
}

uint32_t low_ascii(uint32_t c) {
    if (c >= 'A' && c <= 'Z')
        return c + 32;
    return c;
}

} // namespace

void Tokenizer::bpe_piece(const unsigned char *p, int a, int b, std::vector<int> &ids) const {
    const int nb = b - a;
    if (nb <= 0)
        return;
    std::string s;
    s.reserve(static_cast<size_t>(nb) * 2);
    for (int i = a; i < b; ++i)
        s += byte2str_[p[i]];
    auto whole = token_to_id_.find(s);
    if (whole != token_to_id_.end()) {
        ids.push_back(whole->second);
        return;
    }
    std::vector<int> soff, slen;
    const unsigned char *us = reinterpret_cast<const unsigned char *>(s.data());
    const int sl = static_cast<int>(s.size());
    for (int i = 0; i < sl;) {
        uint32_t cp = 0;
        int k = u8_next(us, sl, i, &cp);
        soff.push_back(i);
        slen.push_back(k);
        i += k;
    }
    int ns = static_cast<int>(soff.size());
    for (;;) {
        int best = std::numeric_limits<int>::max();
        int bp = -1;
        for (int i = 0; i + 1 < ns; ++i) {
            int rk = -1;
            if (rank_bpe_) {
                auto it = token_to_id_.find(s.substr(static_cast<size_t>(soff[i]),
                                                     static_cast<size_t>(slen[i] + slen[i + 1])));
                if (it != token_to_id_.end())
                    rk = it->second;
            } else {
                std::string key = s.substr(static_cast<size_t>(soff[i]), static_cast<size_t>(slen[i]));
                key.push_back('\0');
                key.append(s, static_cast<size_t>(soff[i + 1]), static_cast<size_t>(slen[i + 1]));
                auto it = merges_.find(key);
                if (it != merges_.end())
                    rk = it->second;
            }
            if (rk >= 0 && rk < best) {
                best = rk;
                bp = i;
            }
        }
        if (bp < 0)
            break;
        slen[bp] = soff[bp + 1] + slen[bp + 1] - soff[bp];
        for (int j = bp + 1; j < ns - 1; ++j) {
            soff[j] = soff[j + 1];
            slen[j] = slen[j + 1];
        }
        --ns;
    }
    for (int i = 0; i < ns; ++i) {
        auto it = token_to_id_.find(
            s.substr(static_cast<size_t>(soff[i]), static_cast<size_t>(slen[i])));
        if (it != token_to_id_.end())
            ids.push_back(it->second);
    }
}

void Tokenizer::pretok_cl100k(const uint32_t *cp, const int *off, int n, const unsigned char *p,
                              std::vector<int> &ids) const {
    int i = 0;
    while (i < n) {
        const int start = i;
        const uint32_t c = cp[i];
        if (c == '\'' && i + 1 < n) {
            uint32_t d = low_ascii(cp[i + 1]);
            if (i + 2 < n) {
                uint32_t d2 = low_ascii(cp[i + 2]);
                if ((d == 'r' && d2 == 'e') || (d == 'v' && d2 == 'e') || (d == 'l' && d2 == 'l')) {
                    i += 3;
                    bpe_piece(p, off[start], off[i], ids);
                    continue;
                }
            }
            if (d == 's' || d == 't' || d == 'm' || d == 'd') {
                i += 2;
                bpe_piece(p, off[start], off[i], ids);
                continue;
            }
        }
        {
            int j = i;
            if (!is_letter(c, false) && !is_nl(c) && !is_num(c)) {
                if (j + 1 < n && is_letter(cp[j + 1], false))
                    ++j;
                else
                    j = -1;
            }
            if (j >= 0 && is_letter(cp[j], false)) {
                while (j < n && is_letter(cp[j], false))
                    ++j;
                i = j;
                bpe_piece(p, off[start], off[i], ids);
                continue;
            }
        }
        if (is_num(c)) {
            int j = i, k = 0;
            while (j < n && is_num(cp[j]) && k < 3) {
                ++j;
                ++k;
            }
            i = j;
            bpe_piece(p, off[start], off[i], ids);
            continue;
        }
        if (!is_space(c) && !is_letter(c, false) && !is_num(c)) {
            int j = i;
            while (j < n && !is_space(cp[j]) && !is_letter(cp[j], false) && !is_num(cp[j]))
                ++j;
            while (j < n && is_nl(cp[j]))
                ++j;
            i = j;
            bpe_piece(p, off[start], off[i], ids);
            continue;
        }
        if (is_space(c)) {
            int r = i;
            while (r < n && is_space(cp[r]))
                ++r;
            int last = -1;
            for (int j = i; j < r; ++j)
                if (is_nl(cp[j]))
                    last = j;
            if (last >= 0)
                i = last + 1;
            else {
                int end = (r < n) ? r - 1 : r;
                if (end <= i)
                    end = i + 1;
                i = end;
            }
            bpe_piece(p, off[start], off[i], ids);
            continue;
        }
        ++i;
        bpe_piece(p, off[start], off[i], ids);
    }
}

void Tokenizer::pretok_kimi(const uint32_t *cp, const int *off, int n, const unsigned char *p,
                            std::vector<int> &ids) const {
    int i = 0;
    while (i < n) {
        const int start = i;
        const uint32_t c = cp[i];
        if (is_han(c)) {
            int j = i;
            while (j < n && is_han(cp[j]))
                ++j;
            i = j;
            bpe_piece(p, off[start], off[i], ids);
            continue;
        }
        if (is_letter(c, true)) {
            int j = i;
            while (j < n && is_letter(cp[j], true))
                ++j;
            i = j;
            bpe_piece(p, off[start], off[i], ids);
            continue;
        }
        if (is_num(c)) {
            int j = i, k = 0;
            while (j < n && is_num(cp[j]) && k < 3) {
                ++j;
                ++k;
            }
            i = j;
            bpe_piece(p, off[start], off[i], ids);
            continue;
        }
        {
            int j = i;
            if (c == ' ' && j + 1 < n && !is_space(cp[j + 1]) && !is_letter(cp[j + 1], true) &&
                !is_num(cp[j + 1]))
                ++j;
            if (j < n && !is_space(cp[j]) && !is_letter(cp[j], true) && !is_num(cp[j])) {
                while (j < n && !is_space(cp[j]) && !is_letter(cp[j], true) && !is_num(cp[j]))
                    ++j;
                while (j < n && is_nl(cp[j]))
                    ++j;
                i = j;
                bpe_piece(p, off[start], off[i], ids);
                continue;
            }
        }
        if (is_space(c)) {
            int r = i;
            while (r < n && is_space(cp[r]))
                ++r;
            int last = -1;
            for (int j = i; j < r; ++j)
                if (is_nl(cp[j]))
                    last = j;
            if (last >= 0)
                i = last + 1;
            else {
                int end = (r < n) ? r - 1 : r;
                if (end <= i)
                    end = i + 1;
                i = end;
            }
            bpe_piece(p, off[start], off[i], ids);
            continue;
        }
        ++i;
        bpe_piece(p, off[start], off[i], ids);
    }
}

void Tokenizer::pretok(const unsigned char *p, int a, int b, std::vector<int> &ids) const {
    if (b <= a)
        return;
    const int nb = b - a;
    std::vector<uint32_t> cp(static_cast<size_t>(nb) + 1);
    std::vector<int> off(static_cast<size_t>(nb) + 2);
    int n = 0;
    for (int i = a; i < b;) {
        uint32_t c = 0;
        int k = u8_next(p, b, i, &c);
        off[n] = i;
        cp[n] = c;
        ++n;
        i += k;
    }
    off[n] = b;
    if (kimi_)
        pretok_kimi(cp.data(), off.data(), n, p, ids);
    else
        pretok_cl100k(cp.data(), off.data(), n, p, ids);
}

Status Tokenizer::load(const std::string &model_dir, std::string &err) {
    token_to_id_.clear();
    merges_.clear();
    id_to_token_.clear();
    id_added_.clear();
    specials_.clear();
    vocab_size_ = 0;
    loaded_ = false;
    rank_bpe_ = false;
    kimi_ = false;

    for (int i = 0; i < 1024; ++i)
        cp2byte_[i] = -1;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        uint32_t cp = direct ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + extra);
        if (!direct)
            ++extra;
        char buf[4];
        int n = u8_put(buf, cp);
        byte2str_[b].assign(buf, static_cast<size_t>(n));
        if (cp < 1024)
            cp2byte_[cp] = static_cast<int16_t>(b);
    }

    std::string path = model_dir;
    if (!path.empty() && path.back() != '/')
        path += '/';
    path += "tokenizer.json";

    if (!file_exists(path)) {
        id_to_token_.resize(256);
        for (int i = 0; i < 256; ++i) {
            std::string t(1, static_cast<char>(static_cast<unsigned char>(i)));
            token_to_id_[t] = i;
            id_to_token_[static_cast<size_t>(i)] = t;
        }
        vocab_size_ = 256;
        loaded_ = true;
        return Status::Ok;
    }

    std::string raw = read_file(path);
    json j = json::parse(raw, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        err = "failed to parse tokenizer.json";
        return Status::ParseError;
    }

    const json *model = j.contains("model") && j["model"].is_object() ? &j["model"] : nullptr;
    const json *vocab = nullptr;
    if (model && model->contains("vocab"))
        vocab = &(*model)["vocab"];
    else if (j.contains("vocab"))
        vocab = &j["vocab"];
    if (vocab && vocab->is_object()) {
        for (auto it = vocab->begin(); it != vocab->end(); ++it) {
            int id = json_id(it.value());
            if (id < 0)
                continue;
            token_to_id_[it.key()] = id;
        }
    }

    bool have_merges = false;
    if (model && model->contains("merges") && (*model)["merges"].is_array() &&
        !(*model)["merges"].empty()) {
        have_merges = true;
        int rank = 0;
        for (const auto &el : (*model)["merges"]) {
            std::string left, right;
            if (el.is_string()) {
                std::string s = el.get<std::string>();
                auto sp = s.find(' ');
                if (sp == std::string::npos)
                    continue;
                left = s.substr(0, sp);
                right = s.substr(sp + 1);
            } else if (el.is_array() && el.size() >= 2 && el[0].is_string() && el[1].is_string()) {
                left = el[0].get<std::string>();
                right = el[1].get<std::string>();
            } else
                continue;
            std::string key = left;
            key.push_back('\0');
            key += right;
            merges_[key] = rank++;
        }
    }
    rank_bpe_ = !have_merges;

    if (j.contains("pre_tokenizer")) {
        std::string dump = j["pre_tokenizer"].dump();
        if (dump.find("Han") != std::string::npos)
            kimi_ = true;
    }

    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto &el : j["added_tokens"]) {
            if (!el.is_object() || !el.contains("content") || !el["content"].is_string() ||
                !el.contains("id"))
                continue;
            std::string tok = el["content"].get<std::string>();
            int id = json_id(el["id"]);
            if (tok.empty() || id < 0)
                continue;
            token_to_id_[tok] = id;
            specials_.push_back({tok, id});
        }
    }

    int max_id = -1;
    for (const auto &kv : token_to_id_)
        if (kv.second > max_id)
            max_id = kv.second;
    if (max_id >= 0) {
        id_to_token_.assign(static_cast<size_t>(max_id) + 1, std::string());
        id_added_.assign(static_cast<size_t>(max_id) + 1, 0);
        for (const auto &kv : token_to_id_)
            id_to_token_[static_cast<size_t>(kv.second)] = kv.first;
        for (const auto &sp : specials_)
            if (sp.second >= 0 && static_cast<size_t>(sp.second) < id_added_.size())
                id_added_[static_cast<size_t>(sp.second)] = 1;
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); });
    vocab_size_ = static_cast<int>(id_to_token_.size());
    loaded_ = true;
    return Status::Ok;
}

Status Tokenizer::encode(const std::string &text, std::vector<int> &ids) const {
    ids.clear();
    if (text.empty())
        return Status::Ok;

    // Byte-fallback (no tokenizer.json): raw bytes.
    if (merges_.empty() && !rank_bpe_ && vocab_size_ == 256 && specials_.empty()) {
        ids.reserve(text.size());
        for (unsigned char c : text)
            ids.push_back(static_cast<int>(c));
        return Status::Ok;
    }
    if (token_to_id_.empty()) {
        for (unsigned char c : text)
            ids.push_back(static_cast<int>(c));
        return Status::Ok;
    }

    const unsigned char *p = reinterpret_cast<const unsigned char *>(text.data());
    const int len = static_cast<int>(text.size());
    int i = 0;
    while (i < len) {
        int hitpos = -1, hitlen = 0, hitid = -1;
        for (int j = i; j < len && hitpos < 0; ++j) {
            for (const auto &sp : specials_) {
                int sl = static_cast<int>(sp.first.size());
                if (sl > 0 && j + sl <= len &&
                    std::memcmp(p + j, sp.first.data(), static_cast<size_t>(sl)) == 0) {
                    hitpos = j;
                    hitlen = sl;
                    hitid = sp.second;
                    break;
                }
            }
        }
        int chunk_end = hitpos < 0 ? len : hitpos;
        if (chunk_end > i)
            pretok(p, i, chunk_end, ids);
        if (hitpos < 0)
            break;
        ids.push_back(hitid);
        i = hitpos + hitlen;
    }
    return Status::Ok;
}

Status Tokenizer::decode(const std::vector<int> &ids, std::string &text) const {
    text.clear();
    for (int id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size())
            continue;
        const std::string &s = id_to_token_[static_cast<size_t>(id)];
        if (id < static_cast<int>(id_added_.size()) && id_added_[static_cast<size_t>(id)]) {
            text += s;
            continue;
        }
        if (vocab_size_ == 256 && s.size() == 1) {
            text += s;
            continue;
        }
        const unsigned char *us = reinterpret_cast<const unsigned char *>(s.data());
        const int sl = static_cast<int>(s.size());
        for (int j = 0; j < sl;) {
            uint32_t c = 0;
            int k = u8_next(us, sl, j, &c);
            j += k;
            if (c < 1024 && cp2byte_[c] >= 0)
                text.push_back(static_cast<char>(static_cast<unsigned char>(cp2byte_[c])));
            else
                text.append(s, static_cast<size_t>(j - k), static_cast<size_t>(k));
        }
    }
    return Status::Ok;
}

int Tokenizer::id_of(const std::string &content) const {
    auto it = token_to_id_.find(content);
    if (it == token_to_id_.end())
        return -1;
    return it->second;
}

bool Tokenizer::has_xtml() const {
    return id_of("<|open|>") >= 0 && id_of("<|close|>") >= 0 && id_of("<|sep|>") >= 0 &&
           id_of("<|end_of_msg|>") >= 0;
}

namespace {

void xtml_append_encode(const Tokenizer &tk, const std::string &s, std::vector<int> &ids) {
    if (s.empty())
        return;
    std::vector<int> piece;
    tk.encode(s, piece);
    ids.insert(ids.end(), piece.begin(), piece.end());
}

} // namespace

Status Tokenizer::encode_chat(Family family, const std::vector<ChatMessage> &msgs, bool think,
                              std::vector<int> &ids) const {
    ids.clear();
    if (family != Family::KimiK3 || !has_xtml())
        return encode(apply_chat(family, msgs, think), ids);

    const int op = id_of("<|open|>");
    const int cl = id_of("<|close|>");
    const int sep = id_of("<|sep|>");
    const int eom = id_of("<|end_of_msg|>");
    auto open_tag = [&](const char *tag, const char *role) {
        ids.push_back(op);
        xtml_append_encode(*this, tag, ids);
        if (role) {
            xtml_append_encode(*this, " role", ids);
            xtml_append_encode(*this, "=\"", ids);
            xtml_append_encode(*this, role, ids);
            xtml_append_encode(*this, "\"", ids);
        }
        ids.push_back(sep);
    };
    auto close_tag = [&](const char *tag) {
        ids.push_back(cl);
        xtml_append_encode(*this, tag, ids);
        ids.push_back(sep);
    };

    for (const auto &m : msgs) {
        std::string role = m.role == "developer" ? "system" : m.role;
        if (role == "assistant") {
            open_tag("message", "assistant");
            if (!m.reasoning.empty()) {
                open_tag("think", nullptr);
                xtml_append_encode(*this, m.reasoning, ids);
                close_tag("think");
            }
            open_tag("response", nullptr);
            xtml_append_encode(*this, m.content, ids);
            close_tag("response");
            close_tag("message");
            ids.push_back(eom);
            continue;
        }
        open_tag("message", role.c_str());
        xtml_append_encode(*this, m.content, ids);
        close_tag("message");
        ids.push_back(eom);
    }
    open_tag("message", "assistant");
    open_tag(think ? "think" : "response", nullptr);
    return Status::Ok;
}

std::string Tokenizer::apply_chat(Family family, const std::vector<ChatMessage> &msgs,
                                  bool think) const {
    if (family == Family::Glm53) {
        std::string p = "[gMASK]<sop>";
        if (think)
            p += "<|system|>Reasoning Effort: Max";
        for (const auto &m : msgs) {
            if (m.role == "system")
                p += "<|system|>" + m.content;
            else if (m.role == "user")
                p += "<|user|>" + m.content;
            else if (m.role == "assistant")
                p += "<|assistant|><think></think>" + m.content;
            else if (m.role == "tool")
                p += "<|observation|><tool_response>" + m.content + "</tool_response>";
        }
        p += think ? "<|assistant|><think>" : "<|assistant|><think></think>";
        return p;
    }
    if (family == Family::KimiK3) {
        if (has_xtml()) {
            std::string p;
            auto open = [&](const char *tag, const char *role) {
                p += "<|open|>";
                p += tag;
                if (role) {
                    p += " role=\"";
                    p += role;
                    p += "\"";
                }
                p += "<|sep|>";
            };
            auto close = [&](const char *tag) {
                p += "<|close|>";
                p += tag;
                p += "<|sep|>";
            };
            for (const auto &m : msgs) {
                std::string role = m.role == "developer" ? "system" : m.role;
                if (role == "assistant") {
                    open("message", "assistant");
                    if (!m.reasoning.empty()) {
                        open("think", nullptr);
                        p += m.reasoning;
                        close("think");
                    }
                    open("response", nullptr);
                    p += m.content;
                    close("response");
                    close("message");
                    p += "<|end_of_msg|>";
                    continue;
                }
                open("message", role.c_str());
                p += m.content;
                close("message");
                p += "<|end_of_msg|>";
            }
            open("message", "assistant");
            open(think ? "think" : "response", nullptr);
            return p;
        }
        std::string p;
        for (const auto &m : msgs) {
            std::string role = m.role == "developer" ? "system" : m.role;
            p += "<|im_start|>" + role + "\n" + m.content + "<|im_end|>\n";
        }
        p += "<|im_start|>assistant\n";
        if (think)
            p += "<think>\n";
        return p;
    }
    std::string p;
    for (const auto &m : msgs) {
        if (!p.empty())
            p += "\n";
        p += m.content;
    }
    return p;
}

} // namespace mvllm
