#include "tokenizer.hpp"
#include "glm_tools.hpp"

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
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

int b64_digit(unsigned char c) {
    if (c >= 'A' && c <= 'Z')
        return c - 'A';
    if (c >= 'a' && c <= 'z')
        return c - 'a' + 26;
    if (c >= '0' && c <= '9')
        return c - '0' + 52;
    if (c == '+')
        return 62;
    if (c == '/')
        return 63;
    return -1;
}

bool b64_decode(const std::string &in, std::string &out) {
    out.clear();
    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t')
            break;
        int d = b64_digit(c);
        if (d < 0)
            return false;
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xff));
            valb -= 8;
        }
    }
    return true;
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
    return c == 0x3005 || c == 0x3007 || (c >= 0x2e80 && c <= 0x2fdf) ||
           (c >= 0x31c0 && c <= 0x31ef) || (c >= 0x3400 && c <= 0x4dbf) ||
           (c >= 0x4e00 && c <= 0x9fff) || (c >= 0xf900 && c <= 0xfaff) ||
           (c >= 0x20000 && c <= 0x2a6df) || (c >= 0x2a700 && c <= 0x2b73f) ||
           (c >= 0x2b740 && c <= 0x2b81f) || (c >= 0x2b820 && c <= 0x2ceaf) ||
           (c >= 0x2ceb0 && c <= 0x2ebe0) || (c >= 0x30000 && c <= 0x323af);
}

bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }

bool is_num(uint32_t c) {
    return (c >= '0' && c <= '9') || (c >= 0xff10 && c <= 0xff19);
}

bool is_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == 0xa0 || c == 0x3000;
}

bool is_cjk_punct(uint32_t c) {
    if (c == 0x3005 || c == 0x3007)
        return false;
    if (c >= 0x3000 && c <= 0x303f)
        return true;
    if (c >= 0x2010 && c <= 0x2027)
        return true;
    if ((c >= 0xff01 && c <= 0xff20) || (c >= 0xff3b && c <= 0xff40) ||
        (c >= 0xff5b && c <= 0xff65))
        return true;
    return false;
}

bool is_letter(uint32_t c, bool exclude_han) {
    if (exclude_han && is_han(c))
        return false;
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
        return true;
    if (c >= 0xff21 && c <= 0xff3a)
        return true;
    if (c >= 0xff41 && c <= 0xff5a)
        return true;
    if (c < 0xc0)
        return false;
    if (is_num(c) || is_space(c) || is_nl(c) || is_cjk_punct(c))
        return false;
    if (c == 0xd7 || c == 0xf7)
        return false;
    if (is_han(c))
        return !exclude_han;
    return c >= 0xc0;
}

bool is_upper(uint32_t c) {
    if (c >= 'A' && c <= 'Z')
        return true;
    if (c >= 0xff21 && c <= 0xff3a)
        return true;
    if (c >= 0xc0 && c <= 0xd6)
        return true;
    if (c >= 0xd8 && c <= 0xde)
        return true;
    return false;
}

bool is_lower(uint32_t c) {
    if (c >= 'a' && c <= 'z')
        return true;
    if (c >= 0xff41 && c <= 0xff5a)
        return true;
    if (c >= 0xdf && c <= 0xf6)
        return true;
    if (c >= 0xf8 && c <= 0xff)
        return true;
    return false;
}

uint32_t low_ascii(uint32_t c) {
    if (c >= 'A' && c <= 'Z')
        return c + 32;
    return c;
}

int contraction_len(const uint32_t *cp, int i, int n) {
    if (i >= n || cp[i] != '\'')
        return 0;
    if (i + 1 >= n)
        return 0;
    uint32_t d = low_ascii(cp[i + 1]);
    if (i + 2 < n) {
        uint32_t d2 = low_ascii(cp[i + 2]);
        if ((d == 'r' && d2 == 'e') || (d == 'v' && d2 == 'e') || (d == 'l' && d2 == 'l'))
            return 3;
    }
    if (d == 's' || d == 't' || d == 'm' || d == 'd')
        return 2;
    return 0;
}

bool is_word_prefix(uint32_t c) {
    return !is_nl(c) && !is_letter(c, true) && !is_num(c) && !is_han(c);
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
        // o200k letter word: optional leading non-L/N, Lu*/Ll+ or Lu+/Ll*, then contraction.
        {
            int j = i;
            if (j < n && is_word_prefix(cp[j]) && j + 1 < n && is_letter(cp[j + 1], true))
                ++j;
            if (j < n && is_letter(cp[j], true)) {
                const int word0 = j;
                if (is_lower(cp[j]) || (is_upper(cp[j]) && j + 1 < n && is_lower(cp[j + 1]))) {
                    while (j < n && is_upper(cp[j]))
                        ++j;
                    if (j < n && is_lower(cp[j])) {
                        while (j < n && is_lower(cp[j]))
                            ++j;
                    } else if (word0 == j) {
                        while (j < n && is_letter(cp[j], true))
                            ++j;
                    }
                } else if (is_upper(cp[j])) {
                    while (j < n && is_upper(cp[j]))
                        ++j;
                    while (j < n && is_lower(cp[j]))
                        ++j;
                } else {
                    while (j < n && is_letter(cp[j], true))
                        ++j;
                }
                j += contraction_len(cp, j, n);
                i = j;
                bpe_piece(p, off[start], off[i], ids);
                continue;
            }
        }
        {
            int cl = contraction_len(cp, i, n);
            if (cl > 0) {
                i += cl;
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
        {
            int j = i;
            if (c == ' ' && j + 1 < n && !is_space(cp[j + 1]) && !is_letter(cp[j + 1], true) &&
                !is_num(cp[j + 1]) && !is_han(cp[j + 1]))
                ++j;
            if (j < n && !is_space(cp[j]) && !is_letter(cp[j], true) && !is_num(cp[j]) &&
                !is_han(cp[j])) {
                while (j < n && !is_space(cp[j]) && !is_letter(cp[j], true) && !is_num(cp[j]) &&
                       !is_han(cp[j]))
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
    from_tiktoken_ = false;

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

    std::string dir = model_dir;
    if (!dir.empty() && dir.back() != '/')
        dir += '/';
    const std::string path = dir + "tokenizer.json";
    const std::string tt_path = dir + "tiktoken.model";

    auto seal = [&]() {
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
    };

    if (!file_exists(path)) {
        if (file_exists(tt_path)) {
            std::ifstream in(tt_path);
            if (!in) {
                err = "cannot open tiktoken.model";
                return Status::IoError;
            }
            std::string line;
            int ntok = 0;
            int max_rank = -1;
            while (std::getline(in, line)) {
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                if (line.empty())
                    continue;
                auto sp = line.find(' ');
                if (sp == std::string::npos)
                    continue;
                std::string raw;
                if (!b64_decode(line.substr(0, sp), raw)) {
                    err = "tiktoken.model: bad base64";
                    return Status::ParseError;
                }
                int rank = std::atoi(line.c_str() + sp + 1);
                if (rank < 0)
                    continue;
                std::string key;
                key.reserve(raw.size() * 2);
                for (unsigned char b : raw)
                    key += byte2str_[b];
                token_to_id_[key] = rank;
                if (rank > max_rank)
                    max_rank = rank;
                ++ntok;
            }
            if (ntok <= 0) {
                err = "tiktoken.model: empty vocab";
                return Status::ParseError;
            }
            int num_base = max_rank + 1;
            json tc;
            std::string tcraw = read_file(dir + "tokenizer_config.json");
            if (!tcraw.empty()) {
                tc = json::parse(tcraw, nullptr, false);
                if (tc.is_discarded())
                    tc = json::object();
            }
            const json *dec = nullptr;
            if (tc.is_object() && tc.contains("added_tokens_decoder") &&
                tc["added_tokens_decoder"].is_object())
                dec = &tc["added_tokens_decoder"];
            for (int i = num_base; i < num_base + 256; ++i) {
                std::string content = "<|reserved_token_" + std::to_string(i) + "|>";
                if (dec && dec->contains(std::to_string(i))) {
                    const json &ent = (*dec)[std::to_string(i)];
                    if (ent.is_object() && ent.contains("content") && ent["content"].is_string())
                        content = ent["content"].get<std::string>();
                }
                if (content.empty())
                    continue;
                token_to_id_[content] = i;
                specials_.push_back({content, i});
            }
            if (dec) {
                for (auto it = dec->begin(); it != dec->end(); ++it) {
                    int id = std::atoi(it.key().c_str());
                    if (id < 0 || !it.value().is_object())
                        continue;
                    if (!it.value().contains("content") || !it.value()["content"].is_string())
                        continue;
                    std::string content = it.value()["content"].get<std::string>();
                    if (content.empty())
                        continue;
                    token_to_id_[content] = id;
                    bool have = false;
                    for (const auto &sp : specials_)
                        if (sp.second == id) {
                            have = true;
                            break;
                        }
                    if (!have)
                        specials_.push_back({content, id});
                }
            }
            rank_bpe_ = true;
            kimi_ = true;
            from_tiktoken_ = true;
            seal();
            return Status::Ok;
        }
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

    seal();
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

std::string xtml_trim(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a])))
        ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])))
        --b;
    return s.substr(a, b - a);
}

bool xtml_looks_json_object(const std::string &s) {
    const std::string t = xtml_trim(s);
    return t.size() >= 2 && t.front() == '{' && t.back() == '}';
}

std::vector<K3ToolCall> xtml_effective_calls(const ChatMessage &m) {
    if (!m.tool_calls.empty())
        return m.tool_calls;
    const bool assistantish = m.role == "assistant" || m.role == "tool_call";
    if (!assistantish || (m.tool_name.empty() && m.role != "tool_call"))
        return {};
    K3ToolCall c;
    c.name = m.tool_name;
    c.index = 1;
    const std::string body = xtml_trim(m.content);
    if (xtml_looks_json_object(body))
        c.json = body;
    else {
        K3ToolArg a;
        a.key = "input";
        a.type = "string";
        a.value = m.content;
        c.args.push_back(std::move(a));
    }
    return {std::move(c)};
}

bool xtml_synthesized_call(const ChatMessage &m) {
    return m.tool_calls.empty() &&
           (m.role == "tool_call" || (m.role == "assistant" && !m.tool_name.empty()));
}

// One plain <image> per URL when the text has no vision token yet (K3 / Llama).
void append_plain_image_placeholders(std::string &body, const std::vector<std::string> &urls) {
    if (urls.empty() || body.find("<image>") != std::string::npos ||
        body.find("<|image|>") != std::string::npos)
        return;
    for (size_t i = 0; i < urls.size(); ++i)
        body += "<image>";
}

void xtml_attr_encode(const Tokenizer &tk, const char *key, const std::string &val,
                      std::vector<int> &ids) {
    xtml_append_encode(tk, std::string(" ") + key, ids);
    xtml_append_encode(tk, "=\"", ids);
    xtml_append_encode(tk, k3_xtml_escape_attr(val), ids);
    xtml_append_encode(tk, "\"", ids);
}

void xtml_encode_tools_block(const Tokenizer &tk, const std::vector<K3ToolCall> &calls, int op,
                             int cl, int sep, std::vector<int> &ids) {
    if (calls.empty())
        return;
    auto open = [&](const char *tag) {
        ids.push_back(op);
        xtml_append_encode(tk, tag, ids);
    };
    auto close = [&](const char *tag) {
        ids.push_back(cl);
        xtml_append_encode(tk, tag, ids);
        ids.push_back(sep);
    };
    open("tools");
    ids.push_back(sep);
    for (size_t i = 0; i < calls.size(); ++i) {
        const auto &c = calls[i];
        const int idx = c.index >= 1 ? c.index : static_cast<int>(i) + 1;
        open("call");
        xtml_attr_encode(tk, "tool", c.name, ids);
        xtml_attr_encode(tk, "index", std::to_string(idx), ids);
        ids.push_back(sep);
        if (!c.json.empty()) {
            open("json");
            xtml_attr_encode(tk, "type", "object", ids);
            ids.push_back(sep);
            xtml_append_encode(tk, c.json, ids);
            close("json");
        } else {
            for (const auto &a : c.args) {
                open("argument");
                xtml_attr_encode(tk, "key", a.key, ids);
                xtml_attr_encode(tk, "type", a.type.empty() ? "string" : a.type, ids);
                ids.push_back(sep);
                xtml_append_encode(tk, a.value, ids);
                close("argument");
            }
        }
        close("call");
    }
    close("tools");
}

} // namespace

Status Tokenizer::encode_chat(Family family, const std::vector<ChatMessage> &msgs, bool think,
                              std::vector<int> &ids, const std::string &effort,
                              const std::vector<K3ToolDecl> *tools) const {
    ids.clear();
    if (family != Family::KimiK3 || !has_xtml())
        return encode(apply_chat(family, msgs, think, effort, tools), ids);

    const int op = id_of("<|open|>");
    const int cl = id_of("<|close|>");
    const int sep = id_of("<|sep|>");
    const int eom = id_of("<|end_of_msg|>");
    auto open_tag = [&](const char *tag) {
        ids.push_back(op);
        xtml_append_encode(*this, tag, ids);
    };
    auto close_tag = [&](const char *tag) {
        ids.push_back(cl);
        xtml_append_encode(*this, tag, ids);
        ids.push_back(sep);
    };
    auto open_message = [&](const std::string &role, const std::string &type = {}) {
        open_tag("message");
        xtml_attr_encode(*this, "role", role, ids);
        if (!type.empty())
            xtml_attr_encode(*this, "type", type, ids);
        ids.push_back(sep);
    };

    if (tools && !tools->empty()) {
        open_message("system", "tool-declare");
        xtml_append_encode(*this, k3_tool_declare_body(*tools), ids);
        close_tag("message");
        ids.push_back(eom);
    }

    for (const auto &m : msgs) {
        std::string role = m.role == "developer" ? "system" : m.role;
        if (role == "tool" || role == "tool_result") {
            open_tag("message");
            xtml_attr_encode(*this, "role", "tool", ids);
            xtml_attr_encode(*this, "tool", m.tool_name, ids);
            xtml_attr_encode(*this, "index",
                             std::to_string(m.tool_index >= 1 ? m.tool_index : 1), ids);
            ids.push_back(sep);
            xtml_append_encode(*this, m.content, ids);
            close_tag("message");
            ids.push_back(eom);
            continue;
        }
        if (role == "assistant" || role == "tool_call") {
            const std::vector<K3ToolCall> calls = xtml_effective_calls(m);
            const bool syn = xtml_synthesized_call(m);
            open_message("assistant");
            if (!m.reasoning.empty()) {
                open_tag("think");
                ids.push_back(sep);
                xtml_append_encode(*this, m.reasoning, ids);
                close_tag("think");
            }
            open_tag("response");
            ids.push_back(sep);
            if (!syn)
                xtml_append_encode(*this, m.content, ids);
            close_tag("response");
            if (!calls.empty())
                xtml_encode_tools_block(*this, calls, op, cl, sep, ids);
            close_tag("message");
            ids.push_back(eom);
            continue;
        }
        open_message(role, m.xtml_type);
        std::string body = m.content;
        append_plain_image_placeholders(body, m.image_urls);
        xtml_append_encode(*this, body, ids);
        close_tag("message");
        ids.push_back(eom);
    }
    open_message("assistant");
    open_tag(think ? "think" : "response");
    ids.push_back(sep);
    return Status::Ok;
}

std::string Tokenizer::apply_chat(Family family, const std::vector<ChatMessage> &msgs,
                                  bool think, const std::string &effort,
                                  const std::vector<K3ToolDecl> *tools) const {
    if (family == Family::Glm53) {
        auto expand_images = [](std::string s) {
            const std::string needle = "<image>";
            const std::string wrap = "<|begin_of_image|><image><|end_of_image|>";
            size_t pos = 0;
            while ((pos = s.find(needle, pos)) != std::string::npos) {
                if (pos >= 18 && s.compare(pos - 18, 18, "<|begin_of_image|>") == 0) {
                    pos += needle.size();
                    continue;
                }
                s.replace(pos, needle.size(), wrap);
                pos += wrap.size();
            }
            return s;
        };
        std::string p = "[gMASK]<sop>";
        if (think) {
            std::string label = "Max";
            if (effort == "low" || effort == "minimal")
                label = "Low";
            else if (effort == "high" || effort == "medium")
                label = "High";
            p += "<|system|>Reasoning Effort: " + label;
        }
        if (tools && !tools->empty())
            p += glm_tool_declare(*tools);
        for (const auto &m : msgs) {
            if (m.role == "system")
                p += "<|system|>" + m.content;
            else if (m.role == "user") {
                std::string body = expand_images(m.content);
                if (!m.image_urls.empty() && body.find("<image>") == std::string::npos &&
                    body.find("<|image|>") == std::string::npos) {
                    for (size_t i = 0; i < m.image_urls.size(); ++i)
                        body += "<|begin_of_image|><image><|end_of_image|>";
                }
                p += "<|user|>" + body;
            }
            else if (m.role == "assistant") {
                p += "<|assistant|><think></think>" + m.content;
                if (!m.tool_calls.empty())
                    p += glm_render_tool_calls(m.tool_calls);
            }
            else if (m.role == "tool")
                p += "<|observation|><tool_response>" + m.content + "</tool_response>";
        }
        p += think ? "<|assistant|><think>" : "<|assistant|><think></think>";
        return p;
    }
    if (family == Family::KimiK3) {
        if (has_xtml()) {
            std::string p;
            auto attr = [&](const char *key, const std::string &val) {
                p += " ";
                p += key;
                p += "=\"";
                p += k3_xtml_escape_attr(val);
                p += "\"";
            };
            auto open = [&](const char *tag) {
                p += "<|open|>";
                p += tag;
            };
            auto close = [&](const char *tag) {
                p += "<|close|>";
                p += tag;
                p += "<|sep|>";
            };
            auto end_open = [&]() { p += "<|sep|>"; };
            auto open_message = [&](const std::string &role, const std::string &type = {}) {
                open("message");
                attr("role", role);
                if (!type.empty())
                    attr("type", type);
                end_open();
            };
            if (tools && !tools->empty()) {
                open_message("system", "tool-declare");
                p += k3_tool_declare_body(*tools);
                close("message");
                p += "<|end_of_msg|>";
            }
            for (const auto &m : msgs) {
                std::string role = m.role == "developer" ? "system" : m.role;
                if (role == "tool" || role == "tool_result") {
                    open("message");
                    attr("role", "tool");
                    attr("tool", m.tool_name);
                    attr("index", std::to_string(m.tool_index >= 1 ? m.tool_index : 1));
                    end_open();
                    p += m.content;
                    close("message");
                    p += "<|end_of_msg|>";
                    continue;
                }
                if (role == "assistant" || role == "tool_call") {
                    const std::vector<K3ToolCall> calls = xtml_effective_calls(m);
                    const bool syn = xtml_synthesized_call(m);
                    open_message("assistant");
                    if (!m.reasoning.empty()) {
                        open("think");
                        end_open();
                        p += m.reasoning;
                        close("think");
                    }
                    open("response");
                    end_open();
                    if (!syn)
                        p += m.content;
                    close("response");
                    if (!calls.empty())
                        p += k3_render_tools_block(calls);
                    close("message");
                    p += "<|end_of_msg|>";
                    continue;
                }
                open_message(role, m.xtml_type);
                std::string body = m.content;
                append_plain_image_placeholders(body, m.image_urls);
                p += body;
                close("message");
                p += "<|end_of_msg|>";
            }
            open_message("assistant");
            open(think ? "think" : "response");
            end_open();
            return p;
        }
        std::string p;
        for (const auto &m : msgs) {
            std::string role = m.role == "developer" ? "system" : m.role;
            std::string body = m.content;
            append_plain_image_placeholders(body, m.image_urls);
            p += "<|im_start|>" + role + "\n" + body + "<|im_end|>\n";
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
        std::string body = m.content;
        append_plain_image_placeholders(body, m.image_urls);
        p += body;
    }
    return p;
}

} // namespace mvllm
