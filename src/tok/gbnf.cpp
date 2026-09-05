#include "gbnf.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mvllm {
namespace {

constexpr int kMaxRules = 1024;
constexpr int kMaxStacks = 64;
constexpr int kMaxDepth = 128;
constexpr int kMaxGroup = 32;

struct ByteSet {
    uint8_t bits[32]{};

    void add(int b) { bits[(b & 255) >> 3] |= static_cast<uint8_t>(1u << (b & 7)); }
    bool has(unsigned char b) const { return (bits[b >> 3] & (1u << (b & 7))) != 0; }
    void invert() {
        for (int i = 0; i < 32; ++i)
            bits[i] = static_cast<uint8_t>(~bits[i]);
    }
    void merge(const ByteSet &o) {
        for (int i = 0; i < 32; ++i)
            bits[i] |= o.bits[i];
    }
};

enum class SymKind : uint8_t { Cls, Ref };

struct Sym {
    SymKind kind = SymKind::Cls;
    int16_t ref = -1;
    ByteSet cls;
};

struct Alt {
    std::vector<Sym> syms;
};

struct Rule {
    std::string name;
    std::vector<Alt> alts;
};

struct Grammar {
    std::vector<Rule> rules;
    int root = -1;
};

struct Frame {
    int16_t r = 0;
    int16_t a = 0;
    int16_t s = 0;
};

struct Stack {
    int n = 0;
    Frame f[kMaxDepth];

    bool same(const Stack &o) const {
        if (n != o.n)
            return false;
        return std::memcmp(f, o.f, sizeof(Frame) * static_cast<size_t>(n)) == 0;
    }
};

struct Pda {
    int n = 0;
    bool dead = false;
    Stack st[kMaxStacks];
};

struct Impl {
    Grammar g;
    Pda live;
};

std::unordered_map<const Gbnf *, std::unique_ptr<Impl>> &table() {
    static std::unordered_map<const Gbnf *, std::unique_ptr<Impl>> t;
    return t;
}

Impl *impl_of(const Gbnf *g) {
    auto it = table().find(g);
    return it == table().end() ? nullptr : it->second.get();
}

void drop_impl(const Gbnf *g) { table().erase(g); }

bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

bool is_id(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
           c == '-';
}

int hex_digit(char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

const char *skip_ws(const char *p) {
    for (;;) {
        while (*p && is_ws(*p))
            ++p;
        if (*p == '#') {
            while (*p && *p != '\n')
                ++p;
            continue;
        }
        return p;
    }
}

int id_len(const char *p) {
    int n = 0;
    while (is_id(p[n]))
        ++n;
    return n;
}

// After '\\': consume an escape and return the byte, or -1.
int parse_esc(const char *&p) {
    const char *s = p;
    int b = -1;
    switch (*s) {
    case 'n':
        b = '\n';
        break;
    case 'r':
        b = '\r';
        break;
    case 't':
        b = '\t';
        break;
    case '"':
        b = '"';
        break;
    case '\\':
        b = '\\';
        break;
    case '[':
        b = '[';
        break;
    case ']':
        b = ']';
        break;
    case '-':
        b = '-';
        break;
    case '^':
        b = '^';
        break;
    case 'x': {
        int hi = hex_digit(s[1]);
        int lo = hex_digit(s[2]);
        if (hi < 0 || lo < 0)
            return -1;
        b = (hi << 4) | lo;
        s += 2;
        break;
    }
    default:
        return -1;
    }
    p = s + 1;
    return b;
}

int intern_rule(Grammar &g, const char *name, int n) {
    std::string key(name, name + n);
    for (int i = 0; i < static_cast<int>(g.rules.size()); ++i)
        if (g.rules[static_cast<size_t>(i)].name == key)
            return i;
    if (static_cast<int>(g.rules.size()) >= kMaxRules)
        return -1;
    Rule r;
    r.name = std::move(key);
    g.rules.push_back(std::move(r));
    return static_cast<int>(g.rules.size()) - 1;
}

int anon_rule(Grammar &g) {
    if (static_cast<int>(g.rules.size()) >= kMaxRules)
        return -1;
    Rule r;
    r.name = "$" + std::to_string(g.rules.size());
    g.rules.push_back(std::move(r));
    return static_cast<int>(g.rules.size()) - 1;
}

bool push_sym(Grammar &g, int ri, int ai, const Sym &sy) {
    g.rules[static_cast<size_t>(ri)].alts[static_cast<size_t>(ai)].syms.push_back(sy);
    return true;
}

bool apply_postfix(Grammar &g, int ri, int ai, int n0, char op, std::string &err) {
    auto &syms = g.rules[static_cast<size_t>(ri)].alts[static_cast<size_t>(ai)].syms;
    const int k = static_cast<int>(syms.size()) - n0;
    if (k <= 0)
        return true;
    const int item = anon_rule(g);
    if (item < 0) {
        err = "grammar is too large";
        return false;
    }
    g.rules[static_cast<size_t>(item)].alts.emplace_back();
    for (int j = 0; j < k; ++j)
        g.rules[static_cast<size_t>(item)].alts[0].syms.push_back(syms[static_cast<size_t>(n0 + j)]);
    syms.resize(static_cast<size_t>(n0));

    const int wrap = anon_rule(g);
    if (wrap < 0) {
        err = "grammar is too large";
        return false;
    }
    Sym I;
    I.kind = SymKind::Ref;
    I.ref = static_cast<int16_t>(item);
    Sym R;
    R.kind = SymKind::Ref;
    R.ref = static_cast<int16_t>(wrap);

    // ?  ->  R ::= I | ε
    // *  ->  R ::= I R | ε
    // +  ->  R ::= I R | I
    auto &W = g.rules[static_cast<size_t>(wrap)];
    W.alts.emplace_back();
    W.alts[0].syms.push_back(I);
    if (op == '*' || op == '+')
        W.alts[0].syms.push_back(R);
    W.alts.emplace_back();
    if (op == '+')
        W.alts[1].syms.push_back(I);

    if (!push_sym(g, ri, ai, R)) {
        err = "out of memory";
        return false;
    }
    return true;
}

bool parse_alts(Grammar &g, int ri, const char *&p, int depth, bool in_group, std::string &err);

bool parse_literal(Grammar &g, int ri, int ai, const char *&p, std::string &err) {
    ++p; // opening "
    while (*p && *p != '"') {
        int b;
        if (*p == '\\') {
            ++p;
            b = parse_esc(p);
            if (b < 0) {
                err = "invalid escape in literal";
                return false;
            }
        } else {
            b = static_cast<unsigned char>(*p++);
        }
        Sym sy;
        sy.kind = SymKind::Cls;
        sy.cls.add(b);
        if (!push_sym(g, ri, ai, sy)) {
            err = "out of memory";
            return false;
        }
    }
    if (*p != '"') {
        err = "unterminated literal";
        return false;
    }
    ++p;
    return true;
}

bool parse_class(Grammar &g, int ri, int ai, const char *&p, std::string &err) {
    ++p; // opening [
    bool neg = false;
    if (*p == '^') {
        neg = true;
        ++p;
    }
    Sym sy;
    sy.kind = SymKind::Cls;
    while (*p && *p != ']') {
        int lo;
        if (*p == '\\') {
            ++p;
            lo = parse_esc(p);
            if (lo < 0) {
                err = "invalid escape in character class";
                return false;
            }
        } else {
            lo = static_cast<unsigned char>(*p++);
        }
        int hi = lo;
        if (*p == '-' && p[1] && p[1] != ']') {
            ++p;
            if (*p == '\\') {
                ++p;
                hi = parse_esc(p);
                if (hi < 0) {
                    err = "invalid escape in character class";
                    return false;
                }
            } else {
                hi = static_cast<unsigned char>(*p++);
            }
        }
        if (hi < lo)
            std::swap(lo, hi);
        for (int b = lo; b <= hi; ++b)
            sy.cls.add(b);
    }
    if (*p != ']') {
        err = "unterminated character class";
        return false;
    }
    ++p;
    if (neg)
        sy.cls.invert();
    return push_sym(g, ri, ai, sy);
}

bool parse_alts(Grammar &g, int ri, const char *&p, int depth, bool in_group, std::string &err) {
    if (depth > kMaxGroup) {
        err = "groups are nested too deeply";
        return false;
    }
    g.rules[static_cast<size_t>(ri)].alts.emplace_back();
    int ai = static_cast<int>(g.rules[static_cast<size_t>(ri)].alts.size()) - 1;
    for (;;) {
        p = skip_ws(p);
        if (!*p) {
            if (in_group) {
                err = "missing ')'";
                return false;
            }
            return true;
        }
        if (*p == ')') {
            if (!in_group) {
                err = "unexpected ')'";
                return false;
            }
            return true;
        }
        if (*p == '|') {
            ++p;
            g.rules[static_cast<size_t>(ri)].alts.emplace_back();
            ai = static_cast<int>(g.rules[static_cast<size_t>(ri)].alts.size()) - 1;
            continue;
        }
        const int n0 =
            static_cast<int>(g.rules[static_cast<size_t>(ri)].alts[static_cast<size_t>(ai)].syms.size());
        if (*p == '"') {
            if (!parse_literal(g, ri, ai, p, err))
                return false;
        } else if (*p == '[') {
            if (!parse_class(g, ri, ai, p, err))
                return false;
        } else if (*p == '(') {
            ++p;
            const int gi = anon_rule(g);
            if (gi < 0) {
                err = "grammar is too large";
                return false;
            }
            if (!parse_alts(g, gi, p, depth + 1, true, err))
                return false;
            p = skip_ws(p);
            if (*p != ')') {
                err = "missing ')'";
                return false;
            }
            ++p;
            Sym sy;
            sy.kind = SymKind::Ref;
            sy.ref = static_cast<int16_t>(gi);
            if (!push_sym(g, ri, ai, sy)) {
                err = "out of memory";
                return false;
            }
        } else if (is_id(*p)) {
            const int nl = id_len(p);
            const char *after = skip_ws(p + nl);
            if (!in_group && after[0] == ':' && after[1] == ':' && after[2] == '=')
                return true; // next top-level rule
            const int ref = intern_rule(g, p, nl);
            if (ref < 0) {
                err = "too many rules";
                return false;
            }
            p += nl;
            Sym sy;
            sy.kind = SymKind::Ref;
            sy.ref = static_cast<int16_t>(ref);
            if (!push_sym(g, ri, ai, sy)) {
                err = "out of memory";
                return false;
            }
        } else {
            err = "unexpected character";
            if (static_cast<unsigned char>(*p) >= 32 && static_cast<unsigned char>(*p) < 127) {
                err += " '";
                err += *p;
                err += "'";
            }
            return false;
        }
        p = skip_ws(p);
        if (*p == '?' || *p == '*' || *p == '+') {
            if (!apply_postfix(g, ri, ai, n0, *p, err))
                return false;
            ++p;
        }
    }
}

bool compile_src(Grammar &g, const std::string &src, std::string &err) {
    g = Grammar{};
    const char *p = src.c_str();
    for (;;) {
        p = skip_ws(p);
        if (!*p)
            break;
        const int nl = id_len(p);
        if (nl <= 0) {
            err = "expected a rule";
            return false;
        }
        const char *name = p;
        const char *q = skip_ws(p + nl);
        if (!(q[0] == ':' && q[1] == ':' && q[2] == '=')) {
            err = "expected '::=' after rule name";
            return false;
        }
        p = q + 3;
        const int ri = intern_rule(g, name, nl);
        if (ri < 0) {
            err = "too many rules";
            return false;
        }
        if (!g.rules[static_cast<size_t>(ri)].alts.empty()) {
            err = "duplicate rule '";
            err.append(name, static_cast<size_t>(nl));
            err += "'";
            return false;
        }
        if (!parse_alts(g, ri, p, 0, false, err))
            return false;
    }
    for (int i = 0; i < static_cast<int>(g.rules.size()); ++i) {
        if (g.rules[static_cast<size_t>(i)].name == "root")
            g.root = i;
        if (g.rules[static_cast<size_t>(i)].alts.empty()) {
            err = "rule '";
            err += g.rules[static_cast<size_t>(i)].name;
            err += "' is used but never defined";
            return false;
        }
    }
    if (g.root < 0) {
        err = "missing root rule";
        return false;
    }
    return true;
}

bool add_stack(Pda &out, const Stack &k, bool &overflow) {
    for (int i = 0; i < out.n; ++i)
        if (out.st[i].same(k))
            return true;
    if (out.n >= kMaxStacks) {
        overflow = true;
        return false;
    }
    out.st[out.n++] = k;
    return true;
}

// Push the stack to a terminal top (or empty = parse complete), forking on refs.
// Last-symbol refs are tail-called so `*` / `+` stay constant-depth.
bool normalize(const Grammar &g, Stack k, Pda &out, bool &overflow, int depth) {
    for (;;) {
        if (k.n == 0)
            return add_stack(out, k, overflow);
        Frame &t = k.f[k.n - 1];
        const auto &alts = g.rules[static_cast<size_t>(t.r)].alts;
        if (t.a < 0 || t.a >= static_cast<int16_t>(alts.size())) {
            overflow = true;
            return false;
        }
        const auto &syms = alts[static_cast<size_t>(t.a)].syms;
        if (t.s >= static_cast<int16_t>(syms.size())) {
            --k.n;
            continue;
        }
        const Sym &sy = syms[static_cast<size_t>(t.s)];
        if (sy.kind == SymKind::Cls)
            return add_stack(out, k, overflow);
        if (sy.ref < 0 || sy.ref >= static_cast<int16_t>(g.rules.size())) {
            overflow = true;
            return false;
        }
        if (depth >= kMaxDepth) {
            overflow = true;
            return false;
        }
        const auto &calts = g.rules[static_cast<size_t>(sy.ref)].alts;
        const bool tail = (t.s + 1 >= static_cast<int16_t>(syms.size()));
        if (tail) {
            for (int a = 0; a < static_cast<int>(calts.size()); ++a) {
                Stack cp = k;
                cp.f[cp.n - 1].r = sy.ref;
                cp.f[cp.n - 1].a = static_cast<int16_t>(a);
                cp.f[cp.n - 1].s = 0;
                if (!normalize(g, cp, out, overflow, depth + 1))
                    return false;
            }
            return true;
        }
        ++t.s;
        if (k.n >= kMaxDepth) {
            overflow = true;
            return false;
        }
        for (int a = 0; a < static_cast<int>(calts.size()); ++a) {
            Stack cp = k;
            cp.f[cp.n].r = sy.ref;
            cp.f[cp.n].a = static_cast<int16_t>(a);
            cp.f[cp.n].s = 0;
            ++cp.n;
            if (!normalize(g, cp, out, overflow, depth + 1))
                return false;
        }
        return true;
    }
}

void pda_start(const Grammar &g, Pda &s) {
    s.n = 0;
    s.dead = false;
    bool overflow = false;
    const auto &alts = g.rules[static_cast<size_t>(g.root)].alts;
    for (int a = 0; a < static_cast<int>(alts.size()); ++a) {
        Stack k{};
        k.n = 1;
        k.f[0].r = static_cast<int16_t>(g.root);
        k.f[0].a = static_cast<int16_t>(a);
        k.f[0].s = 0;
        if (!normalize(g, k, s, overflow, 0)) {
            s.dead = true;
            s.n = 0;
            return;
        }
    }
    if (s.n == 0)
        s.dead = true;
}

// Advance one byte. Illegal input or overflow kills `s`.
bool pda_accept(const Grammar &g, Pda &s, unsigned char b) {
    if (s.dead)
        return false;
    Pda nxt{};
    bool overflow = false;
    for (int i = 0; i < s.n; ++i) {
        const Stack &k = s.st[i];
        if (k.n == 0)
            continue;
        const Frame &t = k.f[k.n - 1];
        const Sym &sy =
            g.rules[static_cast<size_t>(t.r)].alts[static_cast<size_t>(t.a)].syms[static_cast<size_t>(t.s)];
        if (!sy.cls.has(b))
            continue;
        Stack cp = k;
        ++cp.f[cp.n - 1].s;
        if (!normalize(g, cp, nxt, overflow, 0)) {
            s.dead = true;
            s.n = 0;
            return false;
        }
    }
    if (overflow || nxt.n == 0) {
        s.dead = true;
        s.n = 0;
        return false;
    }
    s.n = nxt.n;
    std::memcpy(s.st, nxt.st, sizeof(Stack) * static_cast<size_t>(nxt.n));
    return true;
}

void fill_bytes(uint8_t *ok, int vocab, uint8_t v) {
    if (ok && vocab > 0)
        std::fill(ok, ok + vocab, v);
}

} // namespace

Status Gbnf::compile(const std::string &src, std::string &err) {
    ready_ = false;
    drop_impl(this);
    err.clear();
    if (src.empty()) {
        err = "empty grammar";
        return Status::InvalidArgument;
    }
    auto im = std::make_unique<Impl>();
    if (!compile_src(im->g, src, err)) {
        ready_ = false;
        return Status::InvalidArgument;
    }
    pda_start(im->g, im->live);
    table()[this] = std::move(im);
    ready_ = true;
    return Status::Ok;
}

void Gbnf::reset() {
    Impl *im = impl_of(this);
    if (!im || !ready_)
        return;
    pda_start(im->g, im->live);
}

bool Gbnf::accept_byte(unsigned char b) {
    Impl *im = impl_of(this);
    if (!im || !ready_)
        return false;
    return pda_accept(im->g, im->live, b);
}

bool Gbnf::accept_bytes(const std::string &s) {
    if (!ready_)
        return false;
    for (char ch : s) {
        if (!accept_byte(static_cast<unsigned char>(ch)))
            return false;
    }
    return true;
}

void Gbnf::allow_mask(const std::function<std::string(int)> &decode, uint8_t *ok, int vocab) const {
    if (!ok || vocab <= 0)
        return;
    if (!ready_) {
        fill_bytes(ok, vocab, 1);
        return;
    }
    const Impl *im = impl_of(this);
    if (!im || im->live.dead || !decode) {
        fill_bytes(ok, vocab, 0);
        return;
    }
    ByteSet first;
    for (int i = 0; i < im->live.n; ++i) {
        const Stack &k = im->live.st[i];
        if (k.n == 0)
            continue;
        const Frame &t = k.f[k.n - 1];
        const Sym &sy = im->g.rules[static_cast<size_t>(t.r)]
                            .alts[static_cast<size_t>(t.a)]
                            .syms[static_cast<size_t>(t.s)];
        first.merge(sy.cls);
    }
    for (int id = 0; id < vocab; ++id) {
        const std::string text = decode(id);
        if (text.empty()) {
            ok[id] = 0;
            continue;
        }
        const auto b0 = static_cast<unsigned char>(text[0]);
        if (!first.has(b0)) {
            ok[id] = 0;
            continue;
        }
        Pda cp{};
        cp.n = im->live.n;
        cp.dead = im->live.dead;
        if (cp.n > 0)
            std::memcpy(cp.st, im->live.st, sizeof(Stack) * static_cast<size_t>(cp.n));
        bool good = true;
        for (char ch : text) {
            if (!pda_accept(im->g, cp, static_cast<unsigned char>(ch))) {
                good = false;
                break;
            }
        }
        ok[id] = good ? 1 : 0;
    }
}

} // namespace mvllm
