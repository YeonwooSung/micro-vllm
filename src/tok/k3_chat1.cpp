#include "k3_chat1.hpp"

#include <cctype>
#include <cstring>
#include <limits>

namespace mvllm {
namespace {

bool fail(std::string &err, const char *msg) {
    err = msg;
    return false;
}

const char *find_nl(const char *p, const char *end) {
    for (const char *q = p; q < end; ++q) {
        if (*q == '\n')
            return q;
    }
    return nullptr;
}

bool skip_sp(const char *&p, const char *lim) {
    while (p < lim && (*p == ' ' || *p == '\t'))
        ++p;
    return p < lim;
}

bool parse_int(const char *&p, const char *lim, int &out) {
    if (!skip_sp(p, lim))
        return false;
    bool neg = false;
    if (*p == '-') {
        neg = true;
        ++p;
    }
    if (p >= lim || *p < '0' || *p > '9')
        return false;
    long long v = 0;
    while (p < lim && *p >= '0' && *p <= '9') {
        v = v * 10 + (*p - '0');
        if (v > static_cast<long long>(std::numeric_limits<int>::max()) + (neg ? 1LL : 0LL))
            return false;
        ++p;
    }
    if (neg)
        v = -v;
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
        return false;
    out = static_cast<int>(v);
    return true;
}

bool payload_fits(const char *nl, const char *end, int a, int b = 0, int c = 0) {
    if (a < 0 || b < 0 || c < 0 || !nl)
        return false;
    const char *start = nl + 1;
    if (start > end)
        return false;
    const size_t avail = static_cast<size_t>(end - start);
    const size_t need = static_cast<size_t>(a) + static_cast<size_t>(b) + static_cast<size_t>(c);
    return need <= avail;
}

std::string take(const char *p, int n) {
    if (n <= 0)
        return {};
    return std::string(p, static_cast<size_t>(n));
}

const char *advance(const char *nl, int a, int b = 0, int c = 0) {
    return nl + 1 + static_cast<size_t>(a) + static_cast<size_t>(b) + static_cast<size_t>(c);
}

void parse_tool_declare_json(const std::string &body, std::vector<K3ToolDecl> &tools) {
    std::string json_text;
    const size_t fence = body.find("```json");
    if (fence != std::string::npos) {
        size_t start = fence + 7;
        if (start < body.size() && body[start] == '\r')
            ++start;
        if (start < body.size() && body[start] == '\n')
            ++start;
        const size_t close = body.find("```", start);
        json_text = body.substr(start, close == std::string::npos ? std::string::npos : close - start);
    } else {
        size_t i = 0;
        while (i < body.size() && std::isspace(static_cast<unsigned char>(body[i])))
            ++i;
        if (i < body.size() && (body[i] == '[' || body[i] == '{'))
            json_text = body.substr(i);
    }
    if (json_text.empty())
        return;
    size_t i = 0;
    while (i < json_text.size() && std::isspace(static_cast<unsigned char>(json_text[i])))
        ++i;
    std::string wrapped = json_text;
    if (i < json_text.size() && json_text[i] == '[')
        wrapped = std::string("{\"tools\":") + json_text + "}";
    std::vector<K3ToolDecl> parsed;
    if (k3_extract_tools_json(wrapped, parsed) && !parsed.empty())
        tools.insert(tools.end(), parsed.begin(), parsed.end());
}

} // namespace

bool k3_chat1_parse(const char *wire, int n, K3Chat1 &out, std::string &err) {
    out = K3Chat1{};
    if (!wire || n < 8 || std::memcmp(wire, "K3CHAT1\n", 8) != 0)
        return fail(err, "not a K3CHAT1 payload");

    const char *p = wire + 8;
    const char *const end = wire + n;
    K3Chat1 acc;
    bool saw_g = false;

    while (p < end) {
        const char *nl = find_nl(p, end);
        if (!nl)
            return fail(err, "missing newline");

        if (*p == 'G') {
            const char *q = p + 1;
            int v = 0;
            if (!parse_int(q, nl, v))
                return fail(err, "bad G record");
            acc.think = v != 0;
            saw_g = true;
            break;
        }

        if (*p == 'A') {
            const char *q = p + 1;
            int nr = -1, nt = -1;
            if (!parse_int(q, nl, nr) || !parse_int(q, nl, nt) || !payload_fits(nl, end, nr, nt))
                return fail(err, "overflow");
            ChatMessage m;
            m.role = "assistant";
            m.reasoning = take(nl + 1, nr);
            m.content = take(nl + 1 + nr, nt);
            acc.msgs.push_back(std::move(m));
            p = advance(nl, nr, nt);
            continue;
        }

        if (*p == 'Y') {
            const char *q = p + 1;
            int ntp = -1, nb = -1;
            if (!parse_int(q, nl, ntp) || !parse_int(q, nl, nb) || ntp < 1 || ntp > 64 ||
                !payload_fits(nl, end, ntp, nb))
                return fail(err, "overflow");
            const std::string typ = take(nl + 1, ntp);
            const std::string body = take(nl + 1 + ntp, nb);
            if (typ == "tool-declare")
                parse_tool_declare_json(body, acc.tools);
            ChatMessage m;
            m.role = "system";
            m.xtml_type = typ;
            m.content = body;
            acc.msgs.push_back(std::move(m));
            p = advance(nl, ntp, nb);
            continue;
        }

        if (*p == 'O') {
            const char *q = p + 1;
            int idx = -1, nn = -1, nb = -1;
            if (!parse_int(q, nl, idx) || !parse_int(q, nl, nn) || !parse_int(q, nl, nb))
                return fail(err, "bad O record");
            if (idx < 1)
                return fail(err, "index<1");
            if (nn < 1 || nn > 256 || !payload_fits(nl, end, nn, nb))
                return fail(err, "overflow");
            ChatMessage m;
            m.role = "tool";
            m.tool_index = idx;
            m.tool_name = take(nl + 1, nn);
            m.content = take(nl + 1 + nn, nb);
            acc.msgs.push_back(std::move(m));
            p = advance(nl, nn, nb);
            continue;
        }

        if (*p == 'B') {
            const char *q = p + 1;
            int th = -1, nr = -1, nt = -1, nc = -1;
            if (!parse_int(q, nl, th) || !parse_int(q, nl, nr) || !parse_int(q, nl, nt) ||
                !parse_int(q, nl, nc))
                return fail(err, "bad B record");
            if (nc < 1 || nc > 64)
                return fail(err, "ncalls out of 1..64");
            if (th < 0 || th > 1 || !payload_fits(nl, end, nr, nt))
                return fail(err, "overflow");
            ChatMessage m;
            m.role = "assistant";
            m.reasoning = take(nl + 1, nr);
            m.content = take(nl + 1 + nr, nt);
            p = advance(nl, nr, nt);
            for (int ci = 0; ci < nc; ++ci) {
                nl = find_nl(p, end);
                if (!nl)
                    return fail(err, "missing tool-call header");
                K3ToolCall call;
                call.index = ci + 1;
                if (*p == 'J') {
                    const char *r = p + 1;
                    int nn = -1, nj = -1;
                    if (!parse_int(r, nl, nn) || !parse_int(r, nl, nj) || nn < 1 || nn > 256 ||
                        !payload_fits(nl, end, nn, nj))
                        return fail(err, "overflow");
                    call.name = take(nl + 1, nn);
                    call.json = take(nl + 1 + nn, nj);
                    p = advance(nl, nn, nj);
                } else if (*p == 'F') {
                    const char *r = p + 1;
                    int nn = -1, na = -1;
                    if (!parse_int(r, nl, nn) || !parse_int(r, nl, na) || nn < 1 || nn > 256 ||
                        na < 0 || na > 64 || !payload_fits(nl, end, nn))
                        return fail(err, "overflow");
                    call.name = take(nl + 1, nn);
                    p = advance(nl, nn);
                    for (int ai = 0; ai < na; ++ai) {
                        nl = find_nl(p, end);
                        if (!nl || *p != 'V')
                            return fail(err, "unknown record");
                        const char *s = p + 1;
                        int nk = -1, ntp = -1, nv = -1;
                        if (!parse_int(s, nl, nk) || !parse_int(s, nl, ntp) ||
                            !parse_int(s, nl, nv) || nk < 1 || nk > 256 || ntp < 1 || ntp > 16 ||
                            !payload_fits(nl, end, nk, ntp, nv))
                            return fail(err, "overflow");
                        K3ToolArg arg;
                        arg.key = take(nl + 1, nk);
                        arg.type = take(nl + 1 + nk, ntp);
                        arg.value = take(nl + 1 + nk + ntp, nv);
                        call.args.push_back(std::move(arg));
                        p = advance(nl, nk, ntp, nv);
                    }
                } else {
                    return fail(err, "unknown record");
                }
                m.tool_calls.push_back(std::move(call));
            }
            acc.msgs.push_back(std::move(m));
            continue;
        }

        if (*p == 'M') {
            const char *q = p + 1;
            if (!skip_sp(q, nl))
                return fail(err, "bad M record");
            const char *role_b = q;
            while (q < nl && *q != ' ' && *q != '\t')
                ++q;
            std::string role(role_b, q);
            int nb = -1;
            if (!parse_int(q, nl, nb) || !payload_fits(nl, end, nb))
                return fail(err, "overflow");
            if (role == "developer")
                role = "system";
            if (role != "system" && role != "user" && role != "assistant")
                return fail(err, "unknown record");
            ChatMessage m;
            m.role = std::move(role);
            m.content = take(nl + 1, nb);
            acc.msgs.push_back(std::move(m));
            p = advance(nl, nb);
            continue;
        }

        return fail(err, "unknown record");
    }

    if (!saw_g)
        return fail(err, "missing G");
    err.clear();
    out = std::move(acc);
    return true;
}

} // namespace mvllm
