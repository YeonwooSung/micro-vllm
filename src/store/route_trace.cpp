#include "route_trace.hpp"

#include <cstdlib>

namespace mvllm {
namespace {

void skip_ws(const char *&p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')
        ++p;
}

bool at_token_end(char c) {
    return c == '\0' || c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

void append_fmt(std::string &out, const char *fmt, int a, int b, int c) {
    char buf[96];
    const int n = std::snprintf(buf, sizeof(buf), fmt, a, b, c);
    if (n > 0)
        out.append(buf, static_cast<size_t>(n < static_cast<int>(sizeof(buf)) ? n
                                                                             : static_cast<int>(sizeof(buf) - 1)));
}

void append_pair(std::string &out, int id, float gate) {
    char buf[96];
    const int n = std::snprintf(buf, sizeof(buf), " %d:%.4f", id, gate);
    if (n > 0)
        out.append(buf, static_cast<size_t>(n < static_cast<int>(sizeof(buf)) ? n
                                                                             : static_cast<int>(sizeof(buf) - 1)));
}

} // namespace

std::string route_trace_format(int call, int row, int layer, const int *ids, const float *gates,
                               int k) {
    const int n = (ids && gates && k > 0) ? k : 0;
    std::string out;
    out.reserve(static_cast<size_t>(32 + n * 24));
    append_fmt(out, "%d %d %d", call, row, layer);
    for (int i = 0; i < n; ++i)
        append_pair(out, ids[i], gates[i]);
    out.push_back('\n');
    return out;
}

bool route_trace_parse(const std::string &line, RouteTraceEvent &out) {
    const char *p = line.c_str();
    char *end = nullptr;

    skip_ws(p);
    const long call = std::strtol(p, &end, 10);
    if (end == p)
        return false;
    p = end;

    const long row = std::strtol(p, &end, 10);
    if (end == p)
        return false;
    p = end;

    const long layer = std::strtol(p, &end, 10);
    if (end == p)
        return false;
    p = end;
    if (!at_token_end(*p))
        return false;

    std::vector<int> ids;
    std::vector<float> gates;
    for (;;) {
        skip_ws(p);
        if (*p == '\0')
            break;

        const long id = std::strtol(p, &end, 10);
        if (end == p || *end != ':')
            return false;
        p = end + 1;

        const float gate = std::strtof(p, &end);
        if (end == p)
            return false;
        p = end;
        if (!at_token_end(*p))
            return false;

        ids.push_back(static_cast<int>(id));
        gates.push_back(gate);
    }

    out.call = static_cast<int>(call);
    out.row = static_cast<int>(row);
    out.layer = static_cast<int>(layer);
    out.ids = std::move(ids);
    out.gates = std::move(gates);
    return true;
}

RouteTrace::~RouteTrace() { close(); }

void RouteTrace::close() {
    if (fp_ && owned_)
        std::fclose(fp_);
    fp_ = nullptr;
    sink_ = nullptr;
    owned_ = false;
}

bool RouteTrace::is_open() const { return fp_ != nullptr || sink_ != nullptr; }

int RouteTrace::call() const { return call_; }

bool RouteTrace::open(const std::string &path, std::string &err) {
    close();
    if (path.empty()) {
        err = "empty route trace path";
        return false;
    }
    fp_ = std::fopen(path.c_str(), "w");
    if (!fp_) {
        err = "cannot open route trace: " + path;
        return false;
    }
    owned_ = true;
    call_ = 0;
    err.clear();
    return true;
}

bool RouteTrace::attach(std::FILE *fp, std::string &err) {
    close();
    if (!fp) {
        err = "null route trace FILE";
        return false;
    }
    fp_ = fp;
    owned_ = false;
    call_ = 0;
    err.clear();
    return true;
}

bool RouteTrace::attach(std::string *sink, std::string &err) {
    close();
    if (!sink) {
        err = "null route trace sink";
        return false;
    }
    sink_ = sink;
    call_ = 0;
    err.clear();
    return true;
}

bool RouteTrace::write_line(const std::string &line, std::string &err) {
    if (sink_) {
        sink_->append(line);
        err.clear();
        return true;
    }
    if (!fp_) {
        err.clear();
        return true;
    }
    if (std::fwrite(line.data(), 1, line.size(), fp_) != line.size()) {
        err = "route trace write failed";
        return false;
    }
    err.clear();
    return true;
}

bool RouteTrace::emit(int row, int layer, const int *ids, const float *gates, int k,
                      std::string &err) {
    if (!is_open())
        return true;
    if (k < 1 || !ids || !gates)
        return true;
    return write_line(route_trace_format(call_, row, layer, ids, gates, k), err);
}

void RouteTrace::end() {
    if (is_open())
        ++call_;
}

} // namespace mvllm
