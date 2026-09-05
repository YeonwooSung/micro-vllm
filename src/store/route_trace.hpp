#pragma once

#include <cstdio>
#include <string>
#include <vector>

namespace mvllm {

struct RouteTraceEvent {
    int call = 0;
    int row = 0;
    int layer = 0;
    std::vector<int> ids;
    std::vector<float> gates;
};

// Official ROUTE_TRACE line; always ends with \n. k = pair count.
std::string route_trace_format(int call, int row, int layer, const int *ids, const float *gates,
                               int k);

// Parse one line. Trailing newline is optional. False on junk.
bool route_trace_parse(const std::string &line, RouteTraceEvent &out);

// Per-call ROUTE_TRACE text stream (not the .coli_usage histogram).
class RouteTrace {
public:
    RouteTrace() = default;
    RouteTrace(const RouteTrace &) = delete;
    RouteTrace &operator=(const RouteTrace &) = delete;
    RouteTrace(RouteTrace &&) = delete;
    RouteTrace &operator=(RouteTrace &&) = delete;
    ~RouteTrace();

    // Open path for write (truncate). Missing dir / I/O → false + err.
    bool open(const std::string &path, std::string &err);
    // Attach a caller-owned FILE. Not closed by this object.
    bool attach(std::FILE *fp, std::string &err);
    // Attach a caller-owned string; emit() appends official lines.
    bool attach(std::string *sink, std::string &err);
    void close();
    bool is_open() const;
    int call() const;

    // Append one row line for the current call. No-op if not open.
    bool emit(int row, int layer, const int *ids, const float *gates, int k, std::string &err);
    // Advance call counter (once per moe invocation).
    void end();

private:
    bool write_line(const std::string &line, std::string &err);

    std::FILE *fp_ = nullptr;
    std::string *sink_ = nullptr;
    bool owned_ = false;
    int call_ = 0;
};

} // namespace mvllm
