#include "mux_frames.hpp"
#include "hwinfo.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace mvllm {
namespace {

constexpr char kHex[] = "0123456789abcdef";

void replace_bar(std::string &s) {
    for (char &c : s)
        if (c == '|')
            c = ' ';
}

void append_f(std::string &out, double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    out.append(buf);
}

std::string format_counted(const char *kind, uint64_t id, const void *data, size_t n) {
    if (!data)
        n = 0;
    std::string out;
    out.reserve(16 + n + 32);
    out.append(kind);
    out.push_back(' ');
    out.append(std::to_string(id));
    out.push_back(' ');
    out.append(std::to_string(n));
    out.push_back('\n');
    if (n)
        out.append(static_cast<const char *>(data), n);
    out.push_back('\n');
    return out;
}

size_t dim_count(int rows, int cols) {
    if (rows <= 0 || cols <= 0)
        return 0;
    return static_cast<size_t>(rows) * static_cast<size_t>(cols);
}

} // namespace

std::string mux_hex_encode(const void *data, size_t n) {
    if (!data || n == 0)
        return {};
    const auto *p = static_cast<const unsigned char *>(data);
    std::string out(n * 2, '\0');
    for (size_t i = 0; i < n; ++i) {
        out[i * 2] = kHex[p[i] >> 4];
        out[i * 2 + 1] = kHex[p[i] & 15];
    }
    return out;
}

std::string mux_format_tool(uint64_t id, const void *data, size_t n) {
    return format_counted("TOOL", id, data, n);
}

std::string mux_format_data(uint64_t id, const void *data, size_t n) {
    return format_counted("DATA", id, data, n);
}

std::string mux_format_accept(uint64_t id, int prompt_tokens) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "ACCEPT %llu %d\n", static_cast<unsigned long long>(id),
                  prompt_tokens);
    return buf;
}

std::string mux_format_error(uint64_t id, const char *code) {
    if (!code || !code[0])
        code = "BAD_FRAME";
    char buf[96];
    std::snprintf(buf, sizeof(buf), "ERROR %llu %s\n", static_cast<unsigned long long>(id), code);
    return buf;
}

std::string mux_format_topk(uint64_t id, const float *logprobs, const std::string *texts, int k) {
    if (k < 0)
        k = 0;
    std::string out = "TOPK ";
    out.append(std::to_string(id));
    out.push_back(' ');
    out.append(std::to_string(k));
    for (int i = 0; i < k; ++i) {
        out.push_back(' ');
        append_f(out, logprobs ? static_cast<double>(logprobs[i]) : 0.0);
        out.push_back(' ');
        if (texts)
            out.append(mux_hex_encode(texts[i].data(), texts[i].size()));
    }
    out.push_back('\n');
    return out;
}

std::string mux_format_hwinfo(int cores, double ram_total_gb, double ram_avail_gb, int ngpu,
                              double vram_total_gb, const std::string &cpu, const std::string &gpu) {
    std::string cpu_s = cpu;
    std::string gpu_s = gpu;
    replace_bar(cpu_s);
    replace_bar(gpu_s);
    char head[96];
    std::snprintf(head, sizeof(head), "HWINFO %d %.2f %.2f %d %.2f ", cores, ram_total_gb,
                  ram_avail_gb, ngpu, vram_total_gb);
    std::string out = head;
    out.append(cpu_s);
    out.push_back('|');
    out.append(gpu_s);
    out.push_back('\n');
    return out;
}

std::string mux_format_tiers(int vram, int ram, int disk, double vram_gb, double ram_gb) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "TIERS %d %d %d %.2f %.2f\n", vram, ram, disk, vram_gb, ram_gb);
    return buf;
}

uint8_t mux_emap_byte(int tier, uint32_t usage) {
    if (tier < 0)
        tier = 0;
    if (tier > 3)
        tier = 3;
    int heat = 0;
    uint32_t u = usage;
    while (u) {
        ++heat;
        u >>= 1;
    }
    if (heat > 63)
        heat = 63;
    return static_cast<uint8_t>((tier << 6) | heat);
}

void mux_emap_fill(uint8_t *out, const int *tier, const uint32_t *usage, int n) {
    if (!out || n <= 0)
        return;
    for (int i = 0; i < n; ++i) {
        const int t = tier ? tier[i] : 0;
        const uint32_t u = usage ? usage[i] : 0u;
        out[i] = mux_emap_byte(t, u);
    }
}

std::string mux_format_emap(int rows, int cols, const uint8_t *bytes) {
    if (rows < 0)
        rows = 0;
    if (cols < 0)
        cols = 0;
    const size_t n = dim_count(rows, cols);
    std::string hex;
    if (n) {
        if (bytes)
            hex = mux_hex_encode(bytes, n);
        else
            hex.assign(n * 2, '0');
    }
    std::string out = "EMAP ";
    out.append(std::to_string(rows));
    out.push_back(' ');
    out.append(std::to_string(cols));
    out.push_back(' ');
    out.append(hex);
    out.push_back('\n');
    return out;
}

void mux_hits_pack(int rows, int cols, const uint8_t *hit, std::vector<uint8_t> &bm) {
    const size_t bits = dim_count(rows, cols);
    const size_t nb = (bits + 7) / 8;
    bm.assign(nb, 0);
    if (!hit)
        return;
    for (size_t bit = 0; bit < bits; ++bit) {
        if (hit[bit])
            bm[bit >> 3] = static_cast<uint8_t>(bm[bit >> 3] | (1u << (bit & 7)));
    }
}

std::string mux_hits_hex(int rows, int cols, const uint8_t *hit) {
    std::vector<uint8_t> bm;
    mux_hits_pack(rows, cols, hit, bm);
    return mux_hex_encode(bm.data(), bm.size());
}

std::string mux_format_hits(int rows, int cols, const uint8_t *hit) {
    if (rows < 0)
        rows = 0;
    if (cols < 0)
        cols = 0;
    std::vector<uint8_t> bm;
    mux_hits_pack(rows, cols, hit, bm);
    std::string out = "HITS ";
    out.append(std::to_string(rows));
    out.push_back(' ');
    out.append(std::to_string(cols));
    out.push_back(' ');
    out.append(mux_hex_encode(bm.data(), bm.size()));
    out.push_back('\n');
    return out;
}

std::string mux_format_experts_json(int rows, int cols, const uint8_t *emap, const uint8_t *hits,
                                    int seq, const float *entropy, int n_entropy, int64_t created) {
    if (rows < 0)
        rows = 0;
    if (cols < 0)
        cols = 0;
    const size_t n = dim_count(rows, cols);
    std::string map;
    if (emap && n)
        map = mux_hex_encode(emap, n);
    std::string hits_hex;
    if (hits)
        hits_hex = mux_hits_hex(rows, cols, hits);
    std::string out = "{\"rows\":";
    out.append(std::to_string(rows));
    out.append(",\"cols\":");
    out.append(std::to_string(cols));
    out.append(",\"map\":\"");
    out.append(map);
    out.append("\",\"hits\":\"");
    out.append(hits_hex);
    out.append("\",\"seq\":");
    out.append(std::to_string(seq));
    if (entropy && n_entropy > 0) {
        out.append(",\"entropy\":[");
        for (int i = 0; i < n_entropy; ++i) {
            if (i)
                out.push_back(',');
            append_f(out, static_cast<double>(entropy[i]));
        }
        out.push_back(']');
    }
    if (created >= 0) {
        out.append(",\"created\":");
        out.append(std::to_string(created));
    }
    out.push_back('}');
    return out;
}

std::string mux_format_perf(uint64_t id, double dt, double t_edisk, double t_ewait, double t_emm,
                            double t_attn, double t_kvb, double t_head) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "PERF %llu %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n",
                  static_cast<unsigned long long>(id), dt, t_edisk, t_ewait, t_emm, t_attn, t_kvb,
                  t_head);
    return buf;
}

std::string mux_format_entropy(const float *h, int n) {
    if (n < 0)
        n = 0;
    std::string out = "ENTROPY";
    for (int i = 0; i < n; ++i) {
        out.push_back(' ');
        append_f(out, h ? static_cast<double>(h[i]) : 0.0);
    }
    out.push_back('\n');
    return out;
}

std::string mux_format_gpus(int n, const double *used_gb, const double *total_gb,
                            const int *experts) {
    if (n <= 0 || !used_gb || !total_gb || !experts)
        return "GPUS 0\n";
    std::string out = "GPUS ";
    out.append(std::to_string(n));
    for (int i = 0; i < n; ++i) {
        char buf[80];
        std::snprintf(buf, sizeof(buf), " %.2f %.2f %d", used_gb[i], total_gb[i], experts[i]);
        out.append(buf);
    }
    out.push_back('\n');
    return out;
}

std::string mux_format_repin(int layer, int eid, int old_tier, int gpu) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "REPIN %d %d %d %d\n", layer, eid, old_tier, gpu);
    return buf;
}

std::string mux_format_prof(double wall_s, int prompt_tokens, int completion_tokens,
                            double t_edisk, double t_ewait, double t_emm, double t_attn,
                            double t_head, uint64_t n_fw) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\n", wall_s,
                  prompt_tokens, completion_tokens, t_edisk, t_ewait, t_emm, t_attn, t_head,
                  static_cast<unsigned long long>(n_fw));
    return buf;
}

std::string mux_format_turn_telem(const std::string &hwinfo_line, uint64_t id, double dt,
                                  double t_edisk, double t_ewait, double t_emm, double t_attn,
                                  double t_kvb, double t_head, const float *entropy, int n_entropy,
                                  int vram, int ram, int disk, double vram_gb, double ram_gb,
                                  int emap_rows, int emap_cols, const uint8_t *emap,
                                  int hits_rows, int hits_cols, const uint8_t *hits) {
    std::string out;
    if (!hwinfo_line.empty()) {
        out.append(hwinfo_line);
        if (hwinfo_line.back() != '\n')
            out.push_back('\n');
    }
    out.append(mux_format_perf(id, dt, t_edisk, t_ewait, t_emm, t_attn, t_kvb, t_head));
    if (n_entropy > 0)
        out.append(mux_format_entropy(entropy, n_entropy));
    {
        const HwInfo h = hw_probe();
        if (h.ngpu > 0) {
            const int n = h.ngpu;
            std::vector<double> used(static_cast<size_t>(n), 0.0);
            std::vector<double> total(static_cast<size_t>(n), h.vram_total_gb);
            std::vector<int> experts(static_cast<size_t>(n), 0);
            out.append(mux_format_gpus(n, used.data(), total.data(), experts.data()));
        } else {
            out.append(mux_format_gpus(0, nullptr, nullptr, nullptr));
        }
    }
    out.append(mux_format_tiers(vram, ram, disk, vram_gb, ram_gb));
    if (emap_rows > 0 && emap)
        out.append(mux_format_emap(emap_rows, emap_cols, emap));
    if (hits_rows > 0 && hits)
        out.append(mux_format_hits(hits_rows, hits_cols, hits));
    return out;
}

} // namespace mvllm
