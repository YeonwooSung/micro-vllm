#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mvllm {

// Official mux sideband + telemetry lines (docs/serve_protocol.md).
// Pure string builders; counted payloads may contain NULs and newlines.

std::string mux_hex_encode(const void *data, size_t n);

// TOOL <id> <nbytes>\n<payload>\n  — zero-byte is TOOL <id> 0\n\n
std::string mux_format_tool(uint64_t id, const void *data, size_t n);
// DATA <id> <nbytes>\n<payload>\n
std::string mux_format_data(uint64_t id, const void *data, size_t n);
// TOPK <id> <k> <logprob> <hextext> ... ×k\n  (hextext = lowercase UTF-8 hex)
std::string mux_format_topk(uint64_t id, const float *logprobs, const std::string *texts, int k);

// HWINFO cores ram_total_gb ram_avail_gb ngpu vram_total_gb cpu|gpu
// '|' inside cpu/gpu is replaced with space so the separator stays unique.
std::string mux_format_hwinfo(int cores, double ram_total_gb, double ram_avail_gb, int ngpu,
                              double vram_total_gb, const std::string &cpu, const std::string &gpu);
// TIERS vram ram disk vram_gb ram_gb
std::string mux_format_tiers(int vram, int ram, int disk, double vram_gb, double ram_gb);

// byte = (tier<<6) | heat; tier 0 disk / 1 RAM / 2 VRAM (0..3);
// heat = log2-bucket of usage (0..63).
uint8_t mux_emap_byte(int tier, uint32_t usage);
void mux_emap_fill(uint8_t *out, const int *tier, const uint32_t *usage, int n);
// EMAP rows cols hex  — 2 lowercase hex digits per expert, row-major.
std::string mux_format_emap(int rows, int cols, const uint8_t *bytes);
// HITS rows cols hex  — 1 bit per expert, bm[bit>>3] |= 1<<(bit&7).
std::string mux_format_hits(int rows, int cols, const uint8_t *hit);

// PERF id dt t_edisk t_ewait t_emm t_attn t_kvb t_head  (seconds)
std::string mux_format_perf(uint64_t id, double dt, double t_edisk, double t_ewait, double t_emm,
                            double t_attn, double t_kvb, double t_head);
// ENTROPY h0 h1 …  — per-sparse-layer routing entropy (bits)
std::string mux_format_entropy(const float *h, int n);

// Concatenate official turn telemetry lines (each formatter already has \n).
// hwinfo may be empty (skip). entropy n<=0 skips ENTROPY.
// emap/hits: rows<=0 or null bytes skips that line.
// PERF always emitted (zeros ok). TIERS always emitted.
std::string mux_format_turn_telem(const std::string &hwinfo_line, uint64_t id, double dt,
                                  double t_edisk, double t_ewait, double t_emm, double t_attn,
                                  double t_kvb, double t_head, const float *entropy, int n_entropy,
                                  int vram, int ram, int disk, double vram_gb, double ram_gb,
                                  int emap_rows, int emap_cols, const uint8_t *emap,
                                  int hits_rows, int hits_cols, const uint8_t *hits);

} // namespace mvllm
