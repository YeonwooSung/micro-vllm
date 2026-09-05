#pragma once

#include "../core/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {

uint32_t route_usage_hash(const std::string &engine);

// Sparse .coli_usage persist (K3 / GLM). Instance-owned; no process-global state.
class RouteUsage {
public:
    Status init(const std::string &engine, int n_layers, int n_experts, std::string &err);
    void drop_row(int layer); // dense layer: no experts
    void count(int layer, const int *ids, int k);
    uint32_t get(int layer, int expert) const;
    bool has_row(int layer) const { return row(layer) != nullptr; }
    int n_layers() const { return n_layers_; } // inclusive
    int n_experts() const { return n_experts_; }

    // Load into counters (additive). Returns accepted record count, or -1 on refuse.
    int64_t load(const std::string &path, bool trusted, std::string &err);
    // Atomic save. All-zero → empty file. Returns false on I/O failure.
    bool save(const std::string &path, std::string &err) const;

    // Optional decay: multiply nonzero counts by factor in (0,1], round +0.5, keep 1 alive.
    void decay(double factor);

private:
    uint32_t *row(int layer);
    const uint32_t *row(int layer) const;
    bool admit(int layer, int expert) const;

    std::string engine_;
    uint32_t engine_id_ = 0;
    int n_layers_ = -1;
    int n_experts_ = 0;
    std::vector<std::vector<uint32_t>> counts_;
};

} // namespace mvllm
