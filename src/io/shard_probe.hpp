#pragma once

#include "../core/config.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {

// Header scan of a checkpoint dir. When expert / DiT tensors exist, also
// preads one payload (one expert slot or one small H3 vector). Does not
// run Engine::load or a forward.
struct H3ComponentReport {
    std::string name;
    std::string path;
    int n_files = 0;
    int n_tensors = 0;
    int64_t bytes = 0;
    bool present = false;
};

struct ShardReport {
    Family family = Family::Unknown;
    std::string prefix;
    std::string root;
    int n_files = 0;
    int n_tensors = 0;
    int n_layers_seen = 0;
    int n_experts_seen = 0;
    int64_t slot_bytes = 0;
    int slots_per_layer = 0;
    int64_t cache_bytes = 0;
    bool prefix_ok = false;
    bool shards_ok = false;
    bool synth = false;
    bool payload_ok = false;
    int64_t payload_bytes = 0;
    std::vector<H3ComponentReport> components;
    std::string note;
};

Status probe_shards(const std::string &model_dir, const RuntimeConfig &rt, ShardReport &out,
                    std::string &err);
std::string format_shard_report(const ShardReport &r);
std::string format_shard_report_json(const ShardReport &r);

int64_t k3_expert_slot_bytes(int latent, int inter);
int64_t glm53_expert_slot_bytes(int hidden, int inter);

} // namespace mvllm
