#pragma once

#include "types.hpp"

#include <string>
#include <vector>

namespace mvllm {

struct KdaConfig {
    int heads = 0;
    int head_dim = 0;
    int conv_k = 4;
    float gate_lower_bound = -5.f;
    bool full_rank_gate = true; // K3 true, GLM53 low-rank
};

struct MoeConfig {
    int n_experts = 0;
    int topk = 0;
    int intermediate = 0;
    int latent = 0; // 0 = no latent projection (GLM53)
    int n_shared = 0;
    float routed_scale = 1.f;
    float situ_b1 = 4.f;
    float situ_b2 = 25.f;
    float swiglu_limit = 0.f; // 0 = plain SiLU; GLM53 sets a ceiling
};

struct MlaConfig {
    int n_heads = 0;
    int q_lora = 0;
    int kv_lora = 0;
    int qk_nope = 0;
    int qk_rope = 0;
    int v_head = 0;
    bool nope = true;
    bool output_gate = true;
    float rope_theta = 10000.f;
};

struct AttnResConfig {
    int block_size = 0; // 0 = disabled
};

struct MhcConfig {
    int mult = 0;
    int iters = 20;
    float eps = 1e-6f;
};

struct DsaConfig {
    int topk = 0;
    int n_heads = 0;
    int head_dim = 0;
    int kpool = 4;
    bool always_select_tail = true;
};

struct VisionConfig {
    int layers = 0;
    int hidden = 0;
    int heads = 0;
    int intermediate = 0;
    int patch = 14;
    int image_size = 448;
    int merge = 2;
    int out_hidden = 0;
};

struct H3Config {
    int dit_layers = 50;
    int default_width = 864;
    int default_height = 480;
    int default_frames = 56;
    int default_steps = 20;
    int stream_slots = 2;
    int hidden = 5376;
    int inner = 7168; // 56 heads * 128
    int ffn = 14336;
    int head_dim = 128;
    int vae_latent_ch = 24;
    int vae_spatial = 16;
    int vae_hidden = 2048;
    int vae_layers = 36;
    int vae_heads = 32;
    int vae_head_dim = 64;
};

struct ModelConfig {
    Family family = Family::Unknown;
    std::string model_type;
    std::string architecture;
    int hidden = 0;
    int n_layers = 0;
    int vocab = 0;
    int first_dense = 0;
    int dense_intermediate = 0;
    int max_position = 0;
    float rms_eps = 1e-5f;
    float rope_theta = 10000.f;
    int bos = 0;
    int eos = 0;
    std::vector<int> eos_ids;
    std::vector<uint8_t> is_kda;  // 1 = KDA layer, 0 = full attn
    std::vector<uint8_t> is_full; // GLM53: 1 = MLA/DSA
    KdaConfig kda;
    MoeConfig moe;
    MlaConfig mla;
    AttnResConfig attn_res;
    MhcConfig mhc;
    DsaConfig dsa;
    VisionConfig vision;
    H3Config h3;
    // Llama / GQA
    int n_q_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
};

struct RuntimeConfig {
    Device device = Device::Cpu;
    int dense_bits = 4;     // 4 / 8 / 32
    int mla_bits = 8;
    int head_bits = 8;
    double expert_gb = 8.0;
    int loader_threads = 4;
    bool pipe = true;
    bool o_direct = true;
    bool idot = true;
    int prefill_chunk = 32;
    int max_seq = 4096;
    int page_size = 16;
    int kv_gb = 2;
    float temperature = 0.f; // 0 = greedy
    float top_p = 1.f;
    int host_port = 8000;
    std::string host = "127.0.0.1";
    std::string model_dir;
};

inline bool is_stop_token(int id, const ModelConfig &cfg, int extra_eos = -1) {
    if (extra_eos >= 0 && id == extra_eos)
        return true;
    if (cfg.eos && id == cfg.eos)
        return true;
    for (int e : cfg.eos_ids)
        if (e == id)
            return true;
    return false;
}

Status load_model_config(const std::string &model_dir, ModelConfig &out, std::string &err);
Family sniff_family(const std::string &model_dir, std::string *model_type = nullptr);
void apply_family_defaults(ModelConfig &cfg);
RuntimeConfig runtime_from_env();

} // namespace mvllm
