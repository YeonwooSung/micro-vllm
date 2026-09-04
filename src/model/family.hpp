#pragma once

#include "../core/config.hpp"
#include "../quant/weight.hpp"
#include "../store/block_store.hpp"
#include "../store/expert_store.hpp"
#include "../tok/tokenizer.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mvllm {

struct GenParams {
    int max_new_tokens = 32;
    float temperature = 0.f;
    float top_p = 1.f;
    int eos = -1;
    bool apply_template = true;
    bool think = false;
    std::string reasoning_effort;
};

struct GenResult {
    std::vector<int> tokens;
    std::string text;
    int prompt_tokens = 0;
    int completion_tokens = 0;
};

struct H3GenParams {
    int width = 864;
    int height = 480;
    int frames = 56;
    int steps = 20;
    int dit_layers = 50;
    int denoise_reuse = 1;
    bool ssd_streaming = true;
    uint64_t seed = 42;
    std::string prompt;
    std::string output_path;
};

struct H3GenResult {
    std::string output_path;
    int blocks_streamed = 0;
    int steps_run = 0;
    int frames = 0;
    int width = 0;
    int height = 0;
    bool vae_used = false;
    std::string note;
};

class FamilyEngine {
public:
    virtual ~FamilyEngine() = default;
    virtual Family family() const = 0;
    virtual const ModelConfig &config() const = 0;
    virtual Status load(const std::string &model_dir, const RuntimeConfig &rt,
                        std::string &err) = 0;
    virtual Status generate(const std::vector<int> &prompt, const GenParams &gp, GenResult &out,
                            std::string &err) = 0;
    virtual Status generate_video(const H3GenParams &, H3GenResult &, std::string &err) {
        err = "this family is not a video model";
        return Status::Unsupported;
    }
    virtual std::string describe() const = 0;
    virtual void expert_stats(ExpertStoreStats &out) const { out = {}; }
    virtual uint64_t block_hits() const { return 0; }
    virtual uint64_t block_misses() const { return 0; }
};

std::unique_ptr<FamilyEngine> make_engine(Family family);

// CPU ops shared by K3 / GLM53.
// out_norm is the official shared GDN scale [head_dim] (not packed [heads, head_dim]).
// dt_n is the length of dt_bias: P, n_heads (broadcast), or 0.
void kda_step(const float *x, int hidden, const KdaConfig &kda,
              const quant::QuantMat *w_q, const quant::QuantMat *w_k, const quant::QuantMat *w_v,
              const quant::QuantMat *w_b, const quant::QuantMat *w_fa, const quant::QuantMat *w_fb,
              const float *dt_bias, int dt_n, const float *a_log, const quant::QuantMat *w_g,
              const quant::QuantMat *w_o, const float *out_norm, float *S, float *y, float eps,
              const float *conv_q = nullptr, const float *conv_k = nullptr,
              const float *conv_v = nullptr, float *win_q = nullptr, float *win_k = nullptr,
              float *win_v = nullptr);

// Causal depthwise conv: shift window, write x[p] into the last tap, then x[p] <- taps · window.
void kda_short_conv(float *x, const float *taps, float *window, int channels, int k);

// Absorbed NoPE MLA. Cache stride is kv_lora+qk_rope (L then raw R).
// score_j = (W_k^T q_nope) · c_j + q_rot · R_j ;  out = W_v (Σ a_j c_j).
// w_kt is [n_heads * kv_lora, qk_nope], w_v is [n_heads * v_head, kv_lora].
// w_g is optional (K3 output gate). Missing absorbed KV (kva/kt/v/cache) falls back
// to a dense Q/O stand-in.
void mla_step(const float *x, int hidden, const MlaConfig &mla, const quant::QuantMat *w_qa,
              const float *qa_ln, const quant::QuantMat *w_qb, const quant::QuantMat *w_kva,
              const float *kva_ln, const quant::QuantMat *w_kt, const quant::QuantMat *w_v,
              const quant::QuantMat *w_o, const quant::QuantMat *w_g, float *cache, int pos,
              float *y, float eps, const int *selected = nullptr, int n_sel = 0);

int dsa_index_width(const DsaConfig &dsa);
// Score k-pool groups with ReLU(q·k), take top-k pools, optional tail. Unused slots = -1.
int dsa_select(int *out, const float *queries, const float *keys, const float *gates,
               const float *head_w, const float *ape, int seq, const DsaConfig &dsa);

void layernorm(const float *x, const float *w, const float *b, float *y, int n, float eps);

// Fold kv_b_proj [H*(qk_nope+v_head), kv_lora] into W_k^T and W_v.
void mla_absorb_kvb(const float *kv_b, int n_heads, int qk_nope, int v_head, int kv_lora,
                    quant::QuantMat &w_kt, quant::QuantMat &w_v, int bits);

void bf16_to_f32(const uint16_t *src, float *dst, int64_t n);

// One DiT residual block on streamed BF16 qkv/out/fc1/fc2 (row-major [O,I]).
void h3_dit_block_cpu(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                      int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                      int tokens, float eps);

void attnres_mix(const std::vector<std::vector<float>> &snapshots, const float *prefix,
                 const float *res_norm, const float *res_proj, float *hidden, int n, float eps);

void mhc_mix(float *streams, int hidden, int mult, const float *alpha, int iters, float eps);

// Top-k on `choice`. Mix weights come from `mix` (official: unbiased σ) or
// from `choice` when mix is null. Selected weights are clipped at 0 and L1-normalized.
int moe_topk(const float *choice, int n, int k, int *idx, float *w, const float *mix = nullptr);

// Unique expert ids from C tokens × topk, sorted. Returns count.
int moe_union_ids(const int *idx, int n_tok, int topk, int *out, int out_cap);

} // namespace mvllm
