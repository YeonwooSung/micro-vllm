#pragma once

#include "../core/config.hpp"
#include "../gpu/metal_ops.hpp"
#include "../quant/weight.hpp"
#include "../store/block_store.hpp"
#include "../store/expert_store.hpp"
#include "../store/kv_persist.hpp"
#include "../tok/tokenizer.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mvllm {

struct GenParams {
    int max_new_tokens = 32;
    float temperature = 0.f;
    float top_p = 1.f;
    uint64_t seed = 1;
    int eos = -1;
    bool apply_template = true;
    bool think = false;
    std::string reasoning_effort;
    const float *image_rgb = nullptr; // HWC [0,1]; GLM ViT when image_token matches
    int image_w = 0;
    int image_h = 0;
    int image_token = -1;
    std::string grammar; // optional GBNF; empty = unconstrained
    std::vector<K3ToolDecl> tools;
    std::function<void(int token)> on_token;          // called after each new token id
    std::function<std::string(int)> token_text;       // decode one id for GBNF / SSE
    int cache_slot = -1; // -1 auto; else 0 … kv_slots-1
    int prefix_reuse = 0; // skip this many prompt tokens (session match)
    std::vector<std::string> stop; // OpenAI stop strings; empty = off
    std::string tool_choice;       // auto|none|required|function name
    float frequency_penalty = 0.f;
    float presence_penalty = 0.f;
    int logprobs = 0;     // 0 = off; >0 request that many top logprobs
    std::vector<std::pair<int, float>> logit_bias; // token id, bias
    int top_k = 0;                 // 0 = off
    float min_p = 0.f;             // 0 = off
    float repetition_penalty = 1.f; // 1 = off
    std::vector<int> stop_ids;     // extra token-id stops (StopSet); empty = config/eos
    bool eos_only = false;         // official SERVE_BATCH: keep only eos
    std::string persist_path;      // optional per-request .coli_kv overlay
    int persist_ver = 0;           // 0 = runtime default; 1/2/3 = COLIKV1/2/3
    int prefix_bytes = 0;          // mux SUBMIT prefix hint; 0 = unused
};

inline bool gen_stop_id(int id, const ModelConfig &cfg, const GenParams &gp) {
    if (is_stop_token(id, cfg, gp.eos))
        return true;
    for (int s : gp.stop_ids)
        if (s == id)
            return true;
    return false;
}

struct GenLogprob {
    int token = 0;
    float logprob = 0.f;
};

struct GenResult {
    std::vector<int> tokens;
    std::string text;
    std::string reasoning;
    bool stopped_by_stop = false;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int reasoning_tokens = 0;
    std::vector<float> token_logprobs;
    std::vector<std::vector<GenLogprob>> top_logprobs;
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
    std::string audio_path;          // optional reference WAV (encode → start latent)
    const float *audio_pcm = nullptr; // channel-major [2, audio_samples], [-1,1]
    int audio_samples = 0;
    // FL2VA first/last-frame anchors. Mutually exclusive with ref_images.
    std::string first_frame;
    std::string last_frame;
    const float *first_rgb = nullptr; // HWC [0,1]
    int first_w = 0;
    int first_h = 0;
    const float *last_rgb = nullptr;
    int last_w = 0;
    int last_h = 0;
    // Ordered Ref2VA image paths (or in-memory first image via ref_rgb).
    // Non-empty selects Ref2VA/transformer when present.
    std::vector<std::string> ref_images;
    const float *ref_rgb = nullptr;
    int ref_w = 0;
    int ref_h = 0;
};

struct H3GenResult {
    std::string output_path;
    std::string audio_path;
    int blocks_streamed = 0;
    int steps_run = 0;
    int frames = 0;
    int width = 0;
    int height = 0;
    bool vae_used = false;
    bool audio_used = false;
    int audio_samples = 0;
    int audio_rate = 32000;
    std::string note;
};

// Mux turn telemetry: EMAP/HITS over sparse MoE rows × experts.
// emap[i] = (tier<<6)|heat; hits[i] is 0/1 per expert (not packed).
// entropy[r] is Shannon bits of this turn's routes on sparse row r.
struct RouteTelem {
    int rows = 0;
    int cols = 0;
    std::vector<uint8_t> emap;
    std::vector<uint8_t> hits;
    std::vector<float> entropy;
    int vram = 0;
    int ram = 0;
    int disk = 0;
    double vram_gb = 0;
    double ram_gb = 0;
};

// Official PERF phases (seconds). t_edisk = expert disk I/O wall;
// t_ewait = compute-thread stall on a miss; t_emm = expert matmul;
// t_attn / t_kvb / t_head = attention / KV bind / lm-head.
struct TurnPerf {
    double t_edisk = 0;
    double t_ewait = 0;
    double t_emm = 0;
    double t_attn = 0;
    double t_kvb = 0;
    double t_head = 0;
};

// Official /profile turn (openai_server.py PROF snapshot). Rolling window of 120.
struct ProfileTurn {
    double wall_s = 0;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    double expert_disk_s = 0;
    double expert_wait_s = 0;
    double expert_matmul_s = 0;
    double attention_s = 0;
    double lm_head_s = 0;
    uint64_t forwards = 0;
};

// Official REPIN layer eid old_tier gpu. old_tier 0 disk / 1 RAM / 2 VRAM.
struct RepinEvent {
    int layer = 0;
    int eid = 0;
    int old_tier = 1;
    int gpu = 0;
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
    // Official mux EMAP/HITS/ENTROPY/TIERS snapshot. consume_hits clears the
    // turn-hit bitmap after copy (hits_emit). Default: empty (no MoE).
    virtual void route_telem(RouteTelem &out, bool consume_hits = true) {
        (void)consume_hits;
        out = {};
    }
    // Snapshot PERF phases. reset clears the accumulators (turn window).
    virtual void turn_perf(TurnPerf &out, bool reset = false) {
        (void)reset;
        out = {};
    }
    // Consume queued REPIN swaps into out[0..cap). Returns count written.
    virtual int take_repin(RepinEvent *out, int cap) {
        (void)out;
        (void)cap;
        return 0;
    }

    // Optional mux hooks. Default: no persistent slot (always full prefill).
    virtual Status begin_generate(int slot, const std::vector<int> &ids, const GenParams &gp,
                                  int &reuse, std::string &err) {
        (void)slot;
        (void)ids;
        (void)gp;
        reuse = 0;
        err = "begin_generate not implemented";
        return Status::Unsupported;
    }
    virtual Status next_token(int slot, int &token, bool &done, std::string &err) {
        (void)slot;
        (void)token;
        (void)done;
        err = "next_token not implemented";
        return Status::Unsupported;
    }
    // Official mux: one decode row per active slot in one forward. Default loops
    // next_token. K3 overrides to union-MoE the continuing rows.
    virtual Status next_tokens(const int *slots, int n, int *tokens, uint8_t *done,
                               std::string &err) {
        if (n < 1)
            return Status::Ok;
        if (!slots || !tokens || !done) {
            err = "next_tokens bad args";
            return Status::InvalidArgument;
        }
        for (int i = 0; i < n; ++i) {
            bool d = false;
            Status st = next_token(slots[i], tokens[i], d, err);
            done[i] = d ? 1 : 0;
            if (st != Status::Ok)
                return st;
        }
        return Status::Ok;
    }
    virtual void end_generate(int slot) { (void)slot; }

    // Copy slot KV at pos0..pos0+n into rows (already sized to persist geometry).
    // L/R are layer-major [n_layers, kv_lora/qk_rope]; I is concatenated DSA rows.
    // Returns how many rows were written. Default: none (caller keeps zeros).
    virtual int export_kv_rows(int slot, int pos0, int n, KvPersistRecord *rows) const {
        (void)slot;
        (void)pos0;
        (void)n;
        (void)rows;
        return 0;
    }
    // Restore slot KV from persist rows (same layout as export). Returns rows written.
    virtual int import_kv_rows(int slot, int pos0, int n, const KvPersistRecord *rows) {
        (void)slot;
        (void)pos0;
        (void)n;
        (void)rows;
        return 0;
    }
};

std::unique_ptr<FamilyEngine> make_engine(Family family);

// CPU ops shared by K3 / GLM53.
// out_norm is the official shared GDN scale [head_dim] (not packed [heads, head_dim]).
// dt_n is the length of dt_bias: P, n_heads (broadcast), or 0.
void kda_step(const float *x, int hidden, const KdaConfig &kda,
              const quant::QuantMat *w_q, const quant::QuantMat *w_k, const quant::QuantMat *w_v,
              const quant::QuantMat *w_b, const quant::QuantMat *w_fa, const quant::QuantMat *w_fb,
              const float *dt_bias, int dt_n, const float *a_log, const quant::QuantMat *w_g,
              const quant::QuantMat *w_gb, const quant::QuantMat *w_o, const float *out_norm,
              float *S, float *y, float eps,
              const float *conv_q = nullptr, const float *conv_k = nullptr,
              const float *conv_v = nullptr, float *win_q = nullptr, float *win_k = nullptr,
              float *win_v = nullptr);

// Q/K/V GEMM + dt/alpha/beta. tok pointers alias the work vectors (must stay alive).
// False if windows/taps/S are missing or dims are invalid. Does not shift windows.
bool kda_fill_token(const float *x, int hidden, const KdaConfig &kda, const quant::QuantMat *w_q,
                    const quant::QuantMat *w_k, const quant::QuantMat *w_v,
                    const quant::QuantMat *w_b, const quant::QuantMat *w_fa,
                    const quant::QuantMat *w_fb, const float *dt_bias, int dt_n, const float *a_log,
                    const float *conv_q, const float *conv_k, const float *conv_v, float *win_q,
                    float *win_k, float *win_v, float *S, std::vector<float> &qt,
                    std::vector<float> &kt, std::vector<float> &tv, std::vector<float> &alpha,
                    std::vector<float> &beta, std::vector<float> &oh, metal_ops::KdaToken &tok);

// Per-head o_norm + sigmoid(W_g x) + W_o. oh is fused head output [P]; not mutated.
void kda_project_out(const float *x, int hidden, const KdaConfig &kda, const quant::QuantMat *w_g,
                     const quant::QuantMat *w_gb, const quant::QuantMat *w_o, const float *out_norm,
                     const float *oh, float *y, float eps);

// layer_decode_kda + official W_o. On success x += y and nrm = rmsnorm(x, post_ln).
// False if windows/taps missing or P < hidden; does not call kda_step or shift windows.
bool kda_try_layer_decode(const float *in_n, float *x, const float *post_ln, float *nrm, float *y,
                          int hidden, const KdaConfig &kda, const quant::QuantMat *w_q,
                          const quant::QuantMat *w_k, const quant::QuantMat *w_v,
                          const quant::QuantMat *w_b, const quant::QuantMat *w_fa,
                          const quant::QuantMat *w_fb, const float *dt_bias, int dt_n,
                          const float *a_log, const quant::QuantMat *w_g,
                          const quant::QuantMat *w_gb, const quant::QuantMat *w_o,
                          const float *out_norm, float *S, float eps, const float *conv_q,
                          const float *conv_k, const float *conv_v, float *win_q, float *win_k,
                          float *win_v);

// layer_decode_mla after Q/cache/RoPE. On success x += y and nrm = rmsnorm(x, post_ln).
// False if absorbed KV/cache missing, sparse DSA, or dense Q/O stand-in.
// Writes the cache row once; does not call mla_step.
bool mla_try_layer_decode(const float *in_n, float *x, const float *post_ln, float *nrm, float *y,
                          int hidden, const MlaConfig &mla, const quant::QuantMat *w_qa,
                          const float *qa_ln, const quant::QuantMat *w_qb,
                          const quant::QuantMat *w_kva, const float *kva_ln,
                          const quant::QuantMat *w_kt, const quant::QuantMat *w_v,
                          const quant::QuantMat *w_o, const quant::QuantMat *w_g, float *cache,
                          int pos, float eps, const int *selected = nullptr, int n_sel = 0);

// Causal depthwise conv: shift window, write x[p] into the last tap, then x[p] <- taps · window.
void kda_short_conv(float *x, const float *taps, float *window, int channels, int k);

// Absorbed MLA. Cache stride is kv_lora+qk_rope (L then R).
// score_j = (W_k^T q_nope) · c_j + q_rot · R_j ;  out = W_v (Σ a_j c_j).
// When qk_rope>0, rope_theta>0 and nope is false, RoPE is applied to q_rot
// and cached k_pe. Official K3 MLA is NoPE (raw rope strip).
// w_kt is [n_heads * kv_lora, qk_nope], w_v is [n_heads * v_head, kv_lora].
// w_kva is [kv_lora+qk_rope, hidden]. w_g is optional (K3 output gate).
// Missing absorbed KV (kva/kt/v/cache) falls back to a dense Q/O stand-in.
void mla_step(const float *x, int hidden, const MlaConfig &mla, const quant::QuantMat *w_qa,
              const float *qa_ln, const quant::QuantMat *w_qb, const quant::QuantMat *w_kva,
              const float *kva_ln, const quant::QuantMat *w_kt, const quant::QuantMat *w_v,
              const quant::QuantMat *w_o, const quant::QuantMat *w_g, float *cache, int pos,
              float *y, float eps, const int *selected = nullptr, int n_sel = 0,
              double *t_kvb = nullptr);

int dsa_index_width(const DsaConfig &dsa);
// Score k-pool groups with ReLU(q·k), take top-k pools, optional tail. Unused slots = -1.
int dsa_select(int *out, const float *queries, const float *keys, const float *gates,
               const float *head_w, const float *ape, int seq, const DsaConfig &dsa);
// Official decode range: keys/gates/valid cover [0, seq); queries/head_w/out cover [q_from, q_to)
// indexed from 0. out is nq * width. Returns 0 or -1.
int dsa_select_range(int *out, const float *queries, const float *keys, const float *gates,
                     const float *head_w, const float *ape, const uint8_t *valid, int seq,
                     const DsaConfig &dsa, int q_from, int q_to);

void layernorm(const float *x, const float *w, const float *b, float *y, int n, float eps);

// Fold kv_b_proj [H*(qk_nope+v_head), kv_lora] into W_k^T and W_v.
void mla_absorb_kvb(const float *kv_b, int n_heads, int qk_nope, int v_head, int kv_lora,
                    quant::QuantMat &w_kt, quant::QuantMat &w_v, int bits);

void bf16_to_f32(const uint16_t *src, float *dst, int64_t n);

// One DiT residual block on streamed BF16 qkv/out/fc1/fc2 (row-major [O,I]).
void h3_dit_block_cpu(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                      int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                      int tokens, float eps, const float *adaln_mod = nullptr,
                      const float *q_norm = nullptr, const float *k_norm = nullptr,
                      const float *rope_cos = nullptr, const float *rope_sin = nullptr);

void attnres_mix(const std::vector<std::vector<float>> &snapshots, const float *prefix,
                 const float *res_norm, const float *res_proj, float *hidden, int n, float eps);

void mhc_mix(float *streams, int hidden, int mult, const float *alpha, int iters, float eps);

// Official mHC: RMS-flatten streams, hc_fn → pre/post/comb (Sinkhorn), collapse with pre.
// fn is [(2+M)*M, M*H], scale [3], base [(2+M)*M]. Returns 0 or -1.
int mhc_pre(float *collapsed, float *post, float *comb, const float *streams, const float *fn,
            const float *scale, const float *base, int mult, int hidden, int iters, float norm_eps,
            float hc_eps);
// Mix residual through comb and add branch · post. streams and residual may alias.
int mhc_post(float *streams, const float *branch, const float *residual, const float *post,
             const float *comb, int mult, int hidden);

// Top-k on `choice`. Mix weights come from `mix` (official: unbiased σ) or
// from `choice` when mix is null. Selected weights are clipped at 0 and L1-normalized.
int moe_topk(const float *choice, int n, int k, int *idx, float *w, const float *mix = nullptr);

// temperature<=0 is greedy argmax. top_p in (0,1) is nucleus. rng is splitmix state.
// allow==null: all tokens legal. allow[i]!=0 keeps token i.
// Apply OpenAI-style penalties in-place. hist may be prompt+so-far tokens.
void apply_penalties(float *logits, int vocab, const int *hist, int hist_n, float frequency,
                     float presence);
// HuggingFace-style: if logit<0 multiply by penalty else divide. 1 = identity.
void apply_repetition_penalty(float *logits, int vocab, const int *hist, int hist_n, float penalty);
// Keep the k largest logits; others → -1e30. k<=0 is identity.
void apply_top_k(float *logits, int vocab, int k);
// Drop tokens with p < min_p * p_max (softmax space). min_p<=0 is identity.
void apply_min_p(float *logits, int vocab, float min_p);
// logit_bias[i] = {token, bias}; add to logits[token].
void apply_logit_bias(float *logits, int vocab, const std::pair<int, float> *bias, int n_bias);
// log p(token) under softmax of logits (allow mask optional).
float token_logprob(const float *logits, int vocab, int token, const uint8_t *allow = nullptr);
void top_logprobs(const float *logits, int vocab, int k, GenLogprob *out, int *n_out,
                  const uint8_t *allow = nullptr);

int sample_token(const float *logits, int vocab, float temperature, float top_p, uint64_t *rng,
                 const uint8_t *allow = nullptr, float *out_logprob = nullptr);

void apply_rope(float *x, int n, int pos, float theta);

struct GlmVitBlock {
    std::vector<float> norm1, norm2;
    std::vector<float> qkv_w, qkv_b, q_norm, k_norm, proj_w, proj_b;
    std::vector<float> gate_w, gate_b, up_w, up_b, down_w, down_b;
};

struct GlmVitTower {
    VisionConfig cfg;
    std::vector<float> patch_w, patch_b, post_norm;
    std::vector<float> down_w, down_b;
    std::vector<float> merger_proj, merger_norm_w, merger_norm_b;
    std::vector<float> merger_gate, merger_up, merger_down;
    std::vector<GlmVitBlock> blocks;
    bool ready() const { return !patch_w.empty() && !blocks.empty(); }
};

// Official tower: pixels [grid_h*grid_w, C*T*P*P] block-major → [N, out_hidden].
int glm_vit_forward(const GlmVitTower &tower, const float *pixels, int grid_h, int grid_w,
                    float *out);
// RGB HWC [0,1] → tokens. Uses the tower when ready, else patch/proj stub.
int glm_vit_embed(const float *rgb, int width, int height, const VisionConfig &v,
                  const quant::QuantMat *patch, const quant::QuantMat *proj, float *out,
                  int out_cap, const GlmVitTower *tower = nullptr);

int h3_dit_patchify(const float *z, int ch, int t, int h, int w, float *rows);
int h3_dit_unpatchify(const float *rows, int ch, int t, int h, int w, float *z);
void h3_sigma_video(int steps, float *sigmas, float shift = 12.f);
void h3_time_features(float t, float *out, int dim);
int h3_euler_step(float *sample, const float *velocity, int n, float sigma, float sigma_next);

// Unique expert ids from C tokens × topk, sorted. Returns count.
int moe_union_ids(const int *idx, int n_tok, int topk, int *out, int out_cap);

} // namespace mvllm
