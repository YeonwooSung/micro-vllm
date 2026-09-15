#pragma once

#include "../core/config.hpp"
#include "../io/safetensors.hpp"
#include "../quant/weight.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {

struct H3TextConfig {
    int hidden = 5120;
    int layers = 50;
    int n_q = 64;
    int n_kv = 8;
    int head_dim = 128;
    int intermediate = 25600;
    int vocab = 151936;
    float rms_eps = 1e-6f;
    float rope_theta = 5000000.f;
};

// Vision pad span spliced into the language stack (h3_text_encoder.c).
constexpr int kH3TextDeepstacks = 3;
struct H3VisionSpan {
    int start = 0; // first pad token
    int tokens = 0;
    const float *embeddings = nullptr; // [tokens, hidden]
    const float *deepstack[kH3TextDeepstacks] = {};
};

// Qwen3-VL language stack used as the H3 text encoder (first 50 layers official).
class H3TextEncoder {
public:
    H3TextEncoder() = default;
    ~H3TextEncoder();
    H3TextEncoder(const H3TextEncoder &) = delete;
    H3TextEncoder &operator=(const H3TextEncoder &) = delete;
    H3TextEncoder(H3TextEncoder &&) = delete;
    H3TextEncoder &operator=(H3TextEncoder &&) = delete;

    Status load(const std::string &model_dir, std::string &err);
    // Drop a resident embed table after encode so DiT can reuse the RAM.
    void release_embed();
    bool embed_resident() const { return !embed_.empty(); }
    void encode(const std::vector<int> &ids, std::vector<float> &out) const;
    // Official multimodal path: splice vision rows, 3-axis mRoPE, add deepstack
    // residuals after layers 0, 1, 2. positions is [3, seq] axis-major.
    // layer_count runs only the first L decoder layers (clamped to cfg_.layers).
    void encode_mm(const std::vector<int> &ids, const H3VisionSpan *spans, int span_count,
                   const uint32_t *positions, const uint8_t *tags, std::vector<float> &out) const;
    void encode_mm(const std::vector<int> &ids, const H3VisionSpan *spans, int span_count,
                   const uint32_t *positions, const uint8_t *tags, std::vector<float> &out,
                   int layer_count) const;
    const H3TextConfig &config() const { return cfg_; }
    bool from_checkpoint() const { return from_checkpoint_; }
    bool ready() const { return ready_; }
    bool streamed() const { return streamed_; }

private:
    struct LayerHits {
        io::StHit wq, wk, wv, wo, gate, up, down;
    };

    void alloc_synth();
    bool load_layer_mats(int l, quant::QuantMat &wq, quant::QuantMat &wk, quant::QuantMat &wv,
                         quant::QuantMat &wo, quant::QuantMat &gate, quant::QuantMat &up,
                         quant::QuantMat &down) const;
    bool gather_embed(const std::vector<int> &ids, std::vector<float> &out) const;
    bool gemm_from_hit(const io::StHit &w, float *y, const float *x, int S) const;
    void apply_layer(int l, int T, const quant::QuantMat *wq, const quant::QuantMat *wk,
                     const quant::QuantMat *wv, const quant::QuantMat *wo,
                     const quant::QuantMat *gate, const quant::QuantMat *up,
                     const quant::QuantMat *down, const LayerHits *hits, const uint32_t *positions,
                     const H3VisionSpan *spans, int span_count, std::vector<float> &out,
                     std::vector<float> &n, std::vector<float> &q, std::vector<float> &k,
                     std::vector<float> &v, std::vector<float> &ctx, std::vector<float> &attn,
                     std::vector<float> &g, std::vector<float> &u, std::vector<float> &d) const;

    H3TextConfig cfg_{};
    bool ready_ = false;
    bool from_checkpoint_ = false;
    bool streamed_ = false;
    std::vector<float> embed_, norm_;
    std::vector<quant::QuantMat> wq_, wk_, wv_, wo_, gate_, up_, down_;
    std::vector<std::vector<float>> in_n_, post_n_, qn_, kn_;
    // Keep fds open so encode() can stream one layer at a time.
    std::vector<io::StFile> files_;
    std::vector<LayerHits> hits_;
    io::StHit embed_hit_;
};

// max_tokens<=0 keeps the full prompt (official). A positive cap keeps a prefix.
void h3_text_ids_from_prompt(const std::string &prompt, int vocab, std::vector<int> &ids,
                            int max_tokens = 0);

} // namespace mvllm
