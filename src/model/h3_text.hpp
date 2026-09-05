#pragma once

#include "../core/config.hpp"
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
    Status load(const std::string &model_dir, std::string &err);
    void encode(const std::vector<int> &ids, std::vector<float> &out) const;
    // Official multimodal path: splice vision rows, 3-axis mRoPE, add deepstack
    // residuals after layers 0, 1, 2. positions is [3, seq] axis-major.
    void encode_mm(const std::vector<int> &ids, const H3VisionSpan *spans, int span_count,
                   const uint32_t *positions, const uint8_t *tags, std::vector<float> &out) const;
    const H3TextConfig &config() const { return cfg_; }
    bool from_checkpoint() const { return from_checkpoint_; }
    bool ready() const { return ready_; }

private:
    void alloc_synth();
    H3TextConfig cfg_{};
    bool ready_ = false;
    bool from_checkpoint_ = false;
    std::vector<float> embed_, norm_;
    std::vector<quant::QuantMat> wq_, wk_, wv_, wo_, gate_, up_, down_;
    std::vector<std::vector<float>> in_n_, post_n_, qn_, kn_;
};

void h3_text_ids_from_prompt(const std::string &prompt, int vocab, std::vector<int> &ids);

} // namespace mvllm
