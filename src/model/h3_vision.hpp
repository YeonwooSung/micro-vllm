#pragma once

#include "../core/config.hpp"

#include <string>
#include <vector>

namespace mvllm {

// Official Qwen3-VL vision tower (h3_vision_encoder.c).
constexpr int kH3VisionHidden = 1152;
constexpr int kH3VisionIntermediate = 4304;
constexpr int kH3VisionLayers = 27;
constexpr int kH3VisionHeads = 16;
constexpr int kH3VisionHeadDim = 72;
constexpr int kH3VisionRopeHalf = 36;
constexpr int kH3VisionPatch = 16;
constexpr int kH3VisionTemporalPatch = 2;
constexpr int kH3VisionMerge = 2;
constexpr int kH3VisionOut = 5120;
constexpr int kH3VisionDeepstacks = 3;
constexpr int kH3VisionPosSide = 48;
// Deepstack mergers run after blocks 8, 16, 24 (0-based).
constexpr int kH3VisionDeepstackAfter[3] = {8, 16, 24};

struct H3VisionOut {
    int grid_h = 0;
    int grid_w = 0;
    int tokens = 0; // (grid_h/2)*(grid_w/2)
    int out_width = 0;
    std::vector<float> merged;                 // [tokens, out_width]
    std::vector<float> deepstack[kH3VisionDeepstacks];
};

struct H3VisionConfig {
    int hidden = kH3VisionHidden;
    int intermediate = kH3VisionIntermediate;
    int layers = kH3VisionLayers;
    int heads = kH3VisionHeads;
    int head_dim = kH3VisionHeadDim;
    int rope_half = kH3VisionRopeHalf;
    int patch = kH3VisionPatch;
    int temporal_patch = kH3VisionTemporalPatch;
    int merge = kH3VisionMerge;
    int out_width = kH3VisionOut;
    int pos_side = kH3VisionPosSide;
    float ln_eps = 1e-6f;
    int deepstack_after[3] = {8, 16, 24};
};

class H3VisionEncoder {
public:
    Status load(const std::string &model_dir, std::string &err);
    // pixels: frame-major RGB HWC [T,H,W,3] in [0,1]. T is 1 or 2.
    // H and W must be multiples of 32 (patch*merge).
    void encode(const float *rgb_hwc, int frames, int height, int width, H3VisionOut &out) const;
    const H3VisionConfig &config() const { return cfg_; }
    bool from_checkpoint() const { return from_checkpoint_; }
    bool ready() const { return ready_; }

private:
    H3VisionConfig cfg_{};
    bool ready_ = false;
    bool from_checkpoint_ = false;
};

} // namespace mvllm
