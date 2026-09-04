#pragma once

#include "../core/config.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {

// MiniMax-H3 visual VAE geometry. Spatial downsample 16; temporal
// T<=5 → 2 else ((T-5)/17)*5+2 after aligning frames so (F-5)%17==0.
int h3_align_frames(int frames);
int h3_video_latent_t(int frames);
int h3_encoder_latent_t(int frames);
void h3_latent_canvas(int width, int height, int spatial, int *lw, int *lh);

struct H3VaeGeom {
    int frames = 0;
    int width = 0;
    int height = 0;
    int latent_t = 0;
    int latent_h = 0;
    int latent_w = 0;
    int latent_ch = 24;
    int spatial = 16;
};

H3VaeGeom h3_vae_geom(int width, int height, int frames, int spatial = 16, int latent_ch = 24);

// Frame-major RGB in [0,1]: rgb[f * H * W * 3 + y * W * 3 + x * 3 + c]
// Latent is channel-major: z[c * T * H * W + t * H * W + y * W + x]
struct H3Vae {
    bool from_checkpoint = false;
    std::string source_dir;
    H3VaeGeom geom{};
    std::vector<float> mean; // ImageNet RGB (3) for pixels; unused for latent
    std::vector<float> std;
    std::vector<float> latents_mean; // 24
    std::vector<float> latents_std;

    Status load(const std::string &model_dir, const H3Config &h3, std::string &err);
    // Pixels [F,H,W,3] -> normalized latent [24,T,lh,lw]. Always a real transform.
    void encode(const float *rgb, const H3VaeGeom &g, float *z) const;
    // Normalized latent -> RGB [F,H,W,3] in [0,1]. Always a real transform.
    void decode(const float *z, const H3VaeGeom &g, float *rgb) const;
};

// PPM (P6) writer for generate_video output.
Status h3_write_ppm(const std::string &path, const float *rgb, int frames, int height, int width,
                    std::string &err);

} // namespace mvllm
