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

// Official decoder constants (h3_video_vae.c).
constexpr int kH3VaeRegisters = 4;
constexpr int kH3VaeSuffix = 5; // 4 register tokens + 1 zero
constexpr int kH3VaeChunkT = 7;
constexpr int kH3VaeFrameOffset = 3;
constexpr int kH3VaeFirstChunkFrames = 22;
constexpr int kH3VaeTilePixels = 256;
constexpr int kH3VaeTileOverlap = 64;
constexpr int kH3VaeRopeHalf = 24;
constexpr int kH3VaeOutPatch = 3072; // 3 * 4 * 16 * 16

// decoded_t used to index 3072-d patches. first-chunk extra +3 when
// output_frames==22 and frame>=17.
int h3_vae_decoded_t(int frame, int output_frames, int offset = kH3VaeFrameOffset);
int h3_vae_tile_count(int pixel_extent, int tile_pixels = kH3VaeTilePixels);
// rows are [T*H*W, 3072] patch-major (registers not included).
void h3_vae_unpack_3072(const float *rows, int latent_t, int latent_h, int latent_w, int frames,
                        int height, int width, const float *im_mean, const float *im_std,
                        int frame_offset, float *rgb);

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

    bool official_decode = false; // register_tokens present
    bool official_encode = false; // encoder.conv_in present
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
