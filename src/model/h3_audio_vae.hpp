#pragma once

#include "../core/types.hpp"

#include <string>
#include <vector>

namespace mvllm {

// Official H3 AudioVAE geometry: latent [32,2,T] channel/stereo/time,
// hop 800, 32 kHz stereo. audio_t = round(aligned_frames * 40 / 24).
constexpr int kH3AudioChannels = 32;
constexpr int kH3AudioStereo = 2;
constexpr int kH3AudioHop = 800;
constexpr int kH3AudioRate = 32000;
constexpr int kH3AudioLatentFps = 40;
constexpr int kH3VideoFps = 24;
constexpr float kH3AudioSigmaShift = 3.f;

int h3_audio_t(int frames);
int h3_audio_pad_samples(int samples);
int h3_audio_samples(int audio_t);

// Official DiT audio pack: [C,2,T] → rows [2*T, C] (stream, then time).
int h3_dit_pack_audio(const float *latent, int channels, int time, float *rows);
int h3_dit_unpack_audio(const float *rows, int channels, int time, float *latent);

// Channel-major PCM [channels, samples] in [-1,1]. 16-bit PCM WAV.
Status h3_write_wav(const std::string &path, const float *pcm, int channels, int samples, int rate,
                    std::string &err);
Status h3_read_wav(const std::string &path, std::vector<float> &pcm, int &channels, int &samples,
                   int &rate, std::string &err);

struct H3AudioVae {
    bool from_checkpoint = false;
    std::string source_dir;
    std::vector<float> latents_mean;
    std::vector<float> latents_std;

    Status load(const std::string &model_dir, std::string &err);
    // PCM [2, samples] → normalized latent [32, 2, T]. Pads to hop 800.
    void encode(const float *pcm, int samples, std::vector<float> &z, int &audio_t) const;
    // Normalized latent [32, 2, T] → PCM [2, T*800] clipped to [-1,1].
    void decode(const float *z, int audio_t, std::vector<float> &pcm) const;
};

} // namespace mvllm
