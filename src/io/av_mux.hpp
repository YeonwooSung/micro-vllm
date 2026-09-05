#pragma once

#include "../core/types.hpp"

#include <string>

namespace mvllm {

// True if H3_FFMPEG or `ffmpeg` on PATH responds to -version.
bool h3_ffmpeg_available();

// RGB HWC [0,1] frame-major + channel-major PCM → H.264/AAC MP4 via ffmpeg.
// Missing ffmpeg returns Unsupported and does not write the path.
Status h3_write_mp4(const std::string &path, const float *rgb, int frames, int height, int width,
                    int fps, const float *pcm, int channels, int samples, int rate,
                    std::string &err);

} // namespace mvllm
