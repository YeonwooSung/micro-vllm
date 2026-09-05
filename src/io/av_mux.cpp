#include "av_mux.hpp"
#include "../model/h3_audio_vae.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace mvllm {
namespace {

const char *ffmpeg_bin() {
    const char *p = std::getenv("H3_FFMPEG");
    return (p && *p) ? p : "ffmpeg";
}

std::string shell_quote(const std::string &s) {
    std::string o = "'";
    for (char c : s) {
        if (c == '\'')
            o += "'\\''";
        else
            o += c;
    }
    o += "'";
    return o;
}

int even(int v) { return v + (v & 1); }

} // namespace

bool h3_ffmpeg_available() {
    std::string cmd = std::string(ffmpeg_bin()) + " -version >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

Status h3_write_mp4(const std::string &path, const float *rgb, int frames, int height, int width,
                    int fps, const float *pcm, int channels, int samples, int rate,
                    std::string &err) {
    if (path.empty() || !rgb || frames < 1 || height < 1 || width < 1) {
        err = "invalid mp4 args";
        return Status::InvalidArgument;
    }
    if (fps < 1)
        fps = 24;
    if (!h3_ffmpeg_available()) {
        err = "ffmpeg not found (set H3_FFMPEG or install ffmpeg)";
        return Status::Unsupported;
    }
    const int ow = even(width);
    const int oh = even(height);
    const std::string raw = path + ".tmp.rgb";
    const std::string wav = path + ".tmp.wav";
    {
        std::ofstream out(raw, std::ios::binary);
        if (!out) {
            err = "cannot write " + raw;
            return Status::IoError;
        }
        std::vector<uint8_t> row(static_cast<size_t>(ow) * 3, 0);
        for (int f = 0; f < frames; ++f) {
            for (int y = 0; y < oh; ++y) {
                std::memset(row.data(), 0, row.size());
                if (y < height) {
                    const float *src =
                        rgb + (static_cast<size_t>(f) * height + y) * width * 3;
                    for (int x = 0; x < width; ++x) {
                        for (int c = 0; c < 3; ++c) {
                            float v = src[x * 3 + c];
                            if (v < 0.f)
                                v = 0.f;
                            if (v > 1.f)
                                v = 1.f;
                            row[static_cast<size_t>(x) * 3 + c] =
                                static_cast<uint8_t>(v * 255.f + 0.5f);
                        }
                    }
                }
                out.write(reinterpret_cast<const char *>(row.data()),
                          static_cast<std::streamsize>(row.size()));
            }
        }
    }
    bool have_wav = false;
    if (pcm && channels > 0 && samples > 0 && rate > 0) {
        std::string werr;
        if (h3_write_wav(wav, pcm, channels, samples, rate, werr) == Status::Ok)
            have_wav = true;
    }
    std::ostringstream cmd;
    cmd << shell_quote(ffmpeg_bin()) << " -y -hide_banner -loglevel error"
        << " -f rawvideo -pix_fmt rgb24 -s " << ow << "x" << oh << " -r " << fps << " -i "
        << shell_quote(raw);
    if (have_wav)
        cmd << " -i " << shell_quote(wav) << " -c:a aac -b:a 192k";
    cmd << " -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart " << shell_quote(path);
    int rc = std::system(cmd.str().c_str());
    std::remove(raw.c_str());
    if (have_wav)
        std::remove(wav.c_str());
    if (rc != 0) {
        err = "ffmpeg mux failed";
        return Status::IoError;
    }
    return Status::Ok;
}

} // namespace mvllm
