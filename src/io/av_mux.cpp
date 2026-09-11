#include "av_mux.hpp"
#include "../model/h3_audio_vae.hpp"

#include <cctype>
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

bool tool_ok(const std::string &bin) {
    std::string cmd = shell_quote(bin) + " -version >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

// Same directory as H3_FFMPEG when that path has a slash; otherwise `ffprobe`.
std::string ffprobe_bin() {
    const char *p = std::getenv("H3_FFMPEG");
    if (p && *p) {
        std::string ff(p);
        auto slash = ff.find_last_of('/');
        if (slash != std::string::npos) {
            std::string cand = ff.substr(0, slash + 1) + "ffprobe";
            if (tool_ok(cand))
                return cand;
        }
    }
    return "ffprobe";
}

std::vector<uint8_t> capture_bytes(const std::string &cmd, int &rc) {
    FILE *fp = popen(cmd.c_str(), "r");
    if (!fp) {
        rc = -1;
        return {};
    }
    std::vector<uint8_t> out;
    uint8_t buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), fp)) > 0)
        out.insert(out.end(), buf, buf + n);
    rc = pclose(fp);
    return out;
}

std::string capture_text(const std::string &cmd) {
    int rc = 0;
    std::vector<uint8_t> raw = capture_bytes(cmd, rc);
    return std::string(raw.begin(), raw.end());
}

int parse_int_after(const std::string &s, size_t i) {
    int v = 0;
    bool any = false;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        any = true;
        v = v * 10 + (s[i] - '0');
        ++i;
    }
    return any ? v : 0;
}

bool parse_key_wh(const std::string &text, int &w, int &h) {
    auto key = [&](const char *k) -> int {
        auto p = text.find(k);
        if (p == std::string::npos)
            return 0;
        return parse_int_after(text, p + std::strlen(k));
    };
    int ww = key("width=");
    int hh = key("height=");
    if (ww > 0 && hh > 0) {
        w = ww;
        h = hh;
        return true;
    }
    return false;
}

// First `<digits>x<digits>` at or after `from` (ffmpeg Stream / ffprobe csv).
bool parse_wxh(const std::string &text, size_t from, int &w, int &h) {
    for (size_t i = from; i < text.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(text[i])))
            continue;
        int ww = parse_int_after(text, i);
        size_t j = i;
        while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j])))
            ++j;
        if (j >= text.size() || text[j] != 'x' || j + 1 >= text.size() ||
            !std::isdigit(static_cast<unsigned char>(text[j + 1]))) {
            i = j;
            continue;
        }
        int hh = parse_int_after(text, j + 1);
        if (ww > 0 && hh > 0) {
            w = ww;
            h = hh;
            return true;
        }
        i = j;
    }
    return false;
}

bool parse_probe_wh(const std::string &text, int &w, int &h) {
    if (parse_key_wh(text, w, h))
        return true;
    return parse_wxh(text, 0, w, h);
}

bool probe_wh(const std::string &path, int &w, int &h) {
    w = 0;
    h = 0;
    std::string probe = ffprobe_bin();
    if (tool_ok(probe)) {
        std::ostringstream cmd;
        cmd << shell_quote(probe) << " -v error -select_streams v:0"
            << " -show_entries stream=width,height -of default=noprint_wrappers=1 "
            << shell_quote(path) << " 2>/dev/null";
        if (parse_probe_wh(capture_text(cmd.str()), w, h))
            return true;
    }
    std::ostringstream cmd;
    cmd << shell_quote(ffmpeg_bin()) << " -nostdin -hide_banner -i " << shell_quote(path)
        << " 2>&1";
    std::string text = capture_text(cmd.str());
    auto vid = text.find("Video:");
    return parse_wxh(text, vid == std::string::npos ? 0 : vid, w, h);
}

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

Status h3_read_mp4(const std::string &path, std::vector<float> &rgb, int &frames, int &height,
                   int &width, std::string &err) {
    frames = 0;
    height = 0;
    width = 0;
    rgb.clear();
    if (path.empty()) {
        err = "invalid mp4 args";
        return Status::InvalidArgument;
    }
    if (!h3_ffmpeg_available()) {
        err = "ffmpeg not found (set H3_FFMPEG or install ffmpeg)";
        return Status::Unsupported;
    }
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            err = "cannot read " + path;
            return Status::IoError;
        }
    }
    int w = 0, h = 0;
    if (!probe_wh(path, w, h) || w < 1 || h < 1) {
        err = "cannot probe mp4 size";
        return Status::ParseError;
    }
    const std::string rawp = path + ".tmp.rgb";
    std::ostringstream cmd;
    cmd << shell_quote(ffmpeg_bin()) << " -y -nostdin -hide_banner -loglevel error -i "
        << shell_quote(path) << " -an -f rawvideo -pix_fmt rgb24 " << shell_quote(rawp);
    int rc = std::system(cmd.str().c_str());
    std::ifstream in(rawp, std::ios::binary);
    if (rc != 0 || !in) {
        std::remove(rawp.c_str());
        err = "ffmpeg decode failed";
        return Status::IoError;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff nbytes = in.tellg();
    in.seekg(0, std::ios::beg);
    const size_t frame_bytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
    if (nbytes < static_cast<std::streamoff>(frame_bytes) || frame_bytes == 0 ||
        static_cast<size_t>(nbytes) % frame_bytes != 0) {
        std::remove(rawp.c_str());
        err = "empty or truncated rgb24";
        return Status::ParseError;
    }
    const int nf = static_cast<int>(static_cast<size_t>(nbytes) / frame_bytes);
    if (nf < 1) {
        std::remove(rawp.c_str());
        err = "empty or truncated rgb24";
        return Status::ParseError;
    }
    std::vector<uint8_t> raw(static_cast<size_t>(nbytes));
    in.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(nbytes));
    const bool short_read = in.gcount() != static_cast<std::streamsize>(nbytes);
    in.close();
    std::remove(rawp.c_str());
    if (short_read) {
        err = "empty or truncated rgb24";
        return Status::IoError;
    }
    rgb.resize(raw.size());
    for (size_t i = 0; i < raw.size(); ++i)
        rgb[i] = static_cast<float>(raw[i]) / 255.f;
    frames = nf;
    height = h;
    width = w;
    return Status::Ok;
}

} // namespace mvllm
