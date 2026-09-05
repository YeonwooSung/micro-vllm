#include "cli_flags.hpp"

#include "../io/image.hpp"
#include "../tok/json_schema.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace mvllm {
namespace {

bool eq(const char *a, const char *b) { return a && b && std::strcmp(a, b) == 0; }

const char *next_val(int argc, char **argv, int i) {
    if (i + 1 >= argc || !argv || !argv[i + 1])
        return nullptr;
    return argv[i + 1];
}

bool parse_int(const char *s, int &out, const char *flag, std::string &err) {
    if (!s || !*s) {
        err = std::string("bad ") + flag;
        return false;
    }
    char *end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (errno || !end || end == s || *end || v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max()) {
        err = std::string("bad ") + flag + ": " + s;
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

bool parse_float(const char *s, float &out, const char *flag, std::string &err) {
    if (!s || !*s) {
        err = std::string("bad ") + flag;
        return false;
    }
    char *end = nullptr;
    errno = 0;
    const float v = std::strtof(s, &end);
    if (errno || !end || end == s || *end || !std::isfinite(v)) {
        err = std::string("bad ") + flag + ": " + s;
        return false;
    }
    out = v;
    return true;
}

bool parse_u64(const char *s, uint64_t &out, const char *flag, std::string &err) {
    if (!s || !*s || s[0] == '-') {
        err = std::string("bad ") + flag + (s && *s ? std::string(": ") + s : "");
        return false;
    }
    char *end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (errno || !end || end == s || *end) {
        err = std::string("bad ") + flag + ": " + s;
        return false;
    }
    out = static_cast<uint64_t>(v);
    return true;
}

bool read_whole_file(const std::string &path, std::string &out, std::string &err,
                     const char *kind) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = std::string("cannot read ") + kind + " file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    if (!in && !in.eof()) {
        err = std::string("cannot read ") + kind + " file: " + path;
        return false;
    }
    out = ss.str();
    return true;
}

bool load_grammar(const char *str, GenParams &gp, std::string &err) {
    const std::string s = str ? str : "";
    if (s.find("root") != std::string::npos || s.find("::=") != std::string::npos) {
        gp.grammar = s;
        return true;
    }
    std::string text;
    if (!read_whole_file(s, text, err, "grammar"))
        return false;
    gp.grammar = std::move(text);
    return true;
}

bool load_image(const char *path, GenParams &gp, CliGenExtras &ex, std::string &err) {
    const std::string p = path ? path : "";
    std::vector<float> rgb;
    int w = 0, h = 0;
    std::string dec_err;
    Status st = decode_image_url(p, rgb, w, h, dec_err);
    if (st != Status::Ok || w <= 0 || h <= 0) {
        std::string bytes;
        std::string io_err;
        if (read_whole_file(p, bytes, io_err, "image")) {
            std::string byte_err;
            st = decode_image_bytes(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size(),
                                    rgb, w, h, byte_err);
            if (st == Status::Ok && w > 0 && h > 0)
                dec_err.clear();
            else if (!byte_err.empty())
                dec_err = byte_err;
        } else if (dec_err.empty()) {
            dec_err = io_err;
        }
    }
    if (st != Status::Ok || w <= 0 || h <= 0) {
        err = dec_err.empty() ? ("cannot read image: " + p) : dec_err;
        return false;
    }
    ex.image_rgb = std::move(rgb);
    gp.image_rgb = ex.image_rgb.empty() ? nullptr : ex.image_rgb.data();
    gp.image_w = w;
    gp.image_h = h;
    return true;
}

} // namespace

bool apply_cli_gen_flags(int argc, char **argv, GenParams &gp, CliGenExtras &ex, std::string &err) {
    err.clear();
    if (argc <= 0 || !argv)
        return true;

    for (int i = 0; i < argc; ++i) {
        const char *a = argv[i];
        if (!a)
            continue;

        if (eq(a, "--stop")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            gp.stop.push_back(v);
            ++i;
            continue;
        }
        if (eq(a, "--grammar")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            if (!load_grammar(v, gp, err))
                return false;
            continue;
        }
        if (eq(a, "--json")) {
            if (gp.grammar.empty())
                gp.grammar = json_object_gbnf();
            continue;
        }
        if (eq(a, "--top-k")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            int n = 0;
            if (!parse_int(v, n, "--top-k", err))
                return false;
            gp.top_k = n < 0 ? 0 : n;
            continue;
        }
        if (eq(a, "--min-p")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            if (!parse_float(v, gp.min_p, "--min-p", err))
                return false;
            continue;
        }
        if (eq(a, "--rep-penalty") || eq(a, "--repetition-penalty")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            if (!parse_float(v, gp.repetition_penalty, a, err))
                return false;
            continue;
        }
        if (eq(a, "--freq-penalty") || eq(a, "--frequency-penalty")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            if (!parse_float(v, gp.frequency_penalty, a, err))
                return false;
            continue;
        }
        if (eq(a, "--presence-penalty")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            if (!parse_float(v, gp.presence_penalty, "--presence-penalty", err))
                return false;
            continue;
        }
        if (eq(a, "--image")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            if (!load_image(v, gp, ex, err))
                return false;
            continue;
        }
        if (eq(a, "--temp") || eq(a, "--temperature")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            if (!parse_float(v, gp.temperature, a, err))
                return false;
            continue;
        }
        if (eq(a, "--top-p")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            if (!parse_float(v, gp.top_p, "--top-p", err))
                return false;
            continue;
        }
        if (eq(a, "--seed")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            if (!parse_u64(v, gp.seed, "--seed", err))
                return false;
            continue;
        }
        if (eq(a, "--think")) {
            gp.think = true;
            continue;
        }
        if (eq(a, "--no-think")) {
            gp.think = false;
            continue;
        }
        if (eq(a, "--chat")) {
            gp.apply_template = true;
            continue;
        }
    }
    return true;
}

bool apply_cli_video_flags(int argc, char **argv, H3GenParams &hp, std::string &err) {
    err.clear();
    if (argc <= 0 || !argv)
        return true;

    for (int i = 0; i < argc; ++i) {
        const char *a = argv[i];
        if (!a)
            continue;

        if (eq(a, "--prompt") || eq(a, "-p")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            hp.prompt = v;
            ++i;
            continue;
        }
        if (eq(a, "-o") || eq(a, "--output") || eq(a, "--output-path")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            hp.output_path = v;
            ++i;
            continue;
        }
        if (eq(a, "--width") || eq(a, "--height") || eq(a, "--frames") || eq(a, "--steps")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            int n = 0;
            if (!parse_int(v, n, a, err))
                return false;
            if (n > 0) {
                if (eq(a, "--width"))
                    hp.width = n;
                else if (eq(a, "--height"))
                    hp.height = n;
                else if (eq(a, "--frames"))
                    hp.frames = n;
                else
                    hp.steps = n;
            }
            continue;
        }
        if (eq(a, "--layers") || eq(a, "--dit-layers")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            int n = 0;
            if (!parse_int(v, n, a, err))
                return false;
            if (n > 0)
                hp.dit_layers = n;
            continue;
        }
        if (eq(a, "--seed")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            if (!parse_u64(v, hp.seed, "--seed", err))
                return false;
            continue;
        }
        if (eq(a, "--audio") || eq(a, "--audio-path")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            hp.audio_path = v;
            ++i;
            continue;
        }
        if (eq(a, "--first-frame")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            hp.first_frame = v;
            ++i;
            continue;
        }
        if (eq(a, "--last-frame")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            hp.last_frame = v;
            ++i;
            continue;
        }
        if (eq(a, "--ref-image")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            hp.ref_images.push_back(v);
            ++i;
            continue;
        }
        if (eq(a, "--denoise-reuse")) {
            const char *v = next_val(argc, argv, i);
            if (!v)
                continue;
            ++i;
            int n = 0;
            if (!parse_int(v, n, "--denoise-reuse", err))
                return false;
            if (n >= 0)
                hp.denoise_reuse = n;
            continue;
        }
    }
    return true;
}

} // namespace mvllm
