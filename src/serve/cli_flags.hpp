#pragma once

#include "../model/family.hpp"

#include <string>
#include <vector>

namespace mvllm {

// Extra buffers owned by the CLI so GenParams pointers stay valid.
struct CliGenExtras {
    std::vector<float> image_rgb;
};

// Apply `micro-vllm generate` flags onto gp. Repeatable --stop. --json sets
// json_object_gbnf(). --image PATH is decoded into extras.image_rgb.
// Returns false only on a bad numeric flag or unreadable --image/--grammar file.
bool apply_cli_gen_flags(int argc, char **argv, GenParams &gp, CliGenExtras &ex, std::string &err);

// Apply `micro-vllm video` flags onto hp. Repeatable --ref-image.
// Width/height/frames/steps/layers apply only if >0. --denoise-reuse if N>=0.
// Returns false only on a bad numeric flag.
bool apply_cli_video_flags(int argc, char **argv, H3GenParams &hp, std::string &err);

} // namespace mvllm
