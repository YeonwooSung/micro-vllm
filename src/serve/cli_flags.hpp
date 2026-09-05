#pragma once

#include "../model/family.hpp"

#include <string>
#include <vector>

namespace mvllm {

// Extra buffers owned by the CLI so GenParams pointers stay valid.
struct CliGenExtras {
    std::vector<float> image_rgb;
};

// Apply `micro-vllm generate` flags onto gp. Repeatable --stop / --stop-id /
// --logit-bias ID:DELTA.
// --max-tokens/--max-new-tokens/--n/-n sets gp.max_new_tokens if >0.
// --logprobs/--top-logprobs N sets gp.logprobs; 0 = off. --no-logprobs
// clears gp.logprobs.
// --json sets json_object_gbnf(). --no-json / --no-grammar clear gp.grammar.
// --image PATH is decoded into extras.image_rgb.
// --persist/--kv PATH, --persist-ver 1..3, --prefix-bytes/--prefix-reuse N>=0,
// --cache-slot/--slot N sets gp.cache_slot if N>=0.
// --reasoning-effort none|low|medium|high.
// --tool-choice/--function-call auto|none|required|NAME.
// --tools PATH (JSON file; OpenAI-style tools array or {"tools":[...]}).
// --no-tools clears gp.tools.
// --raw/--no-chat clears gp.apply_template.
// --eos-only. --no-eos-only clears gp.eos_only. Returns false on a bad numeric
// flag or unreadable --image/--grammar/--tools.
bool apply_cli_gen_flags(int argc, char **argv, GenParams &gp, CliGenExtras &ex, std::string &err);

// Apply `micro-vllm video` flags onto hp. Repeatable --ref-image.
// Width/height/frames/steps/layers apply only if >0. --denoise-reuse if N>=0.
// Returns false only on a bad numeric flag.
bool apply_cli_video_flags(int argc, char **argv, H3GenParams &hp, std::string &err);

} // namespace mvllm
