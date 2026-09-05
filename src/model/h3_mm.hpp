#pragma once

#include "h3_text.hpp"
#include "h3_vision.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {

class Tokenizer;

// Official multimodal specials (h3_multimodal.c).
constexpr uint32_t kH3VisionStart = 151652;
constexpr uint32_t kH3VisionEnd = 151653;
constexpr uint32_t kH3ImagePad = 151655;
constexpr uint32_t kH3VideoPad = 151656;

enum class H3PresKind { Image = 1, Video = 2, Audio = 3 };

struct H3RefPres {
    H3PresKind kind = H3PresKind::Image;
    bool has_audio = false;
    const H3VisionOut *vision = nullptr;
    int vision_count = 0;
    const double *timestamps = nullptr;
};

struct H3MmSeq {
    std::vector<int> ids;
    std::vector<H3VisionSpan> spans;
    std::vector<uint32_t> positions; // 3 * seq, axis-major [axis, t]
    std::vector<uint8_t> tags;       // 0 = vision wrapper, 1 = text
};

// FL2VA / image-only anchors: "<Picture n>: " + vision_start + pads + vision_end, then prompt.
bool h3_mm_build_fl2va(const std::string &prompt, const H3VisionOut *images, int image_count,
                       const Tokenizer *tok, int vocab, H3MmSeq &out);

// Ordered Ref2VA labels. Audio-only refs require at least one visual. Video blocks use
// "<Video n>: " once then "<%.1f seconds>" per block.
bool h3_mm_build_ref2va(const std::string &prompt, const H3RefPres *refs, int ref_count,
                        const Tokenizer *tok, int vocab, H3MmSeq &out);

// mRoPE: text positions are sequential; each vision span uses (base, base+row, base+col)
// on the merged grid. Offset after a span is length_max - tokens.
bool h3_mm_mrope_positions(const H3VisionOut *const *visions, const H3VisionSpan *spans,
                           int span_count, int seq, uint32_t *positions);

} // namespace mvllm
