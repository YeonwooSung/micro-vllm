#include "h3_mm.hpp"
#include "../tok/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace mvllm {
namespace {

bool tokenize_append(const Tokenizer *tok, const std::string &text, int vocab,
                     std::vector<int> &ids) {
    std::vector<int> tmp;
    if (tok && tok->loaded()) {
        if (tok->encode(text, tmp) != Status::Ok)
            return false;
    } else {
        h3_text_ids_from_prompt(text, vocab, tmp);
    }
    ids.insert(ids.end(), tmp.begin(), tmp.end());
    return true;
}

bool append_vision(std::vector<int> &ids, H3VisionSpan &span, const H3VisionOut &vision,
                   uint32_t pad) {
    if (vision.tokens <= 0)
        return false;
    const int ow = vision.out_width;
    if (ow > 0 &&
        vision.merged.size() < static_cast<size_t>(vision.tokens) * static_cast<size_t>(ow))
        return false;
    ids.push_back(static_cast<int>(kH3VisionStart));
    span.start = static_cast<int>(ids.size());
    span.tokens = vision.tokens;
    span.embeddings = vision.merged.data();
    for (int i = 0; i < kH3TextDeepstacks; ++i)
        span.deepstack[i] = vision.deepstack[i].empty() ? nullptr : vision.deepstack[i].data();
    for (int t = 0; t < vision.tokens; ++t)
        ids.push_back(static_cast<int>(pad));
    ids.push_back(static_cast<int>(kH3VisionEnd));
    return true;
}

bool fill_tags_and_mrope(const std::vector<const H3VisionOut *> &visions, H3MmSeq &out) {
    const int seq = static_cast<int>(out.ids.size());
    if (seq < 1)
        return false;
    out.tags.assign(static_cast<size_t>(seq), 1);
    for (const H3VisionSpan &sp : out.spans) {
        if (sp.start < 1)
            return false;
        const int first = sp.start - 1;
        const int count = sp.tokens + 2;
        if (first > seq || count > seq - first)
            return false;
        std::fill(out.tags.begin() + first, out.tags.begin() + first + count,
                  static_cast<uint8_t>(0));
    }
    out.positions.assign(static_cast<size_t>(3) * static_cast<size_t>(seq), 0);
    return h3_mm_mrope_positions(visions.data(), out.spans.data(),
                                 static_cast<int>(out.spans.size()), seq, out.positions.data());
}

} // namespace

bool h3_mm_build_fl2va(const std::string &prompt, const H3VisionOut *images, int image_count,
                       const Tokenizer *tok, int vocab, H3MmSeq &out) {
    out = H3MmSeq{};
    if (prompt.empty() || !images || image_count < 1)
        return false;
    std::vector<const H3VisionOut *> visions;
    visions.reserve(static_cast<size_t>(image_count));
    for (int i = 0; i < image_count; ++i) {
        char prefix[64];
        const int n = std::snprintf(prefix, sizeof(prefix), "<Picture %d>: ", i + 1);
        if (n < 0 || static_cast<size_t>(n) >= sizeof(prefix) ||
            !tokenize_append(tok, prefix, vocab, out.ids)) {
            out = H3MmSeq{};
            return false;
        }
        out.spans.push_back({});
        if (!append_vision(out.ids, out.spans.back(), images[i], kH3ImagePad)) {
            out = H3MmSeq{};
            return false;
        }
        visions.push_back(&images[i]);
    }
    if (!tokenize_append(tok, prompt, vocab, out.ids) || !fill_tags_and_mrope(visions, out)) {
        out = H3MmSeq{};
        return false;
    }
    return true;
}

bool h3_mm_build_ref2va(const std::string &prompt, const H3RefPres *refs, int ref_count,
                        const Tokenizer *tok, int vocab, H3MmSeq &out) {
    out = H3MmSeq{};
    if (prompt.empty() || !refs || ref_count < 1)
        return false;
    int span_count = 0;
    bool have_visual = false;
    for (int i = 0; i < ref_count; ++i) {
        const H3RefPres &ref = refs[i];
        if (ref.kind == H3PresKind::Image) {
            if (ref.has_audio || ref.vision_count != 1 || !ref.vision)
                return false;
            ++span_count;
            have_visual = true;
        } else if (ref.kind == H3PresKind::Video) {
            if (!ref.vision || ref.vision_count < 1 || !ref.timestamps)
                return false;
            span_count += ref.vision_count;
            have_visual = true;
        } else if (ref.kind == H3PresKind::Audio) {
            if (!ref.has_audio || ref.vision || ref.vision_count != 0)
                return false;
        } else {
            return false;
        }
    }
    if (!have_visual || span_count < 1)
        return false;

    std::vector<const H3VisionOut *> visions;
    visions.reserve(static_cast<size_t>(span_count));
    int image_ord = 0, video_ord = 0, audio_ord = 0;
    for (int i = 0; i < ref_count; ++i) {
        const H3RefPres &ref = refs[i];
        char prefix[96];
        if (ref.has_audio) {
            const int n = std::snprintf(prefix, sizeof(prefix), "<Audio %d>: ", ++audio_ord);
            if (n < 0 || static_cast<size_t>(n) >= sizeof(prefix) ||
                !tokenize_append(tok, prefix, vocab, out.ids)) {
                out = H3MmSeq{};
                return false;
            }
        }
        if (ref.kind == H3PresKind::Audio)
            continue;
        if (ref.kind == H3PresKind::Image) {
            const int n = std::snprintf(prefix, sizeof(prefix), "<Picture %d>: ", ++image_ord);
            if (n < 0 || static_cast<size_t>(n) >= sizeof(prefix) ||
                !tokenize_append(tok, prefix, vocab, out.ids)) {
                out = H3MmSeq{};
                return false;
            }
            out.spans.push_back({});
            if (!append_vision(out.ids, out.spans.back(), *ref.vision, kH3ImagePad)) {
                out = H3MmSeq{};
                return false;
            }
            visions.push_back(ref.vision);
            continue;
        }
        ++video_ord;
        for (int block = 0; block < ref.vision_count; ++block) {
            if (!std::isfinite(ref.timestamps[block]) || ref.timestamps[block] < 0.0) {
                out = H3MmSeq{};
                return false;
            }
            if (block == 0) {
                const int n = std::snprintf(prefix, sizeof(prefix), "<Video %d>: ", video_ord);
                if (n < 0 || static_cast<size_t>(n) >= sizeof(prefix) ||
                    !tokenize_append(tok, prefix, vocab, out.ids)) {
                    out = H3MmSeq{};
                    return false;
                }
            }
            const int n =
                std::snprintf(prefix, sizeof(prefix), "<%.1f seconds>", ref.timestamps[block]);
            if (n < 0 || static_cast<size_t>(n) >= sizeof(prefix) ||
                !tokenize_append(tok, prefix, vocab, out.ids)) {
                out = H3MmSeq{};
                return false;
            }
            out.spans.push_back({});
            if (!append_vision(out.ids, out.spans.back(), ref.vision[block], kH3VideoPad)) {
                out = H3MmSeq{};
                return false;
            }
            visions.push_back(&ref.vision[block]);
        }
    }
    if (static_cast<int>(out.spans.size()) != span_count ||
        !tokenize_append(tok, prompt, vocab, out.ids) || !fill_tags_and_mrope(visions, out)) {
        out = H3MmSeq{};
        return false;
    }
    return true;
}

bool h3_mm_mrope_positions(const H3VisionOut *const *visions, const H3VisionSpan *spans,
                           int span_count, int seq, uint32_t *positions) {
    if (!positions || seq < 1)
        return false;
    if (span_count < 1 || !spans) {
        for (int a = 0; a < 3; ++a)
            for (int t = 0; t < seq; ++t)
                positions[a * seq + t] = static_cast<uint32_t>(t);
        return true;
    }
    if (!visions)
        return false;
    int64_t offset = 0;
    for (int image = 0; image < span_count; ++image) {
        const H3VisionOut *vision = visions[image];
        const int start = spans[image].start;
        const int tokens = spans[image].tokens;
        const int end = start + tokens;
        if (!vision || start < 0 || tokens < 0 || start > seq || end > seq ||
            vision->grid_h < 2 || vision->grid_w < 2 || (vision->grid_h % 2) ||
            (vision->grid_w % 2))
            return false;
        if (image == 0) {
            for (int a = 0; a < 3; ++a)
                for (int t = 0; t < start; ++t)
                    positions[a * seq + t] = static_cast<uint32_t>(t);
        }
        const int merged_h = vision->grid_h / 2;
        const int merged_w = vision->grid_w / 2;
        const int length_max = std::max(merged_h, merged_w);
        const int64_t base = static_cast<int64_t>(start) + offset;
        const int64_t next = static_cast<int64_t>(start) + length_max + offset;
        if (base < 0 || next < 0)
            return false;
        for (int t = start; t < end; ++t)
            positions[t] = static_cast<uint32_t>(base);
        int cursor = start;
        for (int row = 0; row < merged_h; ++row) {
            for (int col = 0; col < merged_w; ++col) {
                if (cursor >= end)
                    return false;
                positions[seq + cursor] = static_cast<uint32_t>(base + row);
                positions[2 * seq + cursor] = static_cast<uint32_t>(base + col);
                ++cursor;
            }
        }
        if (cursor != end)
            return false;
        for (int a = 0; a < 3; ++a)
            for (int t = end; t < seq; ++t)
                positions[a * seq + t] = static_cast<uint32_t>(next + (t - end));
        offset += static_cast<int64_t>(length_max) - tokens;
    }
    return true;
}

} // namespace mvllm
