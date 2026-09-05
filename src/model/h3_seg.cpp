#include "h3_seg.hpp"

#include <cstring>

namespace mvllm {
namespace {

struct SegName {
    H3SegKind kind;
    const char *name;
};

constexpr SegName kSegNames[] = {
    {H3SegKind::Text, "text"},
    {H3SegKind::Cond, "cond"},
    {H3SegKind::RefImage, "ref_img"},
    {H3SegKind::RefAudio, "ref_audio"},
    {H3SegKind::Audio, "audio"},
    {H3SegKind::Video, "video"},
};

void write_int(int *dst, int value) {
    if (dst)
        *dst = value;
}

} // namespace

const char *h3_segment_name(H3SegKind kind) {
    for (const SegName &entry : kSegNames) {
        if (entry.kind == kind)
            return entry.name;
    }
    return "unknown";
}

bool h3_seg_kind_from_name(const char *name, H3SegKind *out) {
    if (!name || !out)
        return false;
    for (const SegName &entry : kSegNames) {
        if (std::strcmp(name, entry.name) == 0) {
            *out = entry.kind;
            return true;
        }
    }
    return false;
}

void h3_layout_sig_pack(int text_len, int latent_t, int latent_h, int latent_w, int audio_t,
                        int sig[kH3SigN]) {
    if (!sig)
        return;
    sig[0] = text_len;
    sig[1] = latent_t;
    sig[2] = latent_h;
    sig[3] = latent_w;
    sig[4] = audio_t;
}

void h3_layout_sig_unpack(const int sig[kH3SigN], int *text_len, int *latent_t, int *latent_h,
                          int *latent_w, int *audio_t) {
    if (!sig)
        return;
    write_int(text_len, sig[0]);
    write_int(latent_t, sig[1]);
    write_int(latent_h, sig[2]);
    write_int(latent_w, sig[3]);
    write_int(audio_t, sig[4]);
}

bool h3_layout_sig_equal(const int *a, const int *b) {
    if (!a || !b)
        return false;
    return std::memcmp(a, b, sizeof(int) * static_cast<size_t>(kH3SigN)) == 0;
}

} // namespace mvllm
