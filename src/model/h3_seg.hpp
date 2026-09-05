#pragma once

#include "h3_layout.hpp"

namespace mvllm {

constexpr int kH3SigN = 5;

const char *h3_segment_name(H3SegKind kind);
bool h3_seg_kind_from_name(const char *name, H3SegKind *out);

// Official slots: text_len, latent_t, latent_h, latent_w, audio_t.
void h3_layout_sig_pack(int text_len, int latent_t, int latent_h, int latent_w, int audio_t,
                        int sig[kH3SigN]);
void h3_layout_sig_unpack(const int sig[kH3SigN], int *text_len, int *latent_t, int *latent_h,
                          int *latent_w, int *audio_t);
bool h3_layout_sig_equal(const int *a, const int *b);

} // namespace mvllm
