#include "family.hpp"
#include "h3_audio_vae.hpp"
#include "h3_layout.hpp"
#include "h3_mm.hpp"
#include "h3_text.hpp"
#include "h3_vae.hpp"
#include "h3_tok.hpp"
#include "h3_vision.hpp"
#include "../gpu/backend.hpp"
#include "../gpu/metal_h3.hpp"
#include "../gpu/vk_ops.hpp"
#include "../io/av_mux.hpp"
#include "../io/image.hpp"
#include "../io/safetensors.hpp"
#include "../quant/quant.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace mvllm {
namespace {

const char *kH3StreamSuffix[4] = {
    "attn.qkv_proj.weight",
    "attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
};

bool ends_with(const std::string &s, const char *suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

uint64_t splitmix64(uint64_t &s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

float gauss01(uint64_t &s) {
    float u = static_cast<float>(splitmix64(s) >> 40) * (1.f / 16777216.f);
    float v = static_cast<float>(splitmix64(s) >> 40) * (1.f / 16777216.f);
    if (u < 1e-7f)
        u = 1e-7f;
    return std::sqrt(-2.f * std::log(u)) * std::cos(6.283185307179586f * v);
}

struct LoadedRgb {
    std::vector<float> rgb;
    int h = 0;
    int w = 0;
};

bool load_rgb(const std::string &path, const float *rgb, int w, int h, LoadedRgb &out) {
    if (rgb && w >= 1 && h >= 1) {
        out.rgb.assign(rgb, rgb + static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
        out.w = w;
        out.h = h;
        return true;
    }
    if (path.empty())
        return false;
    std::string e;
    return decode_image_url(path, out.rgb, out.w, out.h, e) == Status::Ok && out.w > 0 && out.h > 0;
}

void resize_nearest(const float *src, int sh, int sw, int dh, int dw, std::vector<float> &dst) {
    dst.assign(static_cast<size_t>(std::max(dh, 0)) * static_cast<size_t>(std::max(dw, 0)) * 3, 0.f);
    if (!src || sh < 1 || sw < 1 || dh < 1 || dw < 1)
        return;
    for (int y = 0; y < dh; ++y) {
        int sy = static_cast<int>(static_cast<int64_t>(y) * sh / dh);
        if (sy >= sh)
            sy = sh - 1;
        for (int x = 0; x < dw; ++x) {
            int sx = static_cast<int>(static_cast<int64_t>(x) * sw / dw);
            if (sx >= sw)
                sx = sw - 1;
            const float *s = src + (static_cast<size_t>(sy) * sw + sx) * 3;
            float *d = dst.data() + (static_cast<size_t>(y) * dw + x) * 3;
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
        }
    }
}

void fit_multiple32(const float *src, int sh, int sw, std::vector<float> &dst, int &oh, int &ow) {
    oh = (sh / 32) * 32;
    ow = (sw / 32) * 32;
    if (oh < 32)
        oh = 32;
    if (ow < 32)
        ow = 32;
    dst.assign(static_cast<size_t>(oh) * static_cast<size_t>(ow) * 3, 0.f);
    if (!src || sh < 1 || sw < 1)
        return;
    const int y0 = (sh - oh) / 2;
    const int x0 = (sw - ow) / 2;
    for (int y = 0; y < oh; ++y) {
        const int sy = y + y0;
        if (sy < 0 || sy >= sh)
            continue;
        for (int x = 0; x < ow; ++x) {
            const int sx = x + x0;
            if (sx < 0 || sx >= sw)
                continue;
            const float *s = src + (static_cast<size_t>(sy) * sw + sx) * 3;
            float *d = dst.data() + (static_cast<size_t>(y) * ow + x) * 3;
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
        }
    }
}

void expand_patch_rows(const float *rows, int tokens, int patch, int hidden, float *out) {
    for (int i = 0; i < tokens; ++i)
        for (int d = 0; d < hidden; ++d)
            out[static_cast<size_t>(i) * hidden + d] =
                rows[static_cast<size_t>(i) * patch + (d % patch)];
}

struct AccTimer {
    double &acc;
    std::chrono::steady_clock::time_point t0;
    explicit AccTimer(double &a) : acc(a), t0(std::chrono::steady_clock::now()) {}
    ~AccTimer() {
        acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

} // namespace


class H3Engine final : public FamilyEngine {
public:
    Family family() const override { return Family::H3; }
    const ModelConfig &config() const override { return cfg_; }

    Status load(const std::string &model_dir, const RuntimeConfig &rt, std::string &err) override {
        rt_ = rt;
        model_dir_ = model_dir;
        Status st = load_model_config(model_dir, cfg_, err);
        if (st != Status::Ok) {
            // MiniMax-H3 directories may have no HF config.json; still accept.
            cfg_.family = Family::H3;
            apply_family_defaults(cfg_);
            err.clear();
        }
        cfg_.family = Family::H3;
        apply_family_defaults(cfg_);

        st = load_checkpoint(model_dir, err);
        if (st == Status::NotFound) {
            err.clear();
            st = write_synthetic_blocks(model_dir, err);
        }
        if (st != Status::Ok)
            return st;
        {
            std::string verr;
            Status vst = vae_.load(model_dir, cfg_.h3, verr);
            if (vst == Status::NotFound)
                verr.clear();
        }
        {
            std::string terr;
            text_.load(model_dir, terr);
        }
        {
            std::string verr;
            vision_.load(model_dir, verr);
        }
        {
            std::string aerr;
            avae_.load(model_dir, aerr);
        }
        {
            std::string terr;
            h3_tok_load(model_dir, h3_tok_, terr);
        }
        loaded_ = true;
        metal_h3::init();
        vk_ops::init();
        return Status::Ok;
    }

    Status generate(const std::vector<int> &, const GenParams &, GenResult &,
                    std::string &err) override {
        err = "H3 is a video/audio DiT — use /v1/videos/generations";
        return Status::Unsupported;
    }

    Status generate_video(const H3GenParams &hp, H3GenResult &out, std::string &err) override {
        if (!loaded_) {
            err = "h3 not loaded";
            return Status::InvalidArgument;
        }
        const bool want_ref =
            !hp.ref_images.empty() || (hp.ref_rgb && hp.ref_w >= 32 && hp.ref_h >= 32);
        const bool want_fl = !hp.first_frame.empty() || !hp.last_frame.empty() || hp.first_rgb ||
                             hp.last_rgb;
        if (want_ref && want_fl) {
            err = "keyframes and references are mutually exclusive";
            return Status::InvalidArgument;
        }
        bool using_ref2va = false;
        bool ref2va_missing = false;
        if (want_ref) {
            if (!ref2va_dir_.empty()) {
                if (transformer_dir_ != ref2va_dir_) {
                    const std::string keep = transformer_dir_;
                    std::string berr;
                    if (bind_transformer(ref2va_dir_, berr) == Status::Ok) {
                        using_ref2va = true;
                    } else {
                        ref2va_missing = true;
                        if (!keep.empty()) {
                            std::string rerr;
                            bind_transformer(keep, rerr);
                        }
                    }
                } else {
                    using_ref2va = true;
                }
            } else {
                ref2va_missing = true;
            }
        } else if (!fl2va_dir_.empty() && transformer_dir_ != fl2va_dir_) {
            std::string rerr;
            bind_transformer(fl2va_dir_, rerr);
        }

        std::vector<LoadedRgb> fl_imgs;
        std::vector<LoadedRgb> ref_imgs;
        bool have_first = false;
        bool have_last = false;
        if (want_fl) {
            LoadedRgb first, last;
            if (load_rgb(hp.first_frame, hp.first_rgb, hp.first_w, hp.first_h, first)) {
                fl_imgs.push_back(std::move(first));
                have_first = true;
            }
            if (load_rgb(hp.last_frame, hp.last_rgb, hp.last_w, hp.last_h, last)) {
                fl_imgs.push_back(std::move(last));
                have_last = true;
            }
        }
        if (want_ref) {
            LoadedRgb mem;
            if (hp.ref_rgb && hp.ref_w >= 32 && hp.ref_h >= 32 &&
                load_rgb({}, hp.ref_rgb, hp.ref_w, hp.ref_h, mem))
                ref_imgs.push_back(std::move(mem));
            for (const std::string &p : hp.ref_images) {
                LoadedRgb im;
                if (load_rgb(p, nullptr, 0, 0, im))
                    ref_imgs.push_back(std::move(im));
            }
        }
        const int picture_n = static_cast<int>(ref_imgs.size());

        int layers = hp.dit_layers > 0 ? hp.dit_layers : cfg_.h3.dit_layers;
        if (layers > cfg_.h3.dit_layers)
            layers = cfg_.h3.dit_layers;
        int steps = hp.steps > 0 ? hp.steps : cfg_.h3.default_steps;
        int reuse = hp.denoise_reuse > 0 ? hp.denoise_reuse : 1;
        int evals = (steps + reuse - 1) / reuse;

        const int hidden = cfg_.h3.hidden > 0 ? cfg_.h3.hidden : 5376;
        const int inner = cfg_.h3.inner > 0 ? cfg_.h3.inner : 7168;
        const int ffn = cfg_.h3.ffn > 0 ? cfg_.h3.ffn : 14336;
        const int hd = cfg_.h3.head_dim > 0 ? cfg_.h3.head_dim : 128;
        const int64_t qkv_b = static_cast<int64_t>(inner) * 3 * hidden * 2;
        const int64_t out_b = static_cast<int64_t>(hidden) * inner * 2;
        const int64_t fc1_b = static_cast<int64_t>(ffn) * 2 * hidden * 2;
        const int64_t fc2_b = static_cast<int64_t>(hidden) * ffn * 2;
        const int64_t expect = qkv_b + out_b + fc1_b + fc2_b;

        const int req_w = hp.width > 0 ? hp.width : cfg_.h3.default_width;
        const int req_h = hp.height > 0 ? hp.height : cfg_.h3.default_height;
        const int req_f = hp.frames > 0 ? hp.frames : cfg_.h3.default_frames;
        H3VaeGeom vg = h3_vae_geom(req_w, req_h, req_f, cfg_.h3.vae_spatial, cfg_.h3.vae_latent_ch);
        const int C = vg.latent_ch > 0 ? vg.latent_ch : 24;
        const int nlat = std::max(vg.latent_t * vg.latent_h * vg.latent_w, 1);
        const int64_t zN = static_cast<int64_t>(C) * nlat;
        std::vector<float> z(static_cast<size_t>(zN), 0.f);
        uint64_t rng = hp.seed ? hp.seed : 42ull;
        for (int64_t i = 0; i < zN; ++i)
            z[static_cast<size_t>(i)] = gauss01(rng);

        const bool can_patch = vg.latent_h >= 2 && vg.latent_w >= 2 && (vg.latent_h % 2 == 0) &&
                               (vg.latent_w % 2 == 0);
        const int patch = 96;
        int tokens = 0;
        std::vector<float> rows;
        std::vector<float> latent;
        {
            AccTimer t(t_emm_);
            if (can_patch) {
                tokens = vg.latent_t * (vg.latent_h / 2) * (vg.latent_w / 2);
                rows.assign(static_cast<size_t>(tokens) * patch, 0.f);
                h3_dit_patchify(z.data(), C, vg.latent_t, vg.latent_h, vg.latent_w, rows.data());
            } else {
                tokens = std::max(nlat, 1);
                rows.assign(static_cast<size_t>(tokens) * patch, 0.f);
                for (int i = 0; i < tokens; ++i)
                    for (int d = 0; d < patch; ++d)
                        rows[static_cast<size_t>(i) * patch + d] =
                            z[static_cast<size_t>(d % C) * nlat + i];
            }
            const int cap = 256;
            if (tokens > cap) {
                const int keep = cap;
                std::vector<float> slim(static_cast<size_t>(keep) * patch);
                for (int i = 0; i < keep; ++i) {
                    int src = i * tokens / keep;
                    std::memcpy(slim.data() + static_cast<size_t>(i) * patch,
                                rows.data() + static_cast<size_t>(src) * patch,
                                static_cast<size_t>(patch) * sizeof(float));
                }
                rows.swap(slim);
                tokens = keep;
            }
            latent.assign(static_cast<size_t>(tokens) * hidden, 0.f);
            for (int i = 0; i < tokens; ++i)
                for (int d = 0; d < hidden; ++d)
                    latent[static_cast<size_t>(i) * hidden + d] =
                        rows[static_cast<size_t>(i) * patch + (d % patch)];
        }

        const int audio_t = h3_audio_t(vg.frames);
        const int AC = kH3AudioChannels;
        std::vector<float> az(static_cast<size_t>(AC) * kH3AudioStereo * audio_t, 0.f);
        bool audio_from_ref = false;
        if (hp.audio_pcm && hp.audio_samples > 0) {
            int at = 0;
            avae_.encode(hp.audio_pcm, hp.audio_samples, az, at);
            audio_from_ref = true;
        } else if (!hp.audio_path.empty()) {
            std::vector<float> pcm;
            int ach = 0, asn = 0, ar = 0;
            std::string aerr;
            if (h3_read_wav(hp.audio_path, pcm, ach, asn, ar, aerr) == Status::Ok && asn > 0) {
                int at = 0;
                avae_.encode(pcm.data(), asn, az, at);
                audio_from_ref = true;
            }
        }
        if (!audio_from_ref) {
            uint64_t arng = rng ^ 0xA5A5A5A5ull;
            for (float &v : az)
                v = gauss01(arng);
        }
        int audio_rows = audio_t * kH3AudioStereo;
        std::vector<float> arows(static_cast<size_t>(audio_rows) * AC, 0.f);
        std::vector<float> alatent(static_cast<size_t>(audio_rows) * hidden, 0.f);
        {
            AccTimer t(t_emm_);
            h3_dit_pack_audio(az.data(), AC, audio_t, arows.data());
            for (int i = 0; i < audio_rows; ++i)
                for (int d = 0; d < hidden; ++d)
                    alatent[static_cast<size_t>(i) * hidden + d] =
                        arows[static_cast<size_t>(i) * AC + (d % AC)];
        }

        int text_tokens = 0;
        std::vector<float> tproj;
        if (text_.ready() && !hp.prompt.empty()) {
            std::vector<float> th;
            const std::vector<LoadedRgb> *vis_src = nullptr;
            if (want_ref && !ref_imgs.empty())
                vis_src = &ref_imgs;
            else if (want_fl && !fl_imgs.empty())
                vis_src = &fl_imgs;
            if (vision_.ready() && vis_src && !vis_src->empty()) {
                std::vector<H3VisionOut> vouts(vis_src->size());
                bool mm_ok = true;
                for (size_t i = 0; i < vis_src->size(); ++i) {
                    std::vector<float> vis;
                    int vh = 0, vw = 0;
                    fit_multiple32(vis_src->at(i).rgb.data(), vis_src->at(i).h, vis_src->at(i).w, vis,
                                   vh, vw);
                    vision_.encode(vis.data(), 1, vh, vw, vouts[i]);
                    if (vouts[i].tokens < 1)
                        mm_ok = false;
                }
                H3MmSeq seq;
                if (mm_ok && want_ref) {
                    std::vector<H3RefPres> pres(vouts.size());
                    for (size_t i = 0; i < vouts.size(); ++i) {
                        pres[i].kind = H3PresKind::Image;
                        pres[i].vision = &vouts[i];
                        pres[i].vision_count = 1;
                    }
                    mm_ok = h3_mm_build_ref2va(hp.prompt, pres.data(), static_cast<int>(pres.size()),
                                               nullptr, text_.config().vocab, seq);
                } else if (mm_ok) {
                    mm_ok = h3_mm_build_fl2va(hp.prompt, vouts.data(), static_cast<int>(vouts.size()),
                                              nullptr, text_.config().vocab, seq);
                }
                if (mm_ok && !seq.ids.empty())
                    text_.encode_mm(seq.ids, seq.spans.empty() ? nullptr : seq.spans.data(),
                                    static_cast<int>(seq.spans.size()),
                                    seq.positions.empty() ? nullptr : seq.positions.data(),
                                    seq.tags.empty() ? nullptr : seq.tags.data(), th);
            }
            if (th.empty()) {
                std::vector<int> tids;
                h3_text_ids_from_prompt(hp.prompt, text_.config().vocab, tids);
                text_.encode(tids, th);
            }
            const int thid = text_.config().hidden;
            if (!th.empty() && thid > 0) {
                text_tokens = static_cast<int>(th.size() / static_cast<size_t>(thid));
                if (text_tokens > 16)
                    text_tokens = 16;
                tproj.assign(static_cast<size_t>(text_tokens) * hidden, 0.f);
                AccTimer tm(t_emm_);
                if (!cond_w_.empty() && static_cast<int>(cond_w_.size()) >= hidden * thid) {
                    quant::matmul_f32(tproj.data(), th.data(), cond_w_.data(), text_tokens, thid,
                                      hidden);
                    if (static_cast<int>(cond_b_.size()) >= hidden) {
                        for (int t = 0; t < text_tokens; ++t)
                            for (int d = 0; d < hidden; ++d)
                                tproj[static_cast<size_t>(t) * hidden + d] +=
                                    cond_b_[static_cast<size_t>(d)];
                    }
                } else {
                    for (int t = 0; t < text_tokens; ++t)
                        for (int d = 0; d < hidden; ++d)
                            tproj[static_cast<size_t>(t) * hidden + d] =
                                th[static_cast<size_t>(t) * thid + (d % thid)];
                }
            }
        }

        int keyframes[2] = {0, 0};
        int n_keyframes = 0;
        if (want_fl) {
            if (have_first)
                keyframes[n_keyframes++] = 0;
            if (have_last)
                keyframes[n_keyframes++] = vg.frames - 1;
        }
        std::vector<H3LayoutRef> layout_refs;
        if (want_ref) {
            layout_refs.reserve(ref_imgs.size());
            for (const LoadedRgb &im : ref_imgs) {
                std::vector<float> fit;
                int fh = 0, fw = 0;
                fit_multiple32(im.rgb.data(), im.h, im.w, fit, fh, fw);
                H3VaeGeom ig =
                    h3_vae_geom(fw, fh, 1, vg.spatial, C);
                H3LayoutRef r;
                r.kind = H3SegKind::RefImage;
                r.latent_h = ig.latent_h;
                r.latent_w = ig.latent_w;
                layout_refs.push_back(r);
            }
        }
        H3Layout layout;
        const bool built_layout = h3_layout_build(
            text_tokens, vg.latent_t, vg.latent_h, vg.latent_w, audio_t, vg.frames, layout,
            n_keyframes > 0 ? keyframes : nullptr, n_keyframes,
            layout_refs.empty() ? nullptr : layout_refs.data(),
            static_cast<int>(layout_refs.size()));

        int cond_rows = 0;
        std::vector<float> clatent;
        if (built_layout && layout.img_cond_rows > 0) {
            const std::vector<LoadedRgb> *cond_src = want_ref ? &ref_imgs : &fl_imgs;
            std::vector<float> packed_cond;
            int packed_rows = 0;
            bool pack_ok = cond_src && !cond_src->empty();
            for (size_t i = 0; pack_ok && i < cond_src->size(); ++i) {
                const LoadedRgb &im = cond_src->at(i);
                int eh = 0, ew = 0;
                std::vector<float> ergb;
                if (want_fl) {
                    eh = vg.height;
                    ew = vg.width;
                    resize_nearest(im.rgb.data(), im.h, im.w, eh, ew, ergb);
                } else {
                    fit_multiple32(im.rgb.data(), im.h, im.w, ergb, eh, ew);
                }
                H3VaeGeom eg = h3_vae_geom(ew, eh, 1, vg.spatial, C);
                if (want_fl) {
                    eg.latent_h = vg.latent_h;
                    eg.latent_w = vg.latent_w;
                    eg.width = vg.width;
                    eg.height = vg.height;
                }
                eg.frames = 1;
                eg.latent_t = 1;
                const int lh = eg.latent_h;
                const int lw = eg.latent_w;
                if (lh < 2 || lw < 2 || (lh % 2) || (lw % 2)) {
                    pack_ok = false;
                    break;
                }
                const int cells = (lh / 2) * (lw / 2);
                std::vector<float> zc(static_cast<size_t>(C) * lh * lw, 0.f);
                vae_.encode(ergb.data(), eg, zc.data());
                std::vector<float> prows(static_cast<size_t>(cells) * patch, 0.f);
                AccTimer t(t_emm_);
                if (h3_dit_patchify(zc.data(), C, 1, lh, lw, prows.data()) != cells * patch) {
                    pack_ok = false;
                    break;
                }
                packed_cond.insert(packed_cond.end(), prows.begin(), prows.end());
                packed_rows += cells;
            }
            if (pack_ok && packed_rows == layout.img_cond_rows && packed_rows > 0) {
                AccTimer t(t_emm_);
                cond_rows = packed_rows;
                clatent.assign(static_cast<size_t>(cond_rows) * hidden, 0.f);
                expand_patch_rows(packed_cond.data(), cond_rows, patch, hidden, clatent.data());
            }
        }

        if (text_tokens > 0 || cond_rows > 0) {
            AccTimer t(t_emm_);
            std::vector<float> packed(
                static_cast<size_t>(text_tokens + cond_rows + audio_rows + tokens) * hidden);
            float *dst = packed.data();
            if (text_tokens > 0) {
                std::memcpy(dst, tproj.data(),
                            static_cast<size_t>(text_tokens) * hidden * sizeof(float));
                dst += static_cast<size_t>(text_tokens) * hidden;
            }
            if (cond_rows > 0) {
                std::memcpy(dst, clatent.data(),
                            static_cast<size_t>(cond_rows) * hidden * sizeof(float));
                dst += static_cast<size_t>(cond_rows) * hidden;
            }
            if (audio_rows > 0) {
                std::memcpy(dst, alatent.data(),
                            static_cast<size_t>(audio_rows) * hidden * sizeof(float));
                dst += static_cast<size_t>(audio_rows) * hidden;
            }
            std::memcpy(dst, latent.data(), static_cast<size_t>(tokens) * hidden * sizeof(float));
            latent.swap(packed);
        } else if (audio_rows > 0) {
            AccTimer t(t_emm_);
            std::vector<float> packed(static_cast<size_t>(audio_rows + tokens) * hidden);
            std::memcpy(packed.data(), alatent.data(),
                        static_cast<size_t>(audio_rows) * hidden * sizeof(float));
            std::memcpy(packed.data() + static_cast<size_t>(audio_rows) * hidden, latent.data(),
                        static_cast<size_t>(tokens) * hidden * sizeof(float));
            latent.swap(packed);
        }
        const int seq = static_cast<int>(latent.size() / static_cast<size_t>(std::max(hidden, 1)));
        const int audio_off = text_tokens + cond_rows;
        const int video_off = text_tokens + cond_rows + audio_rows;
        const bool layout_ok = built_layout && layout.seq_len == seq;
        if (!layout_ok) {
            layout = {};
            layout.seq_len = seq;
            layout.positions.resize(static_cast<size_t>(seq));
            for (int i = 0; i < seq; ++i)
                layout.positions[static_cast<size_t>(i)] = H3Position{static_cast<float>(i), 0.f, 0.f};
        }
        float inv_freq[kH3RopeFreqs];
        if (static_cast<int>(rope_inv_.size()) >= kH3RopeFreqs)
            std::memcpy(inv_freq, rope_inv_.data(), sizeof(inv_freq));
        else
            h3_dit_default_inv_freq(inv_freq);
        const float spatial_scale =
            (vg.width == 256 && vg.height == 256) ? 0.5f : 1.f;
        std::vector<float> rope_cos, rope_sin;
        h3_dit_rope_tables(layout, inv_freq, spatial_scale, rope_cos, rope_sin);
        const float *r_cos = rope_cos.empty() ? nullptr : rope_cos.data();
        const float *r_sin = rope_sin.empty() ? nullptr : rope_sin.data();
        auto unpack_z = [&]() {
            const float *vid = latent.data() + static_cast<size_t>(video_off) * hidden;
            for (int i = 0; i < tokens; ++i)
                for (int d = 0; d < patch; ++d) {
                    float acc = 0.f;
                    int n = 0;
                    for (int h = d; h < hidden; h += patch) {
                        acc += vid[static_cast<size_t>(i) * hidden + h];
                        ++n;
                    }
                    rows[static_cast<size_t>(i) * patch + d] = n > 0 ? acc / static_cast<float>(n) : 0.f;
                }
            if (can_patch && tokens == vg.latent_t * (vg.latent_h / 2) * (vg.latent_w / 2))
                h3_dit_unpatchify(rows.data(), C, vg.latent_t, vg.latent_h, vg.latent_w, z.data());
            else {
                for (int i = 0; i < std::min(tokens, nlat); ++i)
                    for (int c = 0; c < C; ++c)
                        z[static_cast<size_t>(c) * nlat + i] = rows[static_cast<size_t>(i) * patch + c];
            }
        };

        int streamed = 0;
        int computed = 0;
        std::vector<float> sigmas(static_cast<size_t>(evals) + 1, 0.f);
        std::vector<float> asigmas(static_cast<size_t>(evals) + 1, 0.f);
        h3_sigma_video(evals, sigmas.data(), cfg_.h3.video_sigma_shift);
        h3_sigma_video(evals, asigmas.data(),
                       cfg_.h3.audio_sigma_shift > 0.f ? cfg_.h3.audio_sigma_shift
                                                       : kH3AudioSigmaShift);
        const int tdim = cfg_.h3.time_input > 0 ? cfg_.h3.time_input : 256;
        std::vector<float> tfeat(static_cast<size_t>(tdim), 0.f);
        uint32_t ph = 0;
        for (unsigned char c : hp.prompt)
            ph = ph * 131u + c;
        for (int s = 0; s < evals; ++s) {
            if (hp.on_progress)
                hp.on_progress(s, evals, "denoise");
            h3_time_features(1.f - sigmas[static_cast<size_t>(s)], tfeat.data(), tdim);
            tfeat[0] += 0.01f * static_cast<float>(ph % 100u);
            std::vector<float> prev = latent;
            for (int b = 0; b < layers; ++b) {
                const uint8_t *data = nullptr;
                int64_t bytes = 0;
                Status st = blocks_.acquire(b, &data, &bytes, err);
                if (st != Status::Ok)
                    return st;
                if (hp.ssd_streaming)
                    blocks_.prefetch_async(b + 1 < layers ? b + 1 : 0);
                if (data && bytes > 0)
                    streamed++;
                if (from_checkpoint_ && data && bytes == expect) {
                    std::vector<float> mod;
                    const float *qn = nullptr, *kn = nullptr;
                    if (b < static_cast<int>(q_norm_.size()) && !q_norm_[static_cast<size_t>(b)].empty())
                        qn = q_norm_[static_cast<size_t>(b)].data();
                    if (b < static_cast<int>(k_norm_.size()) && !k_norm_[static_cast<size_t>(b)].empty())
                        kn = k_norm_[static_cast<size_t>(b)].data();
                    if (b < static_cast<int>(adaln_w_.size()) &&
                        !adaln_w_[static_cast<size_t>(b)].empty() && !time_out_w_.empty() &&
                        !time_in_w_.empty()) {
                        const int tin = tdim;
                        const int th = static_cast<int>(time_in_w_.size() / std::max(tin, 1));
                        const int td = static_cast<int>(time_out_w_.size() / std::max(th, 1));
                        std::vector<float> hid(static_cast<size_t>(std::max(th, 1)), 0.f),
                            temb(static_cast<size_t>(std::max(td, 1)), 0.f);
                        if (th > 0)
                            quant::matmul_f32(hid.data(), tfeat.data(), time_in_w_.data(), 1, tin,
                                              th);
                        for (int i = 0; i < th && i < static_cast<int>(time_in_b_.size()); ++i)
                            hid[static_cast<size_t>(i)] += time_in_b_[static_cast<size_t>(i)];
                        for (float &v : hid)
                            v = v * quant::sigmoid(v);
                        if (td > 0)
                            quant::matmul_f32(temb.data(), hid.data(), time_out_w_.data(), 1, th,
                                              td);
                        for (int i = 0; i < td && i < static_cast<int>(time_out_b_.size()); ++i)
                            temb[static_cast<size_t>(i)] += time_out_b_[static_cast<size_t>(i)];
                        const int mrows = 6 * hidden;
                        if (static_cast<int>(adaln_w_[static_cast<size_t>(b)].size()) >=
                            mrows * td) {
                            mod.assign(static_cast<size_t>(mrows), 0.f);
                            quant::matmul_f32(mod.data(), temb.data(),
                                              adaln_w_[static_cast<size_t>(b)].data(), 1, td,
                                              mrows);
                            if (static_cast<int>(adaln_b_[static_cast<size_t>(b)].size()) >= mrows)
                                for (int i = 0; i < mrows; ++i)
                                    mod[static_cast<size_t>(i)] +=
                                        adaln_b_[static_cast<size_t>(b)][static_cast<size_t>(i)];
                        }
                    }
                    {
                        AccTimer t(t_attn_);
                        if (!metal_h3::dit_residual(data, qkv_b, out_b, fc1_b, fc2_b, hidden, inner,
                                                    ffn, hd, latent.data(), seq, 1e-6f,
                                                    mod.empty() ? nullptr : mod.data(), qn, kn,
                                                    r_cos, r_sin)) {
                            if (!mod.empty() || r_cos)
                                h3_dit_block_cpu(data, qkv_b, out_b, fc1_b, fc2_b, hidden, inner,
                                                 ffn, hd, latent.data(), seq, 1e-6f,
                                                 mod.empty() ? nullptr : mod.data(), qn, kn, r_cos,
                                                 r_sin);
                            else
                                gpu::dit_block(data, qkv_b, out_b, fc1_b, fc2_b, hidden, inner, ffn,
                                               hd, latent.data(), seq, 1e-6f);
                        }
                    }
                    ++computed;
                }
                if (hp.ssd_streaming) {
                    Status pst = blocks_.wait_prefetch(err);
                    if (pst != Status::Ok) {
                        blocks_.release(b);
                        return pst;
                    }
                }
                blocks_.release(b);
            }
            if (computed) {
                std::vector<float> vel(latent.size());
                for (size_t i = 0; i < latent.size(); ++i)
                    vel[i] = latent[i] - prev[i];
                latent = prev;
                if (audio_rows > 0) {
                    if (audio_off > 0)
                        h3_euler_step(latent.data(), vel.data(), audio_off * hidden,
                                      sigmas[static_cast<size_t>(s)],
                                      sigmas[static_cast<size_t>(s) + 1]);
                    h3_euler_step(latent.data() + static_cast<size_t>(audio_off) * hidden,
                                  vel.data() + static_cast<size_t>(audio_off) * hidden,
                                  audio_rows * hidden, asigmas[static_cast<size_t>(s)],
                                  asigmas[static_cast<size_t>(s) + 1]);
                    h3_euler_step(latent.data() + static_cast<size_t>(video_off) * hidden,
                                  vel.data() + static_cast<size_t>(video_off) * hidden,
                                  tokens * hidden, sigmas[static_cast<size_t>(s)],
                                  sigmas[static_cast<size_t>(s) + 1]);
                } else {
                    h3_euler_step(latent.data(), vel.data(), static_cast<int>(latent.size()),
                                  sigmas[static_cast<size_t>(s)],
                                  sigmas[static_cast<size_t>(s) + 1]);
                }
            }
        }
        {
            AccTimer t(t_emm_);
            unpack_z();
            if (audio_rows > 0) {
                const float *aud = latent.data() + static_cast<size_t>(audio_off) * hidden;
                for (int i = 0; i < audio_rows; ++i)
                    for (int d = 0; d < AC; ++d) {
                        float acc = 0.f;
                        int n = 0;
                        for (int h = d; h < hidden; h += AC) {
                            acc += aud[static_cast<size_t>(i) * hidden + h];
                            ++n;
                        }
                        arows[static_cast<size_t>(i) * AC + d] =
                            n > 0 ? acc / static_cast<float>(n) : 0.f;
                    }
                h3_dit_unpack_audio(arows.data(), AC, audio_t, az.data());
            }
        }
        float checksum = 0.f;
        for (float v : latent)
            checksum += v * v;

        std::vector<float> rgb(static_cast<size_t>(vg.frames) * vg.height * vg.width * 3, 0.f);
        vae_.decode(z.data(), vg, rgb.data());
        if (hp.on_progress)
            hp.on_progress(evals, evals, "vae");
        vae_.geom = vg;
        std::vector<float> pcm;
        avae_.decode(az.data(), audio_t, pcm);

        out.blocks_streamed = streamed;
        out.steps_run = evals;
        out.frames = vg.frames;
        out.width = vg.width;
        out.height = vg.height;
        out.vae_used = true;
        out.audio_used = true;
        out.audio_samples = static_cast<int>(pcm.size() / 2);
        out.audio_rate = kH3AudioRate;
        out.output_path = hp.output_path.empty() ? (model_dir_ + "/h3_dryrun.txt") : hp.output_path;
        const char *dit_tag = "CPU";
        if (std::strcmp(gpu::name(), "metal") == 0)
            dit_tag = "Metal";
        else if (std::strcmp(gpu::name(), "cuda") == 0)
            dit_tag = "CUDA";
        const bool ppm = ends_with(out.output_path, ".ppm");
        const bool raw = ends_with(out.output_path, ".rgb");
        const bool wav = ends_with(out.output_path, ".wav");
        const bool mp4 = ends_with(out.output_path, ".mp4");
        bool muxed = false;
        bool mux_skip = false;
        out.audio_path = wav ? out.output_path : (out.output_path + ".wav");
        {
            std::string werr;
            h3_write_wav(out.audio_path, pcm.data(), kH3AudioStereo, out.audio_samples, out.audio_rate,
                         werr);
        }
        if (mp4) {
            std::string merr;
            Status mst = h3_write_mp4(out.output_path, rgb.data(), vg.frames, vg.height, vg.width, 24,
                                      pcm.data(), kH3AudioStereo, out.audio_samples, out.audio_rate,
                                      merr);
            if (mst == Status::Unsupported) {
                mux_skip = true;
                h3_write_ppm(out.output_path + ".ppm", rgb.data(), vg.frames, vg.height, vg.width,
                             err);
                err.clear();
            } else if (mst != Status::Ok) {
                err = merr;
                return mst;
            } else {
                muxed = true;
            }
        } else if (ppm) {
            Status wst = h3_write_ppm(out.output_path, rgb.data(), vg.frames, vg.height, vg.width, err);
            if (wst != Status::Ok)
                return wst;
        } else if (raw) {
            std::ofstream rf(out.output_path, std::ios::binary);
            if (!rf) {
                err = "h3: open rgb failed: " + out.output_path;
                return Status::IoError;
            }
            const size_t npix = static_cast<size_t>(vg.frames) * vg.height * vg.width * 3;
            std::vector<unsigned char> bytes(npix);
            for (size_t i = 0; i < npix; ++i) {
                float v = rgb[i];
                if (v < 0.f)
                    v = 0.f;
                if (v > 1.f)
                    v = 1.f;
                bytes[i] = static_cast<unsigned char>(v * 255.f + 0.5f);
            }
            rf.write(reinterpret_cast<const char *>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        } else if (!wav && !mp4) {
            {
                std::ofstream f(out.output_path);
                f << "micro-vllm H3\n"
                  << "prompt: " << hp.prompt << "\n"
                  << "canvas: " << vg.width << "x" << vg.height << " frames=" << vg.frames << "\n"
                  << "steps=" << steps << " reuse=" << reuse << " layers=" << layers << "\n"
                  << "blocks_streamed=" << streamed << " cpu_blocks=" << computed
                  << " slots=" << blocks_.n_slots() << "\n"
                  << "latent_l2=" << checksum << "\n"
                  << "vae="
                  << ((vae_.official_decode || vae_.official_encode)
                          ? "official"
                          : (vae_.from_checkpoint ? "real" : "synth"))
                  << "\n"
                  << "note: " << (computed ? std::string(dit_tag) + " DiT residual on streamed BF16"
                                          : std::string("stream only"))
                  << "\n";
            }
            std::string ppath = out.output_path + ".ppm";
            h3_write_ppm(ppath, rgb.data(), vg.frames, vg.height, vg.width, err);
            err.clear();
        }
        out.note = computed ? std::string("checkpoint: ") + dit_tag +
                                  " DiT residual on VAE latent tokens"
                            : "dry-run: DiT blocks streamed from SSD (2-slot)";
        out.note += (vae_.official_decode || vae_.official_encode)
                        ? " vae=official"
                        : (vae_.from_checkpoint ? " vae=real" : " vae=synth");
        out.note += " latent_tokens=" + std::to_string(tokens);
        out.note += can_patch ? " patchify=2x2" : " patchify=flat";
        out.note += " euler";
        if (text_tokens > 0)
            out.note += text_.from_checkpoint() ? " text=qwen" : " text=enc";
        out.note += avae_.from_checkpoint ? " audio=real" : " audio=synth";
        out.note += " audio_t=" + std::to_string(audio_t);
        out.note += audio_from_ref ? " audio=ref" : " pack=text+audio+video";
        out.note += layout_ok ? " rope=3axis" : " rope=index";
        if (using_ref2va)
            out.note += " ref2va";
        else if (want_fl)
            out.note += " fl2va";
        else
            out.note += " t2va";
        if (want_ref)
            out.note += " picture=" + std::to_string(picture_n);
        out.note += vision_.from_checkpoint() ? " vision=qwen" : " vision=off";
        if (ref2va_missing)
            out.note += " ref2va=missing";
        if (muxed)
            out.note += " mux=mp4";
        else if (mux_skip)
            out.note += " mux=skip";
        err.clear();
        return Status::Ok;
    }

    void turn_perf(TurnPerf &out, bool reset) override {
        out = {};
        out.t_attn = t_attn_;
        out.t_emm = t_emm_;
        if (reset)
            t_attn_ = t_emm_ = 0;
    }

    std::string describe() const override {
        std::ostringstream os;
        os << "h3  dit_layers=" << cfg_.h3.dit_layers << " slots=" << cfg_.h3.stream_slots
           << " hidden=" << cfg_.h3.hidden << " inner=" << cfg_.h3.inner << " ffn=" << cfg_.h3.ffn
           << " block_bytes=" << block_bytes_ << " checkpoint="
           << (from_checkpoint_ ? "yes" : "synthetic")
           << " transformer=" << (transformer_dir_.empty() ? "-" : transformer_dir_)
           << " tok=" << h3_tok_backend()
           << " default=" << cfg_.h3.default_width << "x" << cfg_.h3.default_height << "@"
           << cfg_.h3.default_frames << " hits=" << blocks_.hits()
           << " misses=" << blocks_.misses()
           << " vae="
           << ((vae_.official_decode || vae_.official_encode)
                   ? "official"
                   : (vae_.from_checkpoint ? "real" : "synth"))
           << " T=" << vae_.geom.latent_t
           << " lh=" << vae_.geom.latent_h << " lw=" << vae_.geom.latent_w
           << " ch=" << vae_.geom.latent_ch
           << " audio=" << (avae_.from_checkpoint ? "real" : "synth")
           << " vision=" << (vision_.from_checkpoint() ? "qwen" : "off")
           << " h3gpu=" << metal_h3::backend_name()
           << " int8=" << (metal_h3::available() ? metal_h3::backend_name() : "off")
           << " nax=" << (metal_h3::available() ? metal_h3::backend_name() : "off")
           << " vk=" << (vk_ops::available() ? vk_ops::backend_name() : "off")
           << " ffmpeg=" << (h3_ffmpeg_available() ? "yes" : "no");
        return os.str();
    }

    uint64_t block_hits() const override { return blocks_.hits(); }
    uint64_t block_misses() const override { return blocks_.misses(); }

private:
    Status write_synthetic_blocks(const std::string &model_dir, std::string &err) {
        const int n = cfg_.h3.dit_layers > 0 ? cfg_.h3.dit_layers : 50;
        const int64_t fake = 4096;
        Status st = blocks_.open(n, fake, cfg_.h3.stream_slots > 0 ? cfg_.h3.stream_slots : 2, err);
        if (st != Status::Ok)
            return st;
        blocks_.set_direct(true);
        block_path_ = model_dir + "/.mvllm_h3_blocks.bin";
        {
            std::ofstream out(block_path_, std::ios::binary | std::ios::trunc);
            std::vector<uint8_t> z(static_cast<size_t>(fake), 0);
            for (int i = 0; i < n; ++i) {
                z[0] = static_cast<uint8_t>(i);
                out.write(reinterpret_cast<const char *>(z.data()), fake);
            }
        }
        for (int i = 0; i < n; ++i) {
            st = blocks_.register_block(i, block_path_, static_cast<int64_t>(i) * fake, fake, err);
            if (st != Status::Ok)
                return st;
        }
        block_bytes_ = fake;
        from_checkpoint_ = false;
        transformer_dir_.clear();
        fl2va_dir_.clear();
        ref2va_dir_.clear();
        return Status::Ok;
    }

    Status bind_transformer(const std::string &dir, std::string &err) {
        std::vector<io::StFile> files;
        Status ost = io::st_open_dir(dir, files, err);
        if (ost != Status::Ok || files.empty() ||
            !io::st_find_dir(files, "blocks.0.attn.qkv_proj.weight").tensor) {
            if (!files.empty())
                io::st_close_dir(files);
            if (err.empty())
                err = "h3 transformer not found: " + dir;
            return Status::NotFound;
        }

        auto has = [&](const std::string &n) { return io::st_find_dir(files, n).tensor != nullptr; };

        int n_blocks = 0;
        for (int i = 0; i < 64; ++i) {
            if (!has("blocks." + std::to_string(i) + ".attn.qkv_proj.weight"))
                break;
            n_blocks = i + 1;
        }
        if (n_blocks < 1) {
            io::st_close_dir(files);
            return Status::NotFound;
        }

        io::StHit qkv0 = io::st_find_dir(files, "blocks.0.attn.qkv_proj.weight");
        if (qkv0.tensor && qkv0.tensor->shape.size() == 2) {
            cfg_.h3.hidden = static_cast<int>(qkv0.tensor->shape[1]);
            cfg_.h3.inner = static_cast<int>(qkv0.tensor->shape[0] / 3);
        }
        io::StHit fc1 = io::st_find_dir(files, "blocks.0.mlp.fc1.weight");
        if (fc1.tensor && fc1.tensor->shape.size() == 2)
            cfg_.h3.ffn = static_cast<int>(fc1.tensor->shape[0] / 2);

        int64_t slot = 0;
        BlockPiece probe[4];
        for (int p = 0; p < 4; ++p) {
            io::StHit hit = io::st_find_dir(files, std::string("blocks.0.") + kH3StreamSuffix[p]);
            if (!hit.tensor) {
                err = "h3 block 0 missing " + std::string(kH3StreamSuffix[p]);
                io::st_close_dir(files);
                return Status::ParseError;
            }
            probe[p].path = hit.file->path;
            probe[p].offset = io::st_file_offset(*hit.file, *hit.tensor);
            probe[p].bytes = io::st_nbytes(*hit.tensor);
            slot += probe[p].bytes;
        }

        cfg_.h3.dit_layers = n_blocks;
        Status st = blocks_.open(n_blocks, slot, cfg_.h3.stream_slots > 0 ? cfg_.h3.stream_slots : 2,
                                 err);
        if (st != Status::Ok) {
            io::st_close_dir(files);
            return st;
        }
        blocks_.set_direct(true);

        for (int i = 0; i < n_blocks; ++i) {
            std::vector<BlockPiece> pieces;
            pieces.reserve(4);
            for (int p = 0; p < 4; ++p) {
                const std::string name = "blocks." + std::to_string(i) + "." + kH3StreamSuffix[p];
                io::StHit hit = io::st_find_dir(files, name);
                if (!hit.tensor || io::st_nbytes(*hit.tensor) != probe[p].bytes) {
                    err = "h3 block " + std::to_string(i) + " missing or wrong size: " + name;
                    io::st_close_dir(files);
                    return Status::ParseError;
                }
                BlockPiece bp;
                bp.path = hit.file->path;
                bp.offset = io::st_file_offset(*hit.file, *hit.tensor);
                bp.bytes = probe[p].bytes;
                pieces.push_back(bp);
            }
            bool contig = true;
            for (int p = 1; p < 4; ++p) {
                if (pieces[p].path != pieces[0].path ||
                    pieces[p].offset != pieces[p - 1].offset + pieces[p - 1].bytes)
                    contig = false;
            }
            if (contig) {
                st = blocks_.register_block(i, pieces[0].path, pieces[0].offset, slot, err);
            } else {
                st = blocks_.register_block_pieces(i, pieces, err);
            }
            if (st != Status::Ok) {
                io::st_close_dir(files);
                return st;
            }
        }
        auto load_f = [&](const std::string &name, std::vector<float> &dst) {
            dst.clear();
            io::StHit hit = io::st_find_dir(files, name);
            if (!hit.tensor)
                return;
            int64_t n = 1;
            for (int64_t d : hit.tensor->shape)
                n *= d;
            dst.assign(static_cast<size_t>(n), 0.f);
            std::string e;
            io::st_read_f32(*hit.file, *hit.tensor, dst.data(), n, e);
        };
        load_f("condition_proj.weight", cond_w_);
        load_f("condition_proj.bias", cond_b_);
        load_f("rope.inv_freq", rope_inv_);
        load_f("time_embedder.proj_in.weight", time_in_w_);
        load_f("time_embedder.proj_in.bias", time_in_b_);
        load_f("time_embedder.proj_out.weight", time_out_w_);
        load_f("time_embedder.proj_out.bias", time_out_b_);
        adaln_w_.assign(static_cast<size_t>(n_blocks), {});
        adaln_b_.assign(static_cast<size_t>(n_blocks), {});
        q_norm_.assign(static_cast<size_t>(n_blocks), {});
        k_norm_.assign(static_cast<size_t>(n_blocks), {});
        for (int i = 0; i < n_blocks; ++i) {
            const std::string p = "blocks." + std::to_string(i) + ".";
            load_f(p + "adaln_proj.linear.weight", adaln_w_[static_cast<size_t>(i)]);
            load_f(p + "adaln_proj.linear.bias", adaln_b_[static_cast<size_t>(i)]);
            load_f(p + "attn.q_norm.weight", q_norm_[static_cast<size_t>(i)]);
            load_f(p + "attn.k_norm.weight", k_norm_[static_cast<size_t>(i)]);
        }
        io::st_close_dir(files);
        transformer_dir_ = dir;
        block_bytes_ = slot;
        from_checkpoint_ = true;
        return Status::Ok;
    }

    Status load_checkpoint(const std::string &model_dir, std::string &err) {
        const char *tails[] = {"/FL2VA/transformer", "/transformer", "/dit", ""};
        std::string chosen;
        for (const char *tail : tails) {
            std::string dir = model_dir + tail;
            std::vector<io::StFile> files;
            err.clear();
            Status st = io::st_open_dir(dir, files, err);
            if (st != Status::Ok || files.empty())
                continue;
            const bool hit = io::st_find_dir(files, "blocks.0.attn.qkv_proj.weight").tensor != nullptr;
            io::st_close_dir(files);
            if (hit) {
                chosen = dir;
                break;
            }
        }
        if (chosen.empty()) {
            err.clear();
            return Status::NotFound;
        }

        ref2va_dir_.clear();
        {
            const std::string rdir = model_dir + "/Ref2VA/transformer";
            std::vector<io::StFile> rfiles;
            std::string rerr;
            if (io::st_open_dir(rdir, rfiles, rerr) == Status::Ok &&
                io::st_find_dir(rfiles, "blocks.0.attn.qkv_proj.weight").tensor)
                ref2va_dir_ = rdir;
            if (!rfiles.empty())
                io::st_close_dir(rfiles);
        }
        fl2va_dir_ = chosen;
        return bind_transformer(chosen, err);
    }

    ModelConfig cfg_{};
    RuntimeConfig rt_{};
    BlockStore blocks_;
    H3Vae vae_;
    H3AudioVae avae_;
    std::string model_dir_;
    std::string block_path_;
    std::string transformer_dir_;
    std::string fl2va_dir_;
    std::string ref2va_dir_;
    int64_t block_bytes_ = 0;
    bool loaded_ = false;
    bool from_checkpoint_ = false;
    std::vector<float> time_in_w_, time_in_b_, time_out_w_, time_out_b_;
    std::vector<float> cond_w_, cond_b_, rope_inv_;
    std::vector<std::vector<float>> adaln_w_, adaln_b_, q_norm_, k_norm_;
    H3TextEncoder text_;
    H3VisionEncoder vision_;
    Tokenizer h3_tok_;
    double t_attn_ = 0, t_emm_ = 0;
};

std::unique_ptr<FamilyEngine> make_h3() { return std::make_unique<H3Engine>(); }

} // namespace mvllm
