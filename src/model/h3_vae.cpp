#include "h3_vae.hpp"
#include "../gpu/metal_h3.hpp"
#include "../io/safetensors.hpp"
#include "../quant/quant.hpp"

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <unordered_map>

#include <sys/stat.h>

namespace mvllm {
namespace {

constexpr float kImMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImStd[3] = {0.229f, 0.224f, 0.225f};
constexpr float kRmsEps = 1e-5f;
constexpr float kLnEps = 1e-5f;

struct DecW {
    int hidden = 0;
    int heads = 0;
    int head_dim = 0;
    int n_blocks = 0;
    int proj_dim = 0;
    std::vector<float> post_w, post_b;
    std::vector<float> embed_w, embed_b;
    std::vector<float> proj_w, proj_b;
    std::vector<float> ln_w, ln_b;
    std::vector<float> enc_w, enc_b;
    std::vector<float> registers; // [4, hidden]
    bool official = false;
};

struct EncConv {
    int in_ch = 0, out_ch = 0, k = 1, st = 1, sh = 1, sw = 1;
    int pad_t = 0, pad_h0 = 0, pad_h1 = 0, pad_w0 = 0, pad_w1 = 0;
    std::vector<float> w, b;
};

struct EncBlock {
    std::vector<float> n1w, n1b, n2w, n2b;
    EncConv conv1, conv2, shortcut;
    bool has_shortcut = false;
};

struct EncW {
    bool ready = false;
    int groups = 32;
    int n_levels = 6;
    int ch[6] = {128, 256, 256, 512, 512, 1024};
    int space_st[6] = {2, 2, 2, 2, 1, 1};
    int time_st[6] = {1, 2, 2, 1, 1, 1};
    EncConv conv_in, conv_out, quant;
    EncBlock block[6][2];
    EncConv down[6];
    bool has_down[6] = {};
    std::vector<float> nout_w, nout_b;
};

std::unordered_map<const H3Vae *, EncW> &enc_map() {
    static std::unordered_map<const H3Vae *, EncW> m;
    return m;
}
EncW &encw(const H3Vae *v) { return enc_map()[v]; }
const EncW *encw_c(const H3Vae *v) {
    auto it = enc_map().find(v);
    return it == enc_map().end() ? nullptr : &it->second;
}

std::unordered_map<const H3Vae *, DecW> &dec_map() {
    static std::unordered_map<const H3Vae *, DecW> m;
    return m;
}

DecW &decw(const H3Vae *v) { return dec_map()[v]; }
const DecW *decw_c(const H3Vae *v) {
    auto it = dec_map().find(v);
    return it == dec_map().end() ? nullptr : &it->second;
}

int clampi(int v, int lo, int hi) {
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}

std::string parent_dir(const std::string &p) {
    auto slash = p.find_last_of('/');
    if (slash == std::string::npos || slash == 0)
        return p;
    return p.substr(0, slash);
}

int64_t numel_shape(const std::vector<int64_t> &sh) {
    int64_t n = 1;
    for (int64_t d : sh) {
        if (d <= 0)
            return 0;
        n *= d;
    }
    return n;
}

void fill_mix_a(float *A, int ch) {
    // A[3, ch] — full-ish rank, not identity.
    for (int c = 0; c < 3; ++c) {
        for (int k = 0; k < ch; ++k) {
            const float x = static_cast<float>(((c + 1) * (k + 3) * 17) % 97) / 97.f;
            A[c * ch + k] = (x - 0.5f) * 0.85f + (k % 3 == c ? 0.22f : 0.f);
        }
    }
}

void fill_enc_b(float *B, int ch) {
    // B[ch, 3]
    for (int k = 0; k < ch; ++k) {
        for (int c = 0; c < 3; ++c) {
            const float x = static_cast<float>(((k + 2) * (c + 5) * 13) % 89) / 89.f;
            B[k * 3 + c] = (x - 0.45f) * 0.9f + (k % 3 == c ? 0.18f : 0.f);
        }
    }
}

void ensure_stats_c(const H3Vae &v, int ch, float *mean, float *stdv, float *lm, float *ls) {
    for (int i = 0; i < 3; ++i) {
        mean[i] = (static_cast<int>(v.mean.size()) > i) ? v.mean[static_cast<size_t>(i)] : kImMean[i];
        stdv[i] = (static_cast<int>(v.std.size()) > i) ? v.std[static_cast<size_t>(i)] : kImStd[i];
        if (!(stdv[i] > 1e-8f))
            stdv[i] = kImStd[i];
    }
    for (int c = 0; c < ch; ++c) {
        lm[c] = (static_cast<int>(v.latents_mean.size()) > c) ? v.latents_mean[static_cast<size_t>(c)]
                                                              : 0.f;
        ls[c] = (static_cast<int>(v.latents_std.size()) > c) ? v.latents_std[static_cast<size_t>(c)]
                                                             : 1.f;
        if (!(ls[c] > 1e-8f))
            ls[c] = 1.f;
    }
}

io::StHit find_named(const std::vector<io::StFile> &files, const std::string &name) {
    static const char *kPref[] = {"", "video_vae.", "model.", "vae.", "visual_vae."};
    for (const char *p : kPref) {
        io::StHit h = io::st_find_dir(files, std::string(p) + name);
        if (h.tensor)
            return h;
    }
    return {};
}

bool read_vec(const std::vector<io::StFile> &files, const std::string &name, std::vector<float> &out,
              std::vector<int64_t> *shape = nullptr) {
    io::StHit h = find_named(files, name);
    if (!h.tensor)
        return false;
    const int64_t n = numel_shape(h.tensor->shape);
    if (n <= 0)
        return false;
    out.resize(static_cast<size_t>(n));
    std::string err;
    if (io::st_read_f32(*h.file, *h.tensor, out.data(), n, err) != Status::Ok) {
        out.clear();
        return false;
    }
    if (shape)
        *shape = h.tensor->shape;
    return true;
}

bool read_first(const std::vector<io::StFile> &files, const char *const *names, int nnames,
                std::vector<float> &out, std::vector<int64_t> *shape = nullptr) {
    for (int i = 0; i < nnames; ++i) {
        if (read_vec(files, names[i], out, shape))
            return true;
    }
    return false;
}

void overlay_latents_json(const std::string &path, std::vector<float> &lm, std::vector<float> &ls) {
    std::ifstream in(path);
    if (!in)
        return;
    std::ostringstream ss;
    ss << in.rdbuf();
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(ss.str());
    } catch (...) {
        return;
    }
    if (!j.is_object())
        return;
    const nlohmann::json *root = &j;
    if (j.contains("video_vae") && j["video_vae"].is_object())
        root = &j["video_vae"];
    auto pull = [](const nlohmann::json &o, const char *key, std::vector<float> &dst) {
        if (!o.contains(key) || !o[key].is_array())
            return;
        const auto &a = o[key];
        const size_t n = std::min(dst.size(), a.size());
        for (size_t i = 0; i < n; ++i) {
            if (a[i].is_number())
                dst[i] = static_cast<float>(a[i].get<double>());
        }
    };
    pull(*root, "latents_mean", lm);
    pull(*root, "latents_std", ls);
}

void add_bias(float *y, const float *b, int S, int O) {
    if (!b || O <= 0)
        return;
    for (int s = 0; s < S; ++s) {
        float *row = y + static_cast<size_t>(s) * O;
        for (int o = 0; o < O; ++o)
            row[o] += b[o];
    }
}

void linear(float *y, const float *x, const float *w, const float *b, int S, int I, int O) {
    quant::matmul_f32(y, x, w, S, I, O);
    add_bias(y, b, S, O);
}

void layernorm_rows(float *x, const float *w, const float *b, int S, int H, float eps) {
    for (int s = 0; s < S; ++s) {
        float *row = x + static_cast<size_t>(s) * H;
        float mean = 0.f;
        for (int i = 0; i < H; ++i)
            mean += row[i];
        mean /= static_cast<float>(H > 0 ? H : 1);
        float var = 0.f;
        for (int i = 0; i < H; ++i) {
            float d = row[i] - mean;
            var += d * d;
        }
        var /= static_cast<float>(H > 0 ? H : 1);
        float inv = 1.f / std::sqrt(var + eps);
        for (int i = 0; i < H; ++i) {
            float v = (row[i] - mean) * inv;
            if (w)
                v *= w[i];
            if (b)
                v += b[i];
            row[i] = v;
        }
    }
}

void rmsnorm_rows(const float *x, const float *w, float *y, int S, int H, float eps) {
    for (int s = 0; s < S; ++s)
        quant::rmsnorm(x + static_cast<size_t>(s) * H, w, y + static_cast<size_t>(s) * H, H, eps);
}

void ensure_metal_h3() {
    static const bool inited = metal_h3::init();
    (void)inited;
}

// x += skip; y = rmsnorm(x, w). Per-row Metal; CPU if a call fails.
void residual_rmsnorm_rows(float *x, const float *skip, const float *w, float *y, int S, int H,
                           float eps) {
    ensure_metal_h3();
    for (int s = 0; s < S; ++s) {
        float *xr = x + static_cast<size_t>(s) * H;
        const float *sr = skip + static_cast<size_t>(s) * H;
        float *yr = y + static_cast<size_t>(s) * H;
        if (metal_h3::vae_rms_add(xr, sr, w, yr, H, eps))
            continue;
        for (int i = 0; i < H; ++i)
            xr[i] += sr[i];
        quant::rmsnorm(xr, w, yr, H, eps);
    }
}

void sdpa(const float *q, const float *k, const float *v, float *out, int seq, int heads, int hd) {
    const float scale = 1.f / std::sqrt(static_cast<float>(hd > 0 ? hd : 1));
    std::vector<float> scores(static_cast<size_t>(seq));
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < seq; ++i) {
            const float *qi = q + (static_cast<size_t>(i) * heads + h) * hd;
            float mx = -1e30f;
            for (int j = 0; j < seq; ++j) {
                const float *kj = k + (static_cast<size_t>(j) * heads + h) * hd;
                float dot = 0.f;
                for (int d = 0; d < hd; ++d)
                    dot += qi[d] * kj[d];
                scores[static_cast<size_t>(j)] = dot * scale;
                if (scores[static_cast<size_t>(j)] > mx)
                    mx = scores[static_cast<size_t>(j)];
            }
            float sum = 0.f;
            for (int j = 0; j < seq; ++j) {
                float e = std::exp(scores[static_cast<size_t>(j)] - mx);
                scores[static_cast<size_t>(j)] = e;
                sum += e;
            }
            const float inv = 1.f / (sum + 1e-12f);
            float *oi = out + (static_cast<size_t>(i) * heads + h) * hd;
            for (int d = 0; d < hd; ++d)
                oi[d] = 0.f;
            for (int j = 0; j < seq; ++j) {
                const float a = scores[static_cast<size_t>(j)] * inv;
                const float *vj = v + (static_cast<size_t>(j) * heads + h) * hd;
                for (int d = 0; d < hd; ++d)
                    oi[d] += a * vj[d];
            }
        }
    }
}

void apply_swiglu(float *gate_up, int S, int inner);

void rms_vec(float *x, int n, float eps) {
    float s = 0.f;
    for (int i = 0; i < n; ++i)
        s += x[i] * x[i];
    float inv = 1.f / std::sqrt(s / static_cast<float>(n > 0 ? n : 1) + eps);
    for (int i = 0; i < n; ++i)
        x[i] *= inv;
}

void vae_rope_tables(int t_ext, int lh, int lw, int extra, std::vector<float> &cos,
                     std::vector<float> &sin) {
    const int patches = t_ext * lh * lw;
    const int seq = patches + extra;
    cos.assign(static_cast<size_t>(seq) * kH3VaeRopeHalf, 1.f);
    sin.assign(static_cast<size_t>(seq) * kH3VaeRopeHalf, 0.f);
    int row = 0;
    for (int t = 0; t < t_ext; ++t)
        for (int h = 0; h < lh; ++h)
            for (int w = 0; w < lw; ++w, ++row) {
                const float axes[3] = {
                    2.f * ((static_cast<float>(t) + 0.5f) / static_cast<float>(t_ext > 0 ? t_ext : 1)) -
                        1.f,
                    2.f * ((static_cast<float>(h) + 0.5f) / static_cast<float>(lh > 0 ? lh : 1)) - 1.f,
                    2.f * ((static_cast<float>(w) + 0.5f) / static_cast<float>(lw > 0 ? lw : 1)) - 1.f};
                for (int axis = 0; axis < 3; ++axis)
                    for (int f = 0; f < 8; ++f) {
                        const float inv = 1.f / std::pow(100.f, static_cast<float>(f) * 0.125f);
                        const float ang = 2.f * 3.14159265358979323846f * axes[axis] * inv;
                        const int idx = row * kH3VaeRopeHalf + axis * 8 + f;
                        cos[static_cast<size_t>(idx)] = std::cos(ang);
                        sin[static_cast<size_t>(idx)] = std::sin(ang);
                    }
            }
}

void apply_block_official(float *x, int N, int hidden, int heads, int hd, const float *rope_cos,
                          const float *rope_sin, const std::vector<float> &norm1,
                          const std::vector<float> &qkv_w, const std::vector<float> &qkv_b,
                          const std::vector<float> &out_w, const std::vector<float> &out_b,
                          const std::vector<float> &scale1, const std::vector<float> &norm2,
                          const std::vector<float> &w1, const std::vector<float> &b1,
                          const std::vector<float> &w2, const std::vector<float> &b2,
                          const std::vector<float> &scale2) {
    const int inner = heads * hd;
    if (inner <= 0 || hidden <= 0)
        return;
    std::vector<float> nrm(static_cast<size_t>(N) * hidden);
    const float *nw = norm1.size() == static_cast<size_t>(hidden) ? norm1.data() : nullptr;
    rmsnorm_rows(x, nw, nrm.data(), N, hidden, kRmsEps);

    if (!qkv_w.empty() && static_cast<int>(qkv_w.size()) == 3 * inner * hidden) {
        std::vector<float> qkv(static_cast<size_t>(N) * 3 * inner);
        linear(qkv.data(), nrm.data(), qkv_w.data(),
               qkv_b.size() == static_cast<size_t>(3 * inner) ? qkv_b.data() : nullptr, N, hidden,
               3 * inner);
        std::vector<float> q(static_cast<size_t>(N) * inner), k(static_cast<size_t>(N) * inner),
            v(static_cast<size_t>(N) * inner), attn(static_cast<size_t>(N) * inner);
        // Official pack: [seq, heads, 3, hd]
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < heads; ++h) {
                const float *src = qkv.data() + (static_cast<size_t>(i) * heads + h) * 3 * hd;
                std::memcpy(q.data() + (static_cast<size_t>(i) * heads + h) * hd, src,
                            static_cast<size_t>(hd) * sizeof(float));
                std::memcpy(k.data() + (static_cast<size_t>(i) * heads + h) * hd, src + hd,
                            static_cast<size_t>(hd) * sizeof(float));
                std::memcpy(v.data() + (static_cast<size_t>(i) * heads + h) * hd, src + 2 * hd,
                            static_cast<size_t>(hd) * sizeof(float));
            }
        }
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < heads; ++h) {
                float *qh = q.data() + (static_cast<size_t>(i) * heads + h) * hd;
                float *kh = k.data() + (static_cast<size_t>(i) * heads + h) * hd;
                rms_vec(qh, hd, kRmsEps);
                rms_vec(kh, hd, kRmsEps);
                if (rope_cos && rope_sin && hd >= 2 * kH3VaeRopeHalf) {
                    const float *c = rope_cos + static_cast<size_t>(i) * kH3VaeRopeHalf;
                    const float *s = rope_sin + static_cast<size_t>(i) * kH3VaeRopeHalf;
                    for (int d = 0; d < kH3VaeRopeHalf; ++d) {
                        float qa = qh[d], qb = qh[d + kH3VaeRopeHalf];
                        qh[d] = qa * c[d] - qb * s[d];
                        qh[d + kH3VaeRopeHalf] = qa * s[d] + qb * c[d];
                        float ka = kh[d], kb = kh[d + kH3VaeRopeHalf];
                        kh[d] = ka * c[d] - kb * s[d];
                        kh[d + kH3VaeRopeHalf] = ka * s[d] + kb * c[d];
                    }
                }
            }
        }
        if (heads > 0 && hd > 0)
            sdpa(q.data(), k.data(), v.data(), attn.data(), N, heads, hd);
        else
            attn = q;
        std::vector<float> ao(static_cast<size_t>(N) * hidden, 0.f);
        if (!out_w.empty() && static_cast<int>(out_w.size()) == hidden * inner)
            linear(ao.data(), attn.data(), out_w.data(),
                   out_b.size() == static_cast<size_t>(hidden) ? out_b.data() : nullptr, N, inner,
                   hidden);
        else if (inner == hidden)
            ao = attn;
        if (static_cast<int>(scale1.size()) == hidden) {
            for (int i = 0; i < N * hidden; ++i)
                ao[static_cast<size_t>(i)] *= scale1[static_cast<size_t>(i % hidden)];
        }
        nw = norm2.size() == static_cast<size_t>(hidden) ? norm2.data() : nullptr;
        residual_rmsnorm_rows(x, ao.data(), nw, nrm.data(), N, hidden, kRmsEps);
    } else {
        nw = norm2.size() == static_cast<size_t>(hidden) ? norm2.data() : nullptr;
        rmsnorm_rows(x, nw, nrm.data(), N, hidden, kRmsEps);
    }
    if (!w1.empty() && hidden > 0 && (static_cast<int>(w1.size()) % hidden) == 0) {
        const int w1o = static_cast<int>(w1.size() / hidden);
        if (w1o >= 2 && (w1o % 2) == 0) {
            const int ffn = w1o / 2;
            std::vector<float> gu(static_cast<size_t>(N) * w1o);
            linear(gu.data(), nrm.data(), w1.data(),
                   b1.size() == static_cast<size_t>(w1o) ? b1.data() : nullptr, N, hidden, w1o);
            apply_swiglu(gu.data(), N, ffn);
            std::vector<float> fo(static_cast<size_t>(N) * hidden, 0.f);
            if (!w2.empty() && static_cast<int>(w2.size()) == hidden * ffn) {
                std::vector<float> hid(static_cast<size_t>(N) * ffn);
                for (int s = 0; s < N; ++s)
                    std::memcpy(hid.data() + static_cast<size_t>(s) * ffn,
                                gu.data() + static_cast<size_t>(s) * w1o,
                                static_cast<size_t>(ffn) * sizeof(float));
                linear(fo.data(), hid.data(), w2.data(),
                       b2.size() == static_cast<size_t>(hidden) ? b2.data() : nullptr, N, ffn, hidden);
            }
            for (int i = 0; i < N * hidden; ++i) {
                float s = (static_cast<int>(scale2.size()) == hidden)
                              ? scale2[static_cast<size_t>(i % hidden)]
                              : 1.f;
                x[i] += fo[static_cast<size_t>(i)] * s;
            }
        }
    }
}

int tile_count_for_extent(int extent, int tile_pixels) {
    if (extent <= tile_pixels)
        return 1;
    int count = (extent + tile_pixels - 1) / tile_pixels;
    while (tile_pixels * count - kH3VaeTileOverlap * (count - 1) < extent)
        ++count;
    return count;
}

struct TileAxis {
    int count = 1;
    int length = 0;
    std::vector<int> starts;
    std::vector<int> overlaps;
};

TileAxis make_tile_axis(int extent, int tile_pixels) {
    TileAxis a;
    a.length = extent <= tile_pixels ? extent : tile_pixels;
    if (extent <= tile_pixels) {
        a.count = 1;
        a.starts = {0};
        return a;
    }
    a.count = tile_count_for_extent(extent, tile_pixels);
    a.starts.assign(static_cast<size_t>(a.count), 0);
    a.overlaps.assign(static_cast<size_t>(a.count - 1), kH3VaeTileOverlap);
    int remaining = tile_pixels * a.count - kH3VaeTileOverlap * (a.count - 1) - extent;
    for (int u = 0; u < remaining / 16; ++u)
        a.overlaps[static_cast<size_t>(u % (a.count - 1))] += 16;
    for (int i = 1; i < a.count; ++i)
        a.starts[static_cast<size_t>(i)] =
            a.starts[static_cast<size_t>(i - 1)] + tile_pixels - a.overlaps[static_cast<size_t>(i - 1)];
    a.length = tile_pixels;
    return a;
}

void extract_latent_tile(const float *z, int full_t, int full_h, int full_w, int ch, int start_t,
                         int start_y, int start_x, int tile_t, int tile_h, int tile_w, float *out) {
    for (int c = 0; c < ch; ++c)
        for (int t = 0; t < tile_t; ++t)
            for (int y = 0; y < tile_h; ++y)
                for (int x = 0; x < tile_w; ++x) {
                    int st = start_t + t;
                    if (st >= full_t)
                        st = full_t - 1;
                    if (st < 0)
                        st = 0;
                    out[((static_cast<size_t>(c) * tile_t + t) * tile_h + y) * tile_w + x] =
                        z[((static_cast<size_t>(c) * full_t + st) * full_h + (start_y + y)) * full_w +
                          (start_x + x)];
                }
}

void apply_swiglu(float *gate_up, int S, int inner) {
    for (int s = 0; s < S; ++s) {
        float *g = gate_up + static_cast<size_t>(s) * inner * 2;
        float *u = g + inner;
        quant::silu_mul(g, u, inner);
    }
}

bool load_block(const std::vector<io::StFile> &files, int b, int hidden, std::vector<float> &norm1,
                std::vector<float> &qkv_w, std::vector<float> &qkv_b, std::vector<float> &out_w,
                std::vector<float> &out_b, std::vector<float> &scale1, std::vector<float> &norm2,
                std::vector<float> &w1, std::vector<float> &b1, std::vector<float> &w2,
                std::vector<float> &b2, std::vector<float> &scale2) {
    const std::string p = "decoder.transformer_blocks." + std::to_string(b) + ".";
    read_vec(files, p + "norm1.weight", norm1);
    if (!read_vec(files, p + "attn.to_qkv.weight", qkv_w)) {
        std::vector<float> q, k, v;
        if (read_vec(files, p + "attn.to_q.weight", q) && read_vec(files, p + "attn.to_k.weight", k) &&
            read_vec(files, p + "attn.to_v.weight", v)) {
            qkv_w.resize(q.size() + k.size() + v.size());
            std::memcpy(qkv_w.data(), q.data(), q.size() * sizeof(float));
            std::memcpy(qkv_w.data() + q.size(), k.data(), k.size() * sizeof(float));
            std::memcpy(qkv_w.data() + q.size() + k.size(), v.data(), v.size() * sizeof(float));
        }
    }
    read_vec(files, p + "attn.to_qkv.bias", qkv_b);
    if (qkv_b.empty()) {
        std::vector<float> qb, kb, vb;
        if (read_vec(files, p + "attn.to_q.bias", qb) && read_vec(files, p + "attn.to_k.bias", kb) &&
            read_vec(files, p + "attn.to_v.bias", vb)) {
            qkv_b.resize(qb.size() + kb.size() + vb.size());
            std::memcpy(qkv_b.data(), qb.data(), qb.size() * sizeof(float));
            std::memcpy(qkv_b.data() + qb.size(), kb.data(), kb.size() * sizeof(float));
            std::memcpy(qkv_b.data() + qb.size() + kb.size(), vb.data(), vb.size() * sizeof(float));
        }
    }
    if (!read_vec(files, p + "attn.to_out.weight", out_w))
        read_vec(files, p + "attn.to_out.0.weight", out_w);
    if (!read_vec(files, p + "attn.to_out.bias", out_b))
        read_vec(files, p + "attn.to_out.0.bias", out_b);
    read_vec(files, p + "scale1", scale1);
    read_vec(files, p + "norm2.weight", norm2);
    if (!read_vec(files, p + "ff.w1.weight", w1))
        read_vec(files, p + "ff.net.0.proj.weight", w1);
    if (!read_vec(files, p + "ff.w1.bias", b1))
        read_vec(files, p + "ff.net.0.proj.bias", b1);
    if (!read_vec(files, p + "ff.w2.weight", w2))
        read_vec(files, p + "ff.net.2.weight", w2);
    if (!read_vec(files, p + "ff.w2.bias", b2))
        read_vec(files, p + "ff.net.2.bias", b2);
    read_vec(files, p + "scale2", scale2);
    (void)hidden;
    return !qkv_w.empty() || !w1.empty();
}

void apply_block(float *x, int N, int hidden, int heads, int hd, const std::vector<float> &norm1,
                 const std::vector<float> &qkv_w, const std::vector<float> &qkv_b,
                 const std::vector<float> &out_w, const std::vector<float> &out_b,
                 const std::vector<float> &scale1, const std::vector<float> &norm2,
                 const std::vector<float> &w1, const std::vector<float> &b1, const std::vector<float> &w2,
                 const std::vector<float> &b2, const std::vector<float> &scale2) {
    int ffn = 0;
    if (!w1.empty() && hidden > 0 && (static_cast<int>(w1.size()) % hidden) == 0) {
        const int w1o = static_cast<int>(w1.size() / hidden);
        if (w1o >= 2 && (w1o % 2) == 0)
            ffn = w1o / 2;
    }
    ensure_metal_h3();
    if (metal_h3::vae_transformer_block(
            x, N, hidden, heads, hd,
            norm1.size() == static_cast<size_t>(hidden) ? norm1.data() : nullptr,
            (!qkv_w.empty() && static_cast<int>(qkv_w.size()) == 3 * hidden * hidden) ? qkv_w.data()
                                                                                     : nullptr,
            qkv_b.size() == static_cast<size_t>(3 * hidden) ? qkv_b.data() : nullptr,
            (!out_w.empty() && static_cast<int>(out_w.size()) == hidden * hidden) ? out_w.data()
                                                                                 : nullptr,
            out_b.size() == static_cast<size_t>(hidden) ? out_b.data() : nullptr,
            scale1.size() == static_cast<size_t>(hidden) ? scale1.data() : nullptr,
            norm2.size() == static_cast<size_t>(hidden) ? norm2.data() : nullptr,
            ffn > 0 ? w1.data() : nullptr,
            (ffn > 0 && b1.size() == static_cast<size_t>(2 * ffn)) ? b1.data() : nullptr,
            (ffn > 0 && !w2.empty() && static_cast<int>(w2.size()) == hidden * ffn) ? w2.data()
                                                                                   : nullptr,
            (ffn > 0 && b2.size() == static_cast<size_t>(hidden)) ? b2.data() : nullptr,
            scale2.size() == static_cast<size_t>(hidden) ? scale2.data() : nullptr, ffn, kRmsEps))
        return;

    std::vector<float> nrm(static_cast<size_t>(N) * hidden);
    const float *nw = norm1.size() == static_cast<size_t>(hidden) ? norm1.data() : nullptr;
    rmsnorm_rows(x, nw, nrm.data(), N, hidden, kRmsEps);

    if (!qkv_w.empty() && static_cast<int>(qkv_w.size()) == 3 * hidden * hidden) {
        std::vector<float> qkv(static_cast<size_t>(N) * 3 * hidden);
        linear(qkv.data(), nrm.data(), qkv_w.data(), qkv_b.size() == static_cast<size_t>(3 * hidden)
                                                         ? qkv_b.data()
                                                         : nullptr,
               N, hidden, 3 * hidden);
        const int qoff = 0;
        const int koff = hidden;
        const int voff = 2 * hidden;
        std::vector<float> q(static_cast<size_t>(N) * hidden), k(static_cast<size_t>(N) * hidden),
            v(static_cast<size_t>(N) * hidden), attn(static_cast<size_t>(N) * hidden);
        for (int i = 0; i < N; ++i) {
            std::memcpy(q.data() + static_cast<size_t>(i) * hidden,
                        qkv.data() + static_cast<size_t>(i) * 3 * hidden + qoff,
                        static_cast<size_t>(hidden) * sizeof(float));
            std::memcpy(k.data() + static_cast<size_t>(i) * hidden,
                        qkv.data() + static_cast<size_t>(i) * 3 * hidden + koff,
                        static_cast<size_t>(hidden) * sizeof(float));
            std::memcpy(v.data() + static_cast<size_t>(i) * hidden,
                        qkv.data() + static_cast<size_t>(i) * 3 * hidden + voff,
                        static_cast<size_t>(hidden) * sizeof(float));
        }
        if (heads > 0 && hd > 0 && heads * hd == hidden)
            sdpa(q.data(), k.data(), v.data(), attn.data(), N, heads, hd);
        else
            attn = q;
        std::vector<float> ao(static_cast<size_t>(N) * hidden);
        if (!out_w.empty() && static_cast<int>(out_w.size()) == hidden * hidden) {
            linear(ao.data(), attn.data(), out_w.data(),
                   out_b.size() == static_cast<size_t>(hidden) ? out_b.data() : nullptr, N, hidden,
                   hidden);
        } else {
            ao.swap(attn);
        }
        if (static_cast<int>(scale1.size()) == hidden) {
            for (int i = 0; i < N * hidden; ++i)
                ao[static_cast<size_t>(i)] *= scale1[static_cast<size_t>(i % hidden)];
        }
        nw = norm2.size() == static_cast<size_t>(hidden) ? norm2.data() : nullptr;
        residual_rmsnorm_rows(x, ao.data(), nw, nrm.data(), N, hidden, kRmsEps);
    } else {
        nw = norm2.size() == static_cast<size_t>(hidden) ? norm2.data() : nullptr;
        rmsnorm_rows(x, nw, nrm.data(), N, hidden, kRmsEps);
    }

    if (!w1.empty() && hidden > 0 && (static_cast<int>(w1.size()) % hidden) == 0) {
        const int w1o = static_cast<int>(w1.size() / hidden);
        if (w1o >= 2 && (w1o % 2) == 0) {
            const int inner = w1o / 2;
            std::vector<float> gu(static_cast<size_t>(N) * w1o);
            linear(gu.data(), nrm.data(), w1.data(),
                   b1.size() == static_cast<size_t>(w1o) ? b1.data() : nullptr, N, hidden, w1o);
            apply_swiglu(gu.data(), N, inner);
            std::vector<float> fo(static_cast<size_t>(N) * hidden, 0.f);
            if (!w2.empty() && static_cast<int>(w2.size()) == hidden * inner) {
                // w2 is [hidden, inner]; after SwiGLU we keep the first `inner` of each row.
                std::vector<float> hid(static_cast<size_t>(N) * inner);
                for (int s = 0; s < N; ++s)
                    std::memcpy(hid.data() + static_cast<size_t>(s) * inner,
                                gu.data() + static_cast<size_t>(s) * w1o,
                                static_cast<size_t>(inner) * sizeof(float));
                linear(fo.data(), hid.data(), w2.data(),
                       b2.size() == static_cast<size_t>(hidden) ? b2.data() : nullptr, N, inner,
                       hidden);
            }
            for (int i = 0; i < N * hidden; ++i) {
                float s = (static_cast<int>(scale2.size()) == hidden)
                              ? scale2[static_cast<size_t>(i % hidden)]
                              : 1.f;
                x[i] += fo[static_cast<size_t>(i)] * s;
            }
        }
    }
}

void mix_decode(const float *z_raw, const H3VaeGeom &g, const float *A, const float *im_mean,
                const float *im_std, float *rgb) {
    const int F = g.frames;
    const int H = g.height;
    const int W = g.width;
    const int T = g.latent_t;
    const int lh = g.latent_h;
    const int lw = g.latent_w;
    const int C = g.latent_ch;
    const int sp = g.spatial > 0 ? g.spatial : 16;
    const int plane = T * lh * lw;
    const int cell = lh * lw;
    if (F <= 0 || H <= 0 || W <= 0 || T <= 0 || lh <= 0 || lw <= 0 || C <= 0)
        return;

    for (int f = 0; f < F; ++f) {
        const int t0 = clampi((f * T) / (F > 0 ? F : 1), 0, T - 1);
        for (int ly = 0; ly < lh; ++ly) {
            const int y0 = ly * sp;
            const int y1 = std::min(H, y0 + sp);
            for (int lx = 0; lx < lw; ++lx) {
                const int x0 = lx * sp;
                const int x1 = std::min(W, x0 + sp);
                float acc[3] = {0.f, 0.f, 0.f};
                const int spat = t0 * cell + ly * lw + lx;
                for (int k = 0; k < C; ++k) {
                    const float zk = z_raw[k * plane + spat];
                    acc[0] += A[0 * C + k] * zk;
                    acc[1] += A[1 * C + k] * zk;
                    acc[2] += A[2 * C + k] * zk;
                }
                float pix[3];
                for (int c = 0; c < 3; ++c) {
                    float v = std::tanh(acc[c]) * im_std[c] + im_mean[c];
                    if (v < 0.f)
                        v = 0.f;
                    if (v > 1.f)
                        v = 1.f;
                    pix[c] = v;
                }
                for (int y = y0; y < y1; ++y) {
                    float *row = rgb + (static_cast<size_t>(f) * H + y) * W * 3;
                    for (int x = x0; x < x1; ++x) {
                        row[x * 3 + 0] = pix[0];
                        row[x * 3 + 1] = pix[1];
                        row[x * 3 + 2] = pix[2];
                    }
                }
            }
        }
        // leftover strips if H/W not a multiple of spatial
        if (lh * sp < H || lw * sp < W) {
            for (int y = 0; y < H; ++y) {
                const int ly = clampi(y / sp, 0, lh - 1);
                for (int x = 0; x < W; ++x) {
                    if (y < lh * sp && x < lw * sp)
                        continue;
                    const int lx = clampi(x / sp, 0, lw - 1);
                    float acc[3] = {0.f, 0.f, 0.f};
                    const int spat = t0 * cell + ly * lw + lx;
                    for (int k = 0; k < C; ++k) {
                        const float zk = z_raw[k * plane + spat];
                        acc[0] += A[0 * C + k] * zk;
                        acc[1] += A[1 * C + k] * zk;
                        acc[2] += A[2 * C + k] * zk;
                    }
                    float *pix = rgb + ((static_cast<size_t>(f) * H + y) * W + x) * 3;
                    for (int c = 0; c < 3; ++c) {
                        float v = std::tanh(acc[c]) * im_std[c] + im_mean[c];
                        if (v < 0.f)
                            v = 0.f;
                        if (v > 1.f)
                            v = 1.f;
                        pix[c] = v;
                    }
                }
            }
        }
    }
}

void unpatch_proj(const float *tok, int N, int hid, const float *proj_w, const float *proj_b, int P,
                  const H3VaeGeom &g, const float *im_mean, const float *im_std, bool official3072,
                  float *rgb) {
    const int F = g.frames, H = g.height, W = g.width;
    const int T = g.latent_t, lh = g.latent_h, lw = g.latent_w;
    const int sp = g.spatial > 0 ? g.spatial : 16;
    std::vector<float> patch(static_cast<size_t>(N) * P);
    linear(patch.data(), tok, proj_w, proj_b, N, hid, P);

    auto denorm = [&](float v, int c) {
        float o = v * im_std[c] + im_mean[c];
        if (o < 0.f)
            o = 0.f;
        if (o > 1.f)
            o = 1.f;
        return o;
    };

    if (official3072 && P >= 3072) {
        for (int f = 0; f < F; ++f) {
            const int lt = clampi(f / 4, 0, T - 1);
            const int wt = f % 4;
            for (int y = 0; y < H; ++y) {
                const int ly = clampi(y / 16, 0, lh - 1);
                const int y0 = y % 16;
                for (int x = 0; x < W; ++x) {
                    const int lx = clampi(x / 16, 0, lw - 1);
                    const int x0 = x % 16;
                    const int ti = lt * lh * lw + ly * lw + lx;
                    float *pix = rgb + ((static_cast<size_t>(f) * H + y) * W + x) * 3;
                    for (int c = 0; c < 3; ++c) {
                        const int comp = ((c * 4 + wt) * 16 + y0) * 16 + x0;
                        pix[c] = denorm(patch[static_cast<size_t>(ti) * P + comp], c);
                    }
                }
            }
        }
        return;
    }

    // Nearest unpatch: P = 3*sp*sp (or any P>=3), one RGB patch per latent cell.
    const int patch_sp = (P >= 3 * sp * sp) ? sp : 1;
    const int stride = (P >= 3 * patch_sp * patch_sp) ? (patch_sp * patch_sp) : 1;
    for (int f = 0; f < F; ++f) {
        const int t0 = clampi((f * T) / (F > 0 ? F : 1), 0, T - 1);
        for (int y = 0; y < H; ++y) {
            const int ly = clampi(y / sp, 0, lh - 1);
            const int y0 = (patch_sp > 1) ? (y % patch_sp) : 0;
            for (int x = 0; x < W; ++x) {
                const int lx = clampi(x / sp, 0, lw - 1);
                const int x0 = (patch_sp > 1) ? (x % patch_sp) : 0;
                const int ti = t0 * lh * lw + ly * lw + lx;
                const float *pr = patch.data() + static_cast<size_t>(ti) * P;
                float *pix = rgb + ((static_cast<size_t>(f) * H + y) * W + x) * 3;
                for (int c = 0; c < 3; ++c) {
                    const int idx = c * stride + y0 * patch_sp + x0;
                    pix[c] = denorm(pr[idx < P ? idx : c], c);
                }
            }
        }
    }
}

void mix_unpatch_from_tok(const float *tok, int N, int hid, const H3VaeGeom &g, const float *A,
                          int ch, const float *im_mean, const float *im_std, float *rgb) {
    const int T = g.latent_t, lh = g.latent_h, lw = g.latent_w;
    std::vector<float> z_raw(static_cast<size_t>(ch) * N, 0.f);
    const int use = std::min(hid, ch);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < use; ++k)
            z_raw[static_cast<size_t>(k) * N + i] = tok[static_cast<size_t>(i) * hid + k];
    }
    (void)T;
    (void)lh;
    (void)lw;
    mix_decode(z_raw.data(), g, A, im_mean, im_std, rgb);
}

bool run_transformer(const H3Vae &v, const DecW &dw, const float *z_raw, const H3VaeGeom &g,
                     const float *A, const float *im_mean, const float *im_std, float *rgb) {
    const int C = g.latent_ch;
    const int T = g.latent_t, lh = g.latent_h, lw = g.latent_w;
    const int N = T * lh * lw;
    if (N <= 0 || C <= 0)
        return false;

    std::vector<float> cur(static_cast<size_t>(N) * C);
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < C; ++c)
            cur[static_cast<size_t>(i) * C + c] = z_raw[c * N + i];

    if (dw.post_w.size() == static_cast<size_t>(C) * C) {
        std::vector<float> post(static_cast<size_t>(N) * C);
        linear(post.data(), cur.data(), dw.post_w.data(),
               dw.post_b.size() == static_cast<size_t>(C) ? dw.post_b.data() : nullptr, N, C, C);
        cur.swap(post);
    }

    int hid = dw.hidden;
    std::vector<float> tok;
    if (!dw.embed_w.empty() && hid > 0 && static_cast<int>(dw.embed_w.size()) == hid * C) {
        tok.assign(static_cast<size_t>(N) * hid, 0.f);
        linear(tok.data(), cur.data(), dw.embed_w.data(),
               dw.embed_b.size() == static_cast<size_t>(hid) ? dw.embed_b.data() : nullptr, N, C, hid);
    } else if (hid > 0) {
        tok.assign(static_cast<size_t>(N) * hid, 0.f);
        const int use = std::min(hid, C);
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < use; ++c)
                tok[static_cast<size_t>(i) * hid + c] = cur[static_cast<size_t>(i) * C + c];
    } else {
        hid = C;
        tok.swap(cur);
    }

    if (dw.n_blocks > 0 && !v.source_dir.empty()) {
        std::vector<io::StFile> files;
        std::string err;
        if (io::st_open_dir(v.source_dir, files, err) == Status::Ok) {
            const int heads = dw.heads > 0 ? dw.heads : 1;
            const int hd = dw.head_dim > 0 ? dw.head_dim : hid / heads;
            for (int b = 0; b < dw.n_blocks; ++b) {
                std::vector<float> n1, qw, qb, ow, ob, s1, n2, w1, b1, w2, b2, s2;
                if (!load_block(files, b, hid, n1, qw, qb, ow, ob, s1, n2, w1, b1, w2, b2, s2))
                    continue;
                apply_block(tok.data(), N, hid, heads, hd, n1, qw, qb, ow, ob, s1, n2, w1, b1, w2, b2,
                            s2);
            }
            io::st_close_dir(files);
        }
    }

    if (!dw.ln_w.empty() && static_cast<int>(dw.ln_w.size()) == hid) {
        layernorm_rows(tok.data(), dw.ln_w.data(),
                       dw.ln_b.size() == static_cast<size_t>(hid) ? dw.ln_b.data() : nullptr, N, hid,
                       kLnEps);
    }

    if (!dw.proj_w.empty() && hid > 0 && dw.proj_dim > 0 &&
        static_cast<int>(dw.proj_w.size()) == dw.proj_dim * hid) {
        unpatch_proj(tok.data(), N, hid, dw.proj_w.data(),
                     dw.proj_b.size() == static_cast<size_t>(dw.proj_dim) ? dw.proj_b.data() : nullptr,
                     dw.proj_dim, g, im_mean, im_std, dw.proj_dim == 3072, rgb);
        return true;
    }

    mix_unpatch_from_tok(tok.data(), N, hid, g, A, C, im_mean, im_std, rgb);
    return true;
}

void official_unpack_3072(const float *rows, int pad_t, int lh, int lw, int frames, int height,
                          int width, const float *im_mean, const float *im_std, int frame_offset,
                          int output_frames, float *rgb) {
    const int P = kH3VaeOutPatch;
    const int patches = pad_t * lh * lw;
    for (int f = 0; f < frames; ++f) {
        int decoded_t = h3_vae_decoded_t(f, output_frames, frame_offset);
        int patch_t = decoded_t / 4;
        int within_t = decoded_t % 4;
        if (patch_t < 0)
            patch_t = 0;
        if (patch_t >= pad_t)
            patch_t = pad_t - 1;
        for (int y = 0; y < height; ++y) {
            int ly = y / 16, y0 = y % 16;
            if (ly >= lh)
                ly = lh - 1;
            for (int x = 0; x < width; ++x) {
                int lx = x / 16, x0 = x % 16;
                if (lx >= lw)
                    lx = lw - 1;
                const int ti = patch_t * lh * lw + ly * lw + lx;
                if (ti < 0 || ti >= patches)
                    continue;
                float *pix = rgb + (static_cast<size_t>(f) * height + y) * width * 3;
                for (int c = 0; c < 3; ++c) {
                    const int comp = ((c * 4 + within_t) * 16 + y0) * 16 + x0;
                    float v = rows[static_cast<size_t>(ti) * P + comp];
                    v = v * im_std[c] + im_mean[c];
                    if (v < 0.f)
                        v = 0.f;
                    if (v > 1.f)
                        v = 1.f;
                    pix[c] = v;
                }
            }
        }
    }
}

// Decode one (possibly padded) latent canvas to RGB [out_f, lh*16, lw*16].
bool official_decode_one(const H3Vae &v, const DecW &dw, const float *z_raw, int src_t, int lh,
                         int lw, int ch, int out_frames, int frame_offset, float *rgb) {
    const int pad_t = kH3VaeChunkT;
    const int C = ch > 0 ? ch : 24;
    if (lh < 1 || lw < 1 || src_t < 1 || out_frames < 1)
        return false;
    const int patches = pad_t * lh * lw;
    const int extra = kH3VaeSuffix;
    const int seq = patches + extra;
    int hid = dw.hidden;
    if (hid <= 0)
        hid = C;

    // Flatten [t,h,w,c] with time padded to 7 by repeating last.
    std::vector<float> rows(static_cast<size_t>(patches) * C);
    for (int t = 0; t < pad_t; ++t) {
        int st = t < src_t ? t : src_t - 1;
        for (int y = 0; y < lh; ++y)
            for (int x = 0; x < lw; ++x)
                for (int c = 0; c < C; ++c) {
                    size_t src = ((static_cast<size_t>(c) * src_t + st) * lh + y) * lw + x;
                    size_t dst = ((static_cast<size_t>(t) * lh + y) * lw + x) * C + c;
                    rows[dst] = z_raw[src];
                }
    }
    if (dw.post_w.size() == static_cast<size_t>(C) * C) {
        std::vector<float> post(static_cast<size_t>(patches) * C);
        linear(post.data(), rows.data(), dw.post_w.data(),
               dw.post_b.size() == static_cast<size_t>(C) ? dw.post_b.data() : nullptr, patches, C, C);
        rows.swap(post);
    }
    std::vector<float> tok(static_cast<size_t>(seq) * hid, 0.f);
    if (!dw.embed_w.empty() && static_cast<int>(dw.embed_w.size()) == hid * C) {
        linear(tok.data(), rows.data(), dw.embed_w.data(),
               dw.embed_b.size() == static_cast<size_t>(hid) ? dw.embed_b.data() : nullptr, patches, C,
               hid);
    } else {
        const int use = std::min(hid, C);
        for (int i = 0; i < patches; ++i)
            for (int c = 0; c < use; ++c)
                tok[static_cast<size_t>(i) * hid + c] = rows[static_cast<size_t>(i) * C + c];
    }
    if (static_cast<int>(dw.registers.size()) >= kH3VaeRegisters * hid) {
        std::memcpy(tok.data() + static_cast<size_t>(patches) * hid, dw.registers.data(),
                    static_cast<size_t>(kH3VaeRegisters) * hid * sizeof(float));
    }
    // suffix token stays zero

    std::vector<float> rcos, rsin;
    vae_rope_tables(pad_t, lh, lw, extra, rcos, rsin);

    int heads = dw.heads > 0 ? dw.heads : 1;
    int hd = dw.head_dim > 0 ? dw.head_dim : hid / heads;
    if (heads * hd != hid) {
        heads = 1;
        hd = hid;
    }

    if (dw.n_blocks > 0 && !v.source_dir.empty()) {
        std::vector<io::StFile> files;
        std::string err;
        if (io::st_open_dir(v.source_dir, files, err) == Status::Ok) {
            for (int b = 0; b < dw.n_blocks; ++b) {
                std::vector<float> n1, qw, qb, ow, ob, s1, n2, w1, b1, w2, b2, s2;
                if (!load_block(files, b, hid, n1, qw, qb, ow, ob, s1, n2, w1, b1, w2, b2, s2))
                    continue;
                apply_block_official(tok.data(), seq, hid, heads, hd, rcos.data(), rsin.data(), n1, qw,
                                     qb, ow, ob, s1, n2, w1, b1, w2, b2, s2);
            }
            io::st_close_dir(files);
        }
    }

    if (!dw.ln_w.empty() && static_cast<int>(dw.ln_w.size()) == hid)
        layernorm_rows(tok.data(), dw.ln_w.data(),
                       dw.ln_b.size() == static_cast<size_t>(hid) ? dw.ln_b.data() : nullptr, seq, hid,
                       kLnEps);

    const int ph = lh * 16, pw = lw * 16;
    if (!dw.proj_w.empty() && dw.proj_dim == kH3VaeOutPatch &&
        static_cast<int>(dw.proj_w.size()) == kH3VaeOutPatch * hid) {
        std::vector<float> proj(static_cast<size_t>(patches) * kH3VaeOutPatch);
        linear(proj.data(), tok.data(), dw.proj_w.data(),
               dw.proj_b.size() == static_cast<size_t>(kH3VaeOutPatch) ? dw.proj_b.data() : nullptr,
               patches, hid, kH3VaeOutPatch);
        official_unpack_3072(proj.data(), pad_t, lh, lw, out_frames, ph, pw, v.mean.data(),
                             v.std.data(), frame_offset, out_frames, rgb);
        return true;
    }
    return false;
}

void stitch_rgb(float *dst, int full_h, int full_w, const float *tile, int tile_h, int tile_w,
                int frames, int y0, int x0, int keep_h, int keep_w, const float *above,
                int overlap_y, const float *left, int overlap_x) {
    for (int f = 0; f < frames; ++f)
        for (int y = 0; y < keep_h; ++y)
            for (int x = 0; x < keep_w; ++x)
                for (int c = 0; c < 3; ++c) {
                    float value = tile[(((static_cast<size_t>(f) * tile_h + y) * tile_w + x) * 3) + c];
                    if (above && y < overlap_y) {
                        size_t top = (((static_cast<size_t>(f) * tile_h + (tile_h - overlap_y + y)) *
                                           tile_w +
                                       x) *
                                          3) +
                                     c;
                        float a = static_cast<float>(y) / static_cast<float>(overlap_y);
                        value = above[top] * (1.f - a) + value * a;
                    }
                    if (left && x < overlap_x) {
                        size_t prior =
                            (((static_cast<size_t>(f) * tile_h + y) * tile_w + (tile_w - overlap_x + x)) *
                                 3) +
                            c;
                        float a = static_cast<float>(x) / static_cast<float>(overlap_x);
                        value = left[prior] * (1.f - a) + value * a;
                    }
                    dst[(((static_cast<size_t>(f) * full_h + (y0 + y)) * full_w + (x0 + x)) * 3) + c] =
                        value;
                }
}

bool official_decode_canvas(const H3Vae &v, const DecW &dw, const float *z_raw, int T, int lh,
                            int lw, int ch, int out_frames, int frame_offset, float *rgb) {
    const int ph = lh * 16, pw = lw * 16;
    const int tile_px = kH3VaeTilePixels;
    if (ph <= tile_px && pw <= tile_px)
        return official_decode_one(v, dw, z_raw, T, lh, lw, ch, out_frames, frame_offset, rgb);

    TileAxis ya = make_tile_axis(ph, tile_px);
    TileAxis xa = make_tile_axis(pw, tile_px);
    const int th = ya.length / 16, tw = xa.length / 16;
    const int ntiles = ya.count * xa.count;
    std::vector<std::vector<float>> tiles(static_cast<size_t>(ntiles));
    for (int ty = 0; ty < ya.count; ++ty) {
        for (int tx = 0; tx < xa.count; ++tx) {
            std::vector<float> ztile(static_cast<size_t>(ch) * T * th * tw);
            extract_latent_tile(z_raw, T, lh, lw, ch, 0, ya.starts[static_cast<size_t>(ty)] / 16,
                                xa.starts[static_cast<size_t>(tx)] / 16, T, th, tw, ztile.data());
            tiles[static_cast<size_t>(ty * xa.count + tx)].assign(
                static_cast<size_t>(out_frames) * ya.length * xa.length * 3, 0.f);
            if (!official_decode_one(v, dw, ztile.data(), T, th, tw, ch, out_frames, frame_offset,
                                     tiles[static_cast<size_t>(ty * xa.count + tx)].data()))
                return false;
        }
    }
    const int full_h = ya.starts.back() + ya.length;
    const int full_w = xa.starts.back() + xa.length;
    std::vector<float> full(static_cast<size_t>(out_frames) * full_h * full_w * 3, 0.f);
    for (int ty = 0; ty < ya.count; ++ty)
        for (int tx = 0; tx < xa.count; ++tx) {
            int idx = ty * xa.count + tx;
            const float *above = ty ? tiles[static_cast<size_t>(idx - xa.count)].data() : nullptr;
            const float *left = tx ? tiles[static_cast<size_t>(idx - 1)].data() : nullptr;
            int oy = ty ? ya.overlaps[static_cast<size_t>(ty - 1)] : 0;
            int ox = tx ? xa.overlaps[static_cast<size_t>(tx - 1)] : 0;
            int kh = ya.length - (ty + 1 < ya.count ? ya.overlaps[static_cast<size_t>(ty)] : 0);
            int kw = xa.length - (tx + 1 < xa.count ? xa.overlaps[static_cast<size_t>(tx)] : 0);
            stitch_rgb(full.data(), full_h, full_w, tiles[static_cast<size_t>(idx)].data(), ya.length,
                       xa.length, out_frames, ya.starts[static_cast<size_t>(ty)],
                       xa.starts[static_cast<size_t>(tx)], kh, kw, above, oy, left, ox);
        }
    // Copy into caller canvas (ph x pw); tile plan covers at least that.
    for (int f = 0; f < out_frames; ++f)
        for (int y = 0; y < ph && y < full_h; ++y)
            std::memcpy(rgb + (static_cast<size_t>(f) * ph + y) * pw * 3,
                        full.data() + (static_cast<size_t>(f) * full_h + y) * full_w * 3,
                        static_cast<size_t>(std::min(pw, full_w)) * 3 * sizeof(float));
    return true;
}

bool run_official(const H3Vae &v, const DecW &dw, const float *z_raw, const H3VaeGeom &g,
                  float *rgb) {
    const int C = g.latent_ch > 0 ? g.latent_ch : 24;
    const int T = g.latent_t, lh = g.latent_h, lw = g.latent_w;
    const int F = g.frames, H = g.height, W = g.width;
    const int ph = lh * 16, pw = lw * 16;
    if (T >= kH3VaeChunkT && (T - 2) % 5 == 0 && T > kH3VaeChunkT) {
        const int chunks = (T - 2) / 5;
        const int out_f = chunks * 17 + 5;
        std::vector<float> final_rgb(static_cast<size_t>(out_f) * ph * pw * 3, 0.f);
        std::vector<float> overlap(static_cast<size_t>(5) * ph * pw * 3, 0.f);
        const size_t fe = static_cast<size_t>(ph) * pw * 3;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            std::vector<float> zc(static_cast<size_t>(C) * kH3VaeChunkT * lh * lw);
            extract_latent_tile(z_raw, T, lh, lw, C, chunk * 5, 0, 0, kH3VaeChunkT, lh, lw,
                                zc.data());
            std::vector<float> dec(static_cast<size_t>(kH3VaeFirstChunkFrames) * ph * pw * 3, 0.f);
            if (!official_decode_canvas(v, dw, zc.data(), kH3VaeChunkT, lh, lw, C,
                                        kH3VaeFirstChunkFrames, kH3VaeFrameOffset, dec.data()))
                return false;
            if (chunk) {
                for (int f = 0; f < 5; ++f) {
                    float a = static_cast<float>(f) / 5.f;
                    float *row = dec.data() + static_cast<size_t>(f) * fe;
                    const float *prev = overlap.data() + static_cast<size_t>(f) * fe;
                    for (size_t i = 0; i < fe; ++i)
                        row[i] = prev[i] * (1.f - a) + row[i] * a;
                }
            }
            std::memcpy(final_rgb.data() + static_cast<size_t>(chunk) * 17 * fe, dec.data(),
                        17 * fe * sizeof(float));
            std::memcpy(overlap.data(), dec.data() + 17 * fe, 5 * fe * sizeof(float));
        }
        std::memcpy(final_rgb.data() + static_cast<size_t>(chunks) * 17 * fe, overlap.data(),
                    5 * fe * sizeof(float));
        const int use_f = std::min(F, out_f);
        const int use_h = std::min(H, ph);
        const int use_w = std::min(W, pw);
        for (int f = 0; f < use_f; ++f)
            for (int y = 0; y < use_h; ++y)
                std::memcpy(rgb + (static_cast<size_t>(f) * H + y) * W * 3,
                            final_rgb.data() + (static_cast<size_t>(f) * ph + y) * pw * 3,
                            static_cast<size_t>(use_w) * 3 * sizeof(float));
        return true;
    }
    const int out_f = (T == 2) ? 5 : kH3VaeFirstChunkFrames;
    std::vector<float> dec(static_cast<size_t>(out_f) * ph * pw * 3, 0.f);
    if (!official_decode_canvas(v, dw, z_raw, T, lh, lw, C, out_f, kH3VaeFrameOffset, dec.data()))
        return false;
    const int use_f = std::min(F, out_f);
    const int use_h = std::min(H, ph);
    const int use_w = std::min(W, pw);
    for (int f = 0; f < use_f; ++f)
        for (int y = 0; y < use_h; ++y)
            std::memcpy(rgb + (static_cast<size_t>(f) * H + y) * W * 3,
                        dec.data() + (static_cast<size_t>(f) * ph + y) * pw * 3,
                        static_cast<size_t>(use_w) * 3 * sizeof(float));
    return true;
}

float silu(float x) { return x / (1.f + std::exp(-x)); }

void conv3d(const float *in, int D, int H, int W, const EncConv &c, std::vector<float> &out, int &oD,
            int &oH, int &oW) {
    const int k = c.k > 0 ? c.k : 1;
    const int st = c.st > 0 ? c.st : 1;
    const int sh = c.sh > 0 ? c.sh : 1;
    const int sw = c.sw > 0 ? c.sw : 1;
    const int pD = D + c.pad_t;
    const int pH = H + c.pad_h0 + c.pad_h1;
    const int pW = W + c.pad_w0 + c.pad_w1;
    oD = (pD - k) / st + 1;
    oH = (pH - k) / sh + 1;
    oW = (pW - k) / sw + 1;
    if (oD < 1)
        oD = 1;
    if (oH < 1)
        oH = 1;
    if (oW < 1)
        oW = 1;
    out.assign(static_cast<size_t>(oD) * oH * oW * c.out_ch, 0.f);
    const bool has_w = static_cast<int>(c.w.size()) == c.out_ch * c.in_ch * k * k * k;
    for (int od = 0; od < oD; ++od)
        for (int oh = 0; oh < oH; ++oh)
            for (int ow = 0; ow < oW; ++ow)
                for (int oc = 0; oc < c.out_ch; ++oc) {
                    float acc = (static_cast<int>(c.b.size()) == c.out_ch)
                                    ? c.b[static_cast<size_t>(oc)]
                                    : 0.f;
                    if (has_w) {
                        for (int ic = 0; ic < c.in_ch; ++ic)
                            for (int kt = 0; kt < k; ++kt)
                                for (int kh = 0; kh < k; ++kh)
                                    for (int kw = 0; kw < k; ++kw) {
                                        const int id = od * st + kt - c.pad_t;
                                        const int ih = oh * sh + kh - c.pad_h0;
                                        const int iw = ow * sw + kw - c.pad_w0;
                                        if (id < 0 || id >= D || ih < 0 || ih >= H || iw < 0 ||
                                            iw >= W)
                                            continue;
                                        const float xv =
                                            in[((static_cast<size_t>(id) * H + ih) * W + iw) * c.in_ch +
                                               ic];
                                        const float wv =
                                            c.w[((((static_cast<size_t>(oc) * c.in_ch + ic) * k + kt) *
                                                      k +
                                                  kh) *
                                                     k +
                                                 kw)];
                                        acc += xv * wv;
                                    }
                    }
                    out[((static_cast<size_t>(od) * oH + oh) * oW + ow) * c.out_ch + oc] = acc;
                }
}

void group_norm_silu(float *x, int D, int H, int W, int C, int groups, const float *gw,
                     const float *gb, float eps) {
    if (groups < 1 || C % groups != 0)
        groups = 1;
    const int cpg = C / groups;
    const int hw = H * W;
    const int elem = hw * cpg;
    for (int d = 0; d < D; ++d) {
        for (int g = 0; g < groups; ++g) {
            float mean = 0.f;
            for (int s = 0; s < hw; ++s)
                for (int ci = 0; ci < cpg; ++ci) {
                    int c = g * cpg + ci;
                    mean += x[(static_cast<size_t>(d) * hw + s) * C + c];
                }
            mean /= static_cast<float>(elem > 0 ? elem : 1);
            float var = 0.f;
            for (int s = 0; s < hw; ++s)
                for (int ci = 0; ci < cpg; ++ci) {
                    int c = g * cpg + ci;
                    float z = x[(static_cast<size_t>(d) * hw + s) * C + c] - mean;
                    var += z * z;
                }
            float inv = 1.f / std::sqrt(var / static_cast<float>(elem > 0 ? elem : 1) + eps);
            for (int s = 0; s < hw; ++s)
                for (int ci = 0; ci < cpg; ++ci) {
                    int c = g * cpg + ci;
                    float *p = &x[(static_cast<size_t>(d) * hw + s) * C + c];
                    float v = (*p - mean) * inv;
                    if (gw)
                        v *= gw[c];
                    if (gb)
                        v += gb[c];
                    *p = silu(v);
                }
        }
    }
}

bool fill_enc_conv(const std::vector<io::StFile> &files, const std::string &prefix, EncConv &c,
                   int in_ch, int out_ch, int k, int st, int sh, int sw, int pad_t, int ph0, int ph1,
                   int pw0, int pw1) {
    c = EncConv{};
    c.in_ch = in_ch;
    c.out_ch = out_ch;
    c.k = k;
    c.st = st;
    c.sh = sh;
    c.sw = sw;
    c.pad_t = pad_t;
    c.pad_h0 = ph0;
    c.pad_h1 = ph1;
    c.pad_w0 = pw0;
    c.pad_w1 = pw1;
    std::vector<int64_t> shs;
    if (!read_vec(files, prefix + ".weight", c.w, &shs))
        return false;
    read_vec(files, prefix + ".bias", c.b);
    if (shs.size() >= 2) {
        c.out_ch = static_cast<int>(shs[0]);
        c.in_ch = static_cast<int>(shs[1]);
    }
    return !c.w.empty();
}

int pick_groups(int *chs, int n) {
    for (int g = 32; g >= 1; --g) {
        bool ok = true;
        for (int i = 0; i < n && ok; ++i)
            if (chs[i] % g)
                ok = false;
        if (ok)
            return g;
    }
    return 1;
}

bool load_encoder(const std::vector<io::StFile> &files, EncW &ew) {
    ew = EncW{};
    if (!fill_enc_conv(files, "encoder.conv_in", ew.conv_in, 3, 128, 3, 1, 1, 1, 2, 1, 1, 1, 1))
        return false;
    int prev = ew.conv_in.out_ch;
    for (int L = 0; L < 6; ++L) {
        const std::string p0 = "encoder.down." + std::to_string(L) + ".block.0.conv1";
        std::vector<int64_t> sh;
        std::vector<float> tmp;
        if (!read_vec(files, p0 + ".weight", tmp, &sh) || sh.empty())
            return false;
        ew.ch[L] = static_cast<int>(sh[0]);
        for (int b = 0; b < 2; ++b) {
            EncBlock &blk = ew.block[L][b];
            int in_ch = b ? ew.ch[L] : prev;
            const std::string bp = "encoder.down." + std::to_string(L) + ".block." + std::to_string(b);
            read_vec(files, bp + ".norm1.weight", blk.n1w);
            read_vec(files, bp + ".norm1.bias", blk.n1b);
            if (!fill_enc_conv(files, bp + ".conv1", blk.conv1, in_ch, ew.ch[L], 3, 1, 1, 1, 2, 1, 1,
                               1, 1))
                return false;
            read_vec(files, bp + ".norm2.weight", blk.n2w);
            read_vec(files, bp + ".norm2.bias", blk.n2b);
            if (!fill_enc_conv(files, bp + ".conv2", blk.conv2, ew.ch[L], ew.ch[L], 3, 1, 1, 1, 2, 1,
                               1, 1, 1))
                return false;
            if (in_ch != ew.ch[L]) {
                blk.has_shortcut = fill_enc_conv(files, bp + ".nin_shortcut", blk.shortcut, in_ch,
                                                 ew.ch[L], 1, 1, 1, 1, 0, 0, 0, 0, 0);
            }
        }
        if (ew.space_st[L] * ew.time_st[L] > 1) {
            int tail = ew.space_st[L] == 2 ? 1 : 0;
            ew.has_down[L] = fill_enc_conv(files, "encoder.down." + std::to_string(L) + ".downsample.conv",
                                           ew.down[L], ew.ch[L], ew.ch[L], 3, ew.time_st[L],
                                           ew.space_st[L], ew.space_st[L], 2, 0, tail, 0, tail);
            if (!ew.has_down[L])
                return false;
        }
        prev = ew.ch[L];
    }
    read_vec(files, "encoder.norm_out.weight", ew.nout_w);
    read_vec(files, "encoder.norm_out.bias", ew.nout_b);
    const int last = ew.ch[5];
    if (!fill_enc_conv(files, "encoder.conv_out", ew.conv_out, last, 48, 3, 1, 1, 1, 2, 1, 1, 1, 1))
        return false;
    if (!fill_enc_conv(files, "quant_conv", ew.quant, ew.conv_out.out_ch, ew.conv_out.out_ch, 1, 1, 1,
                       1, 0, 0, 0, 0, 0))
        return false;
    int gch[7];
    gch[0] = ew.conv_in.out_ch;
    for (int i = 0; i < 6; ++i)
        gch[i + 1] = ew.ch[i];
    ew.groups = pick_groups(gch, 7);
    ew.ready = !ew.conv_in.w.empty() && !ew.conv_out.w.empty() && !ew.quant.w.empty();
    return ew.ready;
}

// Residual block: GN-SiLU-Conv, GN-SiLU-Conv, add shortcut or identity.
void enc_residual(std::vector<float> &act, int &D, int &H, int &W, int in_ch, const EncBlock &blk,
                  int groups) {
    const int oD = D, oH = H, oW = W;
    std::vector<float> branch = act;
    const float *nw = blk.n1w.size() == static_cast<size_t>(in_ch) ? blk.n1w.data() : nullptr;
    const float *nb = blk.n1b.size() == static_cast<size_t>(in_ch) ? blk.n1b.data() : nullptr;
    group_norm_silu(branch.data(), D, H, W, in_ch, groups, nw, nb, 1e-6f);
    std::vector<float> h;
    int nd = D, nh = H, nw_ = W;
    conv3d(branch.data(), D, H, W, blk.conv1, h, nd, nh, nw_);
    const int och = blk.conv1.out_ch;
    nw = blk.n2w.size() == static_cast<size_t>(och) ? blk.n2w.data() : nullptr;
    nb = blk.n2b.size() == static_cast<size_t>(och) ? blk.n2b.data() : nullptr;
    group_norm_silu(h.data(), nd, nh, nw_, och, groups, nw, nb, 1e-6f);
    std::vector<float> out;
    int od = nd, oh = nh, ow = nw_;
    conv3d(h.data(), nd, nh, nw_, blk.conv2, out, od, oh, ow);
    std::vector<float> residual;
    if (blk.has_shortcut) {
        int rd = oD, rh = oH, rw = oW;
        conv3d(act.data(), oD, oH, oW, blk.shortcut, residual, rd, rh, rw);
    } else {
        residual = act;
    }
    const int n = static_cast<int>(out.size());
    if (static_cast<int>(residual.size()) == n)
        for (int i = 0; i < n; ++i)
            out[static_cast<size_t>(i)] += residual[static_cast<size_t>(i)];
    act.swap(out);
    D = od;
    H = oh;
    W = ow;
}

bool encode_tile_official(const EncW &ew, const float *rgb_hwc, int F, int Ht, int Wt, int ch,
                          const float *im_mean, const float *im_std, const float *lm,
                          const float *ls, std::vector<float> &z_ch, int &t_out, int &h_out,
                          int &w_out) {
    // rgb_hwc is frame-major HWC. Build [D,H,W,3] ImageNet-normalized.
    std::vector<float> act(static_cast<size_t>(F) * Ht * Wt * 3);
    for (int t = 0; t < F; ++t)
        for (int y = 0; y < Ht; ++y)
            for (int x = 0; x < Wt; ++x)
                for (int c = 0; c < 3; ++c) {
                    float v = rgb_hwc[((static_cast<size_t>(t) * Ht + y) * Wt + x) * 3 + c];
                    act[((static_cast<size_t>(t) * Ht + y) * Wt + x) * 3 + c] =
                        (v - im_mean[c]) / im_std[c];
                }
    int D = F, H = Ht, W = Wt;
    std::vector<float> next;
    int nD, nH, nW;
    conv3d(act.data(), D, H, W, ew.conv_in, next, nD, nH, nW);
    act.swap(next);
    D = nD;
    H = nH;
    W = nW;
    int prev = ew.conv_in.out_ch;
    for (int L = 0; L < 6; ++L) {
        for (int b = 0; b < 2; ++b) {
            int in_ch = b ? ew.ch[L] : prev;
            enc_residual(act, D, H, W, in_ch, ew.block[L][b], ew.groups);
        }
        if (ew.has_down[L]) {
            conv3d(act.data(), D, H, W, ew.down[L], next, nD, nH, nW);
            act.swap(next);
            D = nD;
            H = nH;
            W = nW;
        }
        prev = ew.ch[L];
    }
    const float *nw = ew.nout_w.size() == static_cast<size_t>(prev) ? ew.nout_w.data() : nullptr;
    const float *nb = ew.nout_b.size() == static_cast<size_t>(prev) ? ew.nout_b.data() : nullptr;
    group_norm_silu(act.data(), D, H, W, prev, ew.groups, nw, nb, 1e-6f);
    conv3d(act.data(), D, H, W, ew.conv_out, next, nD, nH, nW);
    act.swap(next);
    D = nD;
    H = nH;
    W = nW;
    conv3d(act.data(), D, H, W, ew.quant, next, nD, nH, nW);
    act.swap(next);
    D = nD;
    H = nH;
    W = nW;
    const int mout = ew.quant.out_ch > 0 ? ew.quant.out_ch : 48;
    const int C = ch > 0 ? ch : 24;
    z_ch.assign(static_cast<size_t>(C) * D * H * W, 0.f);
    for (int c = 0; c < C && c < mout; ++c) {
        float m = lm ? lm[c] : 0.f;
        float s = ls ? ls[c] : 1.f;
        if (!(s > 1e-8f))
            s = 1.f;
        for (int t = 0; t < D; ++t)
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    float v = act[((static_cast<size_t>(t) * H + y) * W + x) * mout + c];
                    z_ch[((static_cast<size_t>(c) * D + t) * H + y) * W + x] = (v - m) / s;
                }
    }
    t_out = D;
    h_out = H;
    w_out = W;
    return true;
}

void extract_rgb_tile(const float *rgb, int F, int full_h, int full_w, int y0, int x0, int th,
                      int tw, float *out) {
    for (int t = 0; t < F; ++t)
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x)
                for (int c = 0; c < 3; ++c) {
                    int yy = y0 + y, xx = x0 + x;
                    float v = 0.f;
                    if (yy >= 0 && yy < full_h && xx >= 0 && xx < full_w)
                        v = rgb[((static_cast<size_t>(t) * full_h + yy) * full_w + xx) * 3 + c];
                    out[((static_cast<size_t>(t) * th + y) * tw + x) * 3 + c] = v;
                }
}

void stitch_latents(const std::vector<std::vector<float>> &tiles, int T, int C, const TileAxis &ya,
                    const TileAxis &xa, float *z, int full_lh, int full_lw) {
    const int tile_h = ya.length / 16;
    const int tile_w = xa.length / 16;
    for (int ty = 0; ty < ya.count; ++ty)
        for (int tx = 0; tx < xa.count; ++tx) {
            const int idx = ty * xa.count + tx;
            const float *cur = tiles[static_cast<size_t>(idx)].data();
            const float *above = ty ? tiles[static_cast<size_t>(idx - xa.count)].data() : nullptr;
            const float *left = tx ? tiles[static_cast<size_t>(idx - 1)].data() : nullptr;
            const int oy = ty ? ya.overlaps[static_cast<size_t>(ty - 1)] / 16 : 0;
            const int ox = tx ? xa.overlaps[static_cast<size_t>(tx - 1)] / 16 : 0;
            const int kh = tile_h - (ty + 1 < ya.count ? ya.overlaps[static_cast<size_t>(ty)] / 16 : 0);
            const int kw = tile_w - (tx + 1 < xa.count ? xa.overlaps[static_cast<size_t>(tx)] / 16 : 0);
            const int dy = ya.starts[static_cast<size_t>(ty)] / 16;
            const int dx = xa.starts[static_cast<size_t>(tx)] / 16;
            for (int c = 0; c < C; ++c)
                for (int t = 0; t < T; ++t)
                    for (int y = 0; y < kh && dy + y < full_lh; ++y)
                        for (int x = 0; x < kw && dx + x < full_lw; ++x) {
                            float v = cur[((static_cast<size_t>(c) * T + t) * tile_h + y) * tile_w + x];
                            if (above && y < oy) {
                                float a = static_cast<float>(y) / static_cast<float>(oy);
                                float u = above[((static_cast<size_t>(c) * T + t) * tile_h +
                                                 (tile_h - oy + y)) *
                                                    tile_w +
                                                x];
                                v = u * (1.f - a) + v * a;
                            }
                            if (left && x < ox) {
                                float a = static_cast<float>(x) / static_cast<float>(ox);
                                float u = left[((static_cast<size_t>(c) * T + t) * tile_h + y) * tile_w +
                                               (tile_w - ox + x)];
                                v = u * (1.f - a) + v * a;
                            }
                            z[((static_cast<size_t>(c) * T + t) * full_lh + (dy + y)) * full_lw +
                              (dx + x)] = v;
                        }
        }
}

bool run_official_encode(const EncW &ew, const float *rgb, const H3VaeGeom &g, const float *im_mean,
                         const float *im_std, const float *lm, const float *ls, float *z) {
    const int F = g.frames, Ht = g.height, Wt = g.width;
    const int C = g.latent_ch > 0 ? g.latent_ch : 24;
    const int lh = g.latent_h, lw = g.latent_w;
    const int Td = g.latent_t;
    if (F < 1 || Ht < 16 || Wt < 16 || (Ht % 16) || (Wt % 16))
        return false;
    TileAxis ya = make_tile_axis(Ht, kH3VaeTilePixels);
    TileAxis xa = make_tile_axis(Wt, kH3VaeTilePixels);
    std::vector<std::vector<float>> tiles(static_cast<size_t>(ya.count * xa.count));
    int t_enc = 0, h0 = 0, w0 = 0;
    for (int ty = 0; ty < ya.count; ++ty)
        for (int tx = 0; tx < xa.count; ++tx) {
            std::vector<float> pix(static_cast<size_t>(F) * ya.length * xa.length * 3);
            extract_rgb_tile(rgb, F, Ht, Wt, ya.starts[static_cast<size_t>(ty)],
                             xa.starts[static_cast<size_t>(tx)], ya.length, xa.length, pix.data());
            int te = 0, he = 0, we = 0;
            if (!encode_tile_official(ew, pix.data(), F, ya.length, xa.length, C, im_mean, im_std, lm,
                                      ls, tiles[static_cast<size_t>(ty * xa.count + tx)], te, he, we))
                return false;
            if (ty == 0 && tx == 0) {
                t_enc = te;
                h0 = he;
                w0 = we;
            } else if (te != t_enc)
                return false;
        }
    const int full_lh = (ya.starts.back() + ya.length) / 16;
    const int full_lw = (xa.starts.back() + xa.length) / 16;
    std::vector<float> full(static_cast<size_t>(C) * t_enc * full_lh * full_lw, 0.f);
    stitch_latents(tiles, t_enc, C, ya, xa, full.data(), full_lh, full_lw);
    const int use_t = std::min(Td, t_enc);
    const int use_h = std::min(lh, full_lh);
    const int use_w = std::min(lw, full_lw);
    std::memset(z, 0, static_cast<size_t>(C) * Td * lh * lw * sizeof(float));
    for (int c = 0; c < C; ++c)
        for (int t = 0; t < use_t; ++t)
            for (int y = 0; y < use_h; ++y)
                for (int x = 0; x < use_w; ++x)
                    z[((static_cast<size_t>(c) * Td + t) * lh + y) * lw + x] =
                        full[((static_cast<size_t>(c) * t_enc + t) * full_lh + y) * full_lw + x];
    (void)h0;
    (void)w0;
    return true;
}

void try_load_encoder_rgb(const std::vector<io::StFile> &files, int ch, DecW &dw) {
    std::vector<int64_t> sh;
    std::vector<float> w, b;
    const char *names[] = {"quant_conv.weight", "encoder.conv_out.weight"};
    if (!read_first(files, names, 2, w, &sh))
        return;
    const int64_t n = static_cast<int64_t>(w.size());
    if (n == static_cast<int64_t>(ch) * 3) {
        dw.enc_w.swap(w);
        const char *bnames[] = {"quant_conv.bias", "encoder.conv_out.bias"};
        read_first(files, bnames, 2, dw.enc_b);
        return;
    }
    if (!sh.empty() && sh[0] == ch && sh.size() >= 2 && sh[1] == 3) {
        // [ch, 3, k, k, k] — only accept 1x1x1
        int64_t rest = 1;
        for (size_t i = 2; i < sh.size(); ++i)
            rest *= sh[i];
        if (rest == 1) {
            dw.enc_w.swap(w);
            const char *bnames[] = {"quant_conv.bias", "encoder.conv_out.bias"};
            read_first(files, bnames, 2, dw.enc_b);
        }
    }
}

int infer_heads(int hidden, const H3Config &h3) {
    if (h3.vae_head_dim > 0 && hidden % h3.vae_head_dim == 0)
        return hidden / h3.vae_head_dim;
    if (h3.vae_heads > 0 && hidden % h3.vae_heads == 0)
        return h3.vae_heads;
    for (int d : {64, 32, 16, 8, 4, 2}) {
        if (hidden % d == 0)
            return hidden / d;
    }
    return 1;
}

} // namespace

int h3_align_frames(int frames) {
    int F = frames < 5 ? 5 : frames;
    while ((F - 5) % 17 != 0)
        ++F;
    return F;
}

int h3_video_latent_t(int frames) {
    const int F = h3_align_frames(frames);
    if (F <= 5)
        return 2;
    return ((F - 5) / 17) * 5 + 2;
}

int h3_encoder_latent_t(int frames) {
    if (frames < 0)
        frames = 0;
    return (frames + 3) / 4;
}

void h3_latent_canvas(int width, int height, int spatial, int *lw, int *lh) {
    const int s = spatial > 0 ? spatial : 16;
    int w = width / s;
    int h = height / s;
    if (w < 1)
        w = 1;
    if (h < 1)
        h = 1;
    if (lw)
        *lw = w;
    if (lh)
        *lh = h;
}

int h3_vae_decoded_t(int frame, int output_frames, int offset) {
    int decoded_t = frame + offset;
    if (output_frames == kH3VaeFirstChunkFrames && frame >= 17)
        decoded_t += 3;
    return decoded_t;
}

int h3_vae_tile_count(int pixel_extent, int tile_pixels) {
    if (pixel_extent < 1)
        return 0;
    if (tile_pixels < 1)
        tile_pixels = kH3VaeTilePixels;
    return tile_count_for_extent(pixel_extent, tile_pixels);
}

void h3_vae_unpack_3072(const float *rows, int latent_t, int latent_h, int latent_w, int frames,
                        int height, int width, const float *im_mean, const float *im_std,
                        int frame_offset, float *rgb) {
    if (!rows || !rgb || latent_t < 1 || latent_h < 1 || latent_w < 1 || frames < 1)
        return;
    float m[3] = {kImMean[0], kImMean[1], kImMean[2]};
    float s[3] = {kImStd[0], kImStd[1], kImStd[2]};
    if (im_mean)
        std::memcpy(m, im_mean, sizeof(m));
    if (im_std)
        std::memcpy(s, im_std, sizeof(s));
    official_unpack_3072(rows, latent_t, latent_h, latent_w, frames, height, width, m, s,
                         frame_offset, frames, rgb);
}

H3VaeGeom h3_vae_geom(int width, int height, int frames, int spatial, int latent_ch) {
    H3VaeGeom g;
    g.frames = h3_align_frames(frames);
    g.width = width > 0 ? width : 1;
    g.height = height > 0 ? height : 1;
    g.spatial = spatial > 0 ? spatial : 16;
    g.latent_ch = latent_ch > 0 ? latent_ch : 24;
    g.latent_t = h3_video_latent_t(g.frames);
    h3_latent_canvas(g.width, g.height, g.spatial, &g.latent_w, &g.latent_h);
    return g;
}

Status H3Vae::load(const ::std::string &model_dir, const H3Config &h3, ::std::string &err) {
    const int ch = h3.vae_latent_ch > 0 ? h3.vae_latent_ch : 24;
    from_checkpoint = false;
    official_decode = false;
    official_encode = false;
    source_dir.clear();
    geom = h3_vae_geom(h3.default_width, h3.default_height, h3.default_frames, h3.vae_spatial, ch);
    mean = {kImMean[0], kImMean[1], kImMean[2]};
    this->std = {kImStd[0], kImStd[1], kImStd[2]};
    latents_mean.assign(static_cast<size_t>(ch), 0.f);
    latents_std.assign(static_cast<size_t>(ch), 1.f);
    decw(this) = DecW{};
    encw(this) = EncW{};

    const char *tails[] = {"/FL2VA/video_vae/source", "/video_vae/source", "/FL2VA/video_vae", ""};
    for (const char *tail : tails) {
        const ::std::string dir = model_dir + tail;
        ::std::vector<io::StFile> files;
        ::std::string oerr;
        Status st = io::st_open_dir(dir, files, oerr);
        if (st != Status::Ok || files.empty()) {
            if (!files.empty())
                io::st_close_dir(files);
            continue;
        }
        const bool dec_hit = find_named(files, "decoder.x_embedder.weight").tensor ||
                             find_named(files, "decoder.proj_in.weight").tensor ||
                             find_named(files, "decoder.transformer_blocks.0.attn.to_qkv.weight")
                                 .tensor ||
                             find_named(files, "decoder.transformer_blocks.0.attn.to_q.weight").tensor;
        const bool enc_hit = find_named(files, "encoder.conv_in.weight").tensor;
        if (!dec_hit && !enc_hit) {
            io::st_close_dir(files);
            continue;
        }

        source_dir = dir;
        from_checkpoint = true;
        DecW &dw = decw(this);

        ::std::vector<int64_t> sh;
        if (!read_vec(files, "decoder.x_embedder.weight", dw.embed_w, &sh))
            read_vec(files, "decoder.proj_in.weight", dw.embed_w, &sh);
        if (sh.size() >= 2)
            dw.hidden = static_cast<int>(sh[0]);
        if (!read_vec(files, "decoder.x_embedder.bias", dw.embed_b))
            read_vec(files, "decoder.proj_in.bias", dw.embed_b);

        read_vec(files, "post_quant_conv.weight", dw.post_w);
        read_vec(files, "post_quant_conv.bias", dw.post_b);

        ::std::vector<int64_t> psh;
        if (read_vec(files, "decoder.proj_out.weight", dw.proj_w, &psh) && psh.size() >= 2)
            dw.proj_dim = static_cast<int>(psh[0]);
        read_vec(files, "decoder.proj_out.bias", dw.proj_b);
        if (!read_vec(files, "decoder.norm_out.weight", dw.ln_w))
            read_vec(files, "decoder.norm_out.gamma", dw.ln_w);
        if (!read_vec(files, "decoder.norm_out.bias", dw.ln_b))
            read_vec(files, "decoder.norm_out.beta", dw.ln_b);
        read_vec(files, "decoder.register_tokens", dw.registers);

        dw.n_blocks = 0;
        for (int i = 0; i < 64; ++i) {
            const ::std::string qn =
                "decoder.transformer_blocks." + ::std::to_string(i) + ".attn.to_qkv.weight";
            const ::std::string qn2 =
                "decoder.transformer_blocks." + ::std::to_string(i) + ".attn.to_q.weight";
            if (!find_named(files, qn).tensor && !find_named(files, qn2).tensor)
                break;
            dw.n_blocks = i + 1;
        }

        if (dw.hidden <= 0) {
            ::std::vector<int64_t> qsh;
            ::std::vector<float> tmp;
            if (read_vec(files, "decoder.transformer_blocks.0.attn.to_qkv.weight", tmp, &qsh) &&
                qsh.size() >= 2 && qsh[1] > 0) {
                dw.hidden = static_cast<int>(qsh[1]);
            } else if (!psh.empty() && psh.size() >= 2 && psh[1] > 0) {
                dw.hidden = static_cast<int>(psh[1]);
            } else if (h3.vae_hidden > 0) {
                dw.hidden = h3.vae_hidden;
            }
        }
        if (dw.hidden > 0) {
            dw.heads = infer_heads(dw.hidden, h3);
            dw.head_dim = dw.heads > 0 ? dw.hidden / dw.heads : dw.hidden;
        }
        if (dw.hidden > 0 &&
            static_cast<int>(dw.registers.size()) >= kH3VaeRegisters * dw.hidden)
            dw.official = true;

        try_load_encoder_rgb(files, ch, dw);
        if (load_encoder(files, encw(this)))
            official_encode = true;
        else
            encw(this) = EncW{};
        io::st_close_dir(files);
        official_decode = dw.official;
        break;
    }

    overlay_latents_json(model_dir + "/FL2VA/video_vae/config.json", latents_mean, latents_std);
    overlay_latents_json(model_dir + "/video_vae/config.json", latents_mean, latents_std);
    overlay_latents_json(model_dir + "/config.json", latents_mean, latents_std);
    if (!source_dir.empty()) {
        overlay_latents_json(source_dir + "/config.json", latents_mean, latents_std);
        overlay_latents_json(parent_dir(source_dir) + "/config.json", latents_mean, latents_std);
    }

    err.clear();
    return Status::Ok;
}

void H3Vae::encode(const float *rgb, const H3VaeGeom &g, float *z) const {
    if (!rgb || !z)
        return;
    const int F = g.frames;
    const int H = g.height;
    const int W = g.width;
    const int T = g.latent_t;
    const int lh = g.latent_h;
    const int lw = g.latent_w;
    const int C = g.latent_ch > 0 ? g.latent_ch : 24;
    const int sp = g.spatial > 0 ? g.spatial : 16;
    if (F <= 0 || H <= 0 || W <= 0 || T <= 0 || lh <= 0 || lw <= 0)
        return;

    float im_mean[3], im_std[3];
    ::std::vector<float> lm(static_cast<size_t>(C)), ls(static_cast<size_t>(C));
    ensure_stats_c(*this, C, im_mean, im_std, lm.data(), ls.data());

    const EncW *ew = encw_c(this);
    if (official_encode && ew && ew->ready &&
        run_official_encode(*ew, rgb, g, im_mean, im_std, lm.data(), ls.data(), z))
        return;

    ::std::vector<float> B(static_cast<size_t>(C) * 3);
    const DecW *dw = decw_c(this);
    const bool use_enc = dw && static_cast<int>(dw->enc_w.size()) == C * 3;
    if (use_enc)
        ::std::memcpy(B.data(), dw->enc_w.data(), B.size() * sizeof(float));
    else
        fill_enc_b(B.data(), C);
    const float *bb = (use_enc && static_cast<int>(dw->enc_b.size()) == C) ? dw->enc_b.data() : nullptr;

    const int plane = T * lh * lw;
    const int cell = lh * lw;
    ::std::memset(z, 0, static_cast<size_t>(C) * plane * sizeof(float));

    for (int t = 0; t < T; ++t) {
        int f0 = (t * F) / T;
        int f1 = ((t + 1) * F) / T;
        if (f1 <= f0)
            f1 = f0 + 1;
        if (f1 > F)
            f1 = F;
        for (int ly = 0; ly < lh; ++ly) {
            const int y0 = ly * sp;
            const int y1 = ::std::min(H, y0 + sp);
            for (int lx = 0; lx < lw; ++lx) {
                const int x0 = lx * sp;
                const int x1 = ::std::min(W, x0 + sp);
                float acc[3] = {0.f, 0.f, 0.f};
                int count = 0;
                for (int f = f0; f < f1; ++f) {
                    for (int y = y0; y < y1; ++y) {
                        const float *row = rgb + (static_cast<size_t>(f) * H + y) * W * 3;
                        for (int x = x0; x < x1; ++x) {
                            acc[0] += row[x * 3 + 0];
                            acc[1] += row[x * 3 + 1];
                            acc[2] += row[x * 3 + 2];
                            ++count;
                        }
                    }
                }
                const float inv = 1.f / static_cast<float>(count > 0 ? count : 1);
                float pix[3];
                for (int c = 0; c < 3; ++c)
                    pix[c] = (acc[c] * inv - im_mean[c]) / im_std[c];
                const int spat = t * cell + ly * lw + lx;
                for (int k = 0; k < C; ++k) {
                    float v = B[k * 3 + 0] * pix[0] + B[k * 3 + 1] * pix[1] + B[k * 3 + 2] * pix[2];
                    if (bb)
                        v += bb[k];
                    z[k * plane + spat] = (v - lm[static_cast<size_t>(k)]) / ls[static_cast<size_t>(k)];
                }
            }
        }
    }
}

void H3Vae::decode(const float *z, const H3VaeGeom &g, float *rgb) const {
    if (!z || !rgb)
        return;
    const int C = g.latent_ch > 0 ? g.latent_ch : 24;
    const int T = g.latent_t, lh = g.latent_h, lw = g.latent_w;
    const int F = g.frames, H = g.height, W = g.width;
    if (F <= 0 || H <= 0 || W <= 0 || T <= 0 || lh <= 0 || lw <= 0)
        return;

    float im_mean[3], im_std[3];
    ::std::vector<float> lm(static_cast<size_t>(C)), ls(static_cast<size_t>(C));
    ensure_stats_c(*this, C, im_mean, im_std, lm.data(), ls.data());

    const int plane = T * lh * lw;
    ::std::vector<float> z_raw(static_cast<size_t>(C) * plane);
    for (int c = 0; c < C; ++c) {
        const float m = lm[static_cast<size_t>(c)];
        const float s = ls[static_cast<size_t>(c)];
        const float *src = z + c * plane;
        float *dst = z_raw.data() + c * plane;
        for (int i = 0; i < plane; ++i)
            dst[i] = src[i] * s + m;
    }

    ::std::vector<float> A(static_cast<size_t>(3) * C);
    fill_mix_a(A.data(), C);

    const DecW *dw = decw_c(this);
    if (from_checkpoint && dw && dw->official &&
        run_official(*this, *dw, z_raw.data(), g, rgb))
        return;
    const bool want_xf = from_checkpoint && dw &&
                         (!dw->embed_w.empty() || dw->n_blocks > 0 || !dw->proj_w.empty());
    if (want_xf && run_transformer(*this, *dw, z_raw.data(), g, A.data(), im_mean, im_std, rgb))
        return;

    mix_decode(z_raw.data(), g, A.data(), im_mean, im_std, rgb);
}

Status h3_write_ppm(const std::string &path, const float *rgb, int frames, int height, int width,
                    std::string &err) {
    if (path.empty() || !rgb || width <= 0 || height <= 0) {
        err = "h3_write_ppm: bad args";
        return Status::InvalidArgument;
    }
    if (frames < 1)
        frames = 1;
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        err = "h3_write_ppm: open failed: " + path;
        return Status::IoError;
    }
    out << "P6\n" << width << " " << height << "\n255\n";
    const int n = height * width * 3;
    std::vector<unsigned char> buf(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        float v = rgb[i];
        if (v < 0.f)
            v = 0.f;
        if (v > 1.f)
            v = 1.f;
        buf[static_cast<size_t>(i)] = static_cast<unsigned char>(v * 255.f + 0.5f);
    }
    out.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!out) {
        err = "h3_write_ppm: write failed: " + path;
        return Status::IoError;
    }
    out.close();
    if (frames > 1) {
        std::string side = path + ".txt";
        std::ofstream t(side);
        t << "frames=" << frames << "\nwidth=" << width << "\nheight=" << height << "\n";
    }
    err.clear();
    return Status::Ok;
}

} // namespace mvllm
