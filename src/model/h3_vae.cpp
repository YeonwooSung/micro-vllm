#include "h3_vae.hpp"
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
};

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
        for (int i = 0; i < N * hidden; ++i) {
            float s = (static_cast<int>(scale1.size()) == hidden) ? scale1[static_cast<size_t>(i % hidden)]
                                                                  : 1.f;
            x[i] += ao[static_cast<size_t>(i)] * s;
        }
    }

    nw = norm2.size() == static_cast<size_t>(hidden) ? norm2.data() : nullptr;
    rmsnorm_rows(x, nw, nrm.data(), N, hidden, kRmsEps);

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
    source_dir.clear();
    geom = h3_vae_geom(h3.default_width, h3.default_height, h3.default_frames, h3.vae_spatial, ch);
    mean = {kImMean[0], kImMean[1], kImMean[2]};
    this->std = {kImStd[0], kImStd[1], kImStd[2]};
    latents_mean.assign(static_cast<size_t>(ch), 0.f);
    latents_std.assign(static_cast<size_t>(ch), 1.f);
    decw(this) = DecW{};

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
        const bool hit = find_named(files, "decoder.x_embedder.weight").tensor ||
                         find_named(files, "decoder.proj_in.weight").tensor ||
                         find_named(files, "decoder.transformer_blocks.0.attn.to_qkv.weight").tensor ||
                         find_named(files, "decoder.transformer_blocks.0.attn.to_q.weight").tensor;
        if (!hit) {
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

        try_load_encoder_rgb(files, ch, dw);
        io::st_close_dir(files);
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
