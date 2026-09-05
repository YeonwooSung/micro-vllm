#include "h3_audio_vae.hpp"
#include "h3_vae.hpp"
#include "../io/safetensors.hpp"
#include "../quant/quant.hpp"

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace mvllm {
namespace {

constexpr int kStages = 7;
constexpr int kEncStages = 5;
constexpr int kResBlocks = 3;
constexpr int kResPairs = 3;
constexpr int kFilt = 12;
constexpr uint32_t kUpRate[kStages] = {5, 5, 2, 2, 2, 2, 2};
constexpr uint32_t kUpKer[kStages] = {9, 9, 4, 4, 4, 4, 4};
constexpr uint32_t kResKer[kResBlocks] = {3, 7, 11};
constexpr uint32_t kResDil[kResPairs] = {1, 3, 5};
constexpr uint32_t kEncStride[kEncStages] = {2, 4, 4, 5, 5};
constexpr uint32_t kEncDil[kResBlocks] = {1, 3, 9};

struct TensorW {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

int64_t numel(const std::vector<int64_t> &sh) {
    int64_t n = 1;
    for (int64_t d : sh) {
        if (d <= 0)
            return 0;
        n *= d;
    }
    return n;
}

void xavier(std::vector<float> &w, int n, uint32_t seed) {
    w.resize(static_cast<size_t>(std::max(n, 0)));
    uint32_t s = seed ? seed : 1u;
    float sc = 1.f / std::sqrt(static_cast<float>(std::max(n, 1)));
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float u = static_cast<float>(s >> 8) * (1.f / 16777216.f);
        w[static_cast<size_t>(i)] = (u * 2.f - 1.f) * sc;
    }
}

struct Conv1d {
    int in_ch = 0, out_ch = 0, k = 1, stride = 1, pad = 0, dil = 1;
    bool transpose = false;
    std::vector<float> w, b;
};

void apply_weight_norm(Conv1d &c, const float *v, const float *g, int outer, int inner) {
    c.w.resize(static_cast<size_t>(outer) * inner);
    for (int o = 0; o < outer; ++o) {
        double n2 = 0;
        for (int i = 0; i < inner; ++i) {
            float x = v[o * inner + i];
            n2 += static_cast<double>(x) * x;
        }
        float inv = 1.f / std::sqrt(static_cast<float>(n2) + 1e-12f);
        float gg = g ? g[o] : 1.f;
        for (int i = 0; i < inner; ++i)
            c.w[static_cast<size_t>(o) * inner + i] = v[o * inner + i] * gg * inv;
    }
}

int conv_out_len(int L, int k, int stride, int pad, int dil) {
    int eff = dil * (k - 1) + 1;
    int padded = L + 2 * pad;
    if (stride < 1 || padded < eff)
        return 0;
    return (padded - eff) / stride + 1;
}

int conv_t_out_len(int L, int k, int stride, int pad) {
    return (L - 1) * stride - 2 * pad + k;
}

// x [B, Cin, L] → y [B, Cout, Lout]
void conv1d(const float *x, int B, int L, const Conv1d &c, std::vector<float> &y, int &Lout) {
    if (c.transpose) {
        Lout = conv_t_out_len(L, c.k, c.stride, c.pad);
        y.assign(static_cast<size_t>(B) * c.out_ch * std::max(Lout, 0), 0.f);
        if (Lout <= 0 || c.w.empty())
            return;
        // w [in, out, k]
        for (int b = 0; b < B; ++b)
            for (int oc = 0; oc < c.out_ch; ++oc) {
                float *yo = y.data() + (static_cast<size_t>(b) * c.out_ch + oc) * Lout;
                if (static_cast<int>(c.b.size()) == c.out_ch)
                    for (int t = 0; t < Lout; ++t)
                        yo[t] = c.b[static_cast<size_t>(oc)];
                for (int ic = 0; ic < c.in_ch; ++ic) {
                    const float *xi = x + (static_cast<size_t>(b) * c.in_ch + ic) * L;
                    const float *wk = c.w.data() + (static_cast<size_t>(ic) * c.out_ch + oc) * c.k;
                    for (int t = 0; t < L; ++t) {
                        float v = xi[t];
                        for (int kk = 0; kk < c.k; ++kk) {
                            int o = t * c.stride + kk - c.pad;
                            if (o >= 0 && o < Lout)
                                yo[o] += v * wk[kk];
                        }
                    }
                }
            }
        return;
    }
    Lout = conv_out_len(L, c.k, c.stride, c.pad, c.dil);
    y.assign(static_cast<size_t>(B) * c.out_ch * std::max(Lout, 0), 0.f);
    if (Lout <= 0 || c.w.empty())
        return;
    for (int b = 0; b < B; ++b)
        for (int oc = 0; oc < c.out_ch; ++oc) {
            float *yo = y.data() + (static_cast<size_t>(b) * c.out_ch + oc) * Lout;
            float bias = static_cast<int>(c.b.size()) == c.out_ch ? c.b[static_cast<size_t>(oc)] : 0.f;
            for (int t = 0; t < Lout; ++t) {
                float acc = bias;
                for (int ic = 0; ic < c.in_ch; ++ic) {
                    const float *xi = x + (static_cast<size_t>(b) * c.in_ch + ic) * L;
                    const float *wk = c.w.data() + (static_cast<size_t>(oc) * c.in_ch + ic) * c.k;
                    for (int kk = 0; kk < c.k; ++kk) {
                        int s = t * c.stride + kk * c.dil - c.pad;
                        if (s >= 0 && s < L)
                            acc += xi[s] * wk[kk];
                    }
                }
                yo[t] = acc;
            }
        }
}

void snake1d(float *x, int B, int C, int L, const float *alpha) {
    for (int c = 0; c < C; ++c) {
        float a = alpha ? alpha[c] : 1.f;
        if (!(std::fabs(a) > 1e-8f))
            continue;
        float inv = 1.f / a;
        for (int b = 0; b < B; ++b) {
            float *row = x + (static_cast<size_t>(b) * C + c) * L;
            for (int t = 0; t < L; ++t) {
                float v = row[t];
                float s = std::sin(a * v);
                row[t] = v + s * s * inv;
            }
        }
    }
}

void snake_beta(float *x, int B, int C, int L, const float *alpha, const float *beta) {
    for (int c = 0; c < C; ++c) {
        float a = alpha ? alpha[c] : 1.f;
        float bt = beta ? beta[c] : a;
        if (!(std::fabs(bt) > 1e-8f))
            continue;
        float inv = 1.f / bt;
        for (int b = 0; b < B; ++b) {
            float *row = x + (static_cast<size_t>(b) * C + c) * L;
            for (int t = 0; t < L; ++t) {
                float v = row[t];
                float s = std::sin(a * v);
                row[t] = v + s * s * inv;
            }
        }
    }
}

void layernorm_vec(float *x, const float *w, const float *b, int rows, int dim, float eps) {
    for (int r = 0; r < rows; ++r) {
        float *row = x + static_cast<size_t>(r) * dim;
        float mean = 0.f;
        for (int i = 0; i < dim; ++i)
            mean += row[i];
        mean /= static_cast<float>(dim > 0 ? dim : 1);
        float var = 0.f;
        for (int i = 0; i < dim; ++i) {
            float d = row[i] - mean;
            var += d * d;
        }
        var /= static_cast<float>(dim > 0 ? dim : 1);
        float inv = 1.f / std::sqrt(var + eps);
        for (int i = 0; i < dim; ++i) {
            float v = (row[i] - mean) * inv;
            if (w)
                v *= w[i];
            if (b)
                v += b[i];
            row[i] = v;
        }
    }
}

float gelu(float x) {
    return 0.5f * x * (1.f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
}

void linear(float *y, const float *x, const float *w, const float *b, int S, int I, int O) {
    quant::matmul_f32(y, x, w, S, I, O);
    if (!b)
        return;
    for (int s = 0; s < S; ++s)
        for (int o = 0; o < O; ++o)
            y[static_cast<size_t>(s) * O + o] += b[o];
}

void sdpa_causal(const float *q, const float *k, const float *v, float *out, int B, int T, int heads,
                 int hd) {
    const float scale = 1.f / std::sqrt(static_cast<float>(hd > 0 ? hd : 1));
    std::vector<float> scores(static_cast<size_t>(T));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int i = 0; i < T; ++i) {
                const float *qi = q + ((static_cast<size_t>(b) * T + i) * heads + h) * hd;
                float mx = -1e30f;
                int lim = i + 1;
                for (int j = 0; j < lim; ++j) {
                    const float *kj = k + ((static_cast<size_t>(b) * T + j) * heads + h) * hd;
                    float dot = 0.f;
                    for (int d = 0; d < hd; ++d)
                        dot += qi[d] * kj[d];
                    scores[static_cast<size_t>(j)] = dot * scale;
                    if (scores[static_cast<size_t>(j)] > mx)
                        mx = scores[static_cast<size_t>(j)];
                }
                float sum = 0.f;
                for (int j = 0; j < lim; ++j) {
                    float e = std::exp(scores[static_cast<size_t>(j)] - mx);
                    scores[static_cast<size_t>(j)] = e;
                    sum += e;
                }
                float inv = 1.f / (sum + 1e-12f);
                float *oi = out + ((static_cast<size_t>(b) * T + i) * heads + h) * hd;
                for (int d = 0; d < hd; ++d)
                    oi[d] = 0.f;
                for (int j = 0; j < lim; ++j) {
                    float a = scores[static_cast<size_t>(j)] * inv;
                    const float *vj = v + ((static_cast<size_t>(b) * T + j) * heads + h) * hd;
                    for (int d = 0; d < hd; ++d)
                        oi[d] += a * vj[d];
                }
            }
        }
    }
}

struct AudioW {
    int enc_dim = 8;
    int latent_dim = 256;
    int dec_dim = 32;
    int dec_in = 64;
    int heads = 4;
    std::unordered_map<std::string, TensorW> t;
};

std::unordered_map<const H3AudioVae *, AudioW> &wmap() {
    static std::unordered_map<const H3AudioVae *, AudioW> m;
    return m;
}

AudioW &aw(const H3AudioVae *v) { return wmap()[v]; }
const AudioW *awc(const H3AudioVae *v) {
    auto it = wmap().find(v);
    return it == wmap().end() ? nullptr : &it->second;
}

TensorW *get(AudioW &w, const std::string &n) {
    auto it = w.t.find(n);
    return it == w.t.end() ? nullptr : &it->second;
}
const TensorW *getc(const AudioW &w, const std::string &n) {
    auto it = w.t.find(n);
    return it == w.t.end() ? nullptr : &it->second;
}

void put(AudioW &w, const std::string &n, std::vector<int64_t> sh, uint32_t seed) {
    TensorW tw;
    tw.shape = std::move(sh);
    xavier(tw.data, static_cast<int>(numel(tw.shape)), seed);
    w.t[n] = std::move(tw);
}

void put_ones(AudioW &w, const std::string &n, int nels, float v = 1.f) {
    TensorW tw;
    tw.shape = {nels};
    tw.data.assign(static_cast<size_t>(nels), v);
    w.t[n] = std::move(tw);
}

uint32_t hash_name(const std::string &s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s)
        h = (h ^ c) * 16777619u;
    return h ? h : 1u;
}

bool fill_conv(const AudioW &w, const std::string &prefix, Conv1d &c, int in_ch, int out_ch, int k,
               int stride, int pad, int dil, bool transpose) {
    c = Conv1d{};
    c.in_ch = in_ch;
    c.out_ch = out_ch;
    c.k = k;
    c.stride = stride;
    c.pad = pad;
    c.dil = dil;
    c.transpose = transpose;
    int outer = transpose ? in_ch : out_ch;
    int inner_ch = transpose ? out_ch : in_ch;
    int inner = inner_ch * k;
    const TensorW *plain = getc(w, prefix + ".weight");
    const TensorW *vv = getc(w, prefix + ".weight_v");
    const TensorW *gg = getc(w, prefix + ".weight_g");
    if (vv && static_cast<int>(vv->data.size()) == outer * inner) {
        apply_weight_norm(c, vv->data.data(),
                          (gg && static_cast<int>(gg->data.size()) >= outer) ? gg->data.data()
                                                                            : nullptr,
                          outer, inner);
    } else if (plain && static_cast<int>(plain->data.size()) == outer * inner) {
        c.w = plain->data;
    } else {
        return false;
    }
    const TensorW *bb = getc(w, prefix + ".bias");
    if (bb && static_cast<int>(bb->data.size()) == out_ch)
        c.b = bb->data;
    return true;
}

void synth_conv(AudioW &w, const std::string &prefix, int in_ch, int out_ch, int k, bool transpose,
                bool bias) {
    int outer = transpose ? in_ch : out_ch;
    int inner_ch = transpose ? out_ch : in_ch;
    put(w, prefix + ".weight_v", {outer, inner_ch, k}, hash_name(prefix + "v"));
    put(w, prefix + ".weight_g", {outer, 1, 1}, hash_name(prefix + "g"));
    if (bias)
        put(w, prefix + ".bias", {out_ch}, hash_name(prefix + "b"));
}

void synth_weights(AudioW &w) {
    w.t.clear();
    const int ED = w.enc_dim;
    const int LD = w.latent_dim;
    const int DD = w.dec_dim;
    const int DI = w.dec_in;
    synth_conv(w, "encoder.block.0", 1, ED, 7, false, true);
    int ch = ED;
    for (int s = 1; s <= kEncStages; ++s) {
        for (int r = 0; r < kResBlocks; ++r) {
            put(w, "encoder.block." + std::to_string(s) + ".block." + std::to_string(r) +
                       ".block.0.alpha",
                {1, ch, 1}, hash_name("ea" + std::to_string(s * 10 + r)));
            synth_conv(w,
                       "encoder.block." + std::to_string(s) + ".block." + std::to_string(r) +
                           ".block.1",
                       ch, ch, 7, false, true);
            put(w, "encoder.block." + std::to_string(s) + ".block." + std::to_string(r) +
                       ".block.2.alpha",
                {1, ch, 1}, hash_name("eb" + std::to_string(s * 10 + r)));
            synth_conv(w,
                       "encoder.block." + std::to_string(s) + ".block." + std::to_string(r) +
                           ".block.3",
                       ch, ch, 1, false, true);
        }
        put(w, "encoder.block." + std::to_string(s) + ".block.3.alpha", {1, ch, 1},
            hash_name("ed" + std::to_string(s)));
        int stride = static_cast<int>(kEncStride[s - 1]);
        synth_conv(w, "encoder.block." + std::to_string(s) + ".block.4", ch, ch * 2, stride * 2,
                   false, true);
        ch *= 2;
    }
    put(w, "encoder.block.6.alpha", {1, LD, 1}, 91);
    synth_conv(w, "encoder.block.7", LD, LD, 3, false, true);
    put_ones(w, "pre_block.norm3.weight", LD, 1.f);
    put(w, "pre_block.norm3.bias", {LD}, 92);
    put(w, "pre_block.proj.weight", {kH3AudioChannels, LD}, 93);
    put(w, "pre_block.proj.bias", {kH3AudioChannels}, 94);
    put_ones(w, "pre_block.norm1.weight", LD, 1.f);
    put(w, "pre_block.norm1.bias", {LD}, 95);
    put(w, "pre_block.attn.qkv.weight", {LD * 3, LD}, 96);
    put(w, "pre_block.attn.q_bias", {LD}, 97);
    put(w, "pre_block.attn.zero_k_bias", {LD}, 98);
    put(w, "pre_block.attn.v_bias", {LD}, 99);
    put(w, "pre_block.attn.proj.weight", {kH3AudioChannels, kH3AudioChannels}, 100);
    put(w, "pre_block.attn.proj.bias", {kH3AudioChannels}, 101);
    put_ones(w, "pre_block.norm2.weight", kH3AudioChannels, 1.f);
    put(w, "pre_block.norm2.bias", {kH3AudioChannels}, 102);
    put_ones(w, "pre_block.mlp.norm.weight", kH3AudioChannels, 1.f);
    put(w, "pre_block.mlp.norm.bias", {kH3AudioChannels}, 103);
    put(w, "pre_block.mlp.w0.weight", {kH3AudioChannels * 2, kH3AudioChannels}, 104);
    put(w, "pre_block.mlp.w0.bias", {kH3AudioChannels * 2}, 105);
    put(w, "pre_block.mlp.w1.weight", {kH3AudioChannels * 2, kH3AudioChannels}, 106);
    put(w, "pre_block.mlp.w1.bias", {kH3AudioChannels * 2}, 107);
    put(w, "pre_block.mlp.w2.weight", {kH3AudioChannels, kH3AudioChannels * 2}, 108);
    put(w, "pre_block.mlp.w2.bias", {kH3AudioChannels}, 109);
    synth_conv(w, "mean_proj", kH3AudioChannels, kH3AudioChannels, 1, false, true);

    synth_conv(w, "dec_in_proj", kH3AudioChannels, DI, 1, false, true);
    synth_conv(w, "decoder.conv_pre", DI, DD, 7, false, true);
    int cin = DD;
    for (int s = 0; s < kStages; ++s) {
        int cout = std::max(1, cin / 2);
        int ker = static_cast<int>(kUpKer[s]);
        synth_conv(w, "decoder.ups." + std::to_string(s) + ".0", cin, cout, ker, true, true);
        for (int b = 0; b < kResBlocks; ++b) {
            int g = s * kResBlocks + b;
            for (int a = 0; a < kResPairs * 2; ++a) {
                put(w, "decoder.resblocks." + std::to_string(g) + ".activations." +
                           std::to_string(a) + ".act.alpha",
                    {cout}, hash_name("da" + std::to_string(g * 10 + a)));
                put(w, "decoder.resblocks." + std::to_string(g) + ".activations." +
                           std::to_string(a) + ".act.beta",
                    {cout}, hash_name("db" + std::to_string(g * 10 + a)));
            }
            for (int p = 0; p < kResPairs; ++p) {
                int rk = static_cast<int>(kResKer[b]);
                synth_conv(w,
                           "decoder.resblocks." + std::to_string(g) + ".convs1." + std::to_string(p),
                           cout, cout, rk, false, true);
                synth_conv(w,
                           "decoder.resblocks." + std::to_string(g) + ".convs2." + std::to_string(p),
                           cout, cout, rk, false, true);
            }
        }
        cin = cout;
    }
    int last = std::max(1, DD >> kStages);
    put(w, "decoder.activation_post.act.alpha", {last}, 200);
    put(w, "decoder.activation_post.act.beta", {last}, 201);
    synth_conv(w, "decoder.conv_post", last, 1, 7, false, false);
    put(w, "decoder.activation_post.upsample.filter", {1, 1, kFilt}, 202);
    put(w, "decoder.activation_post.downsample.lowpass.filter", {1, 1, kFilt}, 203);
}

io::StHit find_named(const std::vector<io::StFile> &files, const std::string &name) {
    static const char *kPref[] = {"", "audio_vae.", "model.", "decoder.", "encoder."};
    for (const char *p : kPref) {
        io::StHit h = io::st_find_dir(files, std::string(p) + name);
        if (h.tensor)
            return h;
    }
    return io::st_find_dir(files, name);
}

bool overlay_tensor(const std::vector<io::StFile> &files, AudioW &w, const std::string &name) {
    io::StHit h = find_named(files, name);
    if (!h.tensor)
        return false;
    int64_t n = numel(h.tensor->shape);
    if (n <= 0)
        return false;
    TensorW tw;
    tw.shape = h.tensor->shape;
    tw.data.resize(static_cast<size_t>(n));
    std::string err;
    if (io::st_read_f32(*h.file, *h.tensor, tw.data.data(), n, err) != Status::Ok)
        return false;
    w.t[name] = std::move(tw);
    return true;
}

void overlay_json_stats(const std::string &path, std::vector<float> &mean, std::vector<float> &stdv) {
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
    if (j.contains("audio_vae") && j["audio_vae"].is_object())
        root = &j["audio_vae"];
    auto pull = [](const nlohmann::json &o, const char *key, std::vector<float> &dst) {
        if (!o.contains(key) || !o[key].is_array())
            return;
        const auto &a = o[key];
        size_t n = std::min(dst.size(), a.size());
        for (size_t i = 0; i < n; ++i)
            if (a[i].is_number())
                dst[i] = static_cast<float>(a[i].get<double>());
    };
    pull(*root, "latents_mean", mean);
    pull(*root, "latents_std", stdv);
}

void bcl_to_btc(const float *x, int B, int C, int L, std::vector<float> &y) {
    y.resize(static_cast<size_t>(B) * L * C);
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < L; ++t)
            for (int c = 0; c < C; ++c)
                y[(static_cast<size_t>(b) * L + t) * C + c] = x[(static_cast<size_t>(b) * C + c) * L + t];
}

void btc_to_bcl(const float *x, int B, int L, int C, std::vector<float> &y) {
    y.resize(static_cast<size_t>(B) * C * L);
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int t = 0; t < L; ++t)
                y[(static_cast<size_t>(b) * C + c) * L + t] = x[(static_cast<size_t>(b) * L + t) * C + c];
}

void add_inplace(float *a, const float *b, int n, float sa = 1.f, float sb = 1.f) {
    for (int i = 0; i < n; ++i)
        a[i] = sa * a[i] + sb * b[i];
}

} // namespace

int h3_audio_t(int frames) {
    int F = h3_align_frames(frames);
    double t = static_cast<double>(F) * static_cast<double>(kH3AudioLatentFps) /
               static_cast<double>(kH3VideoFps);
    int v = static_cast<int>(std::llround(t));
    return v < 1 ? 1 : v;
}

int h3_audio_pad_samples(int samples) {
    if (samples < 1)
        return kH3AudioHop;
    return ((samples + kH3AudioHop - 1) / kH3AudioHop) * kH3AudioHop;
}

int h3_audio_samples(int audio_t) {
    if (audio_t < 1)
        audio_t = 1;
    return audio_t * kH3AudioHop;
}

int h3_dit_pack_audio(const float *latent, int channels, int time, float *rows) {
    if (!latent || !rows || channels < 1 || time < 1)
        return 0;
    int n = 0;
    for (int stream = 0; stream < 2; ++stream)
        for (int t = 0; t < time; ++t)
            for (int c = 0; c < channels; ++c)
                rows[n++] = latent[(static_cast<size_t>(c) * 2 + stream) * time + t];
    return n;
}

int h3_dit_unpack_audio(const float *rows, int channels, int time, float *latent) {
    if (!rows || !latent || channels < 1 || time < 1)
        return 0;
    int n = 0;
    for (int stream = 0; stream < 2; ++stream)
        for (int t = 0; t < time; ++t)
            for (int c = 0; c < channels; ++c)
                latent[(static_cast<size_t>(c) * 2 + stream) * time + t] = rows[n++];
    return n;
}

Status h3_write_wav(const std::string &path, const float *pcm, int channels, int samples, int rate,
                    std::string &err) {
    if (!pcm || channels < 1 || samples < 1 || rate < 1) {
        err = "invalid wav args";
        return Status::InvalidArgument;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        err = "cannot write " + path;
        return Status::IoError;
    }
    const int bps = 16;
    const uint32_t data_bytes = static_cast<uint32_t>(samples) * channels * 2;
    const uint32_t riff = 36 + data_bytes;
    auto u16 = [&](uint16_t v) {
        out.put(static_cast<char>(v & 255));
        out.put(static_cast<char>((v >> 8) & 255));
    };
    auto u32 = [&](uint32_t v) {
        out.put(static_cast<char>(v & 255));
        out.put(static_cast<char>((v >> 8) & 255));
        out.put(static_cast<char>((v >> 16) & 255));
        out.put(static_cast<char>((v >> 24) & 255));
    };
    out.write("RIFF", 4);
    u32(riff);
    out.write("WAVEfmt ", 8);
    u32(16);
    u16(1);
    u16(static_cast<uint16_t>(channels));
    u32(static_cast<uint32_t>(rate));
    u32(static_cast<uint32_t>(rate * channels * bps / 8));
    u16(static_cast<uint16_t>(channels * bps / 8));
    u16(static_cast<uint16_t>(bps));
    out.write("data", 4);
    u32(data_bytes);
    for (int t = 0; t < samples; ++t) {
        for (int c = 0; c < channels; ++c) {
            float v = pcm[static_cast<size_t>(c) * samples + t];
            if (v > 1.f)
                v = 1.f;
            if (v < -1.f)
                v = -1.f;
            int s = static_cast<int>(std::lrint(v * 32767.f));
            if (s > 32767)
                s = 32767;
            if (s < -32768)
                s = -32768;
            u16(static_cast<uint16_t>(static_cast<int16_t>(s)));
        }
    }
    return Status::Ok;
}

Status h3_read_wav(const std::string &path, std::vector<float> &pcm, int &channels, int &samples,
                   int &rate, std::string &err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "cannot read " + path;
        return Status::IoError;
    }
    char tag[12];
    in.read(tag, 12);
    if (!in || std::memcmp(tag, "RIFF", 4) != 0 || std::memcmp(tag + 8, "WAVE", 4) != 0) {
        err = "not a wav file";
        return Status::ParseError;
    }
    channels = 0;
    samples = 0;
    rate = 0;
    int bps = 0, fmt = 0;
    std::vector<uint8_t> data;
    while (in) {
        char ck[4];
        uint8_t ln[4];
        if (!in.read(ck, 4) || !in.read(reinterpret_cast<char *>(ln), 4))
            break;
        uint32_t n = uint32_t(ln[0]) | (uint32_t(ln[1]) << 8) | (uint32_t(ln[2]) << 16) |
                     (uint32_t(ln[3]) << 24);
        if (std::memcmp(ck, "fmt ", 4) == 0) {
            std::vector<uint8_t> f(n);
            in.read(reinterpret_cast<char *>(f.data()), n);
            if (n >= 16) {
                fmt = f[0] | (f[1] << 8);
                channels = f[2] | (f[3] << 8);
                rate = int(f[4] | (f[5] << 8) | (f[6] << 16) | (f[7] << 24));
                bps = f[14] | (f[15] << 8);
            }
        } else if (std::memcmp(ck, "data", 4) == 0) {
            data.resize(n);
            in.read(reinterpret_cast<char *>(data.data()), n);
        } else {
            in.seekg(n, std::ios::cur);
        }
    }
    if (channels < 1 || rate < 1 || data.empty()) {
        err = "bad wav";
        return Status::ParseError;
    }
    int ch_use = channels >= 2 ? 2 : 1;
    if (fmt == 3 && bps == 32) {
        samples = static_cast<int>(data.size() / (4 * channels));
        pcm.assign(static_cast<size_t>(2) * samples, 0.f);
        const float *src = reinterpret_cast<const float *>(data.data());
        for (int t = 0; t < samples; ++t)
            for (int c = 0; c < ch_use; ++c)
                pcm[static_cast<size_t>(c) * samples + t] = src[t * channels + c];
        if (ch_use == 1)
            std::memcpy(pcm.data() + samples, pcm.data(), static_cast<size_t>(samples) * sizeof(float));
        channels = 2;
        return Status::Ok;
    }
    if (bps != 16) {
        err = "wav must be 16-bit PCM or f32";
        return Status::Unsupported;
    }
    samples = static_cast<int>(data.size() / (2 * channels));
    pcm.assign(static_cast<size_t>(2) * samples, 0.f);
    for (int t = 0; t < samples; ++t) {
        for (int c = 0; c < ch_use; ++c) {
            size_t o = static_cast<size_t>(t * channels + c) * 2;
            int16_t s = static_cast<int16_t>(data[o] | (data[o + 1] << 8));
            pcm[static_cast<size_t>(c) * samples + t] = s / 32768.f;
        }
    }
    if (ch_use == 1)
        std::memcpy(pcm.data() + samples, pcm.data(), static_cast<size_t>(samples) * sizeof(float));
    channels = 2;
    return Status::Ok;
}

Status H3AudioVae::load(const std::string &model_dir, std::string &err) {
    from_checkpoint = false;
    source_dir.clear();
    latents_mean.assign(kH3AudioChannels, 0.f);
    latents_std.assign(kH3AudioChannels, 1.f);
    AudioW &w = aw(this);
    w = AudioW{};
    w.enc_dim = 8;
    w.latent_dim = 256;
    w.dec_dim = 32;
    w.dec_in = 64;
    w.heads = 4;
    synth_weights(w);

    const char *tails[] = {"/FL2VA/audio_vae", "/Ref2VA/audio_vae", "/audio_vae", "/AudioVAE", ""};
    for (const char *tail : tails) {
        const std::string dir = model_dir + tail;
        overlay_json_stats(dir + "/config.json", latents_mean, latents_std);
        std::vector<io::StFile> files;
        std::string oerr;
        Status st = io::st_open_dir(dir, files, oerr);
        if (st != Status::Ok || files.empty()) {
            if (!files.empty())
                io::st_close_dir(files);
            continue;
        }
        bool hit = overlay_tensor(files, w, "dec_in_proj.weight") ||
                   overlay_tensor(files, w, "dec_in_proj.weight_v") ||
                   overlay_tensor(files, w, "encoder.block.0.weight") ||
                   overlay_tensor(files, w, "encoder.block.0.weight_v") ||
                   overlay_tensor(files, w, "mean_proj.weight");
        if (!hit) {
            io::st_close_dir(files);
            continue;
        }
        // Promote to official dims when the released tensors are present.
        const TensorW *din = get(w, "dec_in_proj.weight");
        if (!din)
            din = get(w, "dec_in_proj.weight_v");
        if (din && !din->shape.empty() && din->shape[0] >= 1024) {
            w.dec_in = static_cast<int>(din->shape[0]);
            w.dec_dim = std::max(8, w.dec_in / 2);
            w.enc_dim = 64;
            w.latent_dim = 2048;
            w.heads = 8;
            synth_weights(w);
            overlay_tensor(files, w, "dec_in_proj.weight");
            overlay_tensor(files, w, "dec_in_proj.weight_v");
            overlay_tensor(files, w, "dec_in_proj.bias");
        }
        std::vector<std::string> names;
        names.reserve(w.t.size());
        for (const auto &kv : w.t)
            names.push_back(kv.first);
        int nload = 0;
        for (const auto &nm : names)
            if (overlay_tensor(files, w, nm))
                ++nload;
        overlay_json_stats(dir + "/config.json", latents_mean, latents_std);
        io::st_close_dir(files);
        if (nload > 0) {
            from_checkpoint = true;
            source_dir = dir;
        }
        err.clear();
        return Status::Ok;
    }
    err.clear();
    return Status::Ok;
}

void H3AudioVae::encode(const float *pcm, int samples, std::vector<float> &z, int &audio_t) const {
    const AudioW *wp = awc(this);
    AudioW local;
    if (!wp) {
        local.enc_dim = 8;
        local.latent_dim = 256;
        local.dec_dim = 32;
        local.dec_in = 64;
        local.heads = 4;
        synth_weights(local);
        wp = &local;
    }
    const AudioW &w = *wp;
    const int B = kH3AudioStereo;
    const int pad = h3_audio_pad_samples(std::max(samples, 1));
    audio_t = pad / kH3AudioHop;
    std::vector<float> hidden(static_cast<size_t>(B) * 1 * pad, 0.f);
    if (pcm && samples > 0) {
        for (int c = 0; c < B; ++c) {
            int n = std::min(samples, pad);
            std::memcpy(hidden.data() + static_cast<size_t>(c) * pad, pcm + static_cast<size_t>(c) * samples,
                        static_cast<size_t>(n) * sizeof(float));
        }
    }
    int L = pad;
    int C = 1;
    Conv1d conv;
    if (fill_conv(w, "encoder.block.0", conv, 1, w.enc_dim, 7, 1, 3, 1, false)) {
        std::vector<float> y;
        int Lo = 0;
        conv1d(hidden.data(), B, L, conv, y, Lo);
        hidden.swap(y);
        L = Lo;
        C = w.enc_dim;
    }
    for (int s = 1; s <= kEncStages; ++s) {
        for (int r = 0; r < kResBlocks; ++r) {
            const TensorW *a1 =
                getc(w, "encoder.block." + std::to_string(s) + ".block." + std::to_string(r) +
                            ".block.0.alpha");
            std::vector<float> work = hidden;
            snake1d(work.data(), B, C, L, a1 ? a1->data.data() : nullptr);
            Conv1d c1;
            int dil = static_cast<int>(kEncDil[r]);
            if (fill_conv(w,
                          "encoder.block." + std::to_string(s) + ".block." + std::to_string(r) +
                              ".block.1",
                          c1, C, C, 7, 1, 3 * dil, dil, false)) {
                std::vector<float> y;
                int Lo = 0;
                conv1d(work.data(), B, L, c1, y, Lo);
                work.swap(y);
                L = Lo;
            }
            const TensorW *a2 =
                getc(w, "encoder.block." + std::to_string(s) + ".block." + std::to_string(r) +
                            ".block.2.alpha");
            snake1d(work.data(), B, C, L, a2 ? a2->data.data() : nullptr);
            Conv1d c2;
            if (fill_conv(w,
                          "encoder.block." + std::to_string(s) + ".block." + std::to_string(r) +
                              ".block.3",
                          c2, C, C, 1, 1, 0, 1, false)) {
                std::vector<float> y;
                int Lo = 0;
                conv1d(work.data(), B, L, c2, y, Lo);
                work.swap(y);
            }
            add_inplace(hidden.data(), work.data(), static_cast<int>(hidden.size()));
        }
        const TensorW *ad = getc(w, "encoder.block." + std::to_string(s) + ".block.3.alpha");
        snake1d(hidden.data(), B, C, L, ad ? ad->data.data() : nullptr);
        int stride = static_cast<int>(kEncStride[s - 1]);
        Conv1d dn;
        if (fill_conv(w, "encoder.block." + std::to_string(s) + ".block.4", dn, C, C * 2, stride * 2,
                      stride, (stride + 1) / 2, 1, false)) {
            std::vector<float> y;
            int Lo = 0;
            conv1d(hidden.data(), B, L, dn, y, Lo);
            hidden.swap(y);
            L = Lo;
            C *= 2;
        }
    }
    const TensorW *af = getc(w, "encoder.block.6.alpha");
    snake1d(hidden.data(), B, C, L, af ? af->data.data() : nullptr);
    Conv1d fin;
    if (fill_conv(w, "encoder.block.7", fin, C, w.latent_dim, 3, 1, 1, 1, false)) {
        std::vector<float> y;
        int Lo = 0;
        conv1d(hidden.data(), B, L, fin, y, Lo);
        hidden.swap(y);
        L = Lo;
        C = w.latent_dim;
    }
    const int rows = B * L;
    std::vector<float> seq;
    bcl_to_btc(hidden.data(), B, C, L, seq);
    std::vector<float> nrm = seq;
    const TensorW *n3w = getc(w, "pre_block.norm3.weight");
    const TensorW *n3b = getc(w, "pre_block.norm3.bias");
    layernorm_vec(nrm.data(), n3w ? n3w->data.data() : nullptr, n3b ? n3b->data.data() : nullptr,
                  rows, C, 1e-5f);
    std::vector<float> base(static_cast<size_t>(rows) * kH3AudioChannels, 0.f);
    const TensorW *pw = getc(w, "pre_block.proj.weight");
    const TensorW *pb = getc(w, "pre_block.proj.bias");
    if (pw && static_cast<int>(pw->data.size()) == kH3AudioChannels * C)
        linear(base.data(), nrm.data(), pw->data.data(),
               pb && static_cast<int>(pb->data.size()) == kH3AudioChannels ? pb->data.data() : nullptr,
               rows, C, kH3AudioChannels);
    else {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < kH3AudioChannels; ++c)
                base[static_cast<size_t>(r) * kH3AudioChannels + c] =
                    nrm[static_cast<size_t>(r) * C + (c % C)];
    }
    const TensorW *n1w = getc(w, "pre_block.norm1.weight");
    const TensorW *n1b = getc(w, "pre_block.norm1.bias");
    std::vector<float> an = seq;
    layernorm_vec(an.data(), n1w ? n1w->data.data() : nullptr, n1b ? n1b->data.data() : nullptr, rows,
                  C, 1e-5f);
    const TensorW *qkv = getc(w, "pre_block.attn.qkv.weight");
    if (qkv && static_cast<int>(qkv->data.size()) == 3 * C * C) {
        std::vector<float> qkv_o(static_cast<size_t>(rows) * 3 * C);
        linear(qkv_o.data(), an.data(), qkv->data.data(), nullptr, rows, C, 3 * C);
        const TensorW *qb = getc(w, "pre_block.attn.q_bias");
        const TensorW *kb = getc(w, "pre_block.attn.zero_k_bias");
        const TensorW *vb = getc(w, "pre_block.attn.v_bias");
        std::vector<float> q(static_cast<size_t>(rows) * C), k(static_cast<size_t>(rows) * C),
            v(static_cast<size_t>(rows) * C), att(static_cast<size_t>(rows) * C);
        for (int r = 0; r < rows; ++r) {
            for (int d = 0; d < C; ++d) {
                q[static_cast<size_t>(r) * C + d] =
                    qkv_o[static_cast<size_t>(r) * 3 * C + d] +
                    (qb && static_cast<int>(qb->data.size()) == C ? qb->data[static_cast<size_t>(d)]
                                                                  : 0.f);
                k[static_cast<size_t>(r) * C + d] =
                    qkv_o[static_cast<size_t>(r) * 3 * C + C + d] +
                    (kb && static_cast<int>(kb->data.size()) == C ? kb->data[static_cast<size_t>(d)]
                                                                  : 0.f);
                v[static_cast<size_t>(r) * C + d] =
                    qkv_o[static_cast<size_t>(r) * 3 * C + 2 * C + d] +
                    (vb && static_cast<int>(vb->data.size()) == C ? vb->data[static_cast<size_t>(d)]
                                                                  : 0.f);
            }
        }
        int hd = C / std::max(w.heads, 1);
        if (hd > 0 && w.heads * hd == C)
            sdpa_causal(q.data(), k.data(), v.data(), att.data(), B, L, w.heads, hd);
        else
            att = q;
        // Official pool: first latent_ch of each head-concat row.
        std::vector<float> pooled(static_cast<size_t>(rows) * kH3AudioChannels, 0.f);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < kH3AudioChannels; ++c)
                pooled[static_cast<size_t>(r) * kH3AudioChannels + c] =
                    att[static_cast<size_t>(r) * C + (c % C)];
        const TensorW *apw = getc(w, "pre_block.attn.proj.weight");
        const TensorW *apb = getc(w, "pre_block.attn.proj.bias");
        std::vector<float> ap(static_cast<size_t>(rows) * kH3AudioChannels, 0.f);
        if (apw && static_cast<int>(apw->data.size()) == kH3AudioChannels * kH3AudioChannels)
            linear(ap.data(), pooled.data(), apw->data.data(),
                   apb && static_cast<int>(apb->data.size()) == kH3AudioChannels ? apb->data.data()
                                                                                : nullptr,
                   rows, kH3AudioChannels, kH3AudioChannels);
        else
            ap.swap(pooled);
        add_inplace(base.data(), ap.data(), static_cast<int>(base.size()));
    }
    const TensorW *n2w = getc(w, "pre_block.norm2.weight");
    const TensorW *n2b = getc(w, "pre_block.norm2.bias");
    std::vector<float> n2 = base;
    layernorm_vec(n2.data(), n2w ? n2w->data.data() : nullptr, n2b ? n2b->data.data() : nullptr, rows,
                  kH3AudioChannels, 1e-5f);
    const TensorW *nmw = getc(w, "pre_block.mlp.norm.weight");
    const TensorW *nmb = getc(w, "pre_block.mlp.norm.bias");
    layernorm_vec(n2.data(), nmw ? nmw->data.data() : nullptr, nmb ? nmb->data.data() : nullptr, rows,
                  kH3AudioChannels, 1e-5f);
    const TensorW *w0 = getc(w, "pre_block.mlp.w0.weight");
    const TensorW *b0 = getc(w, "pre_block.mlp.w0.bias");
    const TensorW *w1 = getc(w, "pre_block.mlp.w1.weight");
    const TensorW *b1 = getc(w, "pre_block.mlp.w1.bias");
    const TensorW *w2 = getc(w, "pre_block.mlp.w2.weight");
    const TensorW *b2 = getc(w, "pre_block.mlp.w2.bias");
    if (w0 && w1 && w2 && static_cast<int>(w0->data.size()) == 2 * kH3AudioChannels * kH3AudioChannels) {
        std::vector<float> gate(static_cast<size_t>(rows) * 2 * kH3AudioChannels),
            lin(static_cast<size_t>(rows) * 2 * kH3AudioChannels),
            gg(static_cast<size_t>(rows) * 2 * kH3AudioChannels);
        linear(gate.data(), n2.data(), w0->data.data(),
               b0 && static_cast<int>(b0->data.size()) == 2 * kH3AudioChannels ? b0->data.data()
                                                                              : nullptr,
               rows, kH3AudioChannels, 2 * kH3AudioChannels);
        linear(lin.data(), n2.data(), w1->data.data(),
               b1 && static_cast<int>(b1->data.size()) == 2 * kH3AudioChannels ? b1->data.data()
                                                                              : nullptr,
               rows, kH3AudioChannels, 2 * kH3AudioChannels);
        for (size_t i = 0; i < gg.size(); ++i)
            gg[i] = gelu(gate[i]) * lin[i];
        std::vector<float> br(static_cast<size_t>(rows) * kH3AudioChannels);
        linear(br.data(), gg.data(), w2->data.data(),
               b2 && static_cast<int>(b2->data.size()) == kH3AudioChannels ? b2->data.data() : nullptr,
               rows, 2 * kH3AudioChannels, kH3AudioChannels);
        add_inplace(base.data(), br.data(), static_cast<int>(base.size()));
    }
    std::vector<float> bcl;
    btc_to_bcl(base.data(), B, L, kH3AudioChannels, bcl);
    Conv1d mp;
    std::vector<float> meanv;
    int Lm = L;
    if (fill_conv(w, "mean_proj", mp, kH3AudioChannels, kH3AudioChannels, 1, 1, 0, 1, false))
        conv1d(bcl.data(), B, L, mp, meanv, Lm);
    else
        meanv = bcl;
    z.assign(static_cast<size_t>(kH3AudioChannels) * B * Lm, 0.f);
    audio_t = Lm;
    for (int c = 0; c < kH3AudioChannels; ++c) {
        float m = c < static_cast<int>(latents_mean.size()) ? latents_mean[static_cast<size_t>(c)] : 0.f;
        float s = c < static_cast<int>(latents_std.size()) ? latents_std[static_cast<size_t>(c)] : 1.f;
        if (!(s > 1e-8f))
            s = 1.f;
        for (int st = 0; st < B; ++st)
            for (int t = 0; t < Lm; ++t) {
                float v = meanv[(static_cast<size_t>(st) * kH3AudioChannels + c) * Lm + t];
                z[(static_cast<size_t>(c) * B + st) * Lm + t] = (v - m) / s;
            }
    }
}

void H3AudioVae::decode(const float *z, int audio_t, std::vector<float> &pcm) const {
    const AudioW *wp = awc(this);
    AudioW local;
    if (!wp) {
        local.enc_dim = 8;
        local.latent_dim = 256;
        local.dec_dim = 32;
        local.dec_in = 64;
        local.heads = 4;
        synth_weights(local);
        wp = &local;
    }
    const AudioW &w = *wp;
    const int B = kH3AudioStereo;
    const int T = audio_t > 0 ? audio_t : 1;
    const int C = kH3AudioChannels;
    // [32,2,T] → [2, 32, T] after denorm
    std::vector<float> lat(static_cast<size_t>(B) * C * T);
    for (int c = 0; c < C; ++c) {
        float m = c < static_cast<int>(latents_mean.size()) ? latents_mean[static_cast<size_t>(c)] : 0.f;
        float s = c < static_cast<int>(latents_std.size()) ? latents_std[static_cast<size_t>(c)] : 1.f;
        if (!(s > 1e-8f))
            s = 1.f;
        for (int st = 0; st < B; ++st)
            for (int t = 0; t < T; ++t) {
                float v = z ? z[(static_cast<size_t>(c) * B + st) * T + t] : 0.f;
                lat[(static_cast<size_t>(st) * C + c) * T + t] = v * s + m;
            }
    }
    int L = T;
    Conv1d pin;
    std::vector<float> hidden;
    if (fill_conv(w, "dec_in_proj", pin, C, w.dec_in, 1, 1, 0, 1, false)) {
        int Lo = 0;
        conv1d(lat.data(), B, L, pin, hidden, Lo);
        L = Lo;
    } else {
        hidden.assign(static_cast<size_t>(B) * w.dec_in * L, 0.f);
        for (int b = 0; b < B; ++b)
            for (int d = 0; d < w.dec_in; ++d)
                for (int t = 0; t < L; ++t)
                    hidden[(static_cast<size_t>(b) * w.dec_in + d) * L + t] =
                        lat[(static_cast<size_t>(b) * C + (d % C)) * L + t];
    }
    Conv1d pre;
    if (fill_conv(w, "decoder.conv_pre", pre, w.dec_in, w.dec_dim, 7, 1, 3, 1, false)) {
        std::vector<float> y;
        int Lo = 0;
        conv1d(hidden.data(), B, L, pre, y, Lo);
        hidden.swap(y);
        L = Lo;
    }
    int ch = w.dec_dim;
    for (int s = 0; s < kStages; ++s) {
        int cout = std::max(1, ch / 2);
        int ker = static_cast<int>(kUpKer[s]);
        int rate = static_cast<int>(kUpRate[s]);
        int pad = (ker - rate) / 2;
        Conv1d up;
        std::vector<float> upsampled;
        if (fill_conv(w, "decoder.ups." + std::to_string(s) + ".0", up, ch, cout, ker, rate, pad, 1,
                      true)) {
            int Lo = 0;
            conv1d(hidden.data(), B, L, up, upsampled, Lo);
            L = Lo;
        } else {
            int Lo = L * rate;
            upsampled.assign(static_cast<size_t>(B) * cout * Lo, 0.f);
            for (int b = 0; b < B; ++b)
                for (int oc = 0; oc < cout; ++oc)
                    for (int t = 0; t < L; ++t)
                        for (int r = 0; r < rate; ++r)
                            upsampled[(static_cast<size_t>(b) * cout + oc) * Lo + t * rate + r] =
                                hidden[(static_cast<size_t>(b) * ch + (oc % ch)) * L + t];
            L = Lo;
        }
        const int n = B * cout * L;
        std::vector<float> sum(static_cast<size_t>(n), 0.f);
        for (int bi = 0; bi < kResBlocks; ++bi) {
            int g = s * kResBlocks + bi;
            std::vector<float> target = upsampled;
            for (int p = 0; p < kResPairs; ++p) {
                std::vector<float> work = target;
                const TensorW *al = getc(w, "decoder.resblocks." + std::to_string(g) +
                                                ".activations." + std::to_string(p * 2) + ".act.alpha");
                const TensorW *be = getc(w, "decoder.resblocks." + std::to_string(g) +
                                                ".activations." + std::to_string(p * 2) + ".act.beta");
                snake_beta(work.data(), B, cout, L, al ? al->data.data() : nullptr,
                           be ? be->data.data() : nullptr);
                Conv1d c1;
                int rk = static_cast<int>(kResKer[bi]);
                int dil = static_cast<int>(kResDil[p]);
                int dp = dil * (rk - 1) / 2;
                if (fill_conv(w,
                              "decoder.resblocks." + std::to_string(g) + ".convs1." +
                                  std::to_string(p),
                              c1, cout, cout, rk, 1, dp, dil, false)) {
                    std::vector<float> y;
                    int Lo = 0;
                    conv1d(work.data(), B, L, c1, y, Lo);
                    if (Lo == L)
                        work.swap(y);
                }
                const TensorW *al2 = getc(w, "decoder.resblocks." + std::to_string(g) +
                                                 ".activations." + std::to_string(p * 2 + 1) +
                                                 ".act.alpha");
                const TensorW *be2 = getc(w, "decoder.resblocks." + std::to_string(g) +
                                                 ".activations." + std::to_string(p * 2 + 1) +
                                                 ".act.beta");
                snake_beta(work.data(), B, cout, L, al2 ? al2->data.data() : nullptr,
                           be2 ? be2->data.data() : nullptr);
                Conv1d c2;
                int p2 = (rk - 1) / 2;
                if (fill_conv(w,
                              "decoder.resblocks." + std::to_string(g) + ".convs2." +
                                  std::to_string(p),
                              c2, cout, cout, rk, 1, p2, 1, false)) {
                    std::vector<float> y;
                    int Lo = 0;
                    conv1d(work.data(), B, L, c2, y, Lo);
                    if (Lo == L)
                        work.swap(y);
                }
                add_inplace(target.data(), work.data(), n);
            }
            if (bi == 0)
                sum = target;
            else if (bi == 1)
                add_inplace(sum.data(), target.data(), n);
            else
                add_inplace(sum.data(), target.data(), n, 1.f / 3.f, 1.f / 3.f);
        }
        hidden.swap(sum);
        ch = cout;
    }
    const TensorW *pa = getc(w, "decoder.activation_post.act.alpha");
    const TensorW *pb = getc(w, "decoder.activation_post.act.beta");
    snake_beta(hidden.data(), B, ch, L, pa ? pa->data.data() : nullptr,
               pb ? pb->data.data() : nullptr);
    Conv1d post;
    std::vector<float> wave;
    int Lw = L;
    if (fill_conv(w, "decoder.conv_post", post, ch, 1, 7, 1, 3, 1, false))
        conv1d(hidden.data(), B, L, post, wave, Lw);
    else {
        wave.assign(static_cast<size_t>(B) * Lw, 0.f);
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < Lw; ++t) {
                float acc = 0.f;
                for (int c = 0; c < ch; ++c)
                    acc += hidden[(static_cast<size_t>(b) * ch + c) * L + t];
                wave[static_cast<size_t>(b) * Lw + t] = acc / static_cast<float>(std::max(ch, 1));
            }
    }
    pcm.assign(static_cast<size_t>(B) * Lw, 0.f);
    const int hop = T > 0 ? std::max(Lw / T, 1) : kH3AudioHop;
    for (int st = 0; st < B; ++st) {
        for (int t = 0; t < T; ++t) {
            float acc = 0.f;
            for (int c = 0; c < C; ++c)
                acc += lat[(static_cast<size_t>(st) * C + c) * T + t];
            acc /= static_cast<float>(C);
            for (int s = 0; s < hop; ++s) {
                int o = t * hop + s;
                if (o >= Lw)
                    break;
                size_t idx = static_cast<size_t>(st) * Lw + o;
                float v = (idx < wave.size() ? wave[idx] : 0.f) + 0.35f * acc;
                if (v > 1.f)
                    v = 1.f;
                if (v < -1.f)
                    v = -1.f;
                pcm[idx] = v;
            }
        }
    }
}

} // namespace mvllm
