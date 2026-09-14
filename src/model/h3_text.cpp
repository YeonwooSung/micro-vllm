#include "h3_text.hpp"
#include "family.hpp"
#include "../io/safetensors.hpp"
#include "../quant/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>

namespace mvllm {
namespace {

void xavier(std::vector<float> &w, int rows, int cols, uint32_t seed) {
    w.assign(static_cast<size_t>(rows) * std::max(cols, 1), 0.f);
    std::mt19937 rng(seed);
    float s = 1.f / std::sqrt(static_cast<float>(std::max(cols, 1)));
    std::normal_distribution<float> dist(0.f, s);
    for (float &v : w)
        v = dist(rng);
}

quant::QuantMat qmat(int O, int I, uint32_t seed) {
    std::vector<float> t;
    xavier(t, O, I, seed);
    quant::QuantMat m;
    m.from_f32(t.data(), O, I, 32);
    return m;
}

void apply_mrope(float *x, int hd, int t, int seq, const uint32_t *positions, float theta) {
    if (!x || !positions || hd < 2 || theta <= 0.f || seq < 1)
        return;
    const int half = hd / 2;
    for (int i = 0; i < half; ++i) {
        int axis = 0;
        if (i < 60 && i % 3 == 1)
            axis = 1;
        else if (i < 60 && i % 3 == 2)
            axis = 2;
        const float coord = static_cast<float>(positions[axis * seq + t]);
        const float inv =
            1.f / std::pow(theta, static_cast<float>(i * 2) / static_cast<float>(hd));
        const float ang = coord * inv;
        const float c = std::cos(ang), s = std::sin(ang);
        const float a = x[2 * i], b = x[2 * i + 1];
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

bool hit_mat(const io::StHit &h) {
    return h.file && h.tensor && h.tensor->shape.size() == 2 && h.tensor->shape[0] > 0 &&
           h.tensor->shape[1] > 0;
}

} // namespace

H3TextEncoder::~H3TextEncoder() { io::st_close_dir(files_); }

void h3_text_ids_from_prompt(const std::string &prompt, int vocab, std::vector<int> &ids) {
    ids.clear();
    if (vocab < 2)
        vocab = 256;
    if (prompt.empty()) {
        ids.push_back(1);
        return;
    }
    for (unsigned char c : prompt)
        ids.push_back(1 + static_cast<int>(c) % (vocab - 1));
    if (static_cast<int>(ids.size()) > 64)
        ids.resize(64);
}

void H3TextEncoder::alloc_synth() {
    streamed_ = false;
    hits_.clear();
    io::st_close_dir(files_);
    if (cfg_.hidden <= 0)
        cfg_.hidden = 32;
    if (cfg_.layers <= 0)
        cfg_.layers = 2;
    if (cfg_.n_q <= 0)
        cfg_.n_q = 4;
    if (cfg_.n_kv <= 0)
        cfg_.n_kv = 1;
    if (cfg_.head_dim <= 0)
        cfg_.head_dim = 8;
    if (cfg_.intermediate <= 0)
        cfg_.intermediate = 64;
    if (cfg_.vocab <= 0)
        cfg_.vocab = 256;
    const int H = cfg_.hidden;
    const int L = cfg_.layers;
    const int Q = cfg_.n_q * cfg_.head_dim;
    const int K = cfg_.n_kv * cfg_.head_dim;
    xavier(embed_, cfg_.vocab, H, 7);
    norm_.assign(static_cast<size_t>(H), 1.f);
    wq_.assign(static_cast<size_t>(L), {});
    wk_.assign(static_cast<size_t>(L), {});
    wv_.assign(static_cast<size_t>(L), {});
    wo_.assign(static_cast<size_t>(L), {});
    gate_.assign(static_cast<size_t>(L), {});
    up_.assign(static_cast<size_t>(L), {});
    down_.assign(static_cast<size_t>(L), {});
    in_n_.assign(static_cast<size_t>(L), {});
    post_n_.assign(static_cast<size_t>(L), {});
    qn_.assign(static_cast<size_t>(L), {});
    kn_.assign(static_cast<size_t>(L), {});
    for (int l = 0; l < L; ++l) {
        in_n_[static_cast<size_t>(l)].assign(static_cast<size_t>(H), 1.f);
        post_n_[static_cast<size_t>(l)].assign(static_cast<size_t>(H), 1.f);
        qn_[static_cast<size_t>(l)].assign(static_cast<size_t>(cfg_.head_dim), 1.f);
        kn_[static_cast<size_t>(l)].assign(static_cast<size_t>(cfg_.head_dim), 1.f);
        wq_[static_cast<size_t>(l)] = qmat(Q, H, 800 + l);
        wk_[static_cast<size_t>(l)] = qmat(K, H, 810 + l);
        wv_[static_cast<size_t>(l)] = qmat(K, H, 820 + l);
        wo_[static_cast<size_t>(l)] = qmat(H, Q, 830 + l);
        gate_[static_cast<size_t>(l)] = qmat(cfg_.intermediate, H, 840 + l);
        up_[static_cast<size_t>(l)] = qmat(cfg_.intermediate, H, 850 + l);
        down_[static_cast<size_t>(l)] = qmat(H, cfg_.intermediate, 860 + l);
    }
    ready_ = true;
    from_checkpoint_ = false;
}

bool H3TextEncoder::load_layer_mats(int l, quant::QuantMat &wq, quant::QuantMat &wk,
                                   quant::QuantMat &wv, quant::QuantMat &wo, quant::QuantMat &gate,
                                   quant::QuantMat &up, quant::QuantMat &down) const {
    wq.clear();
    wk.clear();
    wv.clear();
    wo.clear();
    gate.clear();
    up.clear();
    down.clear();
    if (l < 0 || static_cast<size_t>(l) >= hits_.size())
        return false;
    const LayerHits &h = hits_[static_cast<size_t>(l)];
    auto one = [](const io::StHit &hit, quant::QuantMat &dst) -> bool {
        if (!hit_mat(hit))
            return false;
        const int O = static_cast<int>(hit.tensor->shape[0]);
        const int I = static_cast<int>(hit.tensor->shape[1]);
        std::vector<float> tmp(static_cast<size_t>(O) * static_cast<size_t>(I));
        std::string err;
        if (io::st_read_f32(*hit.file, *hit.tensor, tmp.data(), static_cast<int64_t>(tmp.size()),
                            err) != Status::Ok)
            return false;
        dst.from_f32(tmp.data(), O, I, 32);
        return true;
    };
    const bool ok = one(h.wq, wq) && one(h.wk, wk) && one(h.wv, wv) && one(h.wo, wo) &&
                    one(h.gate, gate) && one(h.up, up) && one(h.down, down);
    return ok;
}

Status H3TextEncoder::load(const std::string &model_dir, std::string &err) {
    ready_ = false;
    from_checkpoint_ = false;
    streamed_ = false;
    hits_.clear();
    io::st_close_dir(files_);
    // Official Qwen3-VL-32B is ~67 GiB BF16; the host materializes f32 mats.
    // Skip on 64 GiB boxes (or any host that sets the env).
    if (const char *sk = std::getenv("MVLLM_H3_SKIP_TEXT"); sk && sk[0] && sk[0] != '0') {
        err.clear();
        cfg_ = H3TextConfig{};
        cfg_.hidden = 32;
        cfg_.layers = 2;
        cfg_.n_q = 4;
        cfg_.n_kv = 1;
        cfg_.head_dim = 8;
        cfg_.intermediate = 64;
        cfg_.vocab = 256;
        alloc_synth();
        return Status::Ok;
    }
    const char *tails[] = {"/FL2VA/text_encoder", "/Ref2VA/text_encoder", "/text_encoder", "/qwen",
                           ""};
    std::vector<io::StFile> files;
    std::string chosen;
    for (const char *tail : tails) {
        files.clear();
        err.clear();
        Status st = io::st_open_dir(model_dir + tail, files, err);
        if (st != Status::Ok || files.empty())
            continue;
        if (io::st_find_dir(files, "model.language_model.embed_tokens.weight").tensor ||
            io::st_find_dir(files, "model.embed_tokens.weight").tensor) {
            chosen = model_dir + tail;
            break;
        }
        io::st_close_dir(files);
    }
    if (chosen.empty()) {
        err.clear();
        cfg_ = H3TextConfig{};
        cfg_.hidden = 32;
        cfg_.layers = 2;
        cfg_.n_q = 4;
        cfg_.n_kv = 1;
        cfg_.head_dim = 8;
        cfg_.intermediate = 64;
        cfg_.vocab = 256;
        alloc_synth();
        return Status::Ok;
    }
    auto has = [&](const std::string &n) { return io::st_find_dir(files, n).tensor != nullptr; };
    const std::string P = has("model.language_model.embed_tokens.weight")
                              ? "model.language_model."
                              : "model.";
    io::StHit emb = io::st_find_dir(files, P + "embed_tokens.weight");
    if (!emb.tensor || emb.tensor->shape.size() != 2) {
        io::st_close_dir(files);
        err.clear();
        alloc_synth();
        return Status::Ok;
    }
    cfg_.vocab = static_cast<int>(emb.tensor->shape[0]);
    cfg_.hidden = static_cast<int>(emb.tensor->shape[1]);
    cfg_.layers = 0;
    for (int i = 0; i < 50; ++i) {
        if (!has(P + "layers." + std::to_string(i) + ".input_layernorm.weight"))
            break;
        cfg_.layers = i + 1;
    }
    if (cfg_.layers < 1) {
        io::st_close_dir(files);
        alloc_synth();
        return Status::Ok;
    }
    io::StHit q0 = io::st_find_dir(files, P + "layers.0.self_attn.q_proj.weight");
    if (q0.tensor && q0.tensor->shape.size() == 2) {
        const int qdim = static_cast<int>(q0.tensor->shape[0]);
        if (qdim % 128 == 0) {
            cfg_.head_dim = 128;
            cfg_.n_q = qdim / 128;
        } else if (qdim > 0) {
            cfg_.n_q = 64;
            cfg_.head_dim = std::max(qdim / std::max(cfg_.n_q, 1), 1);
        }
    }
    io::StHit k0 = io::st_find_dir(files, P + "layers.0.self_attn.k_proj.weight");
    if (k0.tensor && k0.tensor->shape.size() == 2 && cfg_.head_dim > 0)
        cfg_.n_kv = static_cast<int>(k0.tensor->shape[0]) / cfg_.head_dim;
    io::StHit g0 = io::st_find_dir(files, P + "layers.0.mlp.gate_proj.weight");
    if (g0.tensor && !g0.tensor->shape.empty())
        cfg_.intermediate = static_cast<int>(g0.tensor->shape[0]);
    if (cfg_.n_kv <= 0)
        cfg_.n_kv = 1;
    if (cfg_.head_dim <= 0)
        cfg_.head_dim = 8;

    // Never resize files_ after this: LayerHits pointers alias tensor storage.
    files_ = std::move(files);
    auto read_f = [&](const std::string &name, std::vector<float> &dst, int expect) -> Status {
        io::StHit hit = io::st_find_dir(files_, name);
        if (!hit.tensor)
            return Status::NotFound;
        int64_t n = 1;
        for (int64_t d : hit.tensor->shape)
            n *= d;
        if (expect > 0)
            n = expect;
        dst.assign(static_cast<size_t>(n), 0.f);
        return io::st_read_f32(*hit.file, *hit.tensor, dst.data(), n, err);
    };
    auto read_m = [&](const std::string &name, quant::QuantMat &dst) -> Status {
        io::StHit hit = io::st_find_dir(files_, name);
        if (!hit_mat(hit))
            return Status::NotFound;
        const int O = static_cast<int>(hit.tensor->shape[0]);
        const int I = static_cast<int>(hit.tensor->shape[1]);
        std::vector<float> tmp(static_cast<size_t>(O) * I);
        Status st = io::st_read_f32(*hit.file, *hit.tensor, tmp.data(),
                                    static_cast<int64_t>(tmp.size()), err);
        if (st != Status::Ok)
            return st;
        dst.from_f32(tmp.data(), O, I, 32);
        return Status::Ok;
    };
    auto find_layer = [&](int l, LayerHits &h) {
        const std::string ly = P + "layers." + std::to_string(l) + ".";
        h.wq = io::st_find_dir(files_, ly + "self_attn.q_proj.weight");
        h.wk = io::st_find_dir(files_, ly + "self_attn.k_proj.weight");
        h.wv = io::st_find_dir(files_, ly + "self_attn.v_proj.weight");
        h.wo = io::st_find_dir(files_, ly + "self_attn.o_proj.weight");
        h.gate = io::st_find_dir(files_, ly + "mlp.gate_proj.weight");
        h.up = io::st_find_dir(files_, ly + "mlp.up_proj.weight");
        h.down = io::st_find_dir(files_, ly + "mlp.down_proj.weight");
    };
    auto complete = [](const LayerHits &h) {
        return hit_mat(h.wq) && hit_mat(h.wk) && hit_mat(h.wv) && hit_mat(h.wo) &&
               hit_mat(h.gate) && hit_mat(h.up) && hit_mat(h.down);
    };

    const int H = cfg_.hidden;
    const int L = cfg_.layers;
    read_f(P + "embed_tokens.weight", embed_, cfg_.vocab * H);
    read_f(P + "norm.weight", norm_, H);
    if (norm_.empty())
        norm_.assign(static_cast<size_t>(H), 1.f);
    in_n_.assign(static_cast<size_t>(L), {});
    post_n_.assign(static_cast<size_t>(L), {});
    qn_.assign(static_cast<size_t>(L), {});
    kn_.assign(static_cast<size_t>(L), {});

    LayerHits h0{};
    find_layer(0, h0);
    bool force_resident = false;
    if (const char *r = std::getenv("MVLLM_H3_TEXT_RESIDENT"); r && r[0] && r[0] != '0')
        force_resident = true;
    if (complete(h0) && !force_resident) {
        // Official 32B must not be fully f32-resident; keep files open and stream one layer.
        streamed_ = true;
        hits_.assign(static_cast<size_t>(L), {});
        wq_.clear();
        wk_.clear();
        wv_.clear();
        wo_.clear();
        gate_.clear();
        up_.clear();
        down_.clear();
        for (int l = 0; l < L; ++l) {
            const size_t li = static_cast<size_t>(l);
            find_layer(l, hits_[li]);
            const std::string ly = P + "layers." + std::to_string(l) + ".";
            read_f(ly + "input_layernorm.weight", in_n_[li], H);
            read_f(ly + "post_attention_layernorm.weight", post_n_[li], H);
            read_f(ly + "self_attn.q_norm.weight", qn_[li], cfg_.head_dim);
            read_f(ly + "self_attn.k_norm.weight", kn_[li], cfg_.head_dim);
            if (in_n_[li].empty())
                in_n_[li].assign(static_cast<size_t>(H), 1.f);
            if (post_n_[li].empty())
                post_n_[li].assign(static_cast<size_t>(H), 1.f);
        }
    } else {
        streamed_ = false;
        hits_.clear();
        wq_.assign(static_cast<size_t>(L), {});
        wk_.assign(static_cast<size_t>(L), {});
        wv_.assign(static_cast<size_t>(L), {});
        wo_.assign(static_cast<size_t>(L), {});
        gate_.assign(static_cast<size_t>(L), {});
        up_.assign(static_cast<size_t>(L), {});
        down_.assign(static_cast<size_t>(L), {});
        for (int l = 0; l < L; ++l) {
            const size_t li = static_cast<size_t>(l);
            const std::string ly = P + "layers." + std::to_string(l) + ".";
            read_f(ly + "input_layernorm.weight", in_n_[li], H);
            read_f(ly + "post_attention_layernorm.weight", post_n_[li], H);
            read_m(ly + "self_attn.q_proj.weight", wq_[li]);
            read_m(ly + "self_attn.k_proj.weight", wk_[li]);
            read_m(ly + "self_attn.v_proj.weight", wv_[li]);
            read_m(ly + "self_attn.o_proj.weight", wo_[li]);
            read_f(ly + "self_attn.q_norm.weight", qn_[li], cfg_.head_dim);
            read_f(ly + "self_attn.k_norm.weight", kn_[li], cfg_.head_dim);
            read_m(ly + "mlp.gate_proj.weight", gate_[li]);
            read_m(ly + "mlp.up_proj.weight", up_[li]);
            read_m(ly + "mlp.down_proj.weight", down_[li]);
            if (in_n_[li].empty())
                in_n_[li].assign(static_cast<size_t>(H), 1.f);
            if (post_n_[li].empty())
                post_n_[li].assign(static_cast<size_t>(H), 1.f);
        }
        io::st_close_dir(files_);
    }

    ready_ = !embed_.empty() && (streamed_ ? (!hits_.empty() && hits_[0].wq.tensor)
                                           : (!wq_.empty() && !wq_[0].empty()));
    from_checkpoint_ = ready_;
    if (!ready_) {
        alloc_synth();
        from_checkpoint_ = false;
    }
    err.clear();
    return Status::Ok;
}

void H3TextEncoder::apply_layer(int l, int T, const quant::QuantMat *wq, const quant::QuantMat *wk,
                               const quant::QuantMat *wv, const quant::QuantMat *wo,
                               const quant::QuantMat *gate, const quant::QuantMat *up,
                               const quant::QuantMat *down, const uint32_t *positions,
                               const H3VisionSpan *spans, int span_count, std::vector<float> &out,
                               std::vector<float> &n, std::vector<float> &q, std::vector<float> &k,
                               std::vector<float> &v, std::vector<float> &ctx,
                               std::vector<float> &attn, std::vector<float> &g, std::vector<float> &u,
                               std::vector<float> &d) const {
    const int H = cfg_.hidden;
    const int hd = cfg_.head_dim;
    const int nq = cfg_.n_q;
    const int nkv = std::max(cfg_.n_kv, 1);
    const int I = cfg_.intermediate;
    const float scale = 1.f / std::sqrt(static_cast<float>(std::max(hd, 1)));
    const int group = std::max(nq / nkv, 1);
    const size_t li = static_cast<size_t>(l);
    auto empty = [](const quant::QuantMat *m) { return !m || m->empty(); };
    const bool all_empty = empty(wq) && empty(wk) && empty(wv) && empty(wo) && empty(gate) &&
                           empty(up) && empty(down);
    if (!all_empty) {
        const float *in_w = (li < in_n_.size() && !in_n_[li].empty()) ? in_n_[li].data() : nullptr;
        const float *post_w =
            (li < post_n_.size() && !post_n_[li].empty()) ? post_n_[li].data() : nullptr;
        for (int t = 0; t < T; ++t)
            quant::rmsnorm(out.data() + t * H, in_w, n.data() + t * H, H, cfg_.rms_eps);
        auto gemm = [](const quant::QuantMat *w, float *y, const float *x, int S, size_t nelt) {
            if (!w || w->empty()) {
                if (y && nelt)
                    std::fill(y, y + nelt, 0.f);
                return;
            }
            w->gemm(y, x, S);
        };
        gemm(wq, q.data(), n.data(), T, q.size());
        gemm(wk, k.data(), n.data(), T, k.size());
        gemm(wv, v.data(), n.data(), T, v.size());
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < nq; ++h) {
                float *qh = q.data() + static_cast<size_t>(t) * nq * hd + h * hd;
                if (li < qn_.size() && !qn_[li].empty())
                    quant::rmsnorm(qh, qn_[li].data(), qh, hd, cfg_.rms_eps);
                if (positions)
                    apply_mrope(qh, hd, t, T, positions, cfg_.rope_theta);
                else
                    apply_rope(qh, hd, t, cfg_.rope_theta);
            }
            for (int h = 0; h < nkv; ++h) {
                float *kh = k.data() + static_cast<size_t>(t) * nkv * hd + h * hd;
                if (li < kn_.size() && !kn_[li].empty())
                    quant::rmsnorm(kh, kn_[li].data(), kh, hd, cfg_.rms_eps);
                if (positions)
                    apply_mrope(kh, hd, t, T, positions, cfg_.rope_theta);
                else
                    apply_rope(kh, hd, t, cfg_.rope_theta);
            }
        }
        std::fill(ctx.begin(), ctx.end(), 0.f);
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < nq; ++h) {
                const int kh = h / group;
                const float *qh = q.data() + static_cast<size_t>(t) * nq * hd + h * hd;
                std::vector<float> sc(static_cast<size_t>(t + 1), 0.f);
                float mx = -1e30f;
                for (int s = 0; s <= t; ++s) {
                    const float *kk = k.data() + static_cast<size_t>(s) * nkv * hd + kh * hd;
                    float acc = 0.f;
                    for (int d0 = 0; d0 < hd; ++d0)
                        acc += qh[d0] * kk[d0];
                    sc[static_cast<size_t>(s)] = acc * scale;
                    if (sc[static_cast<size_t>(s)] > mx)
                        mx = sc[static_cast<size_t>(s)];
                }
                float z = 0.f;
                for (int s = 0; s <= t; ++s) {
                    sc[static_cast<size_t>(s)] = std::exp(sc[static_cast<size_t>(s)] - mx);
                    z += sc[static_cast<size_t>(s)];
                }
                float *o = ctx.data() + static_cast<size_t>(t) * nq * hd + h * hd;
                for (int s = 0; s <= t; ++s) {
                    const float *vv = v.data() + static_cast<size_t>(s) * nkv * hd + kh * hd;
                    float a = sc[static_cast<size_t>(s)] / z;
                    for (int d0 = 0; d0 < hd; ++d0)
                        o[d0] += a * vv[d0];
                }
            }
        }
        gemm(wo, attn.data(), ctx.data(), T, attn.size());
        for (int i = 0; i < T * H; ++i)
            out[static_cast<size_t>(i)] += attn[static_cast<size_t>(i)];
        for (int t = 0; t < T; ++t)
            quant::rmsnorm(out.data() + t * H, post_w, n.data() + t * H, H, cfg_.rms_eps);
        gemm(gate, g.data(), n.data(), T, g.size());
        gemm(up, u.data(), n.data(), T, u.size());
        for (int i = 0; i < T * I; ++i)
            g[static_cast<size_t>(i)] =
                g[static_cast<size_t>(i)] * quant::sigmoid(g[static_cast<size_t>(i)]) *
                u[static_cast<size_t>(i)];
        gemm(down, d.data(), g.data(), T, d.size());
        for (int i = 0; i < T * H; ++i)
            out[static_cast<size_t>(i)] += d[static_cast<size_t>(i)];
    }
    if (l < kH3TextDeepstacks && spans && span_count > 0) {
        for (int s = 0; s < span_count; ++s) {
            const H3VisionSpan &sp = spans[s];
            if (!sp.deepstack[l] || sp.tokens <= 0 || sp.start < 0 ||
                sp.start > T || sp.tokens > T - sp.start)
                continue;
            for (int i = 0; i < sp.tokens; ++i) {
                float *dst = out.data() + static_cast<size_t>(sp.start + i) * H;
                const float *src = sp.deepstack[l] + static_cast<size_t>(i) * H;
                for (int d0 = 0; d0 < H; ++d0)
                    dst[d0] += src[d0];
            }
        }
    }
}

void H3TextEncoder::encode(const std::vector<int> &ids, std::vector<float> &out) const {
    encode_mm(ids, nullptr, 0, nullptr, nullptr, out, cfg_.layers);
}

void H3TextEncoder::encode_mm(const std::vector<int> &ids, const H3VisionSpan *spans, int span_count,
                              const uint32_t *positions, const uint8_t *tags,
                              std::vector<float> &out) const {
    encode_mm(ids, spans, span_count, positions, tags, out, cfg_.layers);
}

void H3TextEncoder::encode_mm(const std::vector<int> &ids, const H3VisionSpan *spans, int span_count,
                              const uint32_t *positions, const uint8_t *tags,
                              std::vector<float> &out, int layer_count) const {
    (void)tags;
    out.clear();
    if (!ready_ || ids.empty())
        return;
    const int T = static_cast<int>(ids.size());
    const int H = cfg_.hidden;
    const int hd = cfg_.head_dim;
    const int nq = cfg_.n_q;
    const int nkv = std::max(cfg_.n_kv, 1);
    const int I = cfg_.intermediate;
    int L = layer_count;
    if (L <= 0 || L > cfg_.layers)
        L = cfg_.layers;
    out.assign(static_cast<size_t>(T) * H, 0.f);
    for (int t = 0; t < T; ++t) {
        int id = ids[static_cast<size_t>(t)];
        if (id < 0 || id >= cfg_.vocab)
            id = 0;
        std::memcpy(out.data() + static_cast<size_t>(t) * H,
                    embed_.data() + static_cast<size_t>(id) * H, static_cast<size_t>(H) * sizeof(float));
    }
    if (spans && span_count > 0) {
        for (int s = 0; s < span_count; ++s) {
            const H3VisionSpan &sp = spans[s];
            if (!sp.embeddings || sp.tokens <= 0 || sp.start < 0 ||
                sp.start > T || sp.tokens > T - sp.start)
                continue;
            for (int i = 0; i < sp.tokens; ++i) {
                float *dst = out.data() + static_cast<size_t>(sp.start + i) * H;
                const float *src = sp.embeddings + static_cast<size_t>(i) * H;
                std::memcpy(dst, src, static_cast<size_t>(H) * sizeof(float));
            }
        }
    }
    std::vector<float> n(static_cast<size_t>(T) * H), q(static_cast<size_t>(T) * nq * hd),
        k(static_cast<size_t>(T) * nkv * hd), v(static_cast<size_t>(T) * nkv * hd),
        ctx(static_cast<size_t>(T) * nq * hd), attn(static_cast<size_t>(T) * H),
        g(static_cast<size_t>(T) * I), u(static_cast<size_t>(T) * I), d(static_cast<size_t>(T) * H);
    for (int l = 0; l < L; ++l) {
        if (streamed_) {
            // Peak: one layer of f32 mats; locals free before the next layer.
            quant::QuantMat lwq, lwk, lwv, lwo, lgate, lup, ldown;
            load_layer_mats(l, lwq, lwk, lwv, lwo, lgate, lup, ldown);
            apply_layer(l, T, &lwq, &lwk, &lwv, &lwo, &lgate, &lup, &ldown, positions, spans,
                        span_count, out, n, q, k, v, ctx, attn, g, u, d);
        } else {
            const size_t li = static_cast<size_t>(l);
            apply_layer(l, T, li < wq_.size() ? &wq_[li] : nullptr,
                        li < wk_.size() ? &wk_[li] : nullptr, li < wv_.size() ? &wv_[li] : nullptr,
                        li < wo_.size() ? &wo_[li] : nullptr,
                        li < gate_.size() ? &gate_[li] : nullptr,
                        li < up_.size() ? &up_[li] : nullptr,
                        li < down_.size() ? &down_[li] : nullptr, positions, spans, span_count, out,
                        n, q, k, v, ctx, attn, g, u, d);
        }
    }
    if (!norm_.empty()) {
        for (int t = 0; t < T; ++t)
            quant::rmsnorm(out.data() + t * H, norm_.data(), out.data() + t * H, H, cfg_.rms_eps);
    }
}

} // namespace mvllm
