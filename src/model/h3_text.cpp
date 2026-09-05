#include "h3_text.hpp"
#include "family.hpp"
#include "../io/safetensors.hpp"
#include "../quant/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

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

} // namespace

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

Status H3TextEncoder::load(const std::string &model_dir, std::string &err) {
    ready_ = false;
    from_checkpoint_ = false;
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

    const int H = cfg_.hidden;
    const int L = cfg_.layers;
    auto read_f = [&](const std::string &name, std::vector<float> &dst, int expect) -> Status {
        io::StHit hit = io::st_find_dir(files, name);
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
        io::StHit hit = io::st_find_dir(files, name);
        if (!hit.tensor || hit.tensor->shape.size() != 2)
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
    read_f(P + "embed_tokens.weight", embed_, cfg_.vocab * H);
    read_f(P + "norm.weight", norm_, H);
    if (norm_.empty())
        norm_.assign(static_cast<size_t>(H), 1.f);
    wq_.resize(static_cast<size_t>(L));
    wk_.resize(static_cast<size_t>(L));
    wv_.resize(static_cast<size_t>(L));
    wo_.resize(static_cast<size_t>(L));
    gate_.resize(static_cast<size_t>(L));
    up_.resize(static_cast<size_t>(L));
    down_.resize(static_cast<size_t>(L));
    in_n_.resize(static_cast<size_t>(L));
    post_n_.resize(static_cast<size_t>(L));
    qn_.resize(static_cast<size_t>(L));
    kn_.resize(static_cast<size_t>(L));
    for (int l = 0; l < L; ++l) {
        const std::string ly = P + "layers." + std::to_string(l) + ".";
        read_f(ly + "input_layernorm.weight", in_n_[static_cast<size_t>(l)], H);
        read_f(ly + "post_attention_layernorm.weight", post_n_[static_cast<size_t>(l)], H);
        read_m(ly + "self_attn.q_proj.weight", wq_[static_cast<size_t>(l)]);
        read_m(ly + "self_attn.k_proj.weight", wk_[static_cast<size_t>(l)]);
        read_m(ly + "self_attn.v_proj.weight", wv_[static_cast<size_t>(l)]);
        read_m(ly + "self_attn.o_proj.weight", wo_[static_cast<size_t>(l)]);
        read_f(ly + "self_attn.q_norm.weight", qn_[static_cast<size_t>(l)], cfg_.head_dim);
        read_f(ly + "self_attn.k_norm.weight", kn_[static_cast<size_t>(l)], cfg_.head_dim);
        read_m(ly + "mlp.gate_proj.weight", gate_[static_cast<size_t>(l)]);
        read_m(ly + "mlp.up_proj.weight", up_[static_cast<size_t>(l)]);
        read_m(ly + "mlp.down_proj.weight", down_[static_cast<size_t>(l)]);
        if (in_n_[static_cast<size_t>(l)].empty())
            in_n_[static_cast<size_t>(l)].assign(static_cast<size_t>(H), 1.f);
        if (post_n_[static_cast<size_t>(l)].empty())
            post_n_[static_cast<size_t>(l)].assign(static_cast<size_t>(H), 1.f);
    }
    io::st_close_dir(files);
    ready_ = !embed_.empty() && !wq_[0].empty();
    from_checkpoint_ = ready_;
    if (!ready_) {
        alloc_synth();
        from_checkpoint_ = false;
    }
    err.clear();
    return Status::Ok;
}

void H3TextEncoder::encode(const std::vector<int> &ids, std::vector<float> &out) const {
    out.clear();
    if (!ready_ || ids.empty())
        return;
    const int T = static_cast<int>(ids.size());
    const int H = cfg_.hidden;
    const int hd = cfg_.head_dim;
    const int nq = cfg_.n_q;
    const int nkv = std::max(cfg_.n_kv, 1);
    const int I = cfg_.intermediate;
    out.assign(static_cast<size_t>(T) * H, 0.f);
    for (int t = 0; t < T; ++t) {
        int id = ids[static_cast<size_t>(t)];
        if (id < 0 || id >= cfg_.vocab)
            id = 0;
        std::memcpy(out.data() + static_cast<size_t>(t) * H,
                    embed_.data() + static_cast<size_t>(id) * H, static_cast<size_t>(H) * sizeof(float));
    }
    std::vector<float> n(static_cast<size_t>(T) * H), q(static_cast<size_t>(T) * nq * hd),
        k(static_cast<size_t>(T) * nkv * hd), v(static_cast<size_t>(T) * nkv * hd),
        ctx(static_cast<size_t>(T) * nq * hd), attn(static_cast<size_t>(T) * H),
        g(static_cast<size_t>(T) * I), u(static_cast<size_t>(T) * I), d(static_cast<size_t>(T) * H);
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    const int group = std::max(nq / nkv, 1);
    for (int l = 0; l < cfg_.layers; ++l) {
        for (int t = 0; t < T; ++t)
            quant::rmsnorm(out.data() + t * H, in_n_[static_cast<size_t>(l)].data(),
                           n.data() + t * H, H, cfg_.rms_eps);
        wq_[static_cast<size_t>(l)].gemm(q.data(), n.data(), T);
        wk_[static_cast<size_t>(l)].gemm(k.data(), n.data(), T);
        wv_[static_cast<size_t>(l)].gemm(v.data(), n.data(), T);
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < nq; ++h) {
                float *qh = q.data() + static_cast<size_t>(t) * nq * hd + h * hd;
                if (!qn_[static_cast<size_t>(l)].empty())
                    quant::rmsnorm(qh, qn_[static_cast<size_t>(l)].data(), qh, hd, cfg_.rms_eps);
                apply_rope(qh, hd, t, cfg_.rope_theta);
            }
            for (int h = 0; h < nkv; ++h) {
                float *kh = k.data() + static_cast<size_t>(t) * nkv * hd + h * hd;
                if (!kn_[static_cast<size_t>(l)].empty())
                    quant::rmsnorm(kh, kn_[static_cast<size_t>(l)].data(), kh, hd, cfg_.rms_eps);
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
        wo_[static_cast<size_t>(l)].gemm(attn.data(), ctx.data(), T);
        for (int i = 0; i < T * H; ++i)
            out[static_cast<size_t>(i)] += attn[static_cast<size_t>(i)];
        for (int t = 0; t < T; ++t)
            quant::rmsnorm(out.data() + t * H, post_n_[static_cast<size_t>(l)].data(),
                           n.data() + t * H, H, cfg_.rms_eps);
        gate_[static_cast<size_t>(l)].gemm(g.data(), n.data(), T);
        up_[static_cast<size_t>(l)].gemm(u.data(), n.data(), T);
        for (int i = 0; i < T * I; ++i)
            g[static_cast<size_t>(i)] =
                g[static_cast<size_t>(i)] * quant::sigmoid(g[static_cast<size_t>(i)]) *
                u[static_cast<size_t>(i)];
        down_[static_cast<size_t>(l)].gemm(d.data(), g.data(), T);
        for (int i = 0; i < T * H; ++i)
            out[static_cast<size_t>(i)] += d[static_cast<size_t>(i)];
    }
    if (!norm_.empty()) {
        for (int t = 0; t < T; ++t)
            quant::rmsnorm(out.data() + t * H, norm_.data(), out.data() + t * H, H, cfg_.rms_eps);
    }
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
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    const int group = std::max(nq / nkv, 1);
    for (int l = 0; l < L; ++l) {
        for (int t = 0; t < T; ++t)
            quant::rmsnorm(out.data() + t * H, in_n_[static_cast<size_t>(l)].data(),
                           n.data() + t * H, H, cfg_.rms_eps);
        wq_[static_cast<size_t>(l)].gemm(q.data(), n.data(), T);
        wk_[static_cast<size_t>(l)].gemm(k.data(), n.data(), T);
        wv_[static_cast<size_t>(l)].gemm(v.data(), n.data(), T);
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < nq; ++h) {
                float *qh = q.data() + static_cast<size_t>(t) * nq * hd + h * hd;
                if (!qn_[static_cast<size_t>(l)].empty())
                    quant::rmsnorm(qh, qn_[static_cast<size_t>(l)].data(), qh, hd, cfg_.rms_eps);
                if (positions)
                    apply_mrope(qh, hd, t, T, positions, cfg_.rope_theta);
                else
                    apply_rope(qh, hd, t, cfg_.rope_theta);
            }
            for (int h = 0; h < nkv; ++h) {
                float *kh = k.data() + static_cast<size_t>(t) * nkv * hd + h * hd;
                if (!kn_[static_cast<size_t>(l)].empty())
                    quant::rmsnorm(kh, kn_[static_cast<size_t>(l)].data(), kh, hd, cfg_.rms_eps);
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
        wo_[static_cast<size_t>(l)].gemm(attn.data(), ctx.data(), T);
        for (int i = 0; i < T * H; ++i)
            out[static_cast<size_t>(i)] += attn[static_cast<size_t>(i)];
        for (int t = 0; t < T; ++t)
            quant::rmsnorm(out.data() + t * H, post_n_[static_cast<size_t>(l)].data(),
                           n.data() + t * H, H, cfg_.rms_eps);
        gate_[static_cast<size_t>(l)].gemm(g.data(), n.data(), T);
        up_[static_cast<size_t>(l)].gemm(u.data(), n.data(), T);
        for (int i = 0; i < T * I; ++i)
            g[static_cast<size_t>(i)] =
                g[static_cast<size_t>(i)] * quant::sigmoid(g[static_cast<size_t>(i)]) *
                u[static_cast<size_t>(i)];
        down_[static_cast<size_t>(l)].gemm(d.data(), g.data(), T);
        for (int i = 0; i < T * H; ++i)
            out[static_cast<size_t>(i)] += d[static_cast<size_t>(i)];
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
    if (!norm_.empty()) {
        for (int t = 0; t < T; ++t)
            quant::rmsnorm(out.data() + t * H, norm_.data(), out.data() + t * H, H, cfg_.rms_eps);
    }
}

} // namespace mvllm
