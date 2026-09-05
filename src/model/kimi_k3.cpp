#include "family.hpp"
#include "../gpu/backend.hpp"
#include "../io/safetensors.hpp"
#include "../quant/quant.hpp"
#include "../quant/weight.hpp"
#include "../serve/session.hpp"
#include "../tok/decode_post.hpp"
#include "../tok/gbnf.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

namespace mvllm {
namespace {

struct Dense {
    std::vector<float> embed;
    std::vector<float> norm;
    quant::QuantMat lm_head;
    std::vector<std::vector<float>> attn_in_n, attn_out_n;
    std::vector<quant::QuantMat> wq, wk, wv, wo, wg;
    std::vector<quant::QuantMat> mla_qa, mla_qb, mla_kva, mla_kt, mla_v, mla_o, mla_g;
    std::vector<std::vector<float>> mla_qa_ln, mla_kva_ln;
    std::vector<quant::QuantMat> wfa, wfb, wb;
    std::vector<std::vector<float>> wdt, alog;
    std::vector<std::vector<float>> conv_q, conv_k, conv_v;
    std::vector<std::vector<float>> out_norm;
    std::vector<std::vector<float>> router, router_bias;
    std::vector<quant::QuantMat> lat_down, lat_up;
    std::vector<std::vector<float>> lat_n;
    std::vector<quant::QuantMat> shared_gate, shared_up, shared_down;
    std::vector<std::vector<float>> res_n, res_p, mlp_res_n, mlp_res_p, attn_sw, mlp_sw;
    std::vector<float> out_sw;
    std::vector<quant::QuantMat> mlp_gate, mlp_up, mlp_down;
};

void xavier(std::vector<float> &w, int rows, int cols, uint32_t seed) {
    w.assign(static_cast<size_t>(rows) * cols, 0.f);
    std::mt19937 rng(seed);
    float s = 1.f / std::sqrt(static_cast<float>(cols));
    std::normal_distribution<float> dist(0.f, s);
    for (float &v : w)
        v = dist(rng);
}

quant::QuantMat qmat_xavier(int O, int I, uint32_t seed, int bits) {
    std::vector<float> t;
    xavier(t, O, I, seed);
    quant::QuantMat m;
    m.from_f32(t.data(), O, I, bits);
    return m;
}

void ones(std::vector<float> &w, int n) { w.assign(n, 1.f); }

void fold_sw(const std::vector<float> &n, const std::vector<float> &p, std::vector<float> &sw,
             int dim) {
    sw.assign(static_cast<size_t>(std::max(dim, 1)), 1.f);
    const int m = std::min(dim, static_cast<int>(std::min(n.size(), p.size())));
    if (m > 0) {
        for (int i = 0; i < m; ++i)
            sw[static_cast<size_t>(i)] = n[static_cast<size_t>(i)] * p[static_cast<size_t>(i)];
    } else if (!n.empty()) {
        const int k = std::min(dim, static_cast<int>(n.size()));
        for (int i = 0; i < k; ++i)
            sw[static_cast<size_t>(i)] = n[static_cast<size_t>(i)];
    }
}

struct K3ExpertGeom {
    int64_t w1p = 0, w1s = 0, w2p = 0, w2s = 0, slot = 0;
};

K3ExpertGeom make_k3_geom(int latent, int inter) {
    K3ExpertGeom g{};
    if (latent >= 32 && inter >= 32 && latent % 32 == 0 && inter % 32 == 0) {
        g.w1p = static_cast<int64_t>(inter) * (latent / 2);
        g.w1s = static_cast<int64_t>(inter) * (latent / 32);
        g.w2p = static_cast<int64_t>(latent) * (inter / 2);
        g.w2s = static_cast<int64_t>(latent) * (inter / 32);
    } else {
        g.w1p = static_cast<int64_t>(inter) * ((latent + 1) / 2);
        g.w1s = static_cast<int64_t>(inter) * ((latent + 31) / 32);
        g.w2p = static_cast<int64_t>(latent) * ((inter + 1) / 2);
        g.w2s = static_cast<int64_t>(latent) * ((inter + 31) / 32);
    }
    g.slot = 2 * (g.w1p + g.w1s) + g.w2p + g.w2s;
    return g;
}

const char *k3_expert_mats[3] = {"w1", "w2", "w3"};
const char *k3_expert_half[2] = {"packed", "scale"};

int sample_penalized(const float *logits, int vocab, const GenParams &gp, const int *hist,
                     int hist_n, uint64_t *rng, const uint8_t *allow, float *lp,
                     std::vector<float> *token_lps,
                     std::vector<std::vector<GenLogprob>> *top_lps) {
    std::vector<float> work(static_cast<size_t>(std::max(vocab, 0)));
    if (logits && vocab > 0)
        std::memcpy(work.data(), logits, static_cast<size_t>(vocab) * sizeof(float));
    apply_penalties(work.data(), vocab, hist, hist_n, gp.frequency_penalty, gp.presence_penalty);
    apply_logit_bias(work.data(), vocab, gp.logit_bias.empty() ? nullptr : gp.logit_bias.data(),
                     static_cast<int>(gp.logit_bias.size()));
    apply_repetition_penalty(work.data(), vocab, hist, hist_n, gp.repetition_penalty);
    apply_top_k(work.data(), vocab, gp.top_k);
    apply_min_p(work.data(), vocab, gp.min_p);
    float local_lp = 0.f;
    int tok = sample_token(work.data(), vocab, gp.temperature, gp.top_p, rng, allow, &local_lp);
    if (lp)
        *lp = local_lp;
    if (gp.logprobs > 0 && token_lps && top_lps) {
        token_lps->push_back(local_lp);
        std::vector<GenLogprob> row(static_cast<size_t>(gp.logprobs));
        int n = 0;
        top_logprobs(work.data(), vocab, gp.logprobs, row.data(), &n, allow);
        row.resize(static_cast<size_t>(std::max(n, 0)));
        top_lps->push_back(std::move(row));
    }
    return tok;
}

} // namespace

class KimiK3Engine final : public FamilyEngine {
public:
    Family family() const override { return Family::KimiK3; }
    const ModelConfig &config() const override { return cfg_; }

    struct K3Slot {
        std::vector<int> history;
        std::vector<std::vector<float>> S, winq, wink, winv, mla_cache;
        std::vector<float> h;
        int pos = 0;
        bool have = false;
        bool live = false;
        GenParams gp;
        uint64_t rng = 1;
        int emitted = 0;
        Gbnf g;
        std::vector<uint8_t> allow;
        std::string acc;
        std::vector<float> token_lps;
        std::vector<std::vector<GenLogprob>> top_lps;
    };

    Status load(const std::string &model_dir, const RuntimeConfig &rt, std::string &err) override {
        rt_ = rt;
        Status st = load_model_config(model_dir, cfg_, err);
        if (st != Status::Ok)
            return st;
        if (cfg_.family != Family::KimiK3) {
            err = "not a Kimi K3 config";
            return Status::Unsupported;
        }
        alloc_synthetic();

        const int64_t ebytes = expert_bytes();
        const int64_t cap = static_cast<int64_t>(rt_.expert_gb * 1024.0 * 1024.0 * 1024.0);
        st = store_.open(cfg_.n_layers, std::max(cfg_.moe.n_experts, 1), ebytes,
                         cap > 0 ? cap : ebytes, err);
        if (st != Status::Ok)
            return st;
        store_.set_direct(rt_.o_direct);

        st = load_checkpoint(model_dir, err);
        if (st == Status::NotFound) {
            err.clear();
            st = write_synthetic_experts(model_dir, err);
        }
        if (st != Status::Ok)
            return st;
        experts_.clear();
        loaded_ = true;
        return Status::Ok;
    }

    Status generate(const std::vector<int> &prompt, const GenParams &gp, GenResult &out,
                    std::string &err) override {
        if (!loaded_) {
            err = "kimi_k3 not loaded";
            return Status::InvalidArgument;
        }
        if (prompt.empty()) {
            err = "empty prompt";
            return Status::InvalidArgument;
        }
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        std::vector<float> h(H, 0.f), prefix(H, 0.f);
        std::vector<std::vector<float>> snaps;
        // KDA state: [layer, heads, D, D]
        const int kd = cfg_.kda.head_dim;
        const int kh = cfg_.kda.heads;
        std::vector<std::vector<float>> S(L);
        const int P = kh * kd;
        const int Kc = cfg_.kda.conv_k > 0 ? cfg_.kda.conv_k : 4;
        std::vector<std::vector<float>> winq(L), wink(L), winv(L);
        const int kvL = std::max(cfg_.mla.kv_lora, 0);
        const int kvStride = kvL + std::max(cfg_.mla.qk_rope, 0);
        int Tmax = std::max(rt_.max_seq, static_cast<int>(prompt.size()) + gp.max_new_tokens);
        if (Tmax <= 0)
            Tmax = 4096;
        std::vector<std::vector<float>> mla_cache(L);
        for (int l = 0; l < L; ++l) {
            S[l].assign(static_cast<size_t>(kh) * kd * kd, 0.f);
            winq[l].assign(static_cast<size_t>(P) * Kc, 0.f);
            wink[l].assign(static_cast<size_t>(P) * Kc, 0.f);
            winv[l].assign(static_cast<size_t>(P) * Kc, 0.f);
            if (kvStride > 0)
                mla_cache[l].assign(static_cast<size_t>(Tmax) * kvStride, 0.f);
        }
        int pos = 0;
        int reuse = 0;
        const int cslot = gp.cache_slot;
        if (cslot >= 0 && cslot < kMaxKvSlots && gp.prefix_reuse > 0) {
            K3Slot &sl = slots_[cslot];
            const int match = sl.have ? kv_common_prefix(sl.history, prompt) : 0;
            const int want = std::min(gp.prefix_reuse, match);
            if (sl.have && sl.pos > 0 && sl.pos <= want &&
                sl.pos <= static_cast<int>(prompt.size())) {
                reuse = sl.pos;
                S = sl.S;
                winq = sl.winq;
                wink = sl.wink;
                winv = sl.winv;
                mla_cache = sl.mla_cache;
                h = sl.h;
                pos = sl.pos;
                if (kvStride > 0) {
                    const size_t need = static_cast<size_t>(Tmax) * kvStride;
                    for (int l = 0; l < L; ++l) {
                        if (mla_cache[l].size() < need)
                            mla_cache[l].resize(need, 0.f);
                    }
                }
            }
        }

        auto embed_tok = [&](int id) {
            int tid = id;
            if (tid < 0 || tid >= cfg_.vocab)
                tid = 0;
            std::memcpy(h.data(), d_.embed.data() + static_cast<size_t>(tid) * H, H * sizeof(float));
        };

        std::vector<int> seq = prompt;
        out.prompt_tokens = static_cast<int>(prompt.size());

        auto sw_attn = [&](int l) -> const float * {
            return (l < static_cast<int>(d_.attn_sw.size()) && !d_.attn_sw[l].empty())
                       ? d_.attn_sw[l].data()
                       : nullptr;
        };
        auto sw_mlp = [&](int l) -> const float * {
            return (l < static_cast<int>(d_.mlp_sw.size()) && !d_.mlp_sw[l].empty())
                       ? d_.mlp_sw[l].data()
                       : nullptr;
        };

        auto step = [&](int token, bool) -> Status {
            embed_tok(token);
            snaps.clear();
            for (int l = 0; l < L; ++l) {
                const bool attnres = cfg_.attn_res.block_size > 0;
                const bool snap = attnres && (l % cfg_.attn_res.block_size) == 0;
                if (attnres) {
                    prefix = h;
                    if (!snaps.empty())
                        attnres_mix(snaps, prefix.data(), sw_attn(l), nullptr, h.data(), H,
                                    cfg_.rms_eps);
                    if (snap)
                        snaps.push_back(prefix);
                }

                std::vector<float> n(H), y(H, 0.f);
                quant::rmsnorm(h.data(), d_.attn_in_n[l].data(), n.data(), H, cfg_.rms_eps);

                bool use_kda = (l < static_cast<int>(cfg_.is_kda.size())) ? cfg_.is_kda[l] : 1;
                if (use_kda) {
                    kda_step(n.data(), H, cfg_.kda, &d_.wq[l], &d_.wk[l], &d_.wv[l], &d_.wb[l],
                             &d_.wfa[l], &d_.wfb[l], d_.wdt[l].data(),
                             static_cast<int>(d_.wdt[l].size()), d_.alog[l].data(), &d_.wg[l],
                             nullptr, &d_.wo[l],
                             d_.out_norm[l].empty() ? nullptr : d_.out_norm[l].data(),
                             S[l].data(), y.data(),
                             cfg_.rms_eps,
                             l < static_cast<int>(d_.conv_q.size()) && !d_.conv_q[l].empty()
                                 ? d_.conv_q[l].data()
                                 : nullptr,
                             l < static_cast<int>(d_.conv_k.size()) && !d_.conv_k[l].empty()
                                 ? d_.conv_k[l].data()
                                 : nullptr,
                             l < static_cast<int>(d_.conv_v.size()) && !d_.conv_v[l].empty()
                                 ? d_.conv_v[l].data()
                                 : nullptr,
                             winq[l].data(), wink[l].data(), winv[l].data());
                } else {
                    mla_step(n.data(), H, cfg_.mla, &d_.mla_qa[l],
                             d_.mla_qa_ln[l].empty() ? nullptr : d_.mla_qa_ln[l].data(),
                             &d_.mla_qb[l], &d_.mla_kva[l],
                             d_.mla_kva_ln[l].empty() ? nullptr : d_.mla_kva_ln[l].data(),
                             &d_.mla_kt[l], &d_.mla_v[l], &d_.mla_o[l], &d_.mla_g[l],
                             mla_cache[l].empty() ? nullptr : mla_cache[l].data(), pos, y.data(),
                             cfg_.rms_eps);
                }
                if (attnres) {
                    if (snap)
                        prefix = std::vector<float>(y.begin(), y.end());
                    else {
                        for (int i = 0; i < H; ++i)
                            prefix[i] += y[i];
                    }
                    attnres_mix(snaps, prefix.data(), sw_mlp(l), nullptr, h.data(), H, cfg_.rms_eps);
                } else {
                    for (int i = 0; i < H; ++i)
                        h[i] += y[i];
                }

                quant::rmsnorm(h.data(), d_.attn_out_n[l].data(), n.data(), H, cfg_.rms_eps);
                if (l < cfg_.first_dense) {
                    std::vector<float> g(cfg_.dense_intermediate), u(cfg_.dense_intermediate),
                        down(H);
                    d_.mlp_gate[l].gemm(g.data(), n.data(), 1);
                    d_.mlp_up[l].gemm(u.data(), n.data(), 1);
                    for (int i = 0; i < cfg_.dense_intermediate; ++i)
                        g[i] = quant::situ_glu(g[i], u[i], cfg_.moe.situ_b1, cfg_.moe.situ_b2);
                    d_.mlp_down[l].gemm(down.data(), g.data(), 1);
                    if (attnres) {
                        for (int i = 0; i < H; ++i)
                            prefix[i] += down[i];
                        h = prefix;
                    } else {
                        for (int i = 0; i < H; ++i)
                            h[i] += down[i];
                    }
                } else {
                    if (attnres) {
                        std::vector<float> dest(H, 0.f);
                        Status mst = moe_layer(l, n.data(), dest.data(), err);
                        if (mst != Status::Ok)
                            return mst;
                        for (int i = 0; i < H; ++i)
                            prefix[i] += dest[i];
                        h = prefix;
                    } else {
                        Status mst = moe_layer(l, n.data(), h.data(), err);
                        if (mst != Status::Ok)
                            return mst;
                    }
                }
            }
            if (cfg_.attn_res.block_size > 0)
                attnres_mix(snaps, h.data(), d_.out_sw.empty() ? nullptr : d_.out_sw.data(), nullptr,
                            h.data(), H, cfg_.rms_eps);
            ++pos;
            return Status::Ok;
        };

        const int chunk = rt_.prefill_chunk > 0 ? rt_.prefill_chunk : static_cast<int>(prompt.size());
        for (size_t i = static_cast<size_t>(reuse); i < prompt.size();) {
            const int C = static_cast<int>(
                std::min(static_cast<size_t>(chunk), prompt.size() - i));
            std::vector<float> act(static_cast<size_t>(C) * H);
            std::vector<std::vector<float>> prefixes(static_cast<size_t>(C), std::vector<float>(H, 0.f));
            std::vector<std::vector<std::vector<float>>> psnaps(static_cast<size_t>(C));
            for (int c = 0; c < C; ++c) {
                int tid = seq[i + static_cast<size_t>(c)];
                if (tid < 0 || tid >= cfg_.vocab)
                    tid = 0;
                std::memcpy(act.data() + static_cast<size_t>(c) * H,
                            d_.embed.data() + static_cast<size_t>(tid) * H, H * sizeof(float));
            }
            for (int l = 0; l < L; ++l) {
                const bool attnres = cfg_.attn_res.block_size > 0;
                const bool snap = attnres && (l % cfg_.attn_res.block_size) == 0;
                for (int c = 0; c < C; ++c) {
                    float *hh = act.data() + static_cast<size_t>(c) * H;
                    if (attnres) {
                        prefixes[static_cast<size_t>(c)].assign(hh, hh + H);
                        if (!psnaps[static_cast<size_t>(c)].empty())
                            attnres_mix(psnaps[static_cast<size_t>(c)],
                                        prefixes[static_cast<size_t>(c)].data(), sw_attn(l),
                                        nullptr, hh, H, cfg_.rms_eps);
                        if (snap)
                            psnaps[static_cast<size_t>(c)].push_back(
                                prefixes[static_cast<size_t>(c)]);
                    }
                    std::vector<float> n(H), y(H, 0.f);
                    quant::rmsnorm(hh, d_.attn_in_n[l].data(), n.data(), H, cfg_.rms_eps);
                    bool use_kda = (l < static_cast<int>(cfg_.is_kda.size())) ? cfg_.is_kda[l] : 1;
                    if (use_kda) {
                        kda_step(n.data(), H, cfg_.kda, &d_.wq[l], &d_.wk[l], &d_.wv[l], &d_.wb[l],
                                 &d_.wfa[l], &d_.wfb[l], d_.wdt[l].data(),
                                 static_cast<int>(d_.wdt[l].size()), d_.alog[l].data(), &d_.wg[l],
                                 nullptr, &d_.wo[l],
                                 d_.out_norm[l].empty() ? nullptr : d_.out_norm[l].data(),
                                 S[l].data(), y.data(),
                                 cfg_.rms_eps,
                                 l < static_cast<int>(d_.conv_q.size()) && !d_.conv_q[l].empty()
                                     ? d_.conv_q[l].data()
                                     : nullptr,
                                 l < static_cast<int>(d_.conv_k.size()) && !d_.conv_k[l].empty()
                                     ? d_.conv_k[l].data()
                                     : nullptr,
                                 l < static_cast<int>(d_.conv_v.size()) && !d_.conv_v[l].empty()
                                     ? d_.conv_v[l].data()
                                     : nullptr,
                                 winq[l].data(), wink[l].data(), winv[l].data());
                    } else {
                        mla_step(n.data(), H, cfg_.mla, &d_.mla_qa[l],
                                 d_.mla_qa_ln[l].empty() ? nullptr : d_.mla_qa_ln[l].data(),
                                 &d_.mla_qb[l], &d_.mla_kva[l],
                                 d_.mla_kva_ln[l].empty() ? nullptr : d_.mla_kva_ln[l].data(),
                                 &d_.mla_kt[l], &d_.mla_v[l], &d_.mla_o[l], &d_.mla_g[l],
                                 mla_cache[l].empty() ? nullptr : mla_cache[l].data(), pos + c,
                                 y.data(), cfg_.rms_eps);
                    }
                    if (attnres) {
                        if (snap)
                            prefixes[static_cast<size_t>(c)] = std::vector<float>(y.begin(), y.end());
                        else {
                            for (int j = 0; j < H; ++j)
                                prefixes[static_cast<size_t>(c)][j] += y[j];
                        }
                        attnres_mix(psnaps[static_cast<size_t>(c)],
                                    prefixes[static_cast<size_t>(c)].data(), sw_mlp(l), nullptr, hh,
                                    H, cfg_.rms_eps);
                    } else {
                        for (int j = 0; j < H; ++j)
                            hh[j] += y[j];
                    }
                }
                if (l < cfg_.first_dense) {
                    for (int c = 0; c < C; ++c) {
                        float *hh = act.data() + static_cast<size_t>(c) * H;
                        std::vector<float> n(H), g(cfg_.dense_intermediate),
                            u(cfg_.dense_intermediate), down(H);
                        quant::rmsnorm(hh, d_.attn_out_n[l].data(), n.data(), H, cfg_.rms_eps);
                        d_.mlp_gate[l].gemm(g.data(), n.data(), 1);
                        d_.mlp_up[l].gemm(u.data(), n.data(), 1);
                        for (int j = 0; j < cfg_.dense_intermediate; ++j)
                            g[j] = quant::situ_glu(g[j], u[j], cfg_.moe.situ_b1, cfg_.moe.situ_b2);
                        d_.mlp_down[l].gemm(down.data(), g.data(), 1);
                        if (attnres) {
                            for (int j = 0; j < H; ++j)
                                prefixes[static_cast<size_t>(c)][j] += down[j];
                            std::memcpy(hh, prefixes[static_cast<size_t>(c)].data(),
                                        static_cast<size_t>(H) * sizeof(float));
                        } else {
                            for (int j = 0; j < H; ++j)
                                hh[j] += down[j];
                        }
                    }
                } else {
                    std::vector<float> norms(static_cast<size_t>(C) * H);
                    std::vector<float> before;
                    if (attnres) {
                        before.assign(static_cast<size_t>(C) * H, 0.f);
                    }
                    for (int c = 0; c < C; ++c)
                        quant::rmsnorm(act.data() + static_cast<size_t>(c) * H,
                                       d_.attn_out_n[l].data(),
                                       norms.data() + static_cast<size_t>(c) * H, H, cfg_.rms_eps);
                    Status mst = moe_layer_n(l, norms.data(), attnres ? before.data() : act.data(),
                                             C, err);
                    if (mst != Status::Ok)
                        return mst;
                    if (attnres) {
                        for (int c = 0; c < C; ++c) {
                            for (int j = 0; j < H; ++j)
                                prefixes[static_cast<size_t>(c)][j] +=
                                    before[static_cast<size_t>(c) * H + j];
                            std::memcpy(act.data() + static_cast<size_t>(c) * H,
                                        prefixes[static_cast<size_t>(c)].data(),
                                        static_cast<size_t>(H) * sizeof(float));
                        }
                    }
                }
            }
            if (cfg_.attn_res.block_size > 0) {
                for (int c = 0; c < C; ++c)
                    attnres_mix(psnaps[static_cast<size_t>(c)],
                                act.data() + static_cast<size_t>(c) * H,
                                d_.out_sw.empty() ? nullptr : d_.out_sw.data(), nullptr,
                                act.data() + static_cast<size_t>(c) * H, H, cfg_.rms_eps);
            }
            std::memcpy(h.data(), act.data() + static_cast<size_t>(C - 1) * H, H * sizeof(float));
            pos += C;
            i += static_cast<size_t>(C);
        }

        out.tokens.clear();
        out.token_logprobs.clear();
        out.top_logprobs.clear();
        uint64_t rng = gp.seed ? gp.seed : 1ull;
        std::vector<uint8_t> allow;
        Gbnf g;
        if (!gp.grammar.empty()) {
            std::string e;
            if (g.compile(gp.grammar, e) == Status::Ok && g.ready() && gp.token_text) {
                allow.assign(cfg_.vocab, 0);
                g.allow_mask(gp.token_text, allow.data(), cfg_.vocab);
            }
        }
        std::string acc;
        for (int n = 0; n < gp.max_new_tokens; ++n) {
            std::vector<float> nrm(H), logits(cfg_.vocab);
            quant::rmsnorm(h.data(), d_.norm.data(), nrm.data(), H, cfg_.rms_eps);
            d_.lm_head.gemm(logits.data(), nrm.data(), 1);
            int next = sample_penalized(logits.data(), cfg_.vocab, gp, seq.data(),
                                       static_cast<int>(seq.size()), &rng,
                                       allow.empty() ? nullptr : allow.data(), nullptr,
                                       &out.token_logprobs, &out.top_logprobs);
            out.tokens.push_back(next);
            seq.push_back(next);
            if (g.ready() && gp.token_text)
                g.accept_bytes(gp.token_text(next));
            if (gp.on_token)
                gp.on_token(next);
            if (gen_stop_id(next, cfg_, gp))
                break;
            if (!gp.stop.empty() && gp.token_text) {
                acc += gp.token_text(next);
                if (stop_cut(acc, gp.stop) != std::string::npos) {
                    out.stopped_by_stop = true;
                    break;
                }
            }
            Status sst = step(next, true);
            if (sst != Status::Ok)
                return sst;
            if (g.ready() && gp.token_text) {
                allow.assign(cfg_.vocab, 0);
                g.allow_mask(gp.token_text, allow.data(), cfg_.vocab);
            }
        }
        out.completion_tokens = static_cast<int>(out.tokens.size());
        if (cslot >= 0 && cslot < kMaxKvSlots) {
            K3Slot &sl = slots_[cslot];
            sl.S = S;
            sl.winq = winq;
            sl.wink = wink;
            sl.winv = winv;
            sl.mla_cache = mla_cache;
            sl.h = h;
            sl.pos = pos;
            sl.history = prompt;
            sl.history.insert(sl.history.end(), out.tokens.begin(), out.tokens.end());
            sl.have = true;
            sl.live = false;
        }
        return Status::Ok;
    }

    std::string describe() const override {
        std::ostringstream os;
        ExpertStoreStats st{};
        store_.stats(st);
        int mla_n = 0;
        for (int l = 0; l < cfg_.n_layers; ++l) {
            const bool use_kda = (l < static_cast<int>(cfg_.is_kda.size())) ? cfg_.is_kda[l] : 1;
            if (!use_kda && l < static_cast<int>(d_.mla_kva.size()) && !d_.mla_kva[l].empty())
                ++mla_n;
        }
        os << "kimi_k3  hidden=" << cfg_.hidden << " layers=" << cfg_.n_layers
           << " experts=" << cfg_.moe.n_experts << " topk=" << cfg_.moe.topk
           << " latent=" << cfg_.moe.latent << " kda_heads=" << cfg_.kda.heads
           << " checkpoint=" << (from_checkpoint_ ? "yes" : "synthetic")
           << " bits=" << rt_.dense_bits << " head_bits=" << rt_.head_bits
           << " prefix=" << (prefix_.empty() ? "-" : prefix_) << " offload=disk"
           << " hits=" << st.hits << " misses=" << st.misses << " mla=" << mla_n
           << " prefill=layer";
        return os.str();
    }

    void expert_stats(ExpertStoreStats &out) const override { store_.stats(out); }

    Status begin_generate(int slot, const std::vector<int> &ids, const GenParams &gp, int &reuse,
                          std::string &err) override {
        if (!loaded_) {
            err = "kimi_k3 not loaded";
            return Status::InvalidArgument;
        }
        if (slot < 0 || slot >= kMaxKvSlots) {
            err = "bad cache slot";
            return Status::InvalidArgument;
        }
        if (ids.empty()) {
            err = "empty prompt";
            return Status::InvalidArgument;
        }
        K3Slot &st = slots_[slot];
        const int match = st.have ? kv_common_prefix(st.history, ids) : 0;
        const int want = std::min(std::max(gp.prefix_reuse, 0), match);
        int applied = 0;
        int Tmax = std::max(rt_.max_seq, static_cast<int>(ids.size()) + gp.max_new_tokens);
        if (Tmax <= 0)
            Tmax = 4096;
        if (st.have && st.pos > 0 && st.pos <= want && st.pos <= static_cast<int>(ids.size())) {
            applied = st.pos;
            slot_ensure_t(st, Tmax);
        } else {
            slot_alloc(st, Tmax);
        }
        Status pst = slot_prefill(st, ids, applied, err);
        if (pst != Status::Ok)
            return pst;
        st.history = ids;
        st.live = true;
        st.have = true;
        st.gp = gp;
        st.rng = gp.seed ? gp.seed : 1ull;
        st.emitted = 0;
        st.allow.clear();
        st.acc.clear();
        st.token_lps.clear();
        st.top_lps.clear();
        st.g = Gbnf{};
        if (!gp.grammar.empty()) {
            std::string e;
            if (st.g.compile(gp.grammar, e) == Status::Ok && st.g.ready() && gp.token_text) {
                st.allow.assign(cfg_.vocab, 0);
                st.g.allow_mask(gp.token_text, st.allow.data(), cfg_.vocab);
            }
        }
        reuse = applied;
        return Status::Ok;
    }

    Status next_token(int slot, int &token, bool &done, std::string &err) override {
        if (slot < 0 || slot >= kMaxKvSlots) {
            err = "bad cache slot";
            return Status::InvalidArgument;
        }
        K3Slot &st = slots_[slot];
        if (!st.live) {
            err = "no live generate";
            return Status::InvalidArgument;
        }
        done = false;
        if (st.gp.max_new_tokens <= 0 || st.emitted >= st.gp.max_new_tokens) {
            done = true;
            token = -1;
            return Status::Ok;
        }
        const uint8_t *allow = st.allow.empty() ? nullptr : st.allow.data();
        token = slot_sample(st, &st.rng, allow);
        ++st.emitted;
        st.history.push_back(token);
        if (st.g.ready() && st.gp.token_text)
            st.g.accept_bytes(st.gp.token_text(token));
        bool hit_stop = false;
        if (!st.gp.stop.empty() && st.gp.token_text) {
            st.acc += st.gp.token_text(token);
            if (stop_cut(st.acc, st.gp.stop) != std::string::npos)
                hit_stop = true;
        }
        if (gen_stop_id(token, cfg_, st.gp) || st.emitted >= st.gp.max_new_tokens ||
            hit_stop) {
            done = true;
        } else {
            Status sst = slot_step(st, token, err);
            if (sst != Status::Ok)
                return sst;
            done = false;
        }
        if (st.g.ready() && st.gp.token_text) {
            st.allow.assign(cfg_.vocab, 0);
            st.g.allow_mask(st.gp.token_text, st.allow.data(), cfg_.vocab);
        }
        return Status::Ok;
    }

    Status next_tokens(const int *slots, int n, int *tokens, uint8_t *done,
                       std::string &err) override {
        if (n < 1)
            return Status::Ok;
        if (!slots || !tokens || !done) {
            err = "next_tokens bad args";
            return Status::InvalidArgument;
        }
        if (n == 1) {
            bool d = false;
            Status st = next_token(slots[0], tokens[0], d, err);
            done[0] = d ? 1 : 0;
            return st;
        }
        for (int i = 0; i < n; ++i) {
            if (slots[i] < 0 || slots[i] >= kMaxKvSlots) {
                err = "bad cache slot";
                return Status::InvalidArgument;
            }
            if (!slots_[slots[i]].live) {
                err = "no live generate";
                return Status::InvalidArgument;
            }
        }
        std::vector<K3Slot *> step_s;
        std::vector<int> step_tok;
        step_s.reserve(static_cast<size_t>(n));
        step_tok.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            K3Slot &st = slots_[slots[i]];
            done[i] = 0;
            if (st.gp.max_new_tokens <= 0 || st.emitted >= st.gp.max_new_tokens) {
                done[i] = 1;
                tokens[i] = -1;
                continue;
            }
            const uint8_t *allow = st.allow.empty() ? nullptr : st.allow.data();
            tokens[i] = slot_sample(st, &st.rng, allow);
            ++st.emitted;
            st.history.push_back(tokens[i]);
            if (st.g.ready() && st.gp.token_text)
                st.g.accept_bytes(st.gp.token_text(tokens[i]));
            bool hit_stop = false;
            if (!st.gp.stop.empty() && st.gp.token_text) {
                st.acc += st.gp.token_text(tokens[i]);
                if (stop_cut(st.acc, st.gp.stop) != std::string::npos)
                    hit_stop = true;
            }
            if (gen_stop_id(tokens[i], cfg_, st.gp) || st.emitted >= st.gp.max_new_tokens ||
                hit_stop)
                done[i] = 1;
            else {
                step_s.push_back(&st);
                step_tok.push_back(tokens[i]);
            }
            if (st.g.ready() && st.gp.token_text) {
                st.allow.assign(cfg_.vocab, 0);
                st.g.allow_mask(st.gp.token_text, st.allow.data(), cfg_.vocab);
            }
        }
        if (!step_s.empty()) {
            Status sst = slot_step_n(step_s.data(), step_tok.data(),
                                     static_cast<int>(step_s.size()), err);
            if (sst != Status::Ok)
                return sst;
        }
        return Status::Ok;
    }

    void end_generate(int slot) override {
        if (slot < 0 || slot >= kMaxKvSlots)
            return;
        slots_[slot].live = false;
    }

private:
    const float *sw_attn(int l) const {
        return (l < static_cast<int>(d_.attn_sw.size()) && !d_.attn_sw[l].empty())
                   ? d_.attn_sw[l].data()
                   : nullptr;
    }
    const float *sw_mlp(int l) const {
        return (l < static_cast<int>(d_.mlp_sw.size()) && !d_.mlp_sw[l].empty())
                   ? d_.mlp_sw[l].data()
                   : nullptr;
    }

    int kv_stride() const {
        return std::max(cfg_.mla.kv_lora, 0) + std::max(cfg_.mla.qk_rope, 0);
    }

    void slot_alloc(K3Slot &s, int t_max) {
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int kd = cfg_.kda.head_dim;
        const int kh = cfg_.kda.heads;
        const int P = kh * kd;
        const int Kc = cfg_.kda.conv_k > 0 ? cfg_.kda.conv_k : 4;
        const int kvStride = kv_stride();
        s.h.assign(static_cast<size_t>(H), 0.f);
        s.S.assign(static_cast<size_t>(L), {});
        s.winq.assign(static_cast<size_t>(L), {});
        s.wink.assign(static_cast<size_t>(L), {});
        s.winv.assign(static_cast<size_t>(L), {});
        s.mla_cache.assign(static_cast<size_t>(L), {});
        for (int l = 0; l < L; ++l) {
            s.S[l].assign(static_cast<size_t>(kh) * kd * kd, 0.f);
            s.winq[l].assign(static_cast<size_t>(P) * Kc, 0.f);
            s.wink[l].assign(static_cast<size_t>(P) * Kc, 0.f);
            s.winv[l].assign(static_cast<size_t>(P) * Kc, 0.f);
            if (kvStride > 0)
                s.mla_cache[l].assign(static_cast<size_t>(std::max(t_max, 1)) * kvStride, 0.f);
        }
        s.pos = 0;
        s.have = false;
        s.live = false;
        s.history.clear();
        s.emitted = 0;
        s.allow.clear();
        s.acc.clear();
        s.token_lps.clear();
        s.top_lps.clear();
        s.g = Gbnf{};
    }

    void slot_ensure_t(K3Slot &s, int t_max) {
        const int L = cfg_.n_layers;
        const int kvStride = kv_stride();
        if (kvStride <= 0)
            return;
        const size_t need = static_cast<size_t>(std::max(t_max, 1)) * kvStride;
        if (static_cast<int>(s.mla_cache.size()) < L)
            s.mla_cache.resize(static_cast<size_t>(L));
        for (int l = 0; l < L; ++l) {
            if (s.mla_cache[l].size() < need)
                s.mla_cache[l].resize(need, 0.f);
        }
    }

    Status slot_step(K3Slot &s, int token, std::string &err) {
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        int tid = token;
        if (tid < 0 || tid >= cfg_.vocab)
            tid = 0;
        if (static_cast<int>(s.h.size()) != H)
            s.h.assign(static_cast<size_t>(H), 0.f);
        std::memcpy(s.h.data(), d_.embed.data() + static_cast<size_t>(tid) * H,
                    static_cast<size_t>(H) * sizeof(float));
        std::vector<float> prefix(static_cast<size_t>(H), 0.f);
        std::vector<std::vector<float>> snaps;
        for (int l = 0; l < L; ++l) {
            const bool attnres = cfg_.attn_res.block_size > 0;
            const bool snap = attnres && (l % cfg_.attn_res.block_size) == 0;
            if (attnres) {
                prefix = s.h;
                if (!snaps.empty())
                    attnres_mix(snaps, prefix.data(), sw_attn(l), nullptr, s.h.data(), H,
                                cfg_.rms_eps);
                if (snap)
                    snaps.push_back(prefix);
            }

            std::vector<float> n(static_cast<size_t>(H)), y(static_cast<size_t>(H), 0.f);
            quant::rmsnorm(s.h.data(), d_.attn_in_n[l].data(), n.data(), H, cfg_.rms_eps);

            bool use_kda = (l < static_cast<int>(cfg_.is_kda.size())) ? cfg_.is_kda[l] : 1;
            if (use_kda) {
                kda_step(n.data(), H, cfg_.kda, &d_.wq[l], &d_.wk[l], &d_.wv[l], &d_.wb[l],
                         &d_.wfa[l], &d_.wfb[l], d_.wdt[l].data(),
                         static_cast<int>(d_.wdt[l].size()), d_.alog[l].data(), &d_.wg[l],
                         nullptr, &d_.wo[l],
                         d_.out_norm[l].empty() ? nullptr : d_.out_norm[l].data(),
                         s.S[l].data(), y.data(), cfg_.rms_eps,
                         l < static_cast<int>(d_.conv_q.size()) && !d_.conv_q[l].empty()
                             ? d_.conv_q[l].data()
                             : nullptr,
                         l < static_cast<int>(d_.conv_k.size()) && !d_.conv_k[l].empty()
                             ? d_.conv_k[l].data()
                             : nullptr,
                         l < static_cast<int>(d_.conv_v.size()) && !d_.conv_v[l].empty()
                             ? d_.conv_v[l].data()
                             : nullptr,
                         s.winq[l].data(), s.wink[l].data(), s.winv[l].data());
            } else {
                mla_step(n.data(), H, cfg_.mla, &d_.mla_qa[l],
                         d_.mla_qa_ln[l].empty() ? nullptr : d_.mla_qa_ln[l].data(),
                         &d_.mla_qb[l], &d_.mla_kva[l],
                         d_.mla_kva_ln[l].empty() ? nullptr : d_.mla_kva_ln[l].data(),
                         &d_.mla_kt[l], &d_.mla_v[l], &d_.mla_o[l], &d_.mla_g[l],
                         s.mla_cache[l].empty() ? nullptr : s.mla_cache[l].data(), s.pos,
                         y.data(), cfg_.rms_eps);
            }
            if (attnres) {
                if (snap)
                    prefix = std::vector<float>(y.begin(), y.end());
                else {
                    for (int i = 0; i < H; ++i)
                        prefix[i] += y[i];
                }
                attnres_mix(snaps, prefix.data(), sw_mlp(l), nullptr, s.h.data(), H, cfg_.rms_eps);
            } else {
                for (int i = 0; i < H; ++i)
                    s.h[i] += y[i];
            }

            quant::rmsnorm(s.h.data(), d_.attn_out_n[l].data(), n.data(), H, cfg_.rms_eps);
            if (l < cfg_.first_dense) {
                std::vector<float> g(cfg_.dense_intermediate), u(cfg_.dense_intermediate),
                    down(static_cast<size_t>(H));
                d_.mlp_gate[l].gemm(g.data(), n.data(), 1);
                d_.mlp_up[l].gemm(u.data(), n.data(), 1);
                for (int i = 0; i < cfg_.dense_intermediate; ++i)
                    g[i] = quant::situ_glu(g[i], u[i], cfg_.moe.situ_b1, cfg_.moe.situ_b2);
                d_.mlp_down[l].gemm(down.data(), g.data(), 1);
                if (attnres) {
                    for (int i = 0; i < H; ++i)
                        prefix[i] += down[i];
                    s.h = prefix;
                } else {
                    for (int i = 0; i < H; ++i)
                        s.h[i] += down[i];
                }
            } else {
                if (attnres) {
                    std::vector<float> dest(static_cast<size_t>(H), 0.f);
                    Status mst = moe_layer(l, n.data(), dest.data(), err);
                    if (mst != Status::Ok)
                        return mst;
                    for (int i = 0; i < H; ++i)
                        prefix[i] += dest[i];
                    s.h = prefix;
                } else {
                    Status mst = moe_layer(l, n.data(), s.h.data(), err);
                    if (mst != Status::Ok)
                        return mst;
                }
            }
        }
        if (cfg_.attn_res.block_size > 0)
            attnres_mix(snaps, s.h.data(), d_.out_sw.empty() ? nullptr : d_.out_sw.data(), nullptr,
                        s.h.data(), H, cfg_.rms_eps);
        ++s.pos;
        return Status::Ok;
    }

    // One decode row per continuing slot. KDA/MLA stay per-slot; MoE is unioned.
    Status slot_step_n(K3Slot **ss, const int *tokens, int S, std::string &err) {
        if (S <= 0)
            return Status::Ok;
        if (S == 1)
            return slot_step(*ss[0], tokens[0], err);
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        std::vector<float> act(static_cast<size_t>(S) * H);
        for (int c = 0; c < S; ++c) {
            int tid = tokens[c];
            if (tid < 0 || tid >= cfg_.vocab)
                tid = 0;
            std::memcpy(act.data() + static_cast<size_t>(c) * H,
                        d_.embed.data() + static_cast<size_t>(tid) * H,
                        static_cast<size_t>(H) * sizeof(float));
        }
        std::vector<std::vector<float>> prefixes(static_cast<size_t>(S),
                                                 std::vector<float>(static_cast<size_t>(H), 0.f));
        std::vector<std::vector<std::vector<float>>> psnaps(static_cast<size_t>(S));
        for (int l = 0; l < L; ++l) {
            const bool attnres = cfg_.attn_res.block_size > 0;
            const bool snap = attnres && (l % cfg_.attn_res.block_size) == 0;
            for (int c = 0; c < S; ++c) {
                K3Slot &s = *ss[c];
                float *hh = act.data() + static_cast<size_t>(c) * H;
                if (attnres) {
                    prefixes[static_cast<size_t>(c)].assign(hh, hh + H);
                    if (!psnaps[static_cast<size_t>(c)].empty())
                        attnres_mix(psnaps[static_cast<size_t>(c)],
                                    prefixes[static_cast<size_t>(c)].data(), sw_attn(l), nullptr,
                                    hh, H, cfg_.rms_eps);
                    if (snap)
                        psnaps[static_cast<size_t>(c)].push_back(prefixes[static_cast<size_t>(c)]);
                }
                std::vector<float> n(static_cast<size_t>(H)), y(static_cast<size_t>(H), 0.f);
                quant::rmsnorm(hh, d_.attn_in_n[l].data(), n.data(), H, cfg_.rms_eps);
                bool use_kda = (l < static_cast<int>(cfg_.is_kda.size())) ? cfg_.is_kda[l] : 1;
                if (use_kda) {
                    kda_step(n.data(), H, cfg_.kda, &d_.wq[l], &d_.wk[l], &d_.wv[l], &d_.wb[l],
                             &d_.wfa[l], &d_.wfb[l], d_.wdt[l].data(),
                             static_cast<int>(d_.wdt[l].size()), d_.alog[l].data(), &d_.wg[l],
                             nullptr, &d_.wo[l],
                             d_.out_norm[l].empty() ? nullptr : d_.out_norm[l].data(),
                             s.S[l].data(), y.data(), cfg_.rms_eps,
                             l < static_cast<int>(d_.conv_q.size()) && !d_.conv_q[l].empty()
                                 ? d_.conv_q[l].data()
                                 : nullptr,
                             l < static_cast<int>(d_.conv_k.size()) && !d_.conv_k[l].empty()
                                 ? d_.conv_k[l].data()
                                 : nullptr,
                             l < static_cast<int>(d_.conv_v.size()) && !d_.conv_v[l].empty()
                                 ? d_.conv_v[l].data()
                                 : nullptr,
                             s.winq[l].data(), s.wink[l].data(), s.winv[l].data());
                } else {
                    mla_step(n.data(), H, cfg_.mla, &d_.mla_qa[l],
                             d_.mla_qa_ln[l].empty() ? nullptr : d_.mla_qa_ln[l].data(),
                             &d_.mla_qb[l], &d_.mla_kva[l],
                             d_.mla_kva_ln[l].empty() ? nullptr : d_.mla_kva_ln[l].data(),
                             &d_.mla_kt[l], &d_.mla_v[l], &d_.mla_o[l], &d_.mla_g[l],
                             s.mla_cache[l].empty() ? nullptr : s.mla_cache[l].data(), s.pos,
                             y.data(), cfg_.rms_eps);
                }
                if (attnres) {
                    if (snap)
                        prefixes[static_cast<size_t>(c)] = std::vector<float>(y.begin(), y.end());
                    else {
                        for (int j = 0; j < H; ++j)
                            prefixes[static_cast<size_t>(c)][j] += y[j];
                    }
                    attnres_mix(psnaps[static_cast<size_t>(c)],
                                prefixes[static_cast<size_t>(c)].data(), sw_mlp(l), nullptr, hh, H,
                                cfg_.rms_eps);
                } else {
                    for (int j = 0; j < H; ++j)
                        hh[j] += y[j];
                }
            }
            if (l < cfg_.first_dense) {
                for (int c = 0; c < S; ++c) {
                    float *hh = act.data() + static_cast<size_t>(c) * H;
                    std::vector<float> n(static_cast<size_t>(H)), g(cfg_.dense_intermediate),
                        u(cfg_.dense_intermediate), down(static_cast<size_t>(H));
                    quant::rmsnorm(hh, d_.attn_out_n[l].data(), n.data(), H, cfg_.rms_eps);
                    d_.mlp_gate[l].gemm(g.data(), n.data(), 1);
                    d_.mlp_up[l].gemm(u.data(), n.data(), 1);
                    for (int j = 0; j < cfg_.dense_intermediate; ++j)
                        g[j] = quant::situ_glu(g[j], u[j], cfg_.moe.situ_b1, cfg_.moe.situ_b2);
                    d_.mlp_down[l].gemm(down.data(), g.data(), 1);
                    if (attnres) {
                        for (int j = 0; j < H; ++j)
                            prefixes[static_cast<size_t>(c)][j] += down[j];
                        std::memcpy(hh, prefixes[static_cast<size_t>(c)].data(),
                                    static_cast<size_t>(H) * sizeof(float));
                    } else {
                        for (int j = 0; j < H; ++j)
                            hh[j] += down[j];
                    }
                }
            } else {
                std::vector<float> norms(static_cast<size_t>(S) * H);
                std::vector<float> before;
                if (attnres)
                    before.assign(static_cast<size_t>(S) * H, 0.f);
                for (int c = 0; c < S; ++c)
                    quant::rmsnorm(act.data() + static_cast<size_t>(c) * H, d_.attn_out_n[l].data(),
                                   norms.data() + static_cast<size_t>(c) * H, H, cfg_.rms_eps);
                Status mst = moe_layer_n(l, norms.data(), attnres ? before.data() : act.data(), S,
                                         err);
                if (mst != Status::Ok)
                    return mst;
                if (attnres) {
                    for (int c = 0; c < S; ++c) {
                        for (int j = 0; j < H; ++j)
                            prefixes[static_cast<size_t>(c)][j] +=
                                before[static_cast<size_t>(c) * H + j];
                        std::memcpy(act.data() + static_cast<size_t>(c) * H,
                                    prefixes[static_cast<size_t>(c)].data(),
                                    static_cast<size_t>(H) * sizeof(float));
                    }
                }
            }
        }
        for (int c = 0; c < S; ++c) {
            float *hh = act.data() + static_cast<size_t>(c) * H;
            if (cfg_.attn_res.block_size > 0)
                attnres_mix(psnaps[static_cast<size_t>(c)], hh,
                            d_.out_sw.empty() ? nullptr : d_.out_sw.data(), nullptr, hh, H,
                            cfg_.rms_eps);
            K3Slot &s = *ss[c];
            if (static_cast<int>(s.h.size()) != H)
                s.h.assign(static_cast<size_t>(H), 0.f);
            std::memcpy(s.h.data(), hh, static_cast<size_t>(H) * sizeof(float));
            ++s.pos;
        }
        return Status::Ok;
    }

    Status slot_prefill(K3Slot &s, const std::vector<int> &ids, int start, std::string &err) {
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        if (start < 0)
            start = 0;
        if (start >= static_cast<int>(ids.size()))
            return Status::Ok;
        const int chunk = rt_.prefill_chunk > 0 ? rt_.prefill_chunk
                                                : static_cast<int>(ids.size()) - start;
        for (size_t i = static_cast<size_t>(start); i < ids.size();) {
            const int C = static_cast<int>(
                std::min(static_cast<size_t>(chunk), ids.size() - i));
            std::vector<float> act(static_cast<size_t>(C) * H);
            std::vector<std::vector<float>> prefixes(static_cast<size_t>(C),
                                                     std::vector<float>(static_cast<size_t>(H), 0.f));
            std::vector<std::vector<std::vector<float>>> psnaps(static_cast<size_t>(C));
            for (int c = 0; c < C; ++c) {
                int tid = ids[i + static_cast<size_t>(c)];
                if (tid < 0 || tid >= cfg_.vocab)
                    tid = 0;
                std::memcpy(act.data() + static_cast<size_t>(c) * H,
                            d_.embed.data() + static_cast<size_t>(tid) * H,
                            static_cast<size_t>(H) * sizeof(float));
            }
            for (int l = 0; l < L; ++l) {
                const bool attnres = cfg_.attn_res.block_size > 0;
                const bool snap = attnres && (l % cfg_.attn_res.block_size) == 0;
                for (int c = 0; c < C; ++c) {
                    float *hh = act.data() + static_cast<size_t>(c) * H;
                    if (attnres) {
                        prefixes[static_cast<size_t>(c)].assign(hh, hh + H);
                        if (!psnaps[static_cast<size_t>(c)].empty())
                            attnres_mix(psnaps[static_cast<size_t>(c)],
                                        prefixes[static_cast<size_t>(c)].data(), sw_attn(l),
                                        nullptr, hh, H, cfg_.rms_eps);
                        if (snap)
                            psnaps[static_cast<size_t>(c)].push_back(
                                prefixes[static_cast<size_t>(c)]);
                    }
                    std::vector<float> n(static_cast<size_t>(H)), y(static_cast<size_t>(H), 0.f);
                    quant::rmsnorm(hh, d_.attn_in_n[l].data(), n.data(), H, cfg_.rms_eps);
                    bool use_kda = (l < static_cast<int>(cfg_.is_kda.size())) ? cfg_.is_kda[l] : 1;
                    if (use_kda) {
                        kda_step(n.data(), H, cfg_.kda, &d_.wq[l], &d_.wk[l], &d_.wv[l], &d_.wb[l],
                                 &d_.wfa[l], &d_.wfb[l], d_.wdt[l].data(),
                                 static_cast<int>(d_.wdt[l].size()), d_.alog[l].data(), &d_.wg[l],
                                 nullptr, &d_.wo[l],
                                 d_.out_norm[l].empty() ? nullptr : d_.out_norm[l].data(),
                                 s.S[l].data(), y.data(), cfg_.rms_eps,
                                 l < static_cast<int>(d_.conv_q.size()) && !d_.conv_q[l].empty()
                                     ? d_.conv_q[l].data()
                                     : nullptr,
                                 l < static_cast<int>(d_.conv_k.size()) && !d_.conv_k[l].empty()
                                     ? d_.conv_k[l].data()
                                     : nullptr,
                                 l < static_cast<int>(d_.conv_v.size()) && !d_.conv_v[l].empty()
                                     ? d_.conv_v[l].data()
                                     : nullptr,
                                 s.winq[l].data(), s.wink[l].data(), s.winv[l].data());
                    } else {
                        mla_step(n.data(), H, cfg_.mla, &d_.mla_qa[l],
                                 d_.mla_qa_ln[l].empty() ? nullptr : d_.mla_qa_ln[l].data(),
                                 &d_.mla_qb[l], &d_.mla_kva[l],
                                 d_.mla_kva_ln[l].empty() ? nullptr : d_.mla_kva_ln[l].data(),
                                 &d_.mla_kt[l], &d_.mla_v[l], &d_.mla_o[l], &d_.mla_g[l],
                                 s.mla_cache[l].empty() ? nullptr : s.mla_cache[l].data(),
                                 s.pos + c, y.data(), cfg_.rms_eps);
                    }
                    if (attnres) {
                        if (snap)
                            prefixes[static_cast<size_t>(c)] =
                                std::vector<float>(y.begin(), y.end());
                        else {
                            for (int j = 0; j < H; ++j)
                                prefixes[static_cast<size_t>(c)][j] += y[j];
                        }
                        attnres_mix(psnaps[static_cast<size_t>(c)],
                                    prefixes[static_cast<size_t>(c)].data(), sw_mlp(l), nullptr,
                                    hh, H, cfg_.rms_eps);
                    } else {
                        for (int j = 0; j < H; ++j)
                            hh[j] += y[j];
                    }
                }
                if (l < cfg_.first_dense) {
                    for (int c = 0; c < C; ++c) {
                        float *hh = act.data() + static_cast<size_t>(c) * H;
                        std::vector<float> n(static_cast<size_t>(H)), g(cfg_.dense_intermediate),
                            u(cfg_.dense_intermediate), down(static_cast<size_t>(H));
                        quant::rmsnorm(hh, d_.attn_out_n[l].data(), n.data(), H, cfg_.rms_eps);
                        d_.mlp_gate[l].gemm(g.data(), n.data(), 1);
                        d_.mlp_up[l].gemm(u.data(), n.data(), 1);
                        for (int j = 0; j < cfg_.dense_intermediate; ++j)
                            g[j] = quant::situ_glu(g[j], u[j], cfg_.moe.situ_b1, cfg_.moe.situ_b2);
                        d_.mlp_down[l].gemm(down.data(), g.data(), 1);
                        if (attnres) {
                            for (int j = 0; j < H; ++j)
                                prefixes[static_cast<size_t>(c)][j] += down[j];
                            std::memcpy(hh, prefixes[static_cast<size_t>(c)].data(),
                                        static_cast<size_t>(H) * sizeof(float));
                        } else {
                            for (int j = 0; j < H; ++j)
                                hh[j] += down[j];
                        }
                    }
                } else {
                    std::vector<float> norms(static_cast<size_t>(C) * H);
                    std::vector<float> before;
                    if (attnres)
                        before.assign(static_cast<size_t>(C) * H, 0.f);
                    for (int c = 0; c < C; ++c)
                        quant::rmsnorm(act.data() + static_cast<size_t>(c) * H,
                                       d_.attn_out_n[l].data(),
                                       norms.data() + static_cast<size_t>(c) * H, H, cfg_.rms_eps);
                    Status mst = moe_layer_n(l, norms.data(), attnres ? before.data() : act.data(),
                                             C, err);
                    if (mst != Status::Ok)
                        return mst;
                    if (attnres) {
                        for (int c = 0; c < C; ++c) {
                            for (int j = 0; j < H; ++j)
                                prefixes[static_cast<size_t>(c)][j] +=
                                    before[static_cast<size_t>(c) * H + j];
                            std::memcpy(act.data() + static_cast<size_t>(c) * H,
                                        prefixes[static_cast<size_t>(c)].data(),
                                        static_cast<size_t>(H) * sizeof(float));
                        }
                    }
                }
            }
            if (cfg_.attn_res.block_size > 0) {
                for (int c = 0; c < C; ++c)
                    attnres_mix(psnaps[static_cast<size_t>(c)],
                                act.data() + static_cast<size_t>(c) * H,
                                d_.out_sw.empty() ? nullptr : d_.out_sw.data(), nullptr,
                                act.data() + static_cast<size_t>(c) * H, H, cfg_.rms_eps);
            }
            std::memcpy(s.h.data(), act.data() + static_cast<size_t>(C - 1) * H,
                        static_cast<size_t>(H) * sizeof(float));
            s.pos += C;
            i += static_cast<size_t>(C);
        }
        return Status::Ok;
    }

    int slot_sample(K3Slot &s, uint64_t *rng, const uint8_t *allow) {
        const int H = cfg_.hidden;
        std::vector<float> nrm(static_cast<size_t>(H)), logits(static_cast<size_t>(cfg_.vocab));
        quant::rmsnorm(s.h.data(), d_.norm.data(), nrm.data(), H, cfg_.rms_eps);
        d_.lm_head.gemm(logits.data(), nrm.data(), 1);
        return sample_penalized(logits.data(), cfg_.vocab, s.gp, s.history.data(),
                               static_cast<int>(s.history.size()), rng, allow, nullptr,
                               &s.token_lps, &s.top_lps);
    }
    int64_t expert_bytes() const {
        const int I = cfg_.moe.latent > 0 ? cfg_.moe.latent : cfg_.hidden;
        const int O = cfg_.moe.intermediate > 0 ? cfg_.moe.intermediate : 32;
        return make_k3_geom(I, O).slot;
    }

    void fill_dummy_expert(uint8_t *p, int layer, int eid) {
        const int I = cfg_.moe.latent > 0 ? cfg_.moe.latent : cfg_.hidden;
        const int O = cfg_.moe.intermediate;
        auto g = make_k3_geom(I, O);
        std::vector<float> w1(static_cast<size_t>(O) * I, 0.02f);
        std::vector<float> w2(static_cast<size_t>(I) * O, 0.02f);
        std::vector<float> w3(static_cast<size_t>(O) * I, 0.02f);
        for (size_t i = 0; i < w1.size(); ++i)
            w1[i] = 0.02f * (((layer + eid + static_cast<int>(i)) & 7) - 3);
        quant::pack_mxfp4(w1.data(), O, I, p, p + g.w1p);
        quant::pack_mxfp4(w2.data(), I, O, p + g.w1p + g.w1s, p + g.w1p + g.w1s + g.w2p);
        quant::pack_mxfp4(w3.data(), O, I, p + g.w1p + g.w1s + g.w2p + g.w2s,
                          p + g.w1p + g.w1s + g.w2p + g.w2s + g.w1p);
        (void)eid;
    }

    Status write_synthetic_experts(const std::string &model_dir, std::string &err) {
        const int64_t ebytes = expert_bytes();
        const std::string epath = model_dir + "/.mvllm_k3_experts.bin";
        std::ofstream out(epath, std::ios::binary | std::ios::trunc);
        if (!out) {
            err = "failed to open expert pack for write";
            return Status::IoError;
        }
        std::vector<uint8_t> blob(static_cast<size_t>(ebytes), 0);
        for (int l = 0; l < cfg_.n_layers; ++l) {
            if (l < cfg_.first_dense)
                continue;
            for (int e = 0; e < cfg_.moe.n_experts; ++e) {
                std::fill(blob.begin(), blob.end(), 0);
                fill_dummy_expert(blob.data(), l, e);
                out.write(reinterpret_cast<const char *>(blob.data()),
                          static_cast<std::streamsize>(ebytes));
            }
        }
        if (!out) {
            err = "failed to write expert pack";
            return Status::IoError;
        }
        out.close();
        int64_t running = 0;
        for (int l = 0; l < cfg_.n_layers; ++l) {
            if (l < cfg_.first_dense)
                continue;
            for (int e = 0; e < cfg_.moe.n_experts; ++e) {
                ExpertLoc loc;
                loc.key = {l, e};
                loc.path = epath;
                loc.offset = running;
                loc.bytes = ebytes;
                Status st = store_.register_expert(loc, err);
                if (st != Status::Ok)
                    return st;
                running += ebytes;
            }
        }
        from_checkpoint_ = false;
        return Status::Ok;
    }

    Status overlay_f32(const std::vector<io::StFile> &files, const std::string &name,
                       std::vector<float> &dst, int expect, std::string &err) {
        io::StHit hit = io::st_find_dir(files, name);
        if (!hit.tensor)
            return Status::NotFound;
        if (expect > 0)
            dst.assign(static_cast<size_t>(expect), 0.f);
        else {
            int64_t n = 1;
            for (int64_t d : hit.tensor->shape)
                n *= d;
            dst.assign(static_cast<size_t>(n), 0.f);
        }
        return io::st_read_f32(*hit.file, *hit.tensor, dst.data(), static_cast<int64_t>(dst.size()),
                               err);
    }

    Status overlay_mat(const std::vector<io::StFile> &files, const std::string &name,
                       quant::QuantMat &dst, int bits, std::string &err) {
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
        dst.from_f32(tmp.data(), O, I, bits);
        return Status::Ok;
    }

    Status overlay_kvb(const std::vector<io::StFile> &files, const std::string &name,
                       quant::QuantMat &w_kt, quant::QuantMat &w_v, std::string &err) {
        io::StHit hit = io::st_find_dir(files, name);
        if (!hit.tensor)
            return Status::NotFound;
        int64_t n = 1;
        for (int64_t d : hit.tensor->shape)
            n *= d;
        std::vector<float> tmp(static_cast<size_t>(n), 0.f);
        Status st = io::st_read_f32(*hit.file, *hit.tensor, tmp.data(), n, err);
        if (st != Status::Ok)
            return st;
        const int nh = std::max(cfg_.mla.n_heads, 1);
        const int qk = cfg_.mla.qk_nope;
        const int vh = cfg_.mla.v_head > 0 ? cfg_.mla.v_head : qk;
        mla_absorb_kvb(tmp.data(), nh, qk, vh, cfg_.mla.kv_lora, w_kt, w_v, rt_.mla_bits);
        return Status::Ok;
    }

    Status load_checkpoint(const std::string &model_dir, std::string &err) {
        std::vector<io::StFile> files;
        Status st = io::st_open_dir(model_dir, files, err);
        if (st != Status::Ok)
            return st;
        if (files.empty())
            return Status::NotFound;

        auto has = [&](const std::string &n) { return io::st_find_dir(files, n).tensor != nullptr; };

        prefix_.clear();
        if (!has("model.layers.0.input_layernorm.weight") &&
            has("language_model.model.layers.0.input_layernorm.weight"))
            prefix_ = "language_model.";
        if (!has(prefix_ + "model.layers.0.input_layernorm.weight") &&
            !has(prefix_ + "model.embed_tokens.weight") &&
            !has(prefix_ + "embed_tokens.weight")) {
            io::st_close_dir(files);
            return Status::NotFound;
        }

        const std::string P = prefix_;
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int V = cfg_.vocab;
        overlay_f32(files, P + "model.embed_tokens.weight", d_.embed, V * H, err);
        if (d_.embed.empty())
            overlay_f32(files, P + "embed_tokens.weight", d_.embed, V * H, err);
        overlay_f32(files, P + "model.norm.weight", d_.norm, H, err);
        const int bits = rt_.dense_bits;
        const int hbits = rt_.head_bits;
        const int mbits = rt_.mla_bits;
        if (has("lm_head.weight"))
            overlay_mat(files, "lm_head.weight", d_.lm_head, hbits, err);
        else if (has(P + "lm_head.weight"))
            overlay_mat(files, P + "lm_head.weight", d_.lm_head, hbits, err);
        else if (!d_.embed.empty())
            d_.lm_head.from_f32(d_.embed.data(), V, H, hbits);

        for (int i = 0; i < L; ++i) {
            const std::string ly = P + "model.layers." + std::to_string(i);
            overlay_f32(files, ly + ".input_layernorm.weight", d_.attn_in_n[i], H, err);
            overlay_f32(files, ly + ".post_attention_layernorm.weight", d_.attn_out_n[i], H, err);
            overlay_mat(files, ly + ".self_attn.q_proj.weight", d_.wq[i], bits, err);
            overlay_mat(files, ly + ".self_attn.k_proj.weight", d_.wk[i], bits, err);
            overlay_mat(files, ly + ".self_attn.v_proj.weight", d_.wv[i], bits, err);
            overlay_mat(files, ly + ".self_attn.o_proj.weight", d_.wo[i], bits, err);
            overlay_mat(files, ly + ".self_attn.g_proj.weight", d_.wg[i], bits, err);
            overlay_mat(files, ly + ".self_attn.q_a_proj.weight", d_.mla_qa[i], mbits, err);
            overlay_f32(files, ly + ".self_attn.q_a_layernorm.weight", d_.mla_qa_ln[i],
                        cfg_.mla.q_lora, err);
            overlay_mat(files, ly + ".self_attn.q_b_proj.weight", d_.mla_qb[i], mbits, err);
            overlay_mat(files, ly + ".self_attn.kv_a_proj_with_mqa.weight", d_.mla_kva[i], mbits,
                        err);
            overlay_f32(files, ly + ".self_attn.kv_a_layernorm.weight", d_.mla_kva_ln[i],
                        cfg_.mla.kv_lora, err);
            overlay_kvb(files, ly + ".self_attn.kv_b_proj.weight", d_.mla_kt[i], d_.mla_v[i], err);
            overlay_mat(files, ly + ".self_attn.o_proj.weight", d_.mla_o[i], hbits, err);
            overlay_mat(files, ly + ".self_attn.g_proj.weight", d_.mla_g[i], hbits, err);
            overlay_mat(files, ly + ".self_attn.f_a_proj.weight", d_.wfa[i], bits, err);
            overlay_mat(files, ly + ".self_attn.f_b_proj.weight", d_.wfb[i], bits, err);
            overlay_mat(files, ly + ".self_attn.b_proj.weight", d_.wb[i], bits, err);
            overlay_f32(files, ly + ".self_attn.dt_bias", d_.wdt[i], 0, err);
            overlay_f32(files, ly + ".self_attn.A_log", d_.alog[i], 0, err);
            overlay_f32(files, ly + ".self_attn.o_norm.weight", d_.out_norm[i], 0, err);
            overlay_f32(files, ly + ".self_attn.q_conv1d.weight", d_.conv_q[i], 0, err);
            overlay_f32(files, ly + ".self_attn.k_conv1d.weight", d_.conv_k[i], 0, err);
            overlay_f32(files, ly + ".self_attn.v_conv1d.weight", d_.conv_v[i], 0, err);
            overlay_f32(files, ly + ".self_attention_res_norm.weight", d_.res_n[i], H, err);
            overlay_f32(files, ly + ".self_attention_res_proj.weight", d_.res_p[i], H, err);
            overlay_f32(files, ly + ".mlp_res_norm.weight", d_.mlp_res_n[i], H, err);
            overlay_f32(files, ly + ".mlp_res_proj.weight", d_.mlp_res_p[i], H, err);
            fold_sw(d_.res_n[i], d_.res_p[i], d_.attn_sw[i], H);
            fold_sw(d_.mlp_res_n[i], d_.mlp_res_p[i], d_.mlp_sw[i], H);
            if (i < cfg_.first_dense) {
                overlay_mat(files, ly + ".mlp.gate_proj.weight", d_.mlp_gate[i], bits, err);
                overlay_mat(files, ly + ".mlp.up_proj.weight", d_.mlp_up[i], bits, err);
                overlay_mat(files, ly + ".mlp.down_proj.weight", d_.mlp_down[i], bits, err);
            } else {
                overlay_f32(files, ly + ".block_sparse_moe.gate.weight", d_.router[i],
                            cfg_.moe.n_experts * H, err);
                overlay_f32(files, ly + ".block_sparse_moe.gate.e_score_correction_bias",
                            d_.router_bias[i], cfg_.moe.n_experts, err);
                overlay_mat(files, ly + ".block_sparse_moe.routed_expert_down_proj.weight",
                            d_.lat_down[i], bits, err);
                overlay_mat(files, ly + ".block_sparse_moe.routed_expert_up_proj.weight",
                            d_.lat_up[i], bits, err);
                overlay_f32(files, ly + ".block_sparse_moe.routed_expert_norm.weight",
                            d_.lat_n[i], cfg_.moe.latent, err);
                overlay_mat(files, ly + ".block_sparse_moe.shared_experts.gate_proj.weight",
                            d_.shared_gate[i], bits, err);
                overlay_mat(files, ly + ".block_sparse_moe.shared_experts.up_proj.weight",
                            d_.shared_up[i], bits, err);
                overlay_mat(files, ly + ".block_sparse_moe.shared_experts.down_proj.weight",
                            d_.shared_down[i], bits, err);
            }
        }
        {
            std::vector<float> on, op;
            overlay_f32(files, P + "model.output_attn_res_norm.weight", on, H, err);
            overlay_f32(files, P + "model.output_attn_res_proj.weight", op, H, err);
            fold_sw(on, op, d_.out_sw, H);
        }
        err.clear();

        const int probe_l = cfg_.first_dense < L ? cfg_.first_dense : 0;
        const std::string first = P + "model.layers." + std::to_string(probe_l) +
                                  ".block_sparse_moe.experts.0.w1.weight_packed";
        io::StHit probe = io::st_find_dir(files, first);
        if (!probe.tensor) {
            io::st_close_dir(files);
            from_checkpoint_ = true;
            return write_synthetic_experts(model_dir, err);
        }

        const int I = cfg_.moe.latent > 0 ? cfg_.moe.latent : H;
        const int O = cfg_.moe.intermediate;
        const auto geom = make_k3_geom(I, O);
        const int64_t want[6] = {geom.w1p, geom.w1s, geom.w2p, geom.w2s, geom.w1p, geom.w1s};
        int registered = 0;
        for (int i = cfg_.first_dense; i < L; ++i) {
            for (int e = 0; e < cfg_.moe.n_experts; ++e) {
                ExpertLoc loc;
                loc.key = {i, e};
                bool ok = true;
                for (int k = 0; k < 6; ++k) {
                    const std::string n = P + "model.layers." + std::to_string(i) +
                                          ".block_sparse_moe.experts." + std::to_string(e) + "." +
                                          k3_expert_mats[k / 2] + ".weight_" + k3_expert_half[k & 1];
                    io::StHit hit = io::st_find_dir(files, n);
                    if (!hit.tensor || io::st_nbytes(*hit.tensor) != want[k]) {
                        ok = false;
                        break;
                    }
                    ExpertPiece ep;
                    ep.path = hit.file->path;
                    ep.offset = io::st_file_offset(*hit.file, *hit.tensor);
                    ep.bytes = want[k];
                    loc.pieces.push_back(ep);
                }
                if (!ok) {
                    err = "k3 MXFP4 expert pieces missing or wrong size at L" + std::to_string(i) +
                          " E" + std::to_string(e);
                    io::st_close_dir(files);
                    return Status::ParseError;
                }
                loc.contig = true;
                for (int k = 1; k < 6; ++k) {
                    if (loc.pieces[k].path != loc.pieces[0].path ||
                        loc.pieces[k].offset != loc.pieces[k - 1].offset + loc.pieces[k - 1].bytes)
                        loc.contig = false;
                }
                if (loc.contig) {
                    loc.path = loc.pieces[0].path;
                    loc.offset = loc.pieces[0].offset;
                    loc.bytes = geom.slot;
                    loc.pieces.clear();
                }
                st = store_.register_expert(loc, err);
                if (st != Status::Ok) {
                    io::st_close_dir(files);
                    return st;
                }
                ++registered;
            }
        }
        io::st_close_dir(files);
        if (registered == 0)
            return Status::NotFound;
        from_checkpoint_ = true;
        return Status::Ok;
    }

    Status moe_layer(int layer, const float *x, float *h, std::string &err) {
        return moe_layer_n(layer, x, h, 1, err);
    }

    Status moe_layer_n(int layer, const float *xs, float *hs, int C, std::string &err) {
        if (C <= 0)
            return Status::Ok;
        const int H = cfg_.hidden;
        const int I = cfg_.moe.latent > 0 ? cfg_.moe.latent : H;
        const int O = cfg_.moe.intermediate;
        const int K = std::max(cfg_.moe.topk, 0);
        std::vector<int> idx(static_cast<size_t>(C) * std::max(K, 1), -1);
        std::vector<float> wt(static_cast<size_t>(C) * std::max(K, 1), 0.f);
        std::vector<float> zs(static_cast<size_t>(C) * I, 0.f);
        for (int c = 0; c < C; ++c) {
            const float *x = xs + static_cast<size_t>(c) * H;
            std::vector<float> scores(cfg_.moe.n_experts), choice(cfg_.moe.n_experts);
            quant::matmul_f32(scores.data(), x, d_.router[layer].data(), 1, H, cfg_.moe.n_experts);
            for (int i = 0; i < cfg_.moe.n_experts; ++i) {
                scores[i] = quant::sigmoid(scores[i]);
                float b = i < static_cast<int>(d_.router_bias[layer].size()) ? d_.router_bias[layer][i]
                                                                            : 0.f;
                choice[i] = scores[i] + b;
            }
            moe_topk(choice.data(), cfg_.moe.n_experts, K, idx.data() + static_cast<size_t>(c) * K,
                     wt.data() + static_cast<size_t>(c) * K, scores.data());
            if (cfg_.moe.latent > 0)
                d_.lat_down[layer].gemm(zs.data() + static_cast<size_t>(c) * I, x, 1);
            else
                std::memcpy(zs.data() + static_cast<size_t>(c) * I, x, H * sizeof(float));
        }
        std::vector<int> uniq(static_cast<size_t>(std::max(cfg_.moe.n_experts, 1)));
        int nu = moe_union_ids(idx.data(), C, K, uniq.data(), static_cast<int>(uniq.size()));
        std::vector<ExpertKey> keys(static_cast<size_t>(nu));
        for (int i = 0; i < nu; ++i)
            keys[static_cast<size_t>(i)] = {layer, uniq[i]};
        std::vector<float> acc(static_cast<size_t>(C) * I, 0.f);
        std::vector<float> xb(static_cast<size_t>(C) * I), yb(static_cast<size_t>(C) * I);
        std::vector<float> ww(static_cast<size_t>(C));
        std::vector<int> cmap(static_cast<size_t>(C));
        const bool idot = rt_.idot && gpu::device() == Device::Cpu;

        for (int ui = 0; ui < nu; ++ui) {
            const int eid = uniq[ui];
            ExpertView v{};
            Status st = store_.lookup({layer, eid}, v, err);
            if (st != Status::Ok || !v.data) {
                store_.release(v);
                std::string perr;
                store_.wait_prefetch(perr);
                if (st == Status::Ok) {
                    err = "empty expert view";
                    return Status::NotFound;
                }
                return st;
            }
            if (rt_.pipe && ui + 1 < nu)
                store_.prefetch_one_async(keys[static_cast<size_t>(ui + 1)]);
            int n = 0;
            for (int c = 0; c < C; ++c) {
                float wsum = 0.f;
                for (int t = 0; t < K; ++t)
                    if (idx[static_cast<size_t>(c) * K + t] == eid)
                        wsum += wt[static_cast<size_t>(c) * K + t];
                if (wsum == 0.f)
                    continue;
                std::memcpy(xb.data() + static_cast<size_t>(n) * I,
                            zs.data() + static_cast<size_t>(c) * I,
                            static_cast<size_t>(I) * sizeof(float));
                ww[static_cast<size_t>(n)] = wsum;
                cmap[static_cast<size_t>(n)] = c;
                ++n;
            }
            if (n > 0) {
                gpu::k3_expert(yb.data(), xb.data(), n, v.data, I, O, cfg_.moe.situ_b1,
                               cfg_.moe.situ_b2, idot);
                for (int i = 0; i < n; ++i) {
                    float *ac = acc.data() + static_cast<size_t>(cmap[static_cast<size_t>(i)]) * I;
                    const float *hz = yb.data() + static_cast<size_t>(i) * I;
                    const float wsum = ww[static_cast<size_t>(i)];
                    for (int d = 0; d < I; ++d)
                        ac[d] += wsum * hz[d];
                }
            }
            store_.release(v);
            if (rt_.pipe) {
                Status pst = store_.wait_prefetch(err);
                if (pst != Status::Ok)
                    return pst;
            }
        }
        for (int c = 0; c < C; ++c) {
            float *ac = acc.data() + static_cast<size_t>(c) * I;
            const float *x = xs + static_cast<size_t>(c) * H;
            if (!d_.lat_n[layer].empty()) {
                std::vector<float> n(I);
                quant::rmsnorm(ac, d_.lat_n[layer].data(), n.data(), I, cfg_.rms_eps);
                std::memcpy(ac, n.data(), static_cast<size_t>(I) * sizeof(float));
            }
            std::vector<float> uph(H, 0.f);
            if (cfg_.moe.latent > 0)
                d_.lat_up[layer].gemm(uph.data(), ac, 1);
            else
                std::memcpy(uph.data(), ac, H * sizeof(float));
            float *h = hs + static_cast<size_t>(c) * H;
            // Official: h += routed * scale + shared (do not scale the shared expert).
            for (int i = 0; i < H; ++i)
                h[i] += uph[i] * cfg_.moe.routed_scale;
            if (cfg_.moe.n_shared > 0 && !d_.shared_gate[layer].empty()) {
                int inter = cfg_.dense_intermediate > 0 ? cfg_.dense_intermediate : O * 2;
                std::vector<float> sg(inter), su(inter), sd(H);
                d_.shared_gate[layer].gemm(sg.data(), x, 1);
                d_.shared_up[layer].gemm(su.data(), x, 1);
                for (int i = 0; i < inter; ++i)
                    sg[i] = quant::situ_glu(sg[i], su[i], cfg_.moe.situ_b1, cfg_.moe.situ_b2);
                d_.shared_down[layer].gemm(sd.data(), sg.data(), 1);
                for (int i = 0; i < H; ++i)
                    h[i] += sd[i];
            }
        }
        return Status::Ok;
    }

    void alloc_synthetic() {
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int V = cfg_.vocab > 0 ? cfg_.vocab : 32;
        cfg_.vocab = V;
        const int P = cfg_.kda.heads * cfg_.kda.head_dim;
        const int inter = cfg_.dense_intermediate > 0 ? cfg_.dense_intermediate : 64;
        cfg_.dense_intermediate = inter;
        const int bits = rt_.dense_bits;
        const int hbits = rt_.head_bits;
        xavier(d_.embed, V, H, 1);
        ones(d_.norm, H);
        d_.lm_head = qmat_xavier(V, H, 2, hbits);
        d_.attn_in_n.resize(L);
        d_.attn_out_n.resize(L);
        d_.wq.resize(L);
        d_.wk.resize(L);
        d_.wv.resize(L);
        d_.wo.resize(L);
        d_.wg.resize(L);
        d_.wfa.resize(L);
        d_.wfb.resize(L);
        d_.conv_q.resize(L);
        d_.conv_k.resize(L);
        d_.conv_v.resize(L);
        d_.wdt.resize(L);
        d_.alog.resize(L);
        d_.wb.resize(L);
        d_.out_norm.resize(L);
        d_.router.resize(L);
        d_.router_bias.resize(L);
        d_.lat_down.resize(L);
        d_.lat_up.resize(L);
        d_.lat_n.resize(L);
        d_.shared_gate.resize(L);
        d_.shared_up.resize(L);
        d_.shared_down.resize(L);
        d_.res_n.resize(L);
        d_.res_p.resize(L);
        d_.mlp_res_n.resize(L);
        d_.mlp_res_p.resize(L);
        d_.attn_sw.resize(L);
        d_.mlp_sw.resize(L);
        d_.mlp_gate.resize(L);
        d_.mlp_up.resize(L);
        d_.mlp_down.resize(L);
        d_.mla_qa.resize(L);
        d_.mla_qb.resize(L);
        d_.mla_kva.resize(L);
        d_.mla_kt.resize(L);
        d_.mla_v.resize(L);
        d_.mla_o.resize(L);
        d_.mla_g.resize(L);
        d_.mla_qa_ln.resize(L);
        d_.mla_kva_ln.resize(L);
        if (cfg_.mla.n_heads <= 0)
            cfg_.mla.n_heads = 1;
        if (cfg_.mla.v_head <= 0)
            cfg_.mla.v_head = cfg_.mla.qk_nope;
        const int nh = cfg_.mla.n_heads;
        const int qk = cfg_.mla.qk_nope;
        const int qr = cfg_.mla.qk_rope;
        const int vh = cfg_.mla.v_head;
        const int ql = cfg_.mla.q_lora;
        const int kv = cfg_.mla.kv_lora;
        const int mbits = rt_.mla_bits;
        for (int l = 0; l < L; ++l) {
            ones(d_.attn_in_n[l], H);
            ones(d_.attn_out_n[l], H);
            d_.wq[l] = qmat_xavier(P > 0 ? P : H, H, 10 + l, bits);
            d_.wk[l] = qmat_xavier(P > 0 ? P : H, H, 20 + l, bits);
            d_.wv[l] = qmat_xavier(P > 0 ? P : H, H, 30 + l, bits);
            d_.wo[l] = qmat_xavier(H, P > 0 ? P : H, 40 + l, bits);
            d_.wg[l] = qmat_xavier(P > 0 ? P : H, H, 50 + l, bits);
            d_.wfa[l] = qmat_xavier(cfg_.kda.head_dim, H, 60 + l, bits);
            d_.wfb[l] = qmat_xavier(P, cfg_.kda.head_dim, 70 + l, bits);
            d_.wdt[l].assign(P, 0.f);
            d_.alog[l].assign(cfg_.kda.heads, 0.f);
            d_.wb[l] = qmat_xavier(cfg_.kda.heads > 0 ? cfg_.kda.heads : 1, H, 80 + l, bits);
            ones(d_.out_norm[l], cfg_.kda.head_dim > 0 ? cfg_.kda.head_dim : 1);
            xavier(d_.router[l], cfg_.moe.n_experts, H, 90 + l);
            d_.router_bias[l].assign(cfg_.moe.n_experts, 0.f);
            if (cfg_.moe.latent > 0) {
                d_.lat_down[l] = qmat_xavier(cfg_.moe.latent, H, 100 + l, bits);
                d_.lat_up[l] = qmat_xavier(H, cfg_.moe.latent, 110 + l, bits);
                ones(d_.lat_n[l], cfg_.moe.latent);
            }
            d_.shared_gate[l] = qmat_xavier(inter, H, 120 + l, bits);
            d_.shared_up[l] = qmat_xavier(inter, H, 130 + l, bits);
            d_.shared_down[l] = qmat_xavier(H, inter, 140 + l, bits);
            ones(d_.res_n[l], H);
            ones(d_.res_p[l], H);
            ones(d_.mlp_res_n[l], H);
            ones(d_.mlp_res_p[l], H);
            fold_sw(d_.res_n[l], d_.res_p[l], d_.attn_sw[l], H);
            fold_sw(d_.mlp_res_n[l], d_.mlp_res_p[l], d_.mlp_sw[l], H);
            d_.mlp_gate[l] = qmat_xavier(inter, H, 150 + l, bits);
            d_.mlp_up[l] = qmat_xavier(inter, H, 160 + l, bits);
            d_.mlp_down[l] = qmat_xavier(H, inter, 170 + l, bits);
            const bool use_kda = (l < static_cast<int>(cfg_.is_kda.size())) ? cfg_.is_kda[l] : 1;
            if (!use_kda && ql > 0 && kv > 0 && qk > 0) {
                d_.mla_qa[l] = qmat_xavier(ql, H, 400 + l, mbits);
                d_.mla_qb[l] = qmat_xavier(nh * (qk + qr), ql, 410 + l, mbits);
                d_.mla_kva[l] = qmat_xavier(kv + qr, H, 420 + l, mbits);
                d_.mla_kt[l] = qmat_xavier(nh * kv, qk, 430 + l, mbits);
                d_.mla_v[l] = qmat_xavier(nh * vh, kv, 440 + l, mbits);
                d_.mla_o[l] = qmat_xavier(H, nh * vh, 450 + l, hbits);
                if (cfg_.mla.output_gate)
                    d_.mla_g[l] = qmat_xavier(nh * vh, H, 460 + l, hbits);
                ones(d_.mla_qa_ln[l], ql);
                ones(d_.mla_kva_ln[l], kv);
            }
        }
        ones(d_.out_sw, H);
    }

    ModelConfig cfg_{};
    RuntimeConfig rt_{};
    Dense d_{};
    ExpertStore store_;
    std::vector<std::vector<uint8_t>> experts_;
    bool loaded_ = false;
    bool from_checkpoint_ = false;
    std::string prefix_;
    K3Slot slots_[kMaxKvSlots];
};

std::unique_ptr<FamilyEngine> make_kimi_k3() { return std::make_unique<KimiK3Engine>(); }

} // namespace mvllm
