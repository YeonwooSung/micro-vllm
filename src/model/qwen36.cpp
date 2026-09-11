#include "family.hpp"
#include "../io/safetensors.hpp"
#include "../quant/quant.hpp"
#include "../serve/session.hpp"
#include "../tok/decode_post.hpp"
#include "../tok/gbnf.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <sstream>

namespace mvllm {
namespace {

void xavier(std::vector<float> &w, int rows, int cols, uint32_t seed) {
    w.assign(static_cast<size_t>(rows) * cols, 0.f);
    std::mt19937 rng(seed);
    float s = 1.f / std::sqrt(static_cast<float>(std::max(cols, 1)));
    std::normal_distribution<float> dist(0.f, s);
    for (float &v : w)
        v = dist(rng);
}
void xavier_tiles(std::vector<float> &w, int n_tiles, int rows, int cols, uint32_t seed) {
    const size_t tile =
        static_cast<size_t>(std::max(rows, 0)) * static_cast<size_t>(std::max(cols, 0));
    w.assign(static_cast<size_t>(std::max(n_tiles, 0)) * tile, 0.f);
    for (int e = 0; e < n_tiles; ++e) {
        std::vector<float> t;
        xavier(t, rows, cols, seed + static_cast<uint32_t>(e));
        if (!t.empty())
            std::memcpy(w.data() + static_cast<size_t>(e) * tile, t.data(), t.size() * sizeof(float));
    }
}
void ones(std::vector<float> &w, int n) { w.assign(n, 1.f); }

int shape0(const io::StTensor *t) {
    return t && !t->shape.empty() ? static_cast<int>(t->shape[0]) : 0;
}

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

struct AccTimer {
    double &acc;
    std::chrono::steady_clock::time_point t0;
    explicit AccTimer(double &a) : acc(a), t0(std::chrono::steady_clock::now()) {}
    ~AccTimer() {
        acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

} // namespace

class Qwen36Engine final : public FamilyEngine {
public:
    Family family() const override { return Family::Qwen36; }
    const ModelConfig &config() const override { return cfg_; }

    struct Qwen36Slot {
        std::vector<int> history;
        std::vector<float> h;
        std::vector<std::vector<float>> k_cache, v_cache;
        std::vector<std::vector<float>> gdn_s;
        std::vector<std::vector<float>> gdn_hq, gdn_hk, gdn_hv;
        int pos = 0;
        int t_max = 0;
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
        apply_family_defaults(cfg_);
        from_checkpoint_ = false;
        st = load_checkpoint(model_dir, err);
        if (st != Status::Ok) {
            err.clear();
            alloc_synthetic();
            from_checkpoint_ = false;
        }
        loaded_ = true;
        return Status::Ok;
    }

    Status generate(const std::vector<int> &prompt, const GenParams &gp, GenResult &out,
                    std::string &err) override {
        if (!loaded_) {
            err = "qwen36 not loaded";
            return Status::InvalidArgument;
        }
        if (prompt.empty()) {
            err = "empty prompt";
            return Status::InvalidArgument;
        }
        const int cslot = gp.cache_slot;
        if (cslot >= 0 && cslot < kMaxKvSlots)
            return generate_cached(cslot, prompt, gp, out, err);
        const int H = std::max(cfg_.hidden, 1);
        const int I = std::max(cfg_.dense_intermediate, 1);
        const int nq = std::max(cfg_.n_q_heads, 1);
        const int nkv = std::max(cfg_.n_kv_heads, 1);
        const int hd = std::max(cfg_.head_dim, 1);
        const int qd = nq * hd;
        const int kvd = nkv * hd;
        const int group = std::max(nq / nkv, 1);
        const float scale = 1.f / std::sqrt(static_cast<float>(hd));
        const float theta = cfg_.rope_theta > 0.f ? cfg_.rope_theta : 10000.f;
        int Tmax = static_cast<int>(prompt.size()) + std::max(gp.max_new_tokens, 0) + 1;
        if (Tmax < 1)
            Tmax = 1;
        std::vector<float> h(H), nrm(H), q(std::max(H, qd)), k(kvd), v(kvd), ctx(qd), g(I), u(I),
            d(H);
        const int L = std::max(cfg_.n_layers, 0);
        std::vector<std::vector<float>> gdn_s(static_cast<size_t>(L)),
            gdn_hq(static_cast<size_t>(L)), gdn_hk(static_cast<size_t>(L)),
            gdn_hv(static_cast<size_t>(L));
        for (int l = 0; l < L; ++l) {
            if (!layer_gdn(l))
                continue;
            gdn_s[static_cast<size_t>(l)].assign(static_cast<size_t>(nq) * hd * hd, 0.f);
            gdn_hq[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * qd, 0.f);
            gdn_hk[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * kvd, 0.f);
            gdn_hv[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * kvd, 0.f);
        }
        std::vector<std::vector<float>> k_cache, v_cache;
        const bool any_gqa = has_gqa();
        if (any_gqa) {
            k_cache.resize(static_cast<size_t>(cfg_.n_layers));
            v_cache.resize(static_cast<size_t>(cfg_.n_layers));
            for (int l = 0; l < cfg_.n_layers; ++l) {
                if (layer_gqa(l)) {
                    k_cache[static_cast<size_t>(l)].assign(static_cast<size_t>(Tmax) * kvd, 0.f);
                    v_cache[static_cast<size_t>(l)].assign(static_cast<size_t>(Tmax) * kvd, 0.f);
                }
            }
        }
        int pos = 0;
        auto embed = [&](int id) {
            int tid = id;
            if (tid < 0 || tid >= cfg_.vocab)
                tid = 0;
            std::memcpy(h.data(), embed_.data() + static_cast<size_t>(tid) * H, H * sizeof(float));
        };
        auto step = [&](int token) {
            embed(token);
            const int tpos = pos;
            for (int l = 0; l < cfg_.n_layers; ++l) {
                quant::rmsnorm(h.data(), in_n_[static_cast<size_t>(l)].data(), nrm.data(), H,
                               cfg_.rms_eps);
                {
                    AccTimer t(t_attn_);
                    if (layer_gdn(l) && layer_gqa(l)) {
                        quant::matmul_f32(q.data(), nrm.data(), wq_[static_cast<size_t>(l)].data(),
                                          1, H, qd);
                        quant::matmul_f32(k.data(), nrm.data(), wk_[static_cast<size_t>(l)].data(),
                                          1, H, kvd);
                        quant::matmul_f32(v.data(), nrm.data(), wv_[static_cast<size_t>(l)].data(),
                                          1, H, kvd);
                        if (l < static_cast<int>(conv_q_.size()) &&
                            !conv_q_[static_cast<size_t>(l)].empty()) {
                            gdn_conv4(q.data(), gdn_hq[static_cast<size_t>(l)].data(),
                                      conv_q_[static_cast<size_t>(l)].data(), qd);
                            gdn_conv4(k.data(), gdn_hk[static_cast<size_t>(l)].data(),
                                      conv_k_[static_cast<size_t>(l)].data(), kvd);
                            gdn_conv4(v.data(), gdn_hv[static_cast<size_t>(l)].data(),
                                      conv_v_[static_cast<size_t>(l)].data(), kvd);
                        }
                        const int ss = nq * hd * hd;
                        if (static_cast<int>(gdn_s[static_cast<size_t>(l)].size()) < ss)
                            gdn_s[static_cast<size_t>(l)].assign(static_cast<size_t>(ss), 0.f);
                        gdn_delta(gdn_s[static_cast<size_t>(l)].data(), ctx.data(), q.data(),
                                  k.data(), v.data(), nq, nkv, hd, group);
                        quant::matmul_f32(d.data(), ctx.data(), wo_[static_cast<size_t>(l)].data(),
                                          1, qd, H);
                    } else if (layer_gqa(l) && tpos < Tmax) {
                        quant::matmul_f32(q.data(), nrm.data(), wq_[static_cast<size_t>(l)].data(),
                                          1, H, qd);
                        quant::matmul_f32(k.data(), nrm.data(), wk_[static_cast<size_t>(l)].data(),
                                          1, H, kvd);
                        quant::matmul_f32(v.data(), nrm.data(), wv_[static_cast<size_t>(l)].data(),
                                          1, H, kvd);
                        for (int hh = 0; hh < nq; ++hh)
                            apply_rope(q.data() + hh * hd, hd, tpos, theta);
                        for (int hh = 0; hh < nkv; ++hh)
                            apply_rope(k.data() + hh * hd, hd, tpos, theta);
                        std::memcpy(k_cache[static_cast<size_t>(l)].data() +
                                        static_cast<size_t>(tpos) * kvd,
                                    k.data(), static_cast<size_t>(kvd) * sizeof(float));
                        std::memcpy(v_cache[static_cast<size_t>(l)].data() +
                                        static_cast<size_t>(tpos) * kvd,
                                    v.data(), static_cast<size_t>(kvd) * sizeof(float));
                        std::fill(ctx.begin(), ctx.end(), 0.f);
                        const int past = tpos + 1;
                        std::vector<float> sc(static_cast<size_t>(past));
                        for (int hh = 0; hh < nq; ++hh) {
                            const int kh = std::min(hh / group, nkv - 1);
                            const float *qh = q.data() + hh * hd;
                            float mx = -1e30f;
                            for (int s = 0; s < past; ++s) {
                                const float *kk = k_cache[static_cast<size_t>(l)].data() +
                                                  static_cast<size_t>(s) * kvd + kh * hd;
                                float acc = 0.f;
                                for (int d0 = 0; d0 < hd; ++d0)
                                    acc += qh[d0] * kk[d0];
                                sc[static_cast<size_t>(s)] = acc * scale;
                                if (sc[static_cast<size_t>(s)] > mx)
                                    mx = sc[static_cast<size_t>(s)];
                            }
                            float z = 0.f;
                            for (int s = 0; s < past; ++s) {
                                sc[static_cast<size_t>(s)] =
                                    std::exp(sc[static_cast<size_t>(s)] - mx);
                                z += sc[static_cast<size_t>(s)];
                            }
                            if (z <= 0.f)
                                z = 1.f;
                            float *oh = ctx.data() + hh * hd;
                            for (int s = 0; s < past; ++s) {
                                const float *vv = v_cache[static_cast<size_t>(l)].data() +
                                                  static_cast<size_t>(s) * kvd + kh * hd;
                                const float a = sc[static_cast<size_t>(s)] / z;
                                for (int d0 = 0; d0 < hd; ++d0)
                                    oh[d0] += a * vv[d0];
                            }
                        }
                        quant::matmul_f32(d.data(), ctx.data(), wo_[static_cast<size_t>(l)].data(),
                                          1, qd, H);
                    } else {
                        quant::matmul_f32(q.data(), nrm.data(), wq_[static_cast<size_t>(l)].data(),
                                          1, H, H);
                        quant::matmul_f32(d.data(), q.data(), wo_[static_cast<size_t>(l)].data(), 1,
                                          H, H);
                    }
                }
                for (int i = 0; i < H; ++i)
                    h[i] += d[i];
                apply_ffn(l, h.data(), nrm.data(), g.data(), u.data(), d.data(), H, I);
            }
            ++pos;
        };
        for (int t : prompt)
            step(t);
        out.tokens.clear();
        out.token_logprobs.clear();
        out.top_logprobs.clear();
        out.prompt_tokens = static_cast<int>(prompt.size());
        uint64_t rng = gp.seed ? gp.seed : 1ull;
        std::vector<uint8_t> allow;
        Gbnf gbnf;
        if (!gp.grammar.empty()) {
            std::string e;
            if (gbnf.compile(gp.grammar, e) == Status::Ok && gbnf.ready() && gp.token_text) {
                allow.assign(cfg_.vocab, 0);
                gbnf.allow_mask(gp.token_text, allow.data(), cfg_.vocab);
            }
        }
        std::string acc;
        std::vector<int> hist = prompt;
        for (int k = 0; k < gp.max_new_tokens; ++k) {
            std::vector<float> logits(cfg_.vocab);
            {
                AccTimer t(t_head_);
                quant::rmsnorm(h.data(), norm_.data(), nrm.data(), H, cfg_.rms_eps);
                quant::matmul_f32(logits.data(), nrm.data(), lm_head_.data(), 1, H, cfg_.vocab);
            }
            int next = sample_penalized(logits.data(), cfg_.vocab, gp, hist.data(),
                                       static_cast<int>(hist.size()), &rng,
                                       allow.empty() ? nullptr : allow.data(), nullptr,
                                       &out.token_logprobs, &out.top_logprobs);
            out.tokens.push_back(next);
            hist.push_back(next);
            if (gbnf.ready() && gp.token_text)
                gbnf.accept_bytes(gp.token_text(next));
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
            step(next);
            if (gbnf.ready() && gp.token_text) {
                allow.assign(cfg_.vocab, 0);
                gbnf.allow_mask(gp.token_text, allow.data(), cfg_.vocab);
            }
        }
        out.completion_tokens = static_cast<int>(out.tokens.size());
        return Status::Ok;
    }

    std::string describe() const override {
        std::ostringstream os;
        os << "qwen36  hidden=" << cfg_.hidden << " layers=" << cfg_.n_layers
           << " q_heads=" << cfg_.n_q_heads << " kv_heads=" << cfg_.n_kv_heads
           << " checkpoint=" << (from_checkpoint_ ? "yes" : "synthetic") << " gdn="
           << (has_gdn() ? "delta" : "off") << " (CPU "
           << (has_gqa() ? "GQA" : "stand-in")
           << "; routed="
           << (have_expert_mats() ? "experts" : (cfg_.moe.n_experts > 0 ? "mix" : "no"))
           << "; CUDA demo is micro-vllm-cuda)";
        return os.str();
    }

    void turn_perf(TurnPerf &out, bool reset) override {
        out = {};
        out.t_attn = t_attn_;
        out.t_head = t_head_;
        if (reset)
            t_attn_ = t_head_ = 0;
    }

    Status begin_generate(int slot, const std::vector<int> &ids, const GenParams &gp, int &reuse,
                          std::string &err) override {
        if (!loaded_) {
            err = "qwen36 not loaded";
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
        Qwen36Slot &st = slots_[slot];
        const int match = st.have ? kv_common_prefix(st.history, ids) : 0;
        const int want = std::min(std::max(gp.prefix_reuse, 0), match);
        int applied = 0;
        int Tmax = std::max(rt_.max_seq, static_cast<int>(ids.size()) + gp.max_new_tokens + 1);
        if (Tmax <= 0)
            Tmax = 4096;
        if (st.have && st.pos > 0 && st.pos <= want && st.pos <= static_cast<int>(ids.size())) {
            applied = st.pos;
            slot_ensure_t(st, Tmax);
        } else {
            slot_alloc(st, Tmax);
        }
        slot_prefill(st, ids, applied);
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
                st.allow.assign(static_cast<size_t>(cfg_.vocab), 0);
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
        Qwen36Slot &st = slots_[slot];
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
        token = slot_sample(st, st.gp, &st.rng, allow);
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
            slot_step(st, token);
            done = false;
        }
        if (st.g.ready() && st.gp.token_text) {
            st.allow.assign(static_cast<size_t>(cfg_.vocab), 0);
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
        std::vector<Qwen36Slot *> step_s;
        std::vector<int> step_tok;
        step_s.reserve(static_cast<size_t>(n));
        step_tok.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            Qwen36Slot &st = slots_[slots[i]];
            done[i] = 0;
            if (st.gp.max_new_tokens <= 0 || st.emitted >= st.gp.max_new_tokens) {
                done[i] = 1;
                tokens[i] = -1;
                continue;
            }
            const uint8_t *allow = st.allow.empty() ? nullptr : st.allow.data();
            tokens[i] = slot_sample(st, st.gp, &st.rng, allow);
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
                st.allow.assign(static_cast<size_t>(cfg_.vocab), 0);
                st.g.allow_mask(st.gp.token_text, st.allow.data(), cfg_.vocab);
            }
        }
        for (size_t i = 0; i < step_s.size(); ++i)
            slot_step(*step_s[i], step_tok[i]);
        return Status::Ok;
    }

    void end_generate(int slot) override {
        if (slot < 0 || slot >= kMaxKvSlots)
            return;
        slots_[slot].live = false;
    }

    // GQA K/V rows → persist L/R [n_layers, nkv*hd]. Caller pre-sizes dest.
    int export_kv_rows(int slot, int pos0, int n, KvPersistRecord *rows) const override {
        if (!rows || n <= 0 || slot < 0 || slot >= kMaxKvSlots)
            return 0;
        const Qwen36Slot &sl = slots_[slot];
        if (!sl.have)
            return 0;
        const int nkv = std::max(cfg_.n_kv_heads, 1);
        const int hd = std::max(cfg_.head_dim, 1);
        const int kvd = nkv * hd;
        const int nL = cfg_.n_layers;
        for (int t = 0; t < n; ++t) {
            const int pos = pos0 + t;
            if (pos < 0 || pos >= sl.pos)
                return t;
            KvPersistRecord &rec = rows[t];
            const size_t src0 = static_cast<size_t>(pos) * static_cast<size_t>(kvd);
            for (int l = 0; l < nL; ++l) {
                const size_t dst = static_cast<size_t>(l) * static_cast<size_t>(kvd);
                if (kvd > 0 && l < static_cast<int>(sl.k_cache.size()) &&
                    src0 + static_cast<size_t>(kvd) <= sl.k_cache[static_cast<size_t>(l)].size() &&
                    dst < rec.L.size()) {
                    int ncopy = std::min(kvd, static_cast<int>(rec.L.size() - dst));
                    if (ncopy > 0)
                        std::memcpy(rec.L.data() + dst,
                                    sl.k_cache[static_cast<size_t>(l)].data() + src0,
                                    static_cast<size_t>(ncopy) * sizeof(float));
                }
                if (kvd > 0 && l < static_cast<int>(sl.v_cache.size()) &&
                    src0 + static_cast<size_t>(kvd) <= sl.v_cache[static_cast<size_t>(l)].size() &&
                    dst < rec.R.size()) {
                    int ncopy = std::min(kvd, static_cast<int>(rec.R.size() - dst));
                    if (ncopy > 0)
                        std::memcpy(rec.R.data() + dst,
                                    sl.v_cache[static_cast<size_t>(l)].data() + src0,
                                    static_cast<size_t>(ncopy) * sizeof(float));
                }
            }
        }
        return n;
    }

    // Inverse: persist L/R → k_cache / v_cache, then history / pos.
    int import_kv_rows(int slot, int pos0, int n, const KvPersistRecord *rows) override {
        if (!rows || n <= 0 || slot < 0 || slot >= kMaxKvSlots)
            return 0;
        Qwen36Slot &sl = slots_[slot];
        const int nkv = std::max(cfg_.n_kv_heads, 1);
        const int hd = std::max(cfg_.head_dim, 1);
        const int kvd = nkv * hd;
        const int nL = cfg_.n_layers;
        int t_max = pos0 + n;
        if (t_max < sl.pos)
            t_max = sl.pos;
        if (t_max < 1)
            t_max = 1;
        t_max += std::max(rt_.max_seq, 64);
        const bool caches_small = static_cast<int>(sl.k_cache.size()) < nL ||
                                  static_cast<int>(sl.v_cache.size()) < nL;
        if (!sl.have || caches_small)
            slot_alloc(sl, t_max);
        else
            slot_ensure_t(sl, t_max);
        int written = 0;
        for (int t = 0; t < n; ++t) {
            const int pos = pos0 + t;
            if (pos < 0)
                return t;
            const KvPersistRecord &rec = rows[t];
            const size_t dst0 = static_cast<size_t>(pos) * static_cast<size_t>(kvd);
            for (int l = 0; l < nL; ++l) {
                const size_t src = static_cast<size_t>(l) * static_cast<size_t>(kvd);
                if (kvd > 0 && l < static_cast<int>(sl.k_cache.size()) &&
                    dst0 < sl.k_cache[static_cast<size_t>(l)].size() && src < rec.L.size()) {
                    int ncopy = std::min(kvd, static_cast<int>(rec.L.size() - src));
                    const int room =
                        static_cast<int>(sl.k_cache[static_cast<size_t>(l)].size() - dst0);
                    if (ncopy > room)
                        ncopy = room;
                    if (ncopy > 0)
                        std::memcpy(sl.k_cache[static_cast<size_t>(l)].data() + dst0,
                                    rec.L.data() + src, static_cast<size_t>(ncopy) * sizeof(float));
                }
                if (kvd > 0 && l < static_cast<int>(sl.v_cache.size()) &&
                    dst0 < sl.v_cache[static_cast<size_t>(l)].size() && src < rec.R.size()) {
                    int ncopy = std::min(kvd, static_cast<int>(rec.R.size() - src));
                    const int room =
                        static_cast<int>(sl.v_cache[static_cast<size_t>(l)].size() - dst0);
                    if (ncopy > room)
                        ncopy = room;
                    if (ncopy > 0)
                        std::memcpy(sl.v_cache[static_cast<size_t>(l)].data() + dst0,
                                    rec.R.data() + src, static_cast<size_t>(ncopy) * sizeof(float));
                }
            }
            if (static_cast<int>(sl.history.size()) <= pos)
                sl.history.resize(static_cast<size_t>(pos) + 1, 0);
            sl.history[static_cast<size_t>(pos)] = rec.token;
            ++written;
        }
        sl.pos = std::max(sl.pos, pos0 + written);
        sl.have = sl.pos > 0;
        return written;
    }

private:
    Status generate_cached(int cslot, const std::vector<int> &prompt, const GenParams &gp,
                           GenResult &out, std::string &err) {
        (void)err;
        Qwen36Slot &sl = slots_[cslot];
        int Tmax = std::max(rt_.max_seq, static_cast<int>(prompt.size()) +
                                             std::max(gp.max_new_tokens, 0) + 1);
        if (Tmax <= 0)
            Tmax = 4096;
        int reuse = 0;
        if (gp.prefix_reuse > 0) {
            const int match = sl.have ? kv_common_prefix(sl.history, prompt) : 0;
            const int want = std::min(gp.prefix_reuse, match);
            if (sl.have && sl.pos > 0 && sl.pos <= want &&
                sl.pos <= static_cast<int>(prompt.size())) {
                reuse = sl.pos;
                slot_ensure_t(sl, Tmax);
            }
        }
        if (reuse == 0)
            slot_alloc(sl, Tmax);
        slot_prefill(sl, prompt, reuse);

        out.tokens.clear();
        out.token_logprobs.clear();
        out.top_logprobs.clear();
        out.prompt_tokens = static_cast<int>(prompt.size());
        uint64_t rng = gp.seed ? gp.seed : 1ull;
        std::vector<uint8_t> allow;
        Gbnf gbnf;
        if (!gp.grammar.empty()) {
            std::string e;
            if (gbnf.compile(gp.grammar, e) == Status::Ok && gbnf.ready() && gp.token_text) {
                allow.assign(static_cast<size_t>(cfg_.vocab), 0);
                gbnf.allow_mask(gp.token_text, allow.data(), cfg_.vocab);
            }
        }
        std::string acc;
        sl.history = prompt;
        sl.token_lps.clear();
        sl.top_lps.clear();
        for (int k = 0; k < gp.max_new_tokens; ++k) {
            int next = slot_sample(sl, gp, &rng, allow.empty() ? nullptr : allow.data());
            out.tokens.push_back(next);
            sl.history.push_back(next);
            if (gbnf.ready() && gp.token_text)
                gbnf.accept_bytes(gp.token_text(next));
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
            slot_step(sl, next);
            if (gbnf.ready() && gp.token_text) {
                allow.assign(static_cast<size_t>(cfg_.vocab), 0);
                gbnf.allow_mask(gp.token_text, allow.data(), cfg_.vocab);
            }
        }
        out.completion_tokens = static_cast<int>(out.tokens.size());
        out.token_logprobs = sl.token_lps;
        out.top_logprobs = sl.top_lps;
        sl.history = prompt;
        sl.history.insert(sl.history.end(), out.tokens.begin(), out.tokens.end());
        sl.have = true;
        sl.live = false;
        return Status::Ok;
    }

    void slot_alloc(Qwen36Slot &s, int t_max) {
        const int H = std::max(cfg_.hidden, 1);
        const int nq = std::max(cfg_.n_q_heads, 1);
        const int nkv = std::max(cfg_.n_kv_heads, 1);
        const int hd = std::max(cfg_.head_dim, 1);
        const int qd = nq * hd;
        const int kvd = nkv * hd;
        const int Tmax = std::max(t_max, 1);
        const int L = std::max(cfg_.n_layers, 0);
        s.h.assign(static_cast<size_t>(H), 0.f);
        s.k_cache.assign(static_cast<size_t>(L), {});
        s.v_cache.assign(static_cast<size_t>(L), {});
        s.gdn_s.assign(static_cast<size_t>(L), {});
        s.gdn_hq.assign(static_cast<size_t>(L), {});
        s.gdn_hk.assign(static_cast<size_t>(L), {});
        s.gdn_hv.assign(static_cast<size_t>(L), {});
        if (has_gqa()) {
            for (int l = 0; l < L; ++l) {
                if (layer_gqa(l)) {
                    s.k_cache[static_cast<size_t>(l)].assign(static_cast<size_t>(Tmax) * kvd, 0.f);
                    s.v_cache[static_cast<size_t>(l)].assign(static_cast<size_t>(Tmax) * kvd, 0.f);
                }
            }
        }
        for (int l = 0; l < L; ++l) {
            if (layer_gdn(l)) {
                s.gdn_s[static_cast<size_t>(l)].assign(static_cast<size_t>(nq) * hd * hd, 0.f);
                s.gdn_hq[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * qd, 0.f);
                s.gdn_hk[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * kvd, 0.f);
                s.gdn_hv[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * kvd, 0.f);
            }
        }
        s.t_max = Tmax;
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

    void slot_ensure_t(Qwen36Slot &s, int t_max) {
        const int Tmax = std::max(t_max, 1);
        const int L = std::max(cfg_.n_layers, 0);
        const int nq = std::max(cfg_.n_q_heads, 1);
        const int hd0 = std::max(cfg_.head_dim, 1);
        const int qd = nq * hd0;
        if (static_cast<int>(s.k_cache.size()) < L)
            s.k_cache.resize(static_cast<size_t>(L));
        if (static_cast<int>(s.v_cache.size()) < L)
            s.v_cache.resize(static_cast<size_t>(L));
        if (static_cast<int>(s.gdn_s.size()) < L)
            s.gdn_s.resize(static_cast<size_t>(L));
        if (static_cast<int>(s.gdn_hq.size()) < L)
            s.gdn_hq.resize(static_cast<size_t>(L));
        if (static_cast<int>(s.gdn_hk.size()) < L)
            s.gdn_hk.resize(static_cast<size_t>(L));
        if (static_cast<int>(s.gdn_hv.size()) < L)
            s.gdn_hv.resize(static_cast<size_t>(L));
        const int nkv0 = std::max(cfg_.n_kv_heads, 1);
        const int kvd0 = nkv0 * hd0;
        for (int l = 0; l < L; ++l) {
            if (!layer_gdn(l))
                continue;
            const size_t needS = static_cast<size_t>(nq) * hd0 * hd0;
            if (s.gdn_s[static_cast<size_t>(l)].size() < needS)
                s.gdn_s[static_cast<size_t>(l)].assign(needS, 0.f);
            if (s.gdn_hq[static_cast<size_t>(l)].size() < static_cast<size_t>(4) * qd)
                s.gdn_hq[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * qd, 0.f);
            if (s.gdn_hk[static_cast<size_t>(l)].size() < static_cast<size_t>(4) * kvd0)
                s.gdn_hk[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * kvd0, 0.f);
            if (s.gdn_hv[static_cast<size_t>(l)].size() < static_cast<size_t>(4) * kvd0)
                s.gdn_hv[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * kvd0, 0.f);
        }
        if (!has_gqa()) {
            if (Tmax > s.t_max)
                s.t_max = Tmax;
            return;
        }
        const int nkv = std::max(cfg_.n_kv_heads, 1);
        const int hd = std::max(cfg_.head_dim, 1);
        const int kvd = nkv * hd;
        const size_t need = static_cast<size_t>(Tmax) * kvd;
        for (int l = 0; l < L; ++l) {
            if (!layer_gqa(l))
                continue;
            if (s.k_cache[static_cast<size_t>(l)].size() < need)
                s.k_cache[static_cast<size_t>(l)].resize(need, 0.f);
            if (s.v_cache[static_cast<size_t>(l)].size() < need)
                s.v_cache[static_cast<size_t>(l)].resize(need, 0.f);
        }
        if (Tmax > s.t_max)
            s.t_max = Tmax;
    }

    void slot_step(Qwen36Slot &s, int token) {
        const int H = std::max(cfg_.hidden, 1);
        const int I = std::max(cfg_.dense_intermediate, 1);
        const int nq = std::max(cfg_.n_q_heads, 1);
        const int nkv = std::max(cfg_.n_kv_heads, 1);
        const int hd = std::max(cfg_.head_dim, 1);
        const int qd = nq * hd;
        const int kvd = nkv * hd;
        const int group = std::max(nq / nkv, 1);
        const float scale = 1.f / std::sqrt(static_cast<float>(hd));
        const float theta = cfg_.rope_theta > 0.f ? cfg_.rope_theta : 10000.f;
        const int Tmax = std::max(s.t_max, 1);
        if (static_cast<int>(s.h.size()) != H)
            s.h.assign(static_cast<size_t>(H), 0.f);
        int tid = token;
        if (tid < 0 || tid >= cfg_.vocab)
            tid = 0;
        std::memcpy(s.h.data(), embed_.data() + static_cast<size_t>(tid) * H,
                    static_cast<size_t>(H) * sizeof(float));
        std::vector<float> nrm(static_cast<size_t>(H)), q(static_cast<size_t>(std::max(H, qd))),
            k(static_cast<size_t>(kvd)), v(static_cast<size_t>(kvd)), ctx(static_cast<size_t>(qd)),
            g(static_cast<size_t>(I)), u(static_cast<size_t>(I)), d(static_cast<size_t>(H));
        const int tpos = s.pos;
        for (int l = 0; l < cfg_.n_layers; ++l) {
            quant::rmsnorm(s.h.data(), in_n_[static_cast<size_t>(l)].data(), nrm.data(), H,
                           cfg_.rms_eps);
            {
                AccTimer t(t_attn_);
                if (layer_gdn(l) && layer_gqa(l)) {
                    quant::matmul_f32(q.data(), nrm.data(), wq_[static_cast<size_t>(l)].data(), 1, H,
                                      qd);
                    quant::matmul_f32(k.data(), nrm.data(), wk_[static_cast<size_t>(l)].data(), 1, H,
                                      kvd);
                    quant::matmul_f32(v.data(), nrm.data(), wv_[static_cast<size_t>(l)].data(), 1, H,
                                      kvd);
                    if (l >= static_cast<int>(s.gdn_s.size())) {
                        s.gdn_s.resize(static_cast<size_t>(l) + 1);
                        s.gdn_hq.resize(static_cast<size_t>(l) + 1);
                        s.gdn_hk.resize(static_cast<size_t>(l) + 1);
                        s.gdn_hv.resize(static_cast<size_t>(l) + 1);
                    }
                    if (s.gdn_hq[static_cast<size_t>(l)].size() < static_cast<size_t>(4) * qd)
                        s.gdn_hq[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * qd, 0.f);
                    if (s.gdn_hk[static_cast<size_t>(l)].size() < static_cast<size_t>(4) * kvd)
                        s.gdn_hk[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * kvd, 0.f);
                    if (s.gdn_hv[static_cast<size_t>(l)].size() < static_cast<size_t>(4) * kvd)
                        s.gdn_hv[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * kvd, 0.f);
                    if (l < static_cast<int>(conv_q_.size()) &&
                        !conv_q_[static_cast<size_t>(l)].empty()) {
                        gdn_conv4(q.data(), s.gdn_hq[static_cast<size_t>(l)].data(),
                                  conv_q_[static_cast<size_t>(l)].data(), qd);
                        gdn_conv4(k.data(), s.gdn_hk[static_cast<size_t>(l)].data(),
                                  conv_k_[static_cast<size_t>(l)].data(), kvd);
                        gdn_conv4(v.data(), s.gdn_hv[static_cast<size_t>(l)].data(),
                                  conv_v_[static_cast<size_t>(l)].data(), kvd);
                    }
                    const int ss = nq * hd * hd;
                    if (static_cast<int>(s.gdn_s[static_cast<size_t>(l)].size()) < ss)
                        s.gdn_s[static_cast<size_t>(l)].assign(static_cast<size_t>(ss), 0.f);
                    gdn_delta(s.gdn_s[static_cast<size_t>(l)].data(), ctx.data(), q.data(), k.data(),
                              v.data(), nq, nkv, hd, group);
                    quant::matmul_f32(d.data(), ctx.data(), wo_[static_cast<size_t>(l)].data(), 1,
                                      qd, H);
                } else if (layer_gqa(l) && tpos < Tmax &&
                    l < static_cast<int>(s.k_cache.size()) &&
                    l < static_cast<int>(s.v_cache.size()) &&
                    !s.k_cache[static_cast<size_t>(l)].empty() &&
                    !s.v_cache[static_cast<size_t>(l)].empty()) {
                    quant::matmul_f32(q.data(), nrm.data(), wq_[static_cast<size_t>(l)].data(), 1, H,
                                      qd);
                    quant::matmul_f32(k.data(), nrm.data(), wk_[static_cast<size_t>(l)].data(), 1, H,
                                      kvd);
                    quant::matmul_f32(v.data(), nrm.data(), wv_[static_cast<size_t>(l)].data(), 1, H,
                                      kvd);
                    for (int hh = 0; hh < nq; ++hh)
                        apply_rope(q.data() + hh * hd, hd, tpos, theta);
                    for (int hh = 0; hh < nkv; ++hh)
                        apply_rope(k.data() + hh * hd, hd, tpos, theta);
                    std::memcpy(s.k_cache[static_cast<size_t>(l)].data() +
                                    static_cast<size_t>(tpos) * kvd,
                                k.data(), static_cast<size_t>(kvd) * sizeof(float));
                    std::memcpy(s.v_cache[static_cast<size_t>(l)].data() +
                                    static_cast<size_t>(tpos) * kvd,
                                v.data(), static_cast<size_t>(kvd) * sizeof(float));
                    std::fill(ctx.begin(), ctx.end(), 0.f);
                    const int past = tpos + 1;
                    std::vector<float> sc(static_cast<size_t>(past));
                    for (int hh = 0; hh < nq; ++hh) {
                        const int kh = std::min(hh / group, nkv - 1);
                        const float *qh = q.data() + hh * hd;
                        float mx = -1e30f;
                        for (int t = 0; t < past; ++t) {
                            const float *kk = s.k_cache[static_cast<size_t>(l)].data() +
                                              static_cast<size_t>(t) * kvd + kh * hd;
                            float acc = 0.f;
                            for (int d0 = 0; d0 < hd; ++d0)
                                acc += qh[d0] * kk[d0];
                            sc[static_cast<size_t>(t)] = acc * scale;
                            if (sc[static_cast<size_t>(t)] > mx)
                                mx = sc[static_cast<size_t>(t)];
                        }
                        float z = 0.f;
                        for (int t = 0; t < past; ++t) {
                            sc[static_cast<size_t>(t)] = std::exp(sc[static_cast<size_t>(t)] - mx);
                            z += sc[static_cast<size_t>(t)];
                        }
                        if (z <= 0.f)
                            z = 1.f;
                        float *oh = ctx.data() + hh * hd;
                        for (int t = 0; t < past; ++t) {
                            const float *vv = s.v_cache[static_cast<size_t>(l)].data() +
                                              static_cast<size_t>(t) * kvd + kh * hd;
                            const float a = sc[static_cast<size_t>(t)] / z;
                            for (int d0 = 0; d0 < hd; ++d0)
                                oh[d0] += a * vv[d0];
                        }
                    }
                    quant::matmul_f32(d.data(), ctx.data(), wo_[static_cast<size_t>(l)].data(), 1,
                                      qd, H);
                } else {
                    quant::matmul_f32(q.data(), nrm.data(), wq_[static_cast<size_t>(l)].data(), 1, H,
                                      H);
                    quant::matmul_f32(d.data(), q.data(), wo_[static_cast<size_t>(l)].data(), 1, H,
                                      H);
                }
            }
            for (int i = 0; i < H; ++i)
                s.h[static_cast<size_t>(i)] += d[static_cast<size_t>(i)];
            apply_ffn(l, s.h.data(), nrm.data(), g.data(), u.data(), d.data(), H, I);
        }
        ++s.pos;
    }

    void slot_prefill(Qwen36Slot &s, const std::vector<int> &ids, int start) {
        if (start < 0)
            start = 0;
        if (start >= static_cast<int>(ids.size()))
            return;
        for (size_t i = static_cast<size_t>(start); i < ids.size(); ++i)
            slot_step(s, ids[i]);
    }

    int slot_sample(Qwen36Slot &s, const GenParams &gp, uint64_t *rng, const uint8_t *allow) {
        const int H = std::max(cfg_.hidden, 1);
        std::vector<float> nrm(static_cast<size_t>(H)),
            logits(static_cast<size_t>(std::max(cfg_.vocab, 1)));
        {
            AccTimer t(t_head_);
            quant::rmsnorm(s.h.data(), norm_.data(), nrm.data(), H, cfg_.rms_eps);
            quant::matmul_f32(logits.data(), nrm.data(), lm_head_.data(), 1, H, cfg_.vocab);
        }
        return sample_penalized(logits.data(), cfg_.vocab, gp, s.history.data(),
                               static_cast<int>(s.history.size()), rng, allow, nullptr,
                               &s.token_lps, &s.top_lps);
    }

    static void gdn_conv4(float *x, float *hist, const float *w, int D) {
        if (!x || !hist || D < 1)
            return;
        std::memmove(hist + D, hist, static_cast<size_t>(3) * D * sizeof(float));
        std::memcpy(hist, x, static_cast<size_t>(D) * sizeof(float));
        if (!w)
            return;
        for (int d = 0; d < D; ++d) {
            float acc = 0.f;
            for (int t = 0; t < 4; ++t)
                acc += w[static_cast<size_t>(t) * D + d] * hist[static_cast<size_t>(t) * D + d];
            x[d] = acc;
        }
    }

    static void gdn_delta(float *S, float *ctx, const float *q, const float *k, const float *v,
                          int nq, int nkv, int hd, int group) {
        if (!S || !ctx || !q || !k || !v || nq < 1 || nkv < 1 || hd < 1)
            return;
        for (int hh = 0; hh < nq; ++hh) {
            const int kh = std::min(hh / std::max(group, 1), nkv - 1);
            const float *kk = k + kh * hd;
            const float *vv = v + kh * hd;
            const float *qq = q + hh * hd;
            float *Sh = S + static_cast<size_t>(hh) * hd * hd;
            const float beta = 1.f / (1.f + std::exp(-kk[0]));
            std::vector<float> attn(static_cast<size_t>(hd), 0.f);
            for (int j = 0; j < hd; ++j) {
                float a = 0.f;
                for (int i = 0; i < hd; ++i)
                    a += Sh[static_cast<size_t>(i) * hd + j] * kk[i];
                attn[static_cast<size_t>(j)] = a;
            }
            for (int i = 0; i < hd; ++i)
                for (int j = 0; j < hd; ++j)
                    Sh[static_cast<size_t>(i) * hd + j] +=
                        beta * (kk[i] * vv[j] - kk[i] * attn[static_cast<size_t>(j)]);
            for (int j = 0; j < hd; ++j) {
                float o = 0.f;
                for (int i = 0; i < hd; ++i)
                    o += qq[i] * Sh[static_cast<size_t>(i) * hd + j];
                ctx[hh * hd + j] = o;
            }
        }
    }

    bool layer_expert_mats(int l) const {
        const int E = cfg_.moe.n_experts;
        const int I = std::max(cfg_.dense_intermediate, 1);
        const int H = std::max(cfg_.hidden, 1);
        const size_t gu = static_cast<size_t>(E) * I * H;
        const size_t dn = static_cast<size_t>(E) * H * I;
        return E > 0 && l >= 0 && l < static_cast<int>(e_gate_.size()) &&
               l < static_cast<int>(e_up_.size()) && l < static_cast<int>(e_down_.size()) &&
               e_gate_[static_cast<size_t>(l)].size() >= gu &&
               e_up_[static_cast<size_t>(l)].size() >= gu &&
               e_down_[static_cast<size_t>(l)].size() >= dn;
    }
    bool have_expert_mats() const {
        for (int l = 0; l < cfg_.n_layers; ++l)
            if (layer_expert_mats(l))
                return true;
        return false;
    }

    void apply_ffn(int l, float *h, float *nrm, float *g, float *u, float *d, int H, int I) {
        quant::rmsnorm(h, out_n_[static_cast<size_t>(l)].data(), nrm, H, cfg_.rms_eps);
        quant::matmul_f32(g, nrm, gate_[static_cast<size_t>(l)].data(), 1, H, I);
        quant::matmul_f32(u, nrm, up_[static_cast<size_t>(l)].data(), 1, H, I);
        quant::silu_mul(g, u, I);
        quant::matmul_f32(d, g, down_[static_cast<size_t>(l)].data(), 1, I, H);
        const int E = cfg_.moe.n_experts;
        const int topk = cfg_.moe.topk;
        const bool router_ok =
            E > 0 && topk > 0 && l >= 0 && l < static_cast<int>(router_.size()) &&
            router_[static_cast<size_t>(l)].size() >=
                static_cast<size_t>(E) * static_cast<size_t>(H);
        if (router_ok && layer_expert_mats(l)) {
            std::vector<float> scores(static_cast<size_t>(E));
            quant::matmul_f32(scores.data(), nrm, router_[static_cast<size_t>(l)].data(), 1, H, E);
            for (int i = 0; i < E; ++i)
                scores[static_cast<size_t>(i)] = quant::sigmoid(scores[static_cast<size_t>(i)]);
            int idx[16] = {};
            float ww[16] = {};
            int K = topk;
            if (K > 16)
                K = 16;
            if (K > E)
                K = E;
            if (K > 0)
                moe_topk(scores.data(), E, K, idx, ww, scores.data());
            std::vector<float> tmp(static_cast<size_t>(H)), ge(static_cast<size_t>(I)),
                ue(static_cast<size_t>(I));
            const size_t tile_gu = static_cast<size_t>(I) * static_cast<size_t>(H);
            const size_t tile_dn = static_cast<size_t>(H) * static_cast<size_t>(I);
            for (int k = 0; k < K; ++k) {
                const int e = idx[k];
                if (e < 0 || e >= E)
                    continue;
                const float *wg =
                    e_gate_[static_cast<size_t>(l)].data() + static_cast<size_t>(e) * tile_gu;
                const float *wu =
                    e_up_[static_cast<size_t>(l)].data() + static_cast<size_t>(e) * tile_gu;
                const float *wd =
                    e_down_[static_cast<size_t>(l)].data() + static_cast<size_t>(e) * tile_dn;
                quant::matmul_f32(ge.data(), nrm, wg, 1, H, I);
                quant::matmul_f32(ue.data(), nrm, wu, 1, H, I);
                quant::silu_mul(ge.data(), ue.data(), I);
                quant::matmul_f32(tmp.data(), ge.data(), wd, 1, I, H);
                for (int i = 0; i < H; ++i)
                    d[i] += ww[k] * tmp[static_cast<size_t>(i)];
            }
        } else if (router_ok) {
            std::vector<float> scores(static_cast<size_t>(E));
            quant::matmul_f32(scores.data(), nrm, router_[static_cast<size_t>(l)].data(), 1, H, E);
            for (int i = 0; i < E; ++i)
                scores[static_cast<size_t>(i)] = quant::sigmoid(scores[static_cast<size_t>(i)]);
            int idx[16] = {};
            float ww[16] = {};
            int K = topk;
            if (K > 16)
                K = 16;
            if (K > E)
                K = E;
            if (K > 0)
                moe_topk(scores.data(), E, K, idx, ww, scores.data());
            float mix = 0.f;
            for (int i = 0; i < K; ++i)
                mix += ww[i];
            for (int i = 0; i < H; ++i)
                d[i] *= mix;
        }
        for (int i = 0; i < H; ++i)
            h[i] += d[i];
    }

    bool layer_gdn(int l) const {
        return l >= 0 && l < (int)cfg_.is_kda.size() && cfg_.is_kda[l];
    }
    bool layer_gqa(int l) const {
        return l >= 0 && l < static_cast<int>(wk_.size()) && l < static_cast<int>(wv_.size()) &&
               !wk_[static_cast<size_t>(l)].empty() && !wv_[static_cast<size_t>(l)].empty();
    }
    bool has_gqa() const {
        for (int l = 0; l < cfg_.n_layers; ++l)
            if (layer_gqa(l))
                return true;
        return false;
    }
    bool has_gdn() const {
        for (int l = 0; l < cfg_.n_layers; ++l)
            if (layer_gdn(l))
                return true;
        return false;
    }

    void alloc_weights(bool gqa) {
        const int H = std::max(cfg_.hidden, 1);
        const int L = std::max(cfg_.n_layers, 0);
        const int V = std::max(cfg_.vocab, 1);
        const int I = std::max(cfg_.dense_intermediate, 1);
        cfg_.hidden = H;
        cfg_.n_layers = L;
        cfg_.vocab = V;
        cfg_.dense_intermediate = I;
        int qd = H;
        int kvd = H;
        if (gqa) {
            const int hd = std::max(cfg_.head_dim, 1);
            const int nq = std::max(cfg_.n_q_heads, 1);
            const int nkv = std::max(cfg_.n_kv_heads, 1);
            qd = nq * hd;
            kvd = nkv * hd;
            cfg_.head_dim = hd;
            cfg_.n_q_heads = nq;
            cfg_.n_kv_heads = nkv;
        }
        xavier(embed_, V, H, 5);
        ones(norm_, H);
        xavier(lm_head_, V, H, 6);
        in_n_.assign(static_cast<size_t>(L), {});
        out_n_.assign(static_cast<size_t>(L), {});
        wq_.assign(static_cast<size_t>(L), {});
        wk_.assign(static_cast<size_t>(L), {});
        wv_.assign(static_cast<size_t>(L), {});
        wo_.assign(static_cast<size_t>(L), {});
        gate_.assign(static_cast<size_t>(L), {});
        up_.assign(static_cast<size_t>(L), {});
        down_.assign(static_cast<size_t>(L), {});
        conv_q_.assign(static_cast<size_t>(L), {});
        conv_k_.assign(static_cast<size_t>(L), {});
        conv_v_.assign(static_cast<size_t>(L), {});
        router_.assign(static_cast<size_t>(L), {});
        e_gate_.assign(static_cast<size_t>(L), {});
        e_up_.assign(static_cast<size_t>(L), {});
        e_down_.assign(static_cast<size_t>(L), {});
        const int E = cfg_.moe.n_experts;
        const bool mats =
            E > 0 && E <= 16 &&
            static_cast<int64_t>(E) * static_cast<int64_t>(I) * static_cast<int64_t>(H) <= 2000000;
        for (int l = 0; l < L; ++l) {
            ones(in_n_[static_cast<size_t>(l)], H);
            ones(out_n_[static_cast<size_t>(l)], H);
            xavier(wq_[static_cast<size_t>(l)], qd, H, 400 + l);
            xavier(wo_[static_cast<size_t>(l)], H, qd, 410 + l);
            if (gqa) {
                xavier(wk_[static_cast<size_t>(l)], kvd, H, 450 + l);
                xavier(wv_[static_cast<size_t>(l)], kvd, H, 460 + l);
            }
            xavier(gate_[static_cast<size_t>(l)], I, H, 420 + l);
            xavier(up_[static_cast<size_t>(l)], I, H, 430 + l);
            xavier(down_[static_cast<size_t>(l)], H, I, 440 + l);
            if (E > 0 && E <= 64)
                xavier(router_[static_cast<size_t>(l)], E, H, 470 + l);
            if (mats) {
                xavier_tiles(e_gate_[static_cast<size_t>(l)], E, I, H, 480 + l * 17);
                xavier_tiles(e_up_[static_cast<size_t>(l)], E, I, H, 490 + l * 17);
                xavier_tiles(e_down_[static_cast<size_t>(l)], E, H, I, 500 + l * 17);
            }
            conv_q_[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * qd, 0.f);
            conv_k_[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * kvd, 0.f);
            conv_v_[static_cast<size_t>(l)].assign(static_cast<size_t>(4) * kvd, 0.f);
            for (int d = 0; d < qd; ++d)
                conv_q_[static_cast<size_t>(l)][static_cast<size_t>(d)] = 1.f;
            for (int d = 0; d < kvd; ++d) {
                conv_k_[static_cast<size_t>(l)][static_cast<size_t>(d)] = 1.f;
                conv_v_[static_cast<size_t>(l)][static_cast<size_t>(d)] = 1.f;
            }
        }
    }

    void alloc_synthetic() { alloc_weights(true); }

    Status overlay_f32(const std::vector<io::StFile> &files, const std::string &name,
                       std::vector<float> &dst, std::string &err) {
        io::StHit hit = io::st_find_dir(files, name);
        if (!hit.tensor)
            return Status::NotFound;
        int64_t n = 1;
        for (int64_t d : hit.tensor->shape)
            n *= d;
        if (n <= 0)
            return Status::NotFound;
        dst.assign(static_cast<size_t>(n), 0.f);
        return io::st_read_f32(*hit.file, *hit.tensor, dst.data(), n, err);
    }

    void infer_heads(const io::StTensor *q, const io::StTensor *k) {
        const int q_rows = shape0(q);
        const int k_rows = shape0(k);
        if (q_rows <= 0)
            return;
        auto apply = [&](int hd) {
            if (hd <= 0 || q_rows % hd != 0 || q_rows / hd < 1)
                return false;
            if (k_rows > 0 && (k_rows % hd != 0 || k_rows / hd < 1))
                return false;
            cfg_.head_dim = hd;
            cfg_.n_q_heads = q_rows / hd;
            if (k_rows > 0)
                cfg_.n_kv_heads = k_rows / hd;
            if (cfg_.n_kv_heads <= 0)
                cfg_.n_kv_heads = 1;
            return true;
        };
        // Config head_dim wins when it actually splits q (and k).
        if (cfg_.head_dim > 0 && apply(cfg_.head_dim))
            return;
        if (cfg_.n_q_heads > 0 && q_rows % cfg_.n_q_heads == 0 &&
            apply(q_rows / cfg_.n_q_heads))
            return;
        // Tiny fixtures: prefer more heads. Full-size Qwen36: 64/128.
        const int large[6] = {64, 128, 32, 16, 8, 4};
        const int small[6] = {4, 8, 16, 32, 64, 128};
        const int *use = cfg_.hidden >= 1024 ? large : small;
        for (int i = 0; i < 6; ++i) {
            if (apply(use[i]))
                return;
        }
        int g = q_rows;
        if (k_rows > 0) {
            int a = q_rows, b = k_rows;
            while (b) {
                int t = a % b;
                a = b;
                b = t;
            }
            g = std::max(a, 1);
        }
        apply(g);
        if (cfg_.n_q_heads <= 0)
            cfg_.n_q_heads = 1;
        if (cfg_.n_kv_heads <= 0)
            cfg_.n_kv_heads = 1;
        if (cfg_.head_dim <= 0)
            cfg_.head_dim = std::max(q_rows / std::max(cfg_.n_q_heads, 1), 1);
    }

    Status load_checkpoint(const std::string &model_dir, std::string &err) {
        std::vector<io::StFile> files;
        Status st = io::st_open_dir(model_dir, files, err);
        if (st != Status::Ok)
            return st;
        if (files.empty())
            return Status::NotFound;

        auto has = [&](const std::string &n) { return io::st_find_dir(files, n).tensor != nullptr; };

        std::string P;
        if (has("model.embed_tokens.weight"))
            P = "model.";
        else if (!has("embed_tokens.weight")) {
            io::st_close_dir(files);
            return Status::NotFound;
        }

        io::StHit emb = io::st_find_dir(files, P + "embed_tokens.weight");
        if (!emb.tensor || emb.tensor->shape.size() != 2 || emb.tensor->shape[0] <= 0 ||
            emb.tensor->shape[1] <= 0) {
            io::st_close_dir(files);
            return Status::NotFound;
        }
        cfg_.vocab = static_cast<int>(emb.tensor->shape[0]);
        cfg_.hidden = static_cast<int>(emb.tensor->shape[1]);

        int counted = 0;
        for (int i = 0; i < 128; ++i) {
            const std::string ly = P + "layers." + std::to_string(i);
            if (!has(ly + ".input_layernorm.weight") && !has(ly + ".self_attn.q_proj.weight"))
                break;
            counted = i + 1;
        }
        if (counted > 0)
            cfg_.n_layers = counted;

        io::StHit q0 = io::st_find_dir(files, P + "layers.0.self_attn.q_proj.weight");
        io::StHit k0 = io::st_find_dir(files, P + "layers.0.self_attn.k_proj.weight");
        io::StHit v0 = io::st_find_dir(files, P + "layers.0.self_attn.v_proj.weight");
        infer_heads(q0.tensor, k0.tensor);
        io::StHit g0 = io::st_find_dir(files, P + "layers.0.mlp.gate_proj.weight");
        if (g0.tensor && !g0.tensor->shape.empty() && g0.tensor->shape[0] > 0)
            cfg_.dense_intermediate = static_cast<int>(g0.tensor->shape[0]);

        // GQA only when K/V exist; otherwise keep the q→o stand-in (synth tests).
        const bool have_kv = k0.tensor != nullptr && v0.tensor != nullptr;
        alloc_weights(have_kv);

        bool got_embed = overlay_f32(files, P + "embed_tokens.weight", embed_, err) == Status::Ok;
        overlay_f32(files, P + "norm.weight", norm_, err);
        if (has("lm_head.weight"))
            overlay_f32(files, "lm_head.weight", lm_head_, err);
        else if (has(P + "lm_head.weight"))
            overlay_f32(files, P + "lm_head.weight", lm_head_, err);
        else if (got_embed)
            lm_head_ = embed_;

        bool got_q0 = false;
        const int L = cfg_.n_layers;
        for (int i = 0; i < L; ++i) {
            const std::string ly = P + "layers." + std::to_string(i);
            overlay_f32(files, ly + ".input_layernorm.weight", in_n_[static_cast<size_t>(i)], err);
            overlay_f32(files, ly + ".post_attention_layernorm.weight",
                        out_n_[static_cast<size_t>(i)], err);
            const bool qok =
                overlay_f32(files, ly + ".self_attn.q_proj.weight", wq_[static_cast<size_t>(i)],
                            err) == Status::Ok;
            if (i == 0)
                got_q0 = qok;
            overlay_f32(files, ly + ".self_attn.k_proj.weight", wk_[static_cast<size_t>(i)], err);
            overlay_f32(files, ly + ".self_attn.v_proj.weight", wv_[static_cast<size_t>(i)], err);
            overlay_f32(files, ly + ".self_attn.o_proj.weight", wo_[static_cast<size_t>(i)], err);
            overlay_f32(files, ly + ".mlp.gate_proj.weight", gate_[static_cast<size_t>(i)], err);
            overlay_f32(files, ly + ".mlp.up_proj.weight", up_[static_cast<size_t>(i)], err);
            overlay_f32(files, ly + ".mlp.down_proj.weight", down_[static_cast<size_t>(i)], err);
        }
        io::st_close_dir(files);
        err.clear();
        from_checkpoint_ = got_embed && got_q0;
        return Status::Ok;
    }

    ModelConfig cfg_{};
    RuntimeConfig rt_{};
    bool loaded_ = false;
    bool from_checkpoint_ = false;
    std::vector<float> embed_, norm_, lm_head_;
    std::vector<std::vector<float>> in_n_, out_n_, wq_, wk_, wv_, wo_, gate_, up_, down_;
    std::vector<std::vector<float>> conv_q_, conv_k_, conv_v_;
    std::vector<std::vector<float>> router_, e_gate_, e_up_, e_down_;
    Qwen36Slot slots_[kMaxKvSlots];
    double t_attn_ = 0, t_head_ = 0;
};

std::unique_ptr<FamilyEngine> make_qwen36() { return std::make_unique<Qwen36Engine>(); }

} // namespace mvllm
