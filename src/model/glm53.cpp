#include "family.hpp"
#include "../gpu/backend.hpp"
#include "../io/safetensors.hpp"
#include "../quant/quant.hpp"
#include "../quant/weight.hpp"
#include "../serve/mux_frames.hpp"
#include "../serve/session.hpp"
#include "../store/route_trace.hpp"
#include "../store/route_usage.hpp"
#include "../tok/decode_post.hpp"
#include "../tok/gbnf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
void ones(std::vector<float> &w, int n) { w.assign(n, 1.f); }

quant::QuantMat qmat_xavier(int O, int I, uint32_t seed, int bits) {
    std::vector<float> t;
    xavier(t, O, I, seed);
    quant::QuantMat m;
    m.from_f32(t.data(), O, I, bits);
    return m;
}

struct ExpertGeom {
    int64_t pack_go = 0, sc_go = 0, pack_d = 0, sc_d = 0, slot = 0;
};

ExpertGeom make_expert_geom(int hidden, int inter) {
    ExpertGeom g{};
    if (hidden > 0 && inter > 0 && hidden % 64 == 0 && inter % 64 == 0) {
        const int64_t pack = static_cast<int64_t>(inter) * hidden / 2;
        const int64_t sc = static_cast<int64_t>(inter) * hidden / 64 * static_cast<int64_t>(sizeof(float));
        g.pack_go = pack;
        g.sc_go = sc;
        g.pack_d = pack;
        g.sc_d = sc;
        g.slot = 3 * (pack + sc);
        return g;
    }
    g.pack_go = static_cast<int64_t>(inter) * ((hidden + 1) / 2);
    g.sc_go = static_cast<int64_t>(inter) * ((hidden + 63) / 64) * static_cast<int64_t>(sizeof(float));
    g.pack_d = static_cast<int64_t>(hidden) * ((inter + 1) / 2);
    g.sc_d = static_cast<int64_t>(hidden) * ((inter + 63) / 64) * static_cast<int64_t>(sizeof(float));
    g.slot = 2 * (g.pack_go + g.sc_go) + g.pack_d + g.sc_d;
    return g;
}

const char *kExpertPieces[6] = {
    "gate_proj.weight", "gate_proj.weight.qs", "up_proj.weight", "up_proj.weight.qs",
    "down_proj.weight", "down_proj.weight.qs",
};

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

class Glm53Engine final : public FamilyEngine {
public:
    Family family() const override { return Family::Glm53; }
    const ModelConfig &config() const override { return cfg_; }

    struct Glm53Slot {
        std::vector<int> history;
        std::vector<std::vector<float>> S, winq, wink, winv, mla_cache, dsa_ikeys, dsa_igates;
        std::vector<float> streams;
        std::vector<float> vis;
        int nvis = 0;
        int vis_i = 0;
        int vod = 0;
        int image_tok = -1;
        int pos = 0;
        bool have = false;
        bool live = false;
        GenParams gp;
        uint64_t rng = 1;
        int emitted = 0;
        Gbnf g;
        std::vector<uint8_t> allow;
        std::string decoded;
        std::vector<float> token_lps;
        std::vector<std::vector<GenLogprob>> top_lps;
    };

    Status load(const std::string &model_dir, const RuntimeConfig &rt, std::string &err) override {
        rt_ = rt;
        Status st = load_model_config(model_dir, cfg_, err);
        if (st != Status::Ok)
            return st;
        if (cfg_.family != Family::Glm53) {
            err = "not a GLM-5.3 config";
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
        Status ust = usage_.init("glm53", cfg_.n_layers, std::max(cfg_.moe.n_experts, 1), err);
        if (ust != Status::Ok)
            return ust;
        {
            const int ne = usage_.n_experts();
            const size_t n = static_cast<size_t>(usage_.n_layers() + 1) * static_cast<size_t>(ne);
            ehit_.assign(n, 0);
            turn_c_.assign(n, 0u);
        }
        for (int l = 0; l < cfg_.first_dense; ++l)
            usage_.drop_row(l);
        std::string uerr;
        usage_.load(rt_.model_dir + "/.coli_usage", false, uerr);
        const char *tpath = std::getenv("MVLLM_ROUTE_TRACE");
        if (!tpath || !tpath[0])
            tpath = std::getenv("COLI_ROUTE_TRACE");
        if (tpath && tpath[0]) {
            std::string terr;
            trace_.open(tpath, terr);
        }
        return Status::Ok;
    }

    Status generate(const std::vector<int> &prompt, const GenParams &gp, GenResult &out,
                    std::string &err) override {
        if (!loaded_) {
            err = "glm53 not loaded";
            return Status::InvalidArgument;
        }
        if (prompt.empty()) {
            err = "empty prompt";
            return Status::InvalidArgument;
        }
        const int cslot = gp.cache_slot;
        const bool use_slot = cslot >= 0 && cslot < kMaxKvSlots;
        Glm53Slot local;
        Glm53Slot *work = use_slot ? &slots_[cslot] : &local;
        int Tmax = std::max(rt_.max_seq, static_cast<int>(prompt.size()) + gp.max_new_tokens + 256);
        if (Tmax <= 0)
            Tmax = 4096;

        std::vector<int> ids = prompt;
        int vis_keep = -1;
        int reuse = 0;
        if (use_slot && gp.prefix_reuse > 0) {
            Glm53Slot &sl = slots_[cslot];
            const int match = sl.have ? kv_common_prefix(sl.history, prompt) : 0;
            const int want = std::min(gp.prefix_reuse, match);
            if (sl.have && sl.pos > 0 && sl.pos <= want &&
                sl.pos <= static_cast<int>(prompt.size())) {
                reuse = sl.pos;
                vis_keep = sl.vis_i;
                slot_ensure_t(sl, Tmax);
            }
        }
        if (reuse == 0)
            slot_alloc(*work, Tmax);
        slot_prep_vision(*work, gp);
        if (vis_keep >= 0)
            work->vis_i = vis_keep;
        expand_image_ids(*work, ids);
        if (reuse > 0 && reuse > static_cast<int>(ids.size()))
            reuse = 0;
        Status pst = slot_prefill(*work, ids, reuse, err);
        if (pst != Status::Ok)
            return pst;

        out.tokens.clear();
        out.token_logprobs.clear();
        out.top_logprobs.clear();
        out.prompt_tokens = static_cast<int>(prompt.size());
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
        std::string decoded;
        work->history = prompt;
        work->token_lps.clear();
        work->top_lps.clear();
        for (int ntok = 0; ntok < gp.max_new_tokens; ++ntok) {
            int next = slot_sample(*work, gp, &rng, allow.empty() ? nullptr : allow.data());
            out.tokens.push_back(next);
            work->history.push_back(next);
            if (g.ready() && gp.token_text)
                g.accept_bytes(gp.token_text(next));
            if (gp.on_token)
                gp.on_token(next);
            bool stop = gen_stop_id(next, cfg_, gp);
            if (!stop && gp.token_text) {
                decoded += gp.token_text(next);
                if (stop_cut(decoded, gp.stop) != std::string::npos) {
                    stop = true;
                    out.stopped_by_stop = true;
                }
            }
            if (stop)
                break;
            Status st = slot_step(*work, next, err);
            if (st != Status::Ok)
                return st;
            if (g.ready() && gp.token_text) {
                allow.assign(cfg_.vocab, 0);
                g.allow_mask(gp.token_text, allow.data(), cfg_.vocab);
            }
        }
        out.completion_tokens = static_cast<int>(out.tokens.size());
        out.token_logprobs = work->token_lps;
        out.top_logprobs = work->top_lps;
        if (use_slot) {
            Glm53Slot &sl = slots_[cslot];
            sl.history = prompt;
            sl.history.insert(sl.history.end(), out.tokens.begin(), out.tokens.end());
            sl.have = true;
            sl.live = false;
        }
        std::string serr;
        usage_.save(rt_.model_dir + "/.coli_usage", serr);
        return Status::Ok;
    }

    std::string describe() const override {
        std::ostringstream os;
        int mla_n = 0;
        for (int l = 0; l < cfg_.n_layers; ++l) {
            const bool full = (l < static_cast<int>(cfg_.is_full.size())) ? cfg_.is_full[l] : 0;
            if (full && l < static_cast<int>(mla_kva_.size()) && !mla_kva_[l].empty())
                ++mla_n;
        }
        os << "glm53  hidden=" << cfg_.hidden << " layers=" << cfg_.n_layers
           << " experts=" << cfg_.moe.n_experts << " topk=" << cfg_.moe.topk
           << " mhc=" << cfg_.mhc.mult << " vision=" << (cfg_.vision.layers > 0 ? "yes" : "no")
           << " kv_lora=" << cfg_.mla.kv_lora
           << " checkpoint=" << (from_checkpoint_ ? "yes" : "synthetic")
           << " bits=" << rt_.dense_bits << " head_bits=" << rt_.head_bits
           << " prefix=" << (prefix_.empty() ? "-" : prefix_) << " mla=" << mla_n
           << " prefill=layer"
           << " dsa=" << (cfg_.dsa.topk > 0 ? cfg_.dsa.topk : 0);
        return os.str();
    }

    void expert_stats(ExpertStoreStats &out) const override { store_.stats(out); }

    void route_telem(RouteTelem &out, bool consume_hits) override {
        if (!loaded_ || usage_.n_experts() < 1) {
            out = {};
            return;
        }
        const int cols = usage_.n_experts();
        const int nL = usage_.n_layers();
        std::vector<int> live;
        live.reserve(static_cast<size_t>(nL + 1));
        for (int l = 0; l <= nL; ++l) {
            if (!usage_.has_row(l))
                continue;
            if (l == nL) {
                bool any = false;
                for (int e = 0; e < cols; ++e) {
                    if (usage_.get(l, e)) {
                        any = true;
                        break;
                    }
                    const size_t i =
                        static_cast<size_t>(l) * static_cast<size_t>(cols) + static_cast<size_t>(e);
                    if ((i < ehit_.size() && ehit_[i]) || (i < turn_c_.size() && turn_c_[i])) {
                        any = true;
                        break;
                    }
                }
                if (!any)
                    continue;
            }
            live.push_back(l);
        }
        const int rows = static_cast<int>(live.size());
        const size_t cells = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        out.rows = rows;
        out.cols = cols;
        out.emap.assign(cells, 0);
        out.hits.assign(cells, 0);
        out.entropy.assign(static_cast<size_t>(rows), 0.f);
        out.vram = 0;
        out.vram_gb = 0;
        int ram = 0;
        int disk = 0;
        for (int r = 0; r < rows; ++r) {
            const int l = live[static_cast<size_t>(r)];
            uint64_t sum = 0;
            for (int e = 0; e < cols; ++e) {
                const size_t dst =
                    static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(e);
                const size_t src =
                    static_cast<size_t>(l) * static_cast<size_t>(cols) + static_cast<size_t>(e);
                const int tier = store_.resident(l, e) ? 1 : 0;
                out.emap[dst] = mux_emap_byte(tier, usage_.get(l, e));
                out.hits[dst] = (src < ehit_.size()) ? ehit_[src] : 0;
                if (tier)
                    ++ram;
                else
                    ++disk;
                if (src < turn_c_.size())
                    sum += turn_c_[src];
            }
            if (sum == 0)
                continue;
            double H = 0.0;
            for (int e = 0; e < cols; ++e) {
                const size_t src =
                    static_cast<size_t>(l) * static_cast<size_t>(cols) + static_cast<size_t>(e);
                const uint32_t c = (src < turn_c_.size()) ? turn_c_[src] : 0u;
                if (!c)
                    continue;
                const double p = static_cast<double>(c) / static_cast<double>(sum);
                H -= p * std::log2(p);
            }
            out.entropy[static_cast<size_t>(r)] = static_cast<float>(H);
        }
        out.ram = ram;
        out.disk = disk;
        out.ram_gb = static_cast<double>(store_.expert_bytes()) * static_cast<double>(ram) / 1e9;
        if (consume_hits) {
            std::fill(ehit_.begin(), ehit_.end(), 0);
            std::fill(turn_c_.begin(), turn_c_.end(), 0u);
        }
    }

    Status begin_generate(int slot, const std::vector<int> &ids, const GenParams &gp, int &reuse,
                          std::string &err) override {
        if (!loaded_) {
            err = "glm53 not loaded";
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
        Glm53Slot &st = slots_[slot];
        const int match = st.have ? kv_common_prefix(st.history, ids) : 0;
        const int want = std::min(std::max(gp.prefix_reuse, 0), match);
        int applied = 0;
        int Tmax = std::max(rt_.max_seq, static_cast<int>(ids.size()) + gp.max_new_tokens + 256);
        if (Tmax <= 0)
            Tmax = 4096;
        int vis_keep = -1;
        if (st.have && st.pos > 0 && st.pos <= want && st.pos <= static_cast<int>(ids.size())) {
            applied = st.pos;
            vis_keep = st.vis_i;
            slot_ensure_t(st, Tmax);
        } else {
            slot_alloc(st, Tmax);
        }
        slot_prep_vision(st, gp);
        if (vis_keep >= 0)
            st.vis_i = vis_keep;
        std::vector<int> fed = ids;
        expand_image_ids(st, fed);
        Status pst = slot_prefill(st, fed, applied, err);
        if (pst != Status::Ok)
            return pst;
        st.history = ids;
        st.live = true;
        st.have = true;
        st.gp = gp;
        st.rng = gp.seed ? gp.seed : 1ull;
        st.emitted = 0;
        st.allow.clear();
        st.decoded.clear();
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
        Glm53Slot &st = slots_[slot];
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
        bool stop = gen_stop_id(token, cfg_, st.gp) || st.emitted >= st.gp.max_new_tokens;
        if (!stop && st.gp.token_text) {
            st.decoded += st.gp.token_text(token);
            if (stop_cut(st.decoded, st.gp.stop) != std::string::npos)
                stop = true;
        }
        if (stop) {
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
        std::vector<Glm53Slot *> step_s;
        std::vector<int> step_tok;
        step_s.reserve(static_cast<size_t>(n));
        step_tok.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            Glm53Slot &st = slots_[slots[i]];
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
            bool stop = gen_stop_id(tokens[i], cfg_, st.gp) ||
                        st.emitted >= st.gp.max_new_tokens;
            if (!stop && st.gp.token_text) {
                st.decoded += st.gp.token_text(tokens[i]);
                if (stop_cut(st.decoded, st.gp.stop) != std::string::npos)
                    stop = true;
            }
            if (stop)
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

    // L/R: layer-major from mla_cache[l][pos]. I: concat DSA keys of is_full layers.
    int export_kv_rows(int slot, int pos0, int n, KvPersistRecord *rows) const override {
        if (!rows || n <= 0 || slot < 0 || slot >= kMaxKvSlots)
            return 0;
        const Glm53Slot &sl = slots_[slot];
        if (!sl.have)
            return 0;
        const int kvL = std::max(cfg_.mla.kv_lora, 0);
        const int qr = std::max(cfg_.mla.qk_rope, 0);
        const int stride = kvL + qr;
        const int ID = std::max(cfg_.dsa.head_dim, 0);
        const int L = cfg_.n_layers;
        for (int t = 0; t < n; ++t) {
            const int pos = pos0 + t;
            if (pos < 0 || pos >= sl.pos)
                return t;
            KvPersistRecord &rec = rows[t];
            if (stride > 0) {
                for (int l = 0; l < L; ++l) {
                    if (l >= static_cast<int>(sl.mla_cache.size()))
                        break;
                    const std::vector<float> &cache = sl.mla_cache[l];
                    const size_t src = static_cast<size_t>(pos) * static_cast<size_t>(stride);
                    if (src + static_cast<size_t>(stride) > cache.size())
                        continue;
                    if (kvL > 0) {
                        const size_t dst = static_cast<size_t>(l) * static_cast<size_t>(kvL);
                        if (dst + static_cast<size_t>(kvL) <= rec.L.size())
                            std::memcpy(rec.L.data() + dst, cache.data() + src,
                                        static_cast<size_t>(kvL) * sizeof(float));
                    }
                    if (qr > 0) {
                        const size_t dst = static_cast<size_t>(l) * static_cast<size_t>(qr);
                        if (dst + static_cast<size_t>(qr) <= rec.R.size())
                            std::memcpy(rec.R.data() + dst, cache.data() + src + kvL,
                                        static_cast<size_t>(qr) * sizeof(float));
                    }
                }
            }
            // Persist I is has_index / is_full layers in layer order; empty is_full = no index.
            if (ID > 0 && !cfg_.is_full.empty()) {
                size_t ioff = 0;
                for (int l = 0; l < L; ++l) {
                    const bool full =
                        (l < static_cast<int>(cfg_.is_full.size())) ? cfg_.is_full[l] : 0;
                    if (!full)
                        continue;
                    if (ioff + static_cast<size_t>(ID) <= rec.I.size() &&
                        l < static_cast<int>(sl.dsa_ikeys.size())) {
                        const std::vector<float> &ik = sl.dsa_ikeys[l];
                        const size_t koff = static_cast<size_t>(pos) * static_cast<size_t>(ID);
                        if (koff + static_cast<size_t>(ID) <= ik.size())
                            std::memcpy(rec.I.data() + ioff, ik.data() + koff,
                                        static_cast<size_t>(ID) * sizeof(float));
                    }
                    ioff += static_cast<size_t>(ID);
                }
            }
        }
        return n;
    }

    // Inverse of export: restore mla_cache / dsa_ikeys / history from persist rows.
    int import_kv_rows(int slot, int pos0, int n, const KvPersistRecord *rows) override {
        if (!rows || n <= 0 || slot < 0 || slot >= kMaxKvSlots)
            return 0;
        Glm53Slot &sl = slots_[slot];
        const int kvL = std::max(cfg_.mla.kv_lora, 0);
        const int qr = std::max(cfg_.mla.qk_rope, 0);
        const int stride = kvL + qr;
        const int ID = std::max(cfg_.dsa.head_dim, 0);
        const int L = cfg_.n_layers;
        const int headroom = std::max(rt_.max_seq, 256);
        const int t_max = std::max(std::max(pos0 + n, sl.pos), 1) + headroom;
        bool caches_small = static_cast<int>(sl.mla_cache.size()) < L;
        if (!caches_small && stride > 0) {
            const size_t need = static_cast<size_t>(std::max(pos0 + n, 1)) * static_cast<size_t>(stride);
            for (int l = 0; l < L; ++l) {
                if (sl.mla_cache[l].size() < need) {
                    caches_small = true;
                    break;
                }
            }
        }
        if (ID > 0 && cfg_.dsa.topk > 0 && static_cast<int>(sl.dsa_ikeys.size()) < L)
            caches_small = true;
        if (!sl.have || caches_small)
            slot_alloc(sl, t_max);
        else
            slot_ensure_t(sl, t_max);

        int written = 0;
        for (int t = 0; t < n; ++t) {
            const int pos = pos0 + t;
            if (pos < 0)
                continue;
            const KvPersistRecord &rec = rows[t];
            if (stride > 0) {
                for (int l = 0; l < L; ++l) {
                    if (l >= static_cast<int>(sl.mla_cache.size()) || sl.mla_cache[l].empty())
                        continue;
                    std::vector<float> &cache = sl.mla_cache[l];
                    const size_t dst0 = static_cast<size_t>(pos) * static_cast<size_t>(stride);
                    if (dst0 + static_cast<size_t>(stride) > cache.size())
                        continue;
                    float *dst = cache.data() + dst0;
                    if (kvL > 0) {
                        const size_t src = static_cast<size_t>(l) * static_cast<size_t>(kvL);
                        if (src < rec.L.size()) {
                            const int ncopy = std::min(kvL, static_cast<int>(rec.L.size() - src));
                            if (ncopy > 0)
                                std::memcpy(dst, rec.L.data() + src,
                                            static_cast<size_t>(ncopy) * sizeof(float));
                        }
                    }
                    if (qr > 0) {
                        const size_t src = static_cast<size_t>(l) * static_cast<size_t>(qr);
                        if (src < rec.R.size()) {
                            const int ncopy = std::min(qr, static_cast<int>(rec.R.size() - src));
                            if (ncopy > 0)
                                std::memcpy(dst + kvL, rec.R.data() + src,
                                            static_cast<size_t>(ncopy) * sizeof(float));
                        }
                    }
                }
            }
            if (ID > 0 && !cfg_.is_full.empty()) {
                size_t ioff = 0;
                for (int l = 0; l < L; ++l) {
                    const bool full =
                        (l < static_cast<int>(cfg_.is_full.size())) ? cfg_.is_full[l] : 0;
                    if (!full)
                        continue;
                    if (ioff + static_cast<size_t>(ID) <= rec.I.size() &&
                        l < static_cast<int>(sl.dsa_ikeys.size())) {
                        std::vector<float> &ik = sl.dsa_ikeys[l];
                        const size_t koff = static_cast<size_t>(pos) * static_cast<size_t>(ID);
                        if (koff + static_cast<size_t>(ID) <= ik.size())
                            std::memcpy(ik.data() + koff, rec.I.data() + ioff,
                                        static_cast<size_t>(ID) * sizeof(float));
                    }
                    ioff += static_cast<size_t>(ID);
                }
            }
            if (static_cast<int>(sl.history.size()) <= pos)
                sl.history.resize(static_cast<size_t>(pos) + 1);
            sl.history[static_cast<size_t>(pos)] = rec.token;
            written = t + 1;
        }
        sl.pos = std::max(sl.pos, pos0 + written);
        sl.have = sl.pos > 0;
        return written;
    }

private:
    int mhc_mult() const { return cfg_.mhc.mult > 0 ? cfg_.mhc.mult : 1; }

    int kv_stride() const {
        return std::max(cfg_.mla.kv_lora, 0) + std::max(cfg_.mla.qk_rope, 0);
    }

    bool official_hc(int l, bool ffn) const {
        const int M = mhc_mult();
        const auto &fn = ffn ? hc_ffn_fn_ : hc_attn_fn_;
        const auto &base = ffn ? hc_ffn_base_ : hc_attn_base_;
        const auto &sc = ffn ? hc_ffn_scale_ : hc_attn_scale_;
        return M > 1 && l >= 0 && l < static_cast<int>(fn.size()) && !fn[l].empty() &&
               !base[l].empty() && sc[l].size() >= 3;
    }

    void apply_mhc(float *st, int l) const {
        const int M = mhc_mult();
        const int H = cfg_.hidden;
        if (M > 1 && l >= 0 && l < static_cast<int>(mhc_alpha_.size()) && !mhc_alpha_[l].empty())
            mhc_mix(st, H, M, mhc_alpha_[l].data(), cfg_.mhc.iters, cfg_.mhc.eps);
    }

    void hc_enter(float *st, int l, bool ffn, float *collapsed, float *post, float *comb) const {
        const int M = mhc_mult();
        const int H = cfg_.hidden;
        const auto &fn = ffn ? hc_ffn_fn_[l] : hc_attn_fn_[l];
        const auto &base = ffn ? hc_ffn_base_[l] : hc_attn_base_[l];
        const auto &sc = ffn ? hc_ffn_scale_[l] : hc_attn_scale_[l];
        mhc_pre(collapsed, post, comb, st, fn.data(), sc.data(), base.data(), M, H, cfg_.mhc.iters,
                cfg_.rms_eps, cfg_.mhc.eps);
    }

    void slot_prep_vision(Glm53Slot &s, const GenParams &gp) {
        const int H = cfg_.hidden;
        s.vod = !vit_proj_.empty() ? vit_proj_.O : H;
        s.nvis = 0;
        s.vis_i = 0;
        s.vis.clear();
        s.image_tok = gp.image_token >= 0 ? gp.image_token : cfg_.vision.image_token;
        if (gp.image_rgb && gp.image_w > 0 && gp.image_h > 0 && gp.image_token >= 0 &&
            (!vit_patch_.empty() || cfg_.vision.layers > 0 || cfg_.vision.hidden > 0)) {
            const int cap = 64;
            s.vis.assign(static_cast<size_t>(cap) * std::max(s.vod, 1), 0.f);
            s.nvis = glm_vit_embed(gp.image_rgb, gp.image_w, gp.image_h, cfg_.vision,
                                   vit_patch_.empty() ? nullptr : &vit_patch_,
                                   vit_proj_.empty() ? nullptr : &vit_proj_, s.vis.data(), cap,
                                   vit_.ready() ? &vit_ : nullptr);
        }
    }

    void expand_image_ids(const Glm53Slot &s, std::vector<int> &ids) const {
        if (s.nvis > 1 && s.image_tok >= 0) {
            std::vector<int> exp;
            exp.reserve(ids.size() + static_cast<size_t>(s.nvis));
            for (int id : ids) {
                if (id == s.image_tok)
                    for (int k = 0; k < s.nvis; ++k)
                        exp.push_back(id);
                else
                    exp.push_back(id);
            }
            ids.swap(exp);
        }
    }

    void slot_embed(Glm53Slot &s, int id) {
        const int H = cfg_.hidden;
        const int M = mhc_mult();
        if (static_cast<int>(s.streams.size()) < M * H)
            s.streams.assign(static_cast<size_t>(M) * H, 0.f);
        float *h = s.streams.data();
        int tid = id;
        if (tid < 0 || tid >= cfg_.vocab)
            tid = 0;
        if (s.nvis > 0 && id == s.image_tok) {
            const float *src = s.vis.data() + static_cast<size_t>(s.vis_i % s.nvis) * s.vod;
            std::memcpy(h, src, static_cast<size_t>(std::min(H, s.vod)) * sizeof(float));
            if (H > s.vod)
                std::memset(h + s.vod, 0, static_cast<size_t>(H - s.vod) * sizeof(float));
            ++s.vis_i;
        } else {
            std::memcpy(h, embed_.data() + static_cast<size_t>(tid) * H, H * sizeof(float));
        }
        const bool rep = official_hc(0, false);
        for (int m = 1; m < M; ++m) {
            if (rep)
                std::memcpy(s.streams.data() + static_cast<size_t>(m) * H, h,
                            static_cast<size_t>(H) * sizeof(float));
            else
                std::memset(s.streams.data() + static_cast<size_t>(m) * H, 0,
                            static_cast<size_t>(H) * sizeof(float));
        }
    }

    void attn_one(Glm53Slot &s, float *hh, int at, int l, float *branch) {
        const int H = cfg_.hidden;
        std::vector<float> n(static_cast<size_t>(H)), y(static_cast<size_t>(H), 0.f);
        quant::rmsnorm(hh, in_n_[l].data(), n.data(), H, cfg_.rms_eps);
        bool full = (l < static_cast<int>(cfg_.is_full.size())) ? cfg_.is_full[l] : 0;
        if (!full && cfg_.kda.heads > 0) {
            kda_step(n.data(), H, cfg_.kda, &wq_[l], &wk_[l], &wv_[l], &wb_[l], &wfa_[l], &wfb_[l],
                     wdt_[l].data(), static_cast<int>(wdt_[l].size()), alog_[l].data(), &wg_[l],
                     (l < static_cast<int>(wgb_.size()) && !wgb_[l].empty()) ? &wgb_[l] : nullptr,
                     &wo_[l], on_[l].empty() ? nullptr : on_[l].data(), s.S[l].data(), y.data(),
                     cfg_.rms_eps,
                     l < static_cast<int>(conv_q_.size()) && !conv_q_[l].empty() ? conv_q_[l].data()
                                                                                : nullptr,
                     l < static_cast<int>(conv_k_.size()) && !conv_k_[l].empty() ? conv_k_[l].data()
                                                                                : nullptr,
                     l < static_cast<int>(conv_v_.size()) && !conv_v_[l].empty() ? conv_v_[l].data()
                                                                                : nullptr,
                     s.winq[l].data(), s.wink[l].data(), s.winv[l].data());
        } else {
            const int *sel = nullptr;
            int nsel = 0;
            std::vector<int> selbuf;
            if (dsa_live(l) && at >= 0) {
                std::vector<float> qn(static_cast<size_t>(std::max(cfg_.mla.q_lora, 1)));
                if (!mla_qa_[l].empty()) {
                    mla_qa_[l].gemm(qn.data(), n.data(), 1);
                    if (!mla_qa_ln_[l].empty())
                        quant::rmsnorm(qn.data(), mla_qa_ln_[l].data(), qn.data(), cfg_.mla.q_lora,
                                       cfg_.rms_eps);
                }
                const int ID = std::max(cfg_.dsa.head_dim, 1);
                const int IH = std::max(cfg_.dsa.n_heads, 1);
                std::vector<float> iq(static_cast<size_t>(IH) * ID, 0.f), hw(static_cast<size_t>(IH),
                                                                            0.f);
                if (!dsa_wq_[l].empty())
                    dsa_wq_[l].gemm(iq.data(), qn.data(), 1);
                std::vector<float> &ikeys =
                    (l < static_cast<int>(s.dsa_ikeys.size()) && !s.dsa_ikeys[l].empty())
                        ? s.dsa_ikeys[l]
                        : dsa_ikeys_[l];
                std::vector<float> &igates =
                    (l < static_cast<int>(s.dsa_igates.size()) && !s.dsa_igates[l].empty())
                        ? s.dsa_igates[l]
                        : dsa_igates_[l];
                if (!dsa_wk_[l].empty() && !ikeys.empty()) {
                    std::vector<float> kraw(static_cast<size_t>(ID), 0.f);
                    dsa_wk_[l].gemm(kraw.data(), n.data(), 1);
                    layernorm(kraw.data(), dsa_knw_[l].empty() ? nullptr : dsa_knw_[l].data(),
                              dsa_knb_[l].empty() ? nullptr : dsa_knb_[l].data(),
                              ikeys.data() + static_cast<size_t>(at) * ID, ID, 1e-5f);
                }
                if (!dsa_kg_[l].empty() && !igates.empty())
                    dsa_kg_[l].gemm(igates.data() + static_cast<size_t>(at) * ID, n.data(), 1);
                if (!dsa_wp_[l].empty()) {
                    dsa_wp_[l].gemm(hw.data(), n.data(), 1);
                    float sc = 1.f / std::sqrt(static_cast<float>(IH));
                    for (int i = 0; i < IH; ++i)
                        hw[static_cast<size_t>(i)] *= sc;
                }
                nsel = dsa_index_width(cfg_.dsa);
                selbuf.assign(static_cast<size_t>(std::max(nsel, 1)), -1);
                // Decode: official range over current token; queries/head_w already [0].
                if (dsa_select_range(selbuf.data(), iq.data(),
                                     ikeys.empty() ? nullptr : ikeys.data(),
                                     igates.empty() ? nullptr : igates.data(), hw.data(),
                                     dsa_ape_[l].empty() ? nullptr : dsa_ape_[l].data(), nullptr,
                                     at + 1, cfg_.dsa, at, at + 1) < 0)
                    dsa_select(selbuf.data(), iq.data(),
                               ikeys.empty() ? nullptr : ikeys.data(),
                               igates.empty() ? nullptr : igates.data(), hw.data(),
                               dsa_ape_[l].empty() ? nullptr : dsa_ape_[l].data(), at + 1,
                               cfg_.dsa);
                sel = selbuf.data();
            }
            mla_step(n.data(), H, cfg_.mla, &mla_qa_[l],
                     mla_qa_ln_[l].empty() ? nullptr : mla_qa_ln_[l].data(), &mla_qb_[l],
                     &mla_kva_[l], mla_kva_ln_[l].empty() ? nullptr : mla_kva_ln_[l].data(),
                     &mla_kt_[l], &mla_v_[l], &mla_o_[l], &mla_g_[l],
                     s.mla_cache[l].empty() ? nullptr : s.mla_cache[l].data(), at, y.data(),
                     cfg_.rms_eps, sel, nsel);
        }
        if (branch)
            std::memcpy(branch, y.data(), static_cast<size_t>(H) * sizeof(float));
        else {
            for (int i = 0; i < H; ++i)
                hh[i] += y[i];
        }
    }

    Status ffn_one(float *hh, int l, std::string &err) {
        const int H = cfg_.hidden;
        std::vector<float> n(static_cast<size_t>(H));
        quant::rmsnorm(hh, out_n_[l].data(), n.data(), H, cfg_.rms_eps);
        if (l < cfg_.first_dense) {
            dense_mlp(l, n.data(), hh);
            return Status::Ok;
        }
        return moe_layer(l, n.data(), hh, err);
    }

    void slot_alloc(Glm53Slot &s, int t_max) {
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int M = mhc_mult();
        const int kd = std::max(cfg_.kda.head_dim, 1);
        const int kh = std::max(cfg_.kda.heads, 1);
        const int P = kh * kd;
        const int Kc = cfg_.kda.conv_k > 0 ? cfg_.kda.conv_k : 4;
        const int kvStride = kv_stride();
        const int tcap = std::max(t_max, 1);
        s.streams.assign(static_cast<size_t>(M) * H, 0.f);
        s.S.assign(static_cast<size_t>(L), {});
        s.winq.assign(static_cast<size_t>(L), {});
        s.wink.assign(static_cast<size_t>(L), {});
        s.winv.assign(static_cast<size_t>(L), {});
        s.mla_cache.assign(static_cast<size_t>(L), {});
        s.dsa_ikeys.assign(static_cast<size_t>(L), {});
        s.dsa_igates.assign(static_cast<size_t>(L), {});
        for (int l = 0; l < L; ++l) {
            s.S[l].assign(static_cast<size_t>(kh) * kd * kd, 0.f);
            s.winq[l].assign(static_cast<size_t>(P) * Kc, 0.f);
            s.wink[l].assign(static_cast<size_t>(P) * Kc, 0.f);
            s.winv[l].assign(static_cast<size_t>(P) * Kc, 0.f);
            if (kvStride > 0)
                s.mla_cache[l].assign(static_cast<size_t>(tcap) * kvStride, 0.f);
            if (cfg_.dsa.topk > 0 && cfg_.dsa.head_dim > 0) {
                s.dsa_ikeys[l].assign(static_cast<size_t>(tcap) * cfg_.dsa.head_dim, 0.f);
                s.dsa_igates[l].assign(static_cast<size_t>(tcap) * cfg_.dsa.head_dim, 0.f);
            }
        }
        s.pos = 0;
        s.have = false;
        s.live = false;
        s.history.clear();
        s.emitted = 0;
        s.allow.clear();
        s.decoded.clear();
        s.token_lps.clear();
        s.top_lps.clear();
        s.g = Gbnf{};
        s.nvis = 0;
        s.vis_i = 0;
        s.vis.clear();
        s.vod = 0;
        s.image_tok = -1;
    }

    void slot_ensure_t(Glm53Slot &s, int t_max) {
        const int L = cfg_.n_layers;
        const int kvStride = kv_stride();
        const int tcap = std::max(t_max, 1);
        if (static_cast<int>(s.mla_cache.size()) < L)
            s.mla_cache.resize(static_cast<size_t>(L));
        if (static_cast<int>(s.dsa_ikeys.size()) < L)
            s.dsa_ikeys.resize(static_cast<size_t>(L));
        if (static_cast<int>(s.dsa_igates.size()) < L)
            s.dsa_igates.resize(static_cast<size_t>(L));
        if (kvStride > 0) {
            const size_t need = static_cast<size_t>(tcap) * kvStride;
            for (int l = 0; l < L; ++l) {
                if (s.mla_cache[l].size() < need)
                    s.mla_cache[l].resize(need, 0.f);
            }
        }
        if (cfg_.dsa.topk > 0 && cfg_.dsa.head_dim > 0) {
            const size_t need = static_cast<size_t>(tcap) * cfg_.dsa.head_dim;
            for (int l = 0; l < L; ++l) {
                if (s.dsa_ikeys[l].size() < need)
                    s.dsa_ikeys[l].resize(need, 0.f);
                if (s.dsa_igates[l].size() < need)
                    s.dsa_igates[l].resize(need, 0.f);
            }
        }
    }

    Status slot_step(Glm53Slot &s, int token, std::string &err) {
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int M = mhc_mult();
        slot_embed(s, token);
        float *h = s.streams.data();
        for (int l = 0; l < L; ++l) {
            if (official_hc(l, false)) {
                std::vector<float> collapsed(static_cast<size_t>(H)), post(static_cast<size_t>(M)),
                    comb(static_cast<size_t>(M) * M), branch(static_cast<size_t>(H), 0.f);
                hc_enter(s.streams.data(), l, false, collapsed.data(), post.data(), comb.data());
                attn_one(s, collapsed.data(), s.pos, l, branch.data());
                mhc_post(s.streams.data(), branch.data(), s.streams.data(), post.data(),
                         comb.data(), M, H);
            } else {
                apply_mhc(s.streams.data(), l);
                attn_one(s, h, s.pos, l, nullptr);
            }
            if (official_hc(l, true)) {
                std::vector<float> collapsed(static_cast<size_t>(H)), post(static_cast<size_t>(M)),
                    comb(static_cast<size_t>(M) * M), branch(static_cast<size_t>(H), 0.f);
                hc_enter(s.streams.data(), l, true, collapsed.data(), post.data(), comb.data());
                std::vector<float> n(static_cast<size_t>(H));
                quant::rmsnorm(collapsed.data(), out_n_[l].data(), n.data(), H, cfg_.rms_eps);
                if (l < cfg_.first_dense)
                    dense_mlp(l, n.data(), branch.data());
                else {
                    Status st = moe_layer(l, n.data(), branch.data(), err);
                    if (st != Status::Ok)
                        return st;
                }
                mhc_post(s.streams.data(), branch.data(), s.streams.data(), post.data(),
                         comb.data(), M, H);
            } else {
                Status st = ffn_one(h, l, err);
                if (st != Status::Ok)
                    return st;
                apply_mhc(s.streams.data(), l);
            }
        }
        ++s.pos;
        return Status::Ok;
    }

    Status slot_step_n(Glm53Slot **ss, const int *tokens, int S, std::string &err) {
        if (S <= 0)
            return Status::Ok;
        if (S == 1)
            return slot_step(*ss[0], tokens[0], err);
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int M = mhc_mult();
        for (int c = 0; c < S; ++c)
            slot_embed(*ss[c], tokens[c]);
        for (int l = 0; l < L; ++l) {
            for (int c = 0; c < S; ++c) {
                Glm53Slot &s = *ss[c];
                float *st = s.streams.data();
                if (official_hc(l, false)) {
                    std::vector<float> collapsed(static_cast<size_t>(H)),
                        post(static_cast<size_t>(M)), comb(static_cast<size_t>(M) * M),
                        branch(static_cast<size_t>(H), 0.f);
                    hc_enter(st, l, false, collapsed.data(), post.data(), comb.data());
                    attn_one(s, collapsed.data(), s.pos, l, branch.data());
                    mhc_post(st, branch.data(), st, post.data(), comb.data(), M, H);
                } else {
                    apply_mhc(st, l);
                    attn_one(s, st, s.pos, l, nullptr);
                }
            }
            if (official_hc(l, true)) {
                if (l < cfg_.first_dense) {
                    for (int c = 0; c < S; ++c) {
                        Glm53Slot &s = *ss[c];
                        std::vector<float> collapsed(static_cast<size_t>(H)),
                            post(static_cast<size_t>(M)), comb(static_cast<size_t>(M) * M),
                            branch(static_cast<size_t>(H), 0.f);
                        hc_enter(s.streams.data(), l, true, collapsed.data(), post.data(),
                                 comb.data());
                        std::vector<float> n(static_cast<size_t>(H));
                        quant::rmsnorm(collapsed.data(), out_n_[l].data(), n.data(), H,
                                       cfg_.rms_eps);
                        dense_mlp(l, n.data(), branch.data());
                        mhc_post(s.streams.data(), branch.data(), s.streams.data(), post.data(),
                                 comb.data(), M, H);
                    }
                } else {
                    std::vector<float> norms(static_cast<size_t>(S) * H),
                        branches(static_cast<size_t>(S) * H, 0.f);
                    std::vector<float> posts(static_cast<size_t>(S) * M),
                        combs(static_cast<size_t>(S) * M * M);
                    for (int c = 0; c < S; ++c) {
                        Glm53Slot &s = *ss[c];
                        std::vector<float> collapsed(static_cast<size_t>(H));
                        hc_enter(s.streams.data(), l, true, collapsed.data(),
                                 posts.data() + static_cast<size_t>(c) * M,
                                 combs.data() + static_cast<size_t>(c) * M * M);
                        quant::rmsnorm(collapsed.data(), out_n_[l].data(),
                                       norms.data() + static_cast<size_t>(c) * H, H, cfg_.rms_eps);
                    }
                    Status st = moe_layer_n(l, norms.data(), branches.data(), S, err);
                    if (st != Status::Ok)
                        return st;
                    for (int c = 0; c < S; ++c) {
                        Glm53Slot &s = *ss[c];
                        mhc_post(s.streams.data(), branches.data() + static_cast<size_t>(c) * H,
                                 s.streams.data(), posts.data() + static_cast<size_t>(c) * M,
                                 combs.data() + static_cast<size_t>(c) * M * M, M, H);
                    }
                }
            } else if (l < cfg_.first_dense) {
                for (int c = 0; c < S; ++c) {
                    float *hh = ss[c]->streams.data();
                    std::vector<float> n(static_cast<size_t>(H));
                    quant::rmsnorm(hh, out_n_[l].data(), n.data(), H, cfg_.rms_eps);
                    dense_mlp(l, n.data(), hh);
                }
                for (int c = 0; c < S; ++c)
                    apply_mhc(ss[c]->streams.data(), l);
            } else {
                std::vector<float> norms(static_cast<size_t>(S) * H),
                    s0(static_cast<size_t>(S) * H);
                for (int c = 0; c < S; ++c) {
                    float *hh = ss[c]->streams.data();
                    std::memcpy(s0.data() + static_cast<size_t>(c) * H, hh,
                                static_cast<size_t>(H) * sizeof(float));
                    quant::rmsnorm(hh, out_n_[l].data(), norms.data() + static_cast<size_t>(c) * H,
                                   H, cfg_.rms_eps);
                }
                Status st = moe_layer_n(l, norms.data(), s0.data(), S, err);
                if (st != Status::Ok)
                    return st;
                for (int c = 0; c < S; ++c) {
                    std::memcpy(ss[c]->streams.data(), s0.data() + static_cast<size_t>(c) * H,
                                static_cast<size_t>(H) * sizeof(float));
                    apply_mhc(ss[c]->streams.data(), l);
                }
            }
        }
        for (int c = 0; c < S; ++c)
            ++ss[c]->pos;
        return Status::Ok;
    }

    Status slot_prefill(Glm53Slot &s, const std::vector<int> &ids, int start, std::string &err) {
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int M = mhc_mult();
        if (start < 0)
            start = 0;
        if (start >= static_cast<int>(ids.size()))
            return Status::Ok;
        const int chunk = rt_.prefill_chunk > 0 ? rt_.prefill_chunk
                                                : static_cast<int>(ids.size()) - start;
        for (size_t i = static_cast<size_t>(start); i < ids.size();) {
            const int C = static_cast<int>(
                std::min(static_cast<size_t>(chunk), ids.size() - i));
            std::vector<float> act(static_cast<size_t>(C) * M * H, 0.f);
            for (int c = 0; c < C; ++c) {
                int tid = ids[i + static_cast<size_t>(c)];
                const int raw = tid;
                if (tid < 0 || tid >= cfg_.vocab)
                    tid = 0;
                float *dst = act.data() + static_cast<size_t>(c) * M * H;
                if (s.nvis > 0 && raw == s.image_tok) {
                    const float *src =
                        s.vis.data() + static_cast<size_t>(s.vis_i % s.nvis) * s.vod;
                    std::memcpy(dst, src,
                                static_cast<size_t>(std::min(H, s.vod)) * sizeof(float));
                    if (H > s.vod)
                        std::memset(dst + s.vod, 0,
                                    static_cast<size_t>(H - s.vod) * sizeof(float));
                    ++s.vis_i;
                } else {
                    std::memcpy(dst, embed_.data() + static_cast<size_t>(tid) * H,
                                static_cast<size_t>(H) * sizeof(float));
                }
                if (official_hc(0, false)) {
                    for (int m = 1; m < M; ++m)
                        std::memcpy(dst + static_cast<size_t>(m) * H, dst,
                                    static_cast<size_t>(H) * sizeof(float));
                }
            }
            for (int l = 0; l < L; ++l) {
                for (int c = 0; c < C; ++c) {
                    float *st = act.data() + static_cast<size_t>(c) * M * H;
                    if (official_hc(l, false)) {
                        std::vector<float> collapsed(static_cast<size_t>(H)),
                            post(static_cast<size_t>(M)), comb(static_cast<size_t>(M) * M),
                            branch(static_cast<size_t>(H), 0.f);
                        hc_enter(st, l, false, collapsed.data(), post.data(), comb.data());
                        attn_one(s, collapsed.data(), s.pos + c, l, branch.data());
                        mhc_post(st, branch.data(), st, post.data(), comb.data(), M, H);
                    } else {
                        apply_mhc(st, l);
                        attn_one(s, st, s.pos + c, l, nullptr);
                    }
                }
                if (official_hc(l, true)) {
                    for (int c = 0; c < C; ++c) {
                        float *st = act.data() + static_cast<size_t>(c) * M * H;
                        std::vector<float> collapsed(static_cast<size_t>(H)),
                            post(static_cast<size_t>(M)), comb(static_cast<size_t>(M) * M),
                            branch(static_cast<size_t>(H), 0.f);
                        hc_enter(st, l, true, collapsed.data(), post.data(), comb.data());
                        std::vector<float> n(static_cast<size_t>(H));
                        quant::rmsnorm(collapsed.data(), out_n_[l].data(), n.data(), H,
                                       cfg_.rms_eps);
                        if (l < cfg_.first_dense)
                            dense_mlp(l, n.data(), branch.data());
                        else {
                            Status mst = moe_layer(l, n.data(), branch.data(), err);
                            if (mst != Status::Ok)
                                return mst;
                        }
                        mhc_post(st, branch.data(), st, post.data(), comb.data(), M, H);
                    }
                } else if (l < cfg_.first_dense) {
                    for (int c = 0; c < C; ++c) {
                        float *hh = act.data() + static_cast<size_t>(c) * M * H;
                        std::vector<float> n(static_cast<size_t>(H));
                        quant::rmsnorm(hh, out_n_[l].data(), n.data(), H, cfg_.rms_eps);
                        dense_mlp(l, n.data(), hh);
                    }
                    for (int c = 0; c < C; ++c)
                        apply_mhc(act.data() + static_cast<size_t>(c) * M * H, l);
                } else {
                    std::vector<float> norms(static_cast<size_t>(C) * H),
                        s0(static_cast<size_t>(C) * H);
                    for (int c = 0; c < C; ++c) {
                        float *hh = act.data() + static_cast<size_t>(c) * M * H;
                        std::memcpy(s0.data() + static_cast<size_t>(c) * H, hh,
                                    static_cast<size_t>(H) * sizeof(float));
                        quant::rmsnorm(hh, out_n_[l].data(),
                                       norms.data() + static_cast<size_t>(c) * H, H,
                                       cfg_.rms_eps);
                    }
                    Status mst = moe_layer_n(l, norms.data(), s0.data(), C, err);
                    if (mst != Status::Ok)
                        return mst;
                    for (int c = 0; c < C; ++c)
                        std::memcpy(act.data() + static_cast<size_t>(c) * M * H,
                                    s0.data() + static_cast<size_t>(c) * H,
                                    static_cast<size_t>(H) * sizeof(float));
                    for (int c = 0; c < C; ++c)
                        apply_mhc(act.data() + static_cast<size_t>(c) * M * H, l);
                }
            }
            std::memcpy(s.streams.data(), act.data() + static_cast<size_t>(C - 1) * M * H,
                        static_cast<size_t>(M) * H * sizeof(float));
            s.pos += C;
            i += static_cast<size_t>(C);
        }
        return Status::Ok;
    }

    int slot_sample(Glm53Slot &s, const GenParams &gp, uint64_t *rng, const uint8_t *allow) {
        const int H = cfg_.hidden;
        const int M = mhc_mult();
        std::vector<float> n(static_cast<size_t>(H)), logits(static_cast<size_t>(cfg_.vocab)),
            pooled(static_cast<size_t>(H));
        const float *head_in = s.streams.data();
        if (official_hc(0, false)) {
            std::fill(pooled.begin(), pooled.end(), 0.f);
            for (int m = 0; m < M; ++m)
                for (int i = 0; i < H; ++i)
                    pooled[static_cast<size_t>(i)] +=
                        s.streams[static_cast<size_t>(m) * H + i];
            for (int i = 0; i < H; ++i)
                pooled[static_cast<size_t>(i)] /= static_cast<float>(M);
            head_in = pooled.data();
        }
        quant::rmsnorm(head_in, norm_.data(), n.data(), H, cfg_.rms_eps);
        lm_head_.gemm(logits.data(), n.data(), 1);
        return sample_penalized(logits.data(), cfg_.vocab, gp, s.history.data(),
                               static_cast<int>(s.history.size()), rng, allow, nullptr,
                               &s.token_lps, &s.top_lps);
    }

    int64_t expert_bytes() const {
        return make_expert_geom(cfg_.hidden, cfg_.moe.intermediate > 0 ? cfg_.moe.intermediate : 32)
            .slot;
    }

    void fill_int4_expert(uint8_t *p, int layer, int eid) {
        const int I = cfg_.hidden;
        const int O = cfg_.moe.intermediate;
        auto g = make_expert_geom(I, O);
        std::vector<float> gw(static_cast<size_t>(O) * I, 0.03f);
        std::vector<float> uw(static_cast<size_t>(O) * I, 0.03f);
        std::vector<float> dw(static_cast<size_t>(I) * O, 0.03f);
        for (size_t i = 0; i < gw.size(); ++i)
            gw[i] = 0.03f * (((layer * 3 + eid + static_cast<int>(i)) & 5) - 2);
        uint8_t *cur = p;
        quant::quantize_int4_g64(gw.data(), O, I, cur, reinterpret_cast<float *>(cur + g.pack_go));
        cur += g.pack_go + g.sc_go;
        quant::quantize_int4_g64(uw.data(), O, I, cur, reinterpret_cast<float *>(cur + g.pack_go));
        cur += g.pack_go + g.sc_go;
        quant::quantize_int4_g64(dw.data(), I, O, cur, reinterpret_cast<float *>(cur + g.pack_d));
    }

    Status write_synthetic_experts(const std::string &model_dir, std::string &err) {
        const int64_t ebytes = expert_bytes();
        const std::string epath = model_dir + "/.mvllm_glm53_experts.bin";
        std::ofstream out(epath, std::ios::binary | std::ios::trunc);
        if (!out) {
            err = "cannot write synthetic experts";
            return Status::IoError;
        }
        std::vector<uint8_t> blob(static_cast<size_t>(ebytes));
        for (int l = 0; l < cfg_.n_layers; ++l) {
            if (l < cfg_.first_dense)
                continue;
            for (int e = 0; e < cfg_.moe.n_experts; ++e) {
                fill_int4_expert(blob.data(), l, e);
                out.write(reinterpret_cast<const char *>(blob.data()),
                          static_cast<std::streamsize>(ebytes));
                if (!out) {
                    err = "expert blob write failed";
                    return Status::IoError;
                }
            }
        }
        out.close();
        int64_t off = 0;
        for (int l = 0; l < cfg_.n_layers; ++l) {
            if (l < cfg_.first_dense)
                continue;
            for (int e = 0; e < cfg_.moe.n_experts; ++e) {
                ExpertLoc loc;
                loc.key = {l, e};
                loc.path = epath;
                loc.offset = off;
                loc.bytes = ebytes;
                Status st = store_.register_expert(loc, err);
                if (st != Status::Ok)
                    return st;
                off += ebytes;
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

        prefix_ = "model.language_model.";
        if (!has(prefix_ + "embed_tokens.weight"))
            prefix_ = "model.";
        if (!has(prefix_ + "embed_tokens.weight")) {
            prefix_.clear();
            if (!has("embed_tokens.weight") && !has("model.embed_tokens.weight")) {
                io::st_close_dir(files);
                return Status::NotFound;
            }
            if (has("model.embed_tokens.weight"))
                prefix_ = "model.";
        }

        const std::string P = prefix_;
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int V = cfg_.vocab;
        overlay_f32(files, P + "embed_tokens.weight", embed_, V * H, err);
        overlay_f32(files, P + "norm.weight", norm_, H, err);
        const int bits = rt_.dense_bits;
        const int hbits = rt_.head_bits;
        const int mbits = rt_.mla_bits;
        if (has("lm_head.weight"))
            overlay_mat(files, "lm_head.weight", lm_head_, hbits, err);
        else if (has(P + "lm_head.weight"))
            overlay_mat(files, P + "lm_head.weight", lm_head_, hbits, err);
        else if (!embed_.empty())
            lm_head_.from_f32(embed_.data(), V, H, hbits);

        for (int i = 0; i < L; ++i) {
            overlay_f32(files, P + "layers." + std::to_string(i) + ".input_layernorm.weight",
                        in_n_[i], H, err);
            overlay_f32(files,
                        P + "layers." + std::to_string(i) + ".post_attention_layernorm.weight",
                        out_n_[i], H, err);
            const int Pdim = std::max(cfg_.kda.heads * cfg_.kda.head_dim, 1);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.q_proj.weight",
                        wq_[i], bits, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.k_proj.weight",
                        wk_[i], bits, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.v_proj.weight",
                        wv_[i], bits, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.o_proj.weight",
                        wo_[i], bits, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.q_a_proj.weight",
                        mla_qa_[i], mbits, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".self_attn.q_a_layernorm.weight",
                        mla_qa_ln_[i], cfg_.mla.q_lora, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.q_b_proj.weight",
                        mla_qb_[i], mbits, err);
            overlay_mat(files,
                        P + "layers." + std::to_string(i) + ".self_attn.kv_a_proj_with_mqa.weight",
                        mla_kva_[i], mbits, err);
            overlay_f32(files,
                        P + "layers." + std::to_string(i) + ".self_attn.kv_a_layernorm.weight",
                        mla_kva_ln_[i], cfg_.mla.kv_lora, err);
            overlay_kvb(files, P + "layers." + std::to_string(i) + ".self_attn.kv_b_proj.weight",
                        mla_kt_[i], mla_v_[i], err);
            overlay_mat(files,
                        P + "layers." + std::to_string(i) + ".self_attn.indexer.wq_b.weight",
                        dsa_wq_[i], mbits, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.indexer.wk.weight",
                        dsa_wk_[i], mbits, err);
            overlay_mat(files,
                        P + "layers." + std::to_string(i) +
                            ".self_attn.indexer.weights_proj.weight",
                        dsa_wp_[i], mbits, err);
            overlay_mat(files,
                        P + "layers." + std::to_string(i) +
                            ".self_attn.indexer.index_kpool_compress_gate",
                        dsa_kg_[i], mbits, err);
            overlay_f32(files,
                        P + "layers." + std::to_string(i) + ".self_attn.indexer.k_norm.weight",
                        dsa_knw_[i], 0, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".self_attn.indexer.k_norm.bias",
                        dsa_knb_[i], 0, err);
            overlay_f32(files,
                        P + "layers." + std::to_string(i) +
                            ".self_attn.indexer.index_kpool_compress_ape",
                        dsa_ape_[i], 0, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.o_proj.weight",
                        mla_o_[i], hbits, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.g_proj.weight",
                        mla_g_[i], hbits, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.g_a_proj.weight",
                        wg_[i], bits, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.g_b_proj.weight",
                        wgb_[i], bits, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.f_a_proj.weight",
                        wfa_[i], bits, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.f_b_proj.weight",
                        wfb_[i], bits, err);
            overlay_mat(files, P + "layers." + std::to_string(i) + ".self_attn.b_proj.weight",
                        wb_[i], bits, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".self_attn.dt_bias", wdt_[i],
                        Pdim, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".self_attn.A_log", alog_[i],
                        cfg_.kda.heads, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".self_attn.o_norm.weight",
                        on_[i], cfg_.kda.head_dim, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".hc_attn_fn", hc_attn_fn_[i],
                        0, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".hc_attn_base",
                        hc_attn_base_[i], 0, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".hc_attn_scale",
                        hc_attn_scale_[i], 0, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".hc_ffn_fn", hc_ffn_fn_[i], 0,
                        err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".hc_ffn_base", hc_ffn_base_[i],
                        0, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".hc_ffn_scale",
                        hc_ffn_scale_[i], 0, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".self_attn.q_conv1d.weight",
                        conv_q_[i], 0, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".self_attn.k_conv1d.weight",
                        conv_k_[i], 0, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".self_attn.v_conv1d.weight",
                        conv_v_[i], 0, err);
            {
                const int Mn = cfg_.mhc.mult > 0 ? cfg_.mhc.mult : 1;
                const int expect = Mn * Mn;
                const std::string ly = P + "layers." + std::to_string(i) + ".";
                const char *hc_names[] = {"mhc.weight", "mhc.alpha", "hyper_connection.weight",
                                          "hc.alpha", "input_mhc.weight", nullptr};
                for (int n = 0; hc_names[n]; ++n) {
                    if (overlay_f32(files, ly + hc_names[n], mhc_alpha_[i], expect, err) ==
                        Status::Ok)
                        break;
                }
            }
            if (i < cfg_.first_dense) {
                overlay_mat(files, P + "layers." + std::to_string(i) + ".mlp.gate_proj.weight",
                            mlp_gate_[i], bits, err);
                overlay_mat(files, P + "layers." + std::to_string(i) + ".mlp.up_proj.weight",
                            mlp_up_[i], bits, err);
                overlay_mat(files, P + "layers." + std::to_string(i) + ".mlp.down_proj.weight",
                            mlp_down_[i], bits, err);
            } else {
                overlay_f32(files, P + "layers." + std::to_string(i) + ".mlp.gate.weight",
                            router_[i], cfg_.moe.n_experts * H, err);
                overlay_f32(files,
                            P + "layers." + std::to_string(i) +
                                ".mlp.gate.e_score_correction_bias",
                            router_bias_[i], cfg_.moe.n_experts, err);
                overlay_mat(files,
                            P + "layers." + std::to_string(i) +
                                ".mlp.shared_experts.gate_proj.weight",
                            shared_gate_[i], bits, err);
                overlay_mat(files,
                            P + "layers." + std::to_string(i) +
                                ".mlp.shared_experts.up_proj.weight",
                            shared_up_[i], bits, err);
                overlay_mat(files,
                            P + "layers." + std::to_string(i) +
                                ".mlp.shared_experts.down_proj.weight",
                            shared_down_[i], bits, err);
            }
        }
        {
            const char *pnames[] = {"model.visual.patch_embed.proj.weight",
                                    "model.vision.patch_embed.proj.weight",
                                    "visual.patch_embed.proj.weight", nullptr};
            for (int n = 0; pnames[n]; ++n) {
                if (overlay_mat(files, pnames[n], vit_patch_, bits, err) == Status::Ok)
                    break;
            }
            const char *mnames[] = {"model.visual.merger.proj.weight",
                                    "model.vision.merger.proj.weight",
                                    "visual.merger.proj.weight", nullptr};
            for (int n = 0; mnames[n]; ++n) {
                if (overlay_mat(files, mnames[n], vit_proj_, bits, err) == Status::Ok)
                    break;
            }
            const char *V = has("model.visual.patch_embed.proj.weight") ? "model.visual."
                            : has("model.vision.patch_embed.proj.weight") ? "model.vision."
                                                                         : "visual.";
            vit_.cfg = cfg_.vision;
            if (vit_.cfg.heads <= 0)
                vit_.cfg.heads = 1;
            if (vit_.cfg.hidden <= 0 && !vit_patch_.empty())
                vit_.cfg.hidden = vit_patch_.O;
            if (vit_.cfg.out_hidden <= 0)
                vit_.cfg.out_hidden = cfg_.hidden;
            overlay_f32(files, std::string(V) + "patch_embed.proj.weight", vit_.patch_w, 0, err);
            overlay_f32(files, std::string(V) + "patch_embed.proj.bias", vit_.patch_b, 0, err);
            overlay_f32(files, std::string(V) + "post_layernorm.weight", vit_.post_norm, 0, err);
            overlay_f32(files, std::string(V) + "downsample.weight", vit_.down_w, 0, err);
            overlay_f32(files, std::string(V) + "downsample.bias", vit_.down_b, 0, err);
            overlay_f32(files, std::string(V) + "merger.proj.weight", vit_.merger_proj, 0, err);
            overlay_f32(files, std::string(V) + "merger.post_projection_norm.weight",
                        vit_.merger_norm_w, 0, err);
            overlay_f32(files, std::string(V) + "merger.post_projection_norm.bias",
                        vit_.merger_norm_b, 0, err);
            overlay_f32(files, std::string(V) + "merger.gate_proj.weight", vit_.merger_gate, 0, err);
            overlay_f32(files, std::string(V) + "merger.up_proj.weight", vit_.merger_up, 0, err);
            overlay_f32(files, std::string(V) + "merger.down_proj.weight", vit_.merger_down, 0, err);
            const int depth = cfg_.vision.layers > 0 ? cfg_.vision.layers : 0;
            vit_.blocks.resize(static_cast<size_t>(depth));
            for (int b = 0; b < depth; ++b) {
                const std::string B = std::string(V) + "blocks." + std::to_string(b) + ".";
                overlay_f32(files, B + "norm1.weight", vit_.blocks[b].norm1, 0, err);
                overlay_f32(files, B + "norm2.weight", vit_.blocks[b].norm2, 0, err);
                overlay_f32(files, B + "attn.qkv.weight", vit_.blocks[b].qkv_w, 0, err);
                overlay_f32(files, B + "attn.qkv.bias", vit_.blocks[b].qkv_b, 0, err);
                overlay_f32(files, B + "attn.q_norm.weight", vit_.blocks[b].q_norm, 0, err);
                overlay_f32(files, B + "attn.k_norm.weight", vit_.blocks[b].k_norm, 0, err);
                overlay_f32(files, B + "attn.proj.weight", vit_.blocks[b].proj_w, 0, err);
                overlay_f32(files, B + "attn.proj.bias", vit_.blocks[b].proj_b, 0, err);
                overlay_f32(files, B + "mlp.gate_proj.weight", vit_.blocks[b].gate_w, 0, err);
                overlay_f32(files, B + "mlp.gate_proj.bias", vit_.blocks[b].gate_b, 0, err);
                overlay_f32(files, B + "mlp.up_proj.weight", vit_.blocks[b].up_w, 0, err);
                overlay_f32(files, B + "mlp.up_proj.bias", vit_.blocks[b].up_b, 0, err);
                overlay_f32(files, B + "mlp.down_proj.weight", vit_.blocks[b].down_w, 0, err);
                overlay_f32(files, B + "mlp.down_proj.bias", vit_.blocks[b].down_b, 0, err);
            }
        }
        err.clear();

        const int probe_l = cfg_.first_dense < L ? cfg_.first_dense : 0;
        const std::string first = P + "layers." + std::to_string(probe_l) +
                                  ".mlp.experts.0.gate_proj.weight";
        io::StHit probe = io::st_find_dir(files, first);
        if (!probe.tensor) {
            // Dense-only overlay; keep synthetic experts.
            io::st_close_dir(files);
            from_checkpoint_ = true;
            return write_synthetic_experts(model_dir, err);
        }

        const bool streaming = (probe.tensor->dtype == "U8" || probe.tensor->dtype == "I8");
        const auto geom = make_expert_geom(H, cfg_.moe.intermediate);
        int registered = 0;
        for (int i = cfg_.first_dense; i < L; ++i) {
            for (int e = 0; e < cfg_.moe.n_experts; ++e) {
                ExpertLoc loc;
                loc.key = {i, e};
                if (streaming) {
                    const int64_t expect[6] = {geom.pack_go, geom.sc_go, geom.pack_go, geom.sc_go,
                                               geom.pack_d, geom.sc_d};
                    bool ok = true;
                    for (int p = 0; p < 6; ++p) {
                        const std::string n = P + "layers." + std::to_string(i) + ".mlp.experts." +
                                              std::to_string(e) + "." + kExpertPieces[p];
                        io::StHit hit = io::st_find_dir(files, n);
                        if (!hit.tensor || io::st_nbytes(*hit.tensor) != expect[p]) {
                            ok = false;
                            break;
                        }
                        ExpertPiece ep;
                        ep.path = hit.file->path;
                        ep.offset = io::st_file_offset(*hit.file, *hit.tensor);
                        ep.bytes = expect[p];
                        loc.pieces.push_back(ep);
                    }
                    if (!ok) {
                        err = "glm53 expert pieces missing or wrong size at L" + std::to_string(i) +
                              " E" + std::to_string(e);
                        io::st_close_dir(files);
                        return Status::ParseError;
                    }
                    loc.contig = true;
                    for (int p = 1; p < 6; ++p) {
                        if (loc.pieces[p].path != loc.pieces[0].path ||
                            loc.pieces[p].offset != loc.pieces[p - 1].offset + loc.pieces[p - 1].bytes)
                            loc.contig = false;
                    }
                    if (loc.contig) {
                        loc.path = loc.pieces[0].path;
                        loc.offset = loc.pieces[0].offset;
                        loc.bytes = geom.slot;
                        loc.pieces.clear();
                    }
                } else {
                    // BF16/F32 oracle: quantize to int4 and append to a pack file.
                    err = "glm53 f32 expert checkpoint: use convert_glm53.py or a U8 container";
                    io::st_close_dir(files);
                    return Status::Unsupported;
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

    void dense_mlp(int layer, const float *x, float *h) {
        const int H = cfg_.hidden;
        const int I = cfg_.dense_intermediate;
        std::vector<float> g(I), u(I), d(H);
        mlp_gate_[layer].gemm(g.data(), x, 1);
        mlp_up_[layer].gemm(u.data(), x, 1);
        for (int i = 0; i < I; ++i)
            g[i] = quant::clamped_swiglu(g[i], u[i], cfg_.moe.swiglu_limit);
        mlp_down_[layer].gemm(d.data(), g.data(), 1);
        for (int i = 0; i < H; ++i)
            h[i] += d[i];
    }

    Status moe_layer(int layer, const float *x, float *h, std::string &err) {
        return moe_layer_n(layer, x, h, 1, err);
    }

    Status moe_layer_n(int layer, const float *xs, float *hs, int C, std::string &err) {
        if (C <= 0)
            return Status::Ok;
        const int H = cfg_.hidden;
        const int O = cfg_.moe.intermediate;
        const int K = std::max(cfg_.moe.topk, 0);
        std::vector<int> idx(static_cast<size_t>(C) * std::max(K, 1), -1);
        std::vector<float> wt(static_cast<size_t>(C) * std::max(K, 1), 0.f);
        std::string terr;
        for (int c = 0; c < C; ++c) {
            std::vector<float> scores(cfg_.moe.n_experts), choice(cfg_.moe.n_experts);
            quant::matmul_f32(scores.data(), xs + static_cast<size_t>(c) * H, router_[layer].data(),
                              1, H, cfg_.moe.n_experts);
            for (int i = 0; i < cfg_.moe.n_experts; ++i) {
                scores[i] = quant::sigmoid(scores[i]);
                float b = i < static_cast<int>(router_bias_[layer].size()) ? router_bias_[layer][i]
                                                                          : 0.f;
                choice[i] = scores[i] + b;
            }
            moe_topk(choice.data(), cfg_.moe.n_experts, K, idx.data() + static_cast<size_t>(c) * K,
                     wt.data() + static_cast<size_t>(c) * K, scores.data());
            const int *ids = idx.data() + static_cast<size_t>(c) * K;
            usage_.count(layer, ids, K);
            mark_hits(layer, ids, K);
            trace_.emit(c, layer, ids, wt.data() + static_cast<size_t>(c) * K, K, terr);
        }
        trace_.end();
        std::vector<int> uniq(static_cast<size_t>(std::max(cfg_.moe.n_experts, 1)));
        int nu = moe_union_ids(idx.data(), C, K, uniq.data(), static_cast<int>(uniq.size()));
        std::vector<ExpertKey> keys(static_cast<size_t>(nu));
        for (int i = 0; i < nu; ++i)
            keys[static_cast<size_t>(i)] = {layer, uniq[i]};
        std::vector<float> acc(static_cast<size_t>(C) * H, 0.f);
        std::vector<float> xb(static_cast<size_t>(C) * H), yb(static_cast<size_t>(C) * H);
        std::vector<float> ww(static_cast<size_t>(C));
        std::vector<int> cmap(static_cast<size_t>(C));
        for (int ui = 0; ui < nu; ++ui) {
            const int eid = uniq[ui];
            ExpertView v{};
            Status st = store_.lookup({layer, eid}, v, err);
            if (st != Status::Ok) {
                std::string perr;
                store_.wait_prefetch(perr);
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
                std::memcpy(xb.data() + static_cast<size_t>(n) * H, xs + static_cast<size_t>(c) * H,
                            static_cast<size_t>(H) * sizeof(float));
                ww[static_cast<size_t>(n)] = wsum;
                cmap[static_cast<size_t>(n)] = c;
                ++n;
            }
            if (n > 0) {
                gpu::glm_expert(yb.data(), xb.data(), n, v.data, H, O, cfg_.moe.swiglu_limit);
                for (int i = 0; i < n; ++i) {
                    float *ac = acc.data() + static_cast<size_t>(cmap[static_cast<size_t>(i)]) * H;
                    const float *d = yb.data() + static_cast<size_t>(i) * H;
                    const float wsum = ww[static_cast<size_t>(i)];
                    for (int j = 0; j < H; ++j)
                        ac[j] += wsum * d[j];
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
            const float *x = xs + static_cast<size_t>(c) * H;
            float *ac = acc.data() + static_cast<size_t>(c) * H;
            float *h = hs + static_cast<size_t>(c) * H;
            // Official: h += routed * scale + shared (do not scale the shared expert).
            for (int i = 0; i < H; ++i)
                h[i] += ac[i] * cfg_.moe.routed_scale;
            if (!shared_gate_[layer].empty()) {
                std::vector<float> sg(O), su(O), sd(H);
                shared_gate_[layer].gemm(sg.data(), x, 1);
                shared_up_[layer].gemm(su.data(), x, 1);
                for (int i = 0; i < O; ++i)
                    sg[i] = quant::clamped_swiglu(sg[i], su[i], cfg_.moe.swiglu_limit);
                shared_down_[layer].gemm(sd.data(), sg.data(), 1);
                for (int i = 0; i < H; ++i)
                    h[i] += sd[i];
            }
        }
        return Status::Ok;
    }

    void alloc_synthetic() {
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int V = cfg_.vocab > 0 ? cfg_.vocab : 64;
        cfg_.vocab = V;
        if (cfg_.kda.heads == 0)
            cfg_.kda.heads = 2;
        if (cfg_.kda.head_dim == 0)
            cfg_.kda.head_dim = 16;
        if (cfg_.dense_intermediate == 0)
            cfg_.dense_intermediate = 64;
        if (cfg_.moe.intermediate == 0)
            cfg_.moe.intermediate = 32;
        if (cfg_.mla.n_heads == 0)
            cfg_.mla.n_heads = cfg_.n_q_heads > 0 ? cfg_.n_q_heads : 2;
        if (cfg_.mla.v_head == 0)
            cfg_.mla.v_head = cfg_.mla.qk_nope;
        const int P = cfg_.kda.heads * cfg_.kda.head_dim;
        const int bits = rt_.dense_bits;
        const int hbits = rt_.head_bits;
        const int mbits = rt_.mla_bits;
        const int nh = cfg_.mla.n_heads;
        const int qk = cfg_.mla.qk_nope;
        const int qr = cfg_.mla.qk_rope;
        const int vh = cfg_.mla.v_head;
        const int ql = cfg_.mla.q_lora;
        const int kv = cfg_.mla.kv_lora;
        xavier(embed_, V, H, 3);
        ones(norm_, H);
        lm_head_ = qmat_xavier(V, H, 4, hbits);
        in_n_.resize(L);
        out_n_.resize(L);
        wq_.resize(L);
        wk_.resize(L);
        wv_.resize(L);
        wo_.resize(L);
        wg_.resize(L);
        wgb_.resize(L);
        wfa_.resize(L);
        wfb_.resize(L);
        wdt_.resize(L);
        alog_.resize(L);
        wb_.resize(L);
        on_.resize(L);
        conv_q_.resize(L);
        conv_k_.resize(L);
        conv_v_.resize(L);
        router_.resize(L);
        router_bias_.resize(L);
        mlp_gate_.resize(L);
        mlp_up_.resize(L);
        mlp_down_.resize(L);
        shared_gate_.resize(L);
        shared_up_.resize(L);
        shared_down_.resize(L);
        mhc_alpha_.resize(L);
        hc_attn_fn_.resize(L);
        hc_attn_base_.resize(L);
        hc_attn_scale_.resize(L);
        hc_ffn_fn_.resize(L);
        hc_ffn_base_.resize(L);
        hc_ffn_scale_.resize(L);
        mla_qa_.resize(L);
        mla_qb_.resize(L);
        mla_kva_.resize(L);
        mla_kt_.resize(L);
        mla_v_.resize(L);
        mla_o_.resize(L);
        mla_g_.resize(L);
        mla_qa_ln_.resize(L);
        mla_kva_ln_.resize(L);
        dsa_wq_.resize(L);
        dsa_wk_.resize(L);
        dsa_wp_.resize(L);
        dsa_kg_.resize(L);
        dsa_knw_.resize(L);
        dsa_knb_.resize(L);
        dsa_ape_.resize(L);
        dsa_ikeys_.resize(L);
        dsa_igates_.resize(L);
        const int M = cfg_.mhc.mult > 0 ? cfg_.mhc.mult : 1;
        for (int l = 0; l < L; ++l) {
            ones(in_n_[l], H);
            ones(out_n_[l], H);
            wq_[l] = qmat_xavier(std::max(P, H), H, 200 + l, bits);
            wk_[l] = qmat_xavier(std::max(std::max(P, cfg_.mla.kv_lora), 1), H, 210 + l, bits);
            wv_[l] = qmat_xavier(std::max(P, 1), H, 220 + l, bits);
            wo_[l] = qmat_xavier(H, std::max(P, H), 230 + l, bits);
            if (!cfg_.kda.full_rank_gate) {
                wg_[l] = qmat_xavier(std::max(cfg_.kda.head_dim, 1), H, 240 + l, bits);
                wgb_[l] = qmat_xavier(std::max(P, 1), std::max(cfg_.kda.head_dim, 1), 245 + l, bits);
            } else {
                wg_[l] = qmat_xavier(std::max(P, 1), H, 240 + l, bits);
            }
            wfa_[l] = qmat_xavier(cfg_.kda.head_dim, H, 250 + l, bits);
            wfb_[l] = qmat_xavier(P, cfg_.kda.head_dim, 260 + l, bits);
            wdt_[l].assign(P, 0.f);
            alog_[l].assign(cfg_.kda.heads, 0.f);
            wb_[l] = qmat_xavier(cfg_.kda.heads > 0 ? cfg_.kda.heads : 1, H, 270 + l, bits);
            ones(on_[l], cfg_.kda.head_dim > 0 ? cfg_.kda.head_dim : 1);
            xavier(router_[l], std::max(cfg_.moe.n_experts, 1), H, 280 + l);
            router_bias_[l].assign(std::max(cfg_.moe.n_experts, 1), 0.f);
            mlp_gate_[l] = qmat_xavier(cfg_.dense_intermediate, H, 290 + l, bits);
            mlp_up_[l] = qmat_xavier(cfg_.dense_intermediate, H, 300 + l, bits);
            mlp_down_[l] = qmat_xavier(H, cfg_.dense_intermediate, 310 + l, bits);
            shared_gate_[l] = qmat_xavier(cfg_.moe.intermediate, H, 320 + l, bits);
            shared_up_[l] = qmat_xavier(cfg_.moe.intermediate, H, 330 + l, bits);
            shared_down_[l] = qmat_xavier(H, cfg_.moe.intermediate, 340 + l, bits);
            mhc_alpha_[l].assign(static_cast<size_t>(M) * M, 1.f / static_cast<float>(M));
            const bool full = (l < static_cast<int>(cfg_.is_full.size())) ? cfg_.is_full[l] : 0;
            if (full && ql > 0 && kv > 0 && qk > 0) {
                mla_qa_[l] = qmat_xavier(ql, H, 400 + l, mbits);
                mla_qb_[l] = qmat_xavier(nh * (qk + qr), ql, 410 + l, mbits);
                mla_kva_[l] = qmat_xavier(kv + qr, H, 420 + l, mbits);
                mla_kt_[l] = qmat_xavier(nh * kv, qk, 430 + l, mbits);
                mla_v_[l] = qmat_xavier(nh * vh, kv, 440 + l, mbits);
                mla_o_[l] = qmat_xavier(H, nh * vh, 450 + l, hbits);
                if (cfg_.mla.output_gate)
                    mla_g_[l] = qmat_xavier(nh * vh, H, 460 + l, hbits);
                ones(mla_qa_ln_[l], ql);
                ones(mla_kva_ln_[l], kv);
            }
        }
        if (cfg_.vision.layers > 0 || cfg_.vision.hidden > 0) {
            const int vp = cfg_.vision.patch > 0 ? cfg_.vision.patch : 14;
            const int vh = cfg_.vision.hidden > 0 ? cfg_.vision.hidden : 32;
            const int oh = cfg_.vision.out_hidden > 0 ? cfg_.vision.out_hidden : H;
            vit_patch_ = qmat_xavier(vh, 3 * vp * vp, 500, bits);
            vit_proj_ = qmat_xavier(oh, vh, 501, bits);
        }
    }

    void mark_hits(int layer, const int *ids, int k) {
        if (!ids || k <= 0 || layer < 0 || layer > usage_.n_layers())
            return;
        const int ne = usage_.n_experts();
        if (ne < 1)
            return;
        const size_t base = static_cast<size_t>(layer) * static_cast<size_t>(ne);
        if (base + static_cast<size_t>(ne) > ehit_.size())
            return;
        for (int i = 0; i < k; ++i) {
            const int e = ids[i];
            if (e >= 0 && e < ne) {
                ehit_[base + static_cast<size_t>(e)] = 1;
                ++turn_c_[base + static_cast<size_t>(e)];
            }
        }
    }

    ModelConfig cfg_{};
    RuntimeConfig rt_{};
    ExpertStore store_;
    RouteUsage usage_;
    std::vector<uint8_t> ehit_;
    std::vector<uint32_t> turn_c_;
    RouteTrace trace_;
    bool loaded_ = false;
    bool from_checkpoint_ = false;
    std::string prefix_;
    std::vector<float> embed_, norm_;
    quant::QuantMat lm_head_;
    std::vector<std::vector<float>> in_n_, out_n_, wdt_, alog_, on_, router_, router_bias_,
        mhc_alpha_, conv_q_, conv_k_, conv_v_, hc_attn_fn_, hc_attn_base_, hc_attn_scale_,
        hc_ffn_fn_, hc_ffn_base_, hc_ffn_scale_;
    std::vector<quant::QuantMat> wq_, wk_, wv_, wo_, wg_, wgb_, wfa_, wfb_, wb_, mlp_gate_, mlp_up_,
        mlp_down_, shared_gate_, shared_up_, shared_down_;
    quant::QuantMat vit_patch_, vit_proj_;
    GlmVitTower vit_;
    std::vector<quant::QuantMat> mla_qa_, mla_qb_, mla_kva_, mla_kt_, mla_v_, mla_o_, mla_g_;
    std::vector<std::vector<float>> mla_qa_ln_, mla_kva_ln_;
    std::vector<quant::QuantMat> dsa_wq_, dsa_wk_, dsa_wp_, dsa_kg_;
    std::vector<std::vector<float>> dsa_knw_, dsa_knb_, dsa_ape_, dsa_ikeys_, dsa_igates_;

    bool dsa_live(int l) const {
        return cfg_.dsa.topk > 0 && cfg_.dsa.n_heads > 0 && cfg_.dsa.head_dim > 0 &&
               l >= 0 && l < static_cast<int>(dsa_wq_.size()) && !dsa_wq_[l].empty() &&
               !dsa_wk_[l].empty();
    }
    std::vector<std::vector<uint8_t>> experts_;
    Glm53Slot slots_[kMaxKvSlots];
};

std::unique_ptr<FamilyEngine> make_glm53() { return std::make_unique<Glm53Engine>(); }

} // namespace mvllm
