#include "family.hpp"
#include "../gpu/backend.hpp"
#include "../gpu/dsv4_cuda.hpp"
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
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <random>
#include <sstream>
#include <sys/stat.h>

namespace mvllm {
namespace {

void xavier(std::vector<float> &w, int rows, int cols, uint32_t seed) {
    w.assign(static_cast<size_t>(std::max(rows, 0)) * std::max(cols, 1), 0.f);
    std::mt19937 rng(seed);
    float s = 1.f / std::sqrt(static_cast<float>(std::max(cols, 1)));
    std::normal_distribution<float> dist(0.f, s);
    for (float &v : w)
        v = dist(rng);
}

void ones(std::vector<float> &w, int n) { w.assign(static_cast<size_t>(std::max(n, 0)), 1.f); }

quant::QuantMat qmat_xavier(int O, int I, uint32_t seed, int bits) {
    std::vector<float> t;
    xavier(t, O, I, seed);
    quant::QuantMat m;
    m.from_f32(t.data(), O, I, bits);
    return m;
}

struct AccTimer {
    double &acc;
    std::chrono::steady_clock::time_point t0;
    explicit AccTimer(double &a) : acc(a), t0(std::chrono::steady_clock::now()) {}
    ~AccTimer() {
        acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

struct MxGeom {
    int64_t w1p = 0, w1s = 0, w2p = 0, w2s = 0, slot = 0;
};

MxGeom make_mx_geom(int hidden, int inter) {
    MxGeom g{};
    const int I = std::max(hidden, 1);
    const int O = std::max(inter, 1);
    if (I >= 32 && O >= 32 && I % 32 == 0 && O % 32 == 0) {
        g.w1p = static_cast<int64_t>(O) * (I / 2);
        g.w1s = static_cast<int64_t>(O) * (I / 32);
        g.w2p = static_cast<int64_t>(I) * (O / 2);
        g.w2s = static_cast<int64_t>(I) * (O / 32);
    } else {
        g.w1p = static_cast<int64_t>(O) * ((I + 1) / 2);
        g.w1s = static_cast<int64_t>(O) * ((I + 31) / 32);
        g.w2p = static_cast<int64_t>(I) * ((O + 1) / 2);
        g.w2s = static_cast<int64_t>(I) * ((O + 31) / 32);
    }
    g.slot = 2 * (g.w1p + g.w1s) + g.w2p + g.w2s;
    return g;
}

float sqrt_softplus(float x) {
    float sp;
    if (x > 20.f)
        sp = x;
    else if (x < -20.f)
        sp = std::exp(x);
    else
        sp = std::log1p(std::exp(x));
    return std::sqrt(std::max(sp, 0.f));
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

void mxfp4_swiglu_expert(float *y, const float *x, int S, const uint8_t *blob, int H, int O,
                         float limit, bool idot) {
    if (!y || !x || !blob || S <= 0 || H <= 0 || O <= 0)
        return;
    const MxGeom g = make_mx_geom(H, O);
    const uint8_t *w1p = blob;
    const uint8_t *w1s = w1p + g.w1p;
    const uint8_t *w2p = w1s + g.w1s;
    const uint8_t *w2s = w2p + g.w2p;
    const uint8_t *w3p = w2s + g.w2s;
    const uint8_t *w3s = w3p + g.w1p;
    if (!idot && dsv4_cuda::available()) {
        dsv4_cuda::Tensor *gate = nullptr;
        dsv4_cuda::Tensor *up = nullptr;
        dsv4_cuda::Tensor *down = nullptr;
        bool ok = dsv4_cuda::upload_fp4(&gate, w1p, w1s, O, H, 0) &&
                  dsv4_cuda::upload_fp4(&up, w3p, w3s, O, H, 0) &&
                  dsv4_cuda::upload_fp4(&down, w2p, w2s, H, O, 0);
        const float one = 1.f;
        for (int s = 0; ok && s < S; ++s)
            ok = dsv4_cuda::expert_group(&gate, &up, &down, &one, 1, limit,
                                         y + static_cast<size_t>(s) * H,
                                         x + static_cast<size_t>(s) * H);
        dsv4_cuda::tensor_free(gate);
        dsv4_cuda::tensor_free(up);
        dsv4_cuda::tensor_free(down);
        if (ok)
            return;
    }
    std::vector<float> gate(static_cast<size_t>(S) * O), up(static_cast<size_t>(S) * O);
    gpu::gemm_mxfp4(gate.data(), x, w1p, w1s, S, H, O, idot);
    gpu::gemm_mxfp4(up.data(), x, w3p, w3s, S, H, O, idot);
    for (int s = 0; s < S; ++s) {
        float *gg = gate.data() + static_cast<size_t>(s) * O;
        const float *uu = up.data() + static_cast<size_t>(s) * O;
        for (int i = 0; i < O; ++i)
            gg[i] = quant::clamped_swiglu(gg[i], uu[i], limit);
    }
    gpu::gemm_mxfp4(y, gate.data(), w2p, w2s, S, O, H, idot);
}

bool collect_pieces(const std::vector<io::StFile> &files, const std::string names[6],
                    const int64_t expect[6], ExpertLoc &loc) {
    loc.pieces.clear();
    for (int p = 0; p < 6; ++p) {
        io::StHit hit = io::st_find_dir(files, names[p]);
        if (!hit.tensor)
            return false;
        const int64_t nb = io::st_nbytes(*hit.tensor);
        if (expect[p] > 0 && nb != expect[p])
            return false;
        ExpertPiece ep;
        ep.path = hit.file->path;
        ep.offset = io::st_file_offset(*hit.file, *hit.tensor);
        ep.bytes = nb;
        loc.pieces.push_back(ep);
    }
    loc.contig = true;
    int64_t sum = 0;
    for (int p = 0; p < 6; ++p) {
        sum += loc.pieces[static_cast<size_t>(p)].bytes;
        if (p == 0)
            continue;
        if (loc.pieces[static_cast<size_t>(p)].path != loc.pieces[0].path ||
            loc.pieces[static_cast<size_t>(p)].offset !=
                loc.pieces[static_cast<size_t>(p - 1)].offset +
                    loc.pieces[static_cast<size_t>(p - 1)].bytes)
            loc.contig = false;
    }
    if (loc.contig) {
        loc.path = loc.pieces[0].path;
        loc.offset = loc.pieces[0].offset;
        loc.bytes = sum;
        loc.pieces.clear();
    }
    return true;
}

const char *ckpt_env_raw(const char *suffix) {
    const std::string mv = std::string("MVLLM_PREFIX_CKPT") + suffix;
    if (const char *s = std::getenv(mv.c_str()); s && s[0])
        return s;
    const std::string v4 = std::string("V4_PREFIX_CKPT") + suffix;
    if (const char *s = std::getenv(v4.c_str()); s && s[0])
        return s;
    return nullptr;
}

int ckpt_env_int(const char *suffix, int def) {
    const char *s = ckpt_env_raw(suffix);
    if (!s)
        return def;
    return std::atoi(s);
}

bool wr_bytes(std::ostream &o, const void *p, size_t n) {
    if (n == 0)
        return true;
    o.write(reinterpret_cast<const char *>(p), static_cast<std::streamsize>(n));
    return static_cast<bool>(o);
}

bool rd_bytes(std::istream &in, void *p, size_t n) {
    if (n == 0)
        return true;
    in.read(reinterpret_cast<char *>(p), static_cast<std::streamsize>(n));
    return static_cast<bool>(in) && in.gcount() == static_cast<std::streamsize>(n);
}

template <class T> bool wr_pod(std::ostream &o, T v) { return wr_bytes(o, &v, sizeof(v)); }

template <class T> bool rd_pod(std::istream &in, T &v) { return rd_bytes(in, &v, sizeof(v)); }

uint64_t fnv1a64(uint64_t h, uint64_t x) {
    h ^= x;
    h *= 1099511628211ull;
    return h;
}

uint64_t fnv1a64_bytes(uint64_t h, const void *p, size_t n) {
    const auto *b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

std::string hex64(uint64_t x) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(x));
    return buf;
}

} // namespace

class Dsv4Engine final : public FamilyEngine {
public:
    Family family() const override { return Family::Dsv4; }
    const ModelConfig &config() const override { return cfg_; }

    struct Slot {
        std::vector<int> history;
        std::vector<std::vector<float>> kv, dsa_ikeys, dsa_igates;
        std::vector<float> streams;
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

    struct PrefixCkpt {
        std::vector<int> ids;
        std::vector<std::vector<float>> kv, dsa_ikeys, dsa_igates;
        std::vector<float> streams;
        int pos = 0;
        bool prompt_end = false;
        uint64_t tick = 0;
    };

    Status load(const std::string &model_dir, const RuntimeConfig &rt, std::string &err) override {
        rt_ = rt;
        if (rt_.model_dir.empty())
            rt_.model_dir = model_dir;
        Status st = load_model_config(model_dir, cfg_, err);
        if (st != Status::Ok)
            return st;
        if (cfg_.family != Family::Dsv4) {
            err = "not a DeepSeek V4 config";
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
        Status ust = usage_.init("dsv4", cfg_.n_layers, std::max(cfg_.moe.n_experts, 1), err);
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
        dsv4_cuda::init(nullptr, 0);
        return Status::Ok;
    }

    Status generate(const std::vector<int> &prompt, const GenParams &gp, GenResult &out,
                    std::string &err) override {
        if (!loaded_) {
            err = "dsv4 not loaded";
            return Status::InvalidArgument;
        }
        if (prompt.empty()) {
            err = "empty prompt";
            return Status::InvalidArgument;
        }
        const int cslot = gp.cache_slot;
        const bool use_slot = cslot >= 0 && cslot < kMaxKvSlots;
        Slot local;
        Slot *work = use_slot ? &slots_[cslot] : &local;
        int Tmax = std::max(rt_.max_seq, static_cast<int>(prompt.size()) + gp.max_new_tokens + 256);
        if (Tmax <= 0)
            Tmax = 4096;
        int kv_reuse = 0;
        if (use_slot && gp.prefix_reuse > 0) {
            Slot &sl = slots_[cslot];
            const int match = sl.have ? kv_common_prefix(sl.history, prompt) : 0;
            const int want = std::min(gp.prefix_reuse, match);
            if (sl.have && sl.pos > 0 && sl.pos <= want &&
                sl.pos <= static_cast<int>(prompt.size())) {
                kv_reuse = sl.pos;
                slot_ensure_t(sl, Tmax);
            }
        }
        int reuse = kv_reuse;
        Status pst = ckpt_run_prefill(*work, prompt, gp, kv_reuse, Tmax, reuse, err);
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
                allow.assign(static_cast<size_t>(cfg_.vocab), 0);
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
                allow.assign(static_cast<size_t>(cfg_.vocab), 0);
                g.allow_mask(gp.token_text, allow.data(), cfg_.vocab);
            }
        }
        out.completion_tokens = static_cast<int>(out.tokens.size());
        out.token_logprobs = work->token_lps;
        out.top_logprobs = work->top_lps;
        if (use_slot) {
            Slot &sl = slots_[cslot];
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
        os << "dsv4  hidden=" << cfg_.hidden << " layers=" << cfg_.n_layers
           << " experts=" << cfg_.moe.n_experts << " topk=" << cfg_.moe.topk
           << " shared=" << cfg_.moe.n_shared << " mhc=" << cfg_.mhc.mult
           << " q_lora=" << cfg_.mla.q_lora << " o_lora=" << cfg_.o_lora
           << " window=" << cfg_.sliding_window << " dsa=" << (cfg_.dsa.topk > 0 ? cfg_.dsa.topk : 0)
           << " checkpoint=" << (from_checkpoint_ ? "yes" : "synthetic")
           << " bits=" << rt_.dense_bits << " prefix=" << (prefix_.empty() ? "-" : prefix_)
           << " ckpt=" << ckpts_.size() << " hits=" << ckpt_hits_
           << " tier=" << (dsv4_cuda::available() ? dsv4_cuda::backend_name() : "off");
        return os.str();
    }

    void expert_stats(ExpertStoreStats &out) const override { store_.stats(out); }
    int take_repin(RepinEvent *out, int cap) override { return store_.take_repin(out, cap); }

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
            live.push_back(l);
        }
        const int rows = static_cast<int>(live.size());
        const size_t cells = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        out.rows = rows;
        out.cols = cols;
        out.emap.assign(cells, 0);
        out.hits.assign(cells, 0);
        out.entropy.assign(static_cast<size_t>(rows), 0.f);
        int ram = 0, disk = 0;
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
        if (consume_hits)
            std::fill(ehit_.begin(), ehit_.end(), 0);
    }

    void turn_perf(TurnPerf &out, bool reset) override {
        out = {};
        store_.take_io_perf(out.t_edisk, out.t_ewait, reset);
        out.t_emm = t_emm_;
        out.t_attn = t_attn_;
        out.t_kvb = t_kvb_;
        out.t_head = t_head_;
        if (reset)
            t_emm_ = t_attn_ = t_kvb_ = t_head_ = 0;
    }

    Status begin_generate(int slot, const std::vector<int> &ids, const GenParams &gp, int &reuse,
                          std::string &err) override {
        if (!loaded_) {
            err = "dsv4 not loaded";
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
        Slot &st = slots_[slot];
        const int match = st.have ? kv_common_prefix(st.history, ids) : 0;
        const int want = std::min(std::max(gp.prefix_reuse, 0), match);
        int kv_reuse = 0;
        int Tmax = std::max(rt_.max_seq, static_cast<int>(ids.size()) + gp.max_new_tokens + 256);
        if (Tmax <= 0)
            Tmax = 4096;
        if (st.have && st.pos > 0 && st.pos <= want && st.pos <= static_cast<int>(ids.size())) {
            kv_reuse = st.pos;
            slot_ensure_t(st, Tmax);
        }
        int applied = kv_reuse;
        Status pst = ckpt_run_prefill(st, ids, gp, kv_reuse, Tmax, applied, err);
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
        Slot &st = slots_[slot];
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
        std::vector<Slot *> step_s;
        std::vector<int> step_tok;
        step_s.reserve(static_cast<size_t>(n));
        step_tok.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            Slot &st = slots_[slots[i]];
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
            bool stop = gen_stop_id(tokens[i], cfg_, st.gp) || st.emitted >= st.gp.max_new_tokens;
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
                st.allow.assign(static_cast<size_t>(cfg_.vocab), 0);
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

    // Official COLIKV: L = first kv_lora of each layer row, R = next qk_rope,
    // I = concatenated DSA keys for layers that have dsa_ikeys.
    int export_kv_rows(int slot, int pos0, int n, KvPersistRecord *rows) const override {
        if (!rows || n <= 0 || slot < 0 || slot >= kMaxKvSlots)
            return 0;
        const Slot &sl = slots_[slot];
        if (!sl.have)
            return 0;
        const int width = kv_width();
        const int kvL = std::max(cfg_.mla.kv_lora, 0);
        const int qr = std::max(cfg_.mla.qk_rope, 0);
        const int ID = std::max(cfg_.dsa.head_dim, 0);
        const int nL = cfg_.n_layers;
        for (int t = 0; t < n; ++t) {
            const int pos = pos0 + t;
            if (pos < 0 || pos >= sl.pos)
                return t;
            KvPersistRecord &rec = rows[t];
            if (width > 0) {
                for (int l = 0; l < nL; ++l) {
                    if (l >= static_cast<int>(sl.kv.size()) || sl.kv[l].empty())
                        continue;
                    const size_t src = static_cast<size_t>(pos) * static_cast<size_t>(width);
                    if (src + static_cast<size_t>(width) > sl.kv[l].size())
                        continue;
                    const float *row = sl.kv[l].data() + src;
                    if (kvL > 0) {
                        const size_t dst = static_cast<size_t>(l) * static_cast<size_t>(kvL);
                        if (dst < rec.L.size()) {
                            const int ncopy =
                                std::min(kvL, std::min(width, static_cast<int>(rec.L.size() - dst)));
                            if (ncopy > 0)
                                std::memcpy(rec.L.data() + dst, row,
                                            static_cast<size_t>(ncopy) * sizeof(float));
                        }
                    }
                    if (qr > 0) {
                        const int off = std::min(kvL, width);
                        const int room = width - off;
                        const size_t dst = static_cast<size_t>(l) * static_cast<size_t>(qr);
                        if (dst < rec.R.size() && room > 0) {
                            const int ncopy =
                                std::min(qr, std::min(room, static_cast<int>(rec.R.size() - dst)));
                            if (ncopy > 0)
                                std::memcpy(rec.R.data() + dst, row + off,
                                            static_cast<size_t>(ncopy) * sizeof(float));
                        }
                    }
                }
            }
            if (ID > 0) {
                size_t ioff = 0;
                for (int l = 0; l < nL; ++l) {
                    if (l >= static_cast<int>(sl.dsa_ikeys.size()) || sl.dsa_ikeys[l].empty())
                        continue;
                    const std::vector<float> &ik = sl.dsa_ikeys[l];
                    const size_t koff = static_cast<size_t>(pos) * static_cast<size_t>(ID);
                    if (ioff + static_cast<size_t>(ID) <= rec.I.size() &&
                        koff + static_cast<size_t>(ID) <= ik.size())
                        std::memcpy(rec.I.data() + ioff, ik.data() + koff,
                                    static_cast<size_t>(ID) * sizeof(float));
                    ioff += static_cast<size_t>(ID);
                }
            }
        }
        return n;
    }

    int import_kv_rows(int slot, int pos0, int n, const KvPersistRecord *rows) override {
        if (!rows || n <= 0 || slot < 0 || slot >= kMaxKvSlots)
            return 0;
        Slot &sl = slots_[slot];
        const int width = kv_width();
        const int kvL = std::max(cfg_.mla.kv_lora, 0);
        const int qr = std::max(cfg_.mla.qk_rope, 0);
        const int ID = std::max(cfg_.dsa.head_dim, 0);
        const int nL = cfg_.n_layers;
        const int headroom = std::max(rt_.max_seq, 256);
        const int t_max = std::max(std::max(pos0 + n, sl.pos), 1) + headroom;
        bool caches_small = static_cast<int>(sl.kv.size()) < nL;
        if (!caches_small && width > 0) {
            const size_t need =
                static_cast<size_t>(std::max(pos0 + n, 1)) * static_cast<size_t>(width);
            for (int l = 0; l < nL; ++l) {
                if (sl.kv[l].size() < need) {
                    caches_small = true;
                    break;
                }
            }
        }
        if (ID > 0 && cfg_.dsa.topk > 0 && static_cast<int>(sl.dsa_ikeys.size()) < nL)
            caches_small = true;
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
            if (width > 0) {
                for (int l = 0; l < nL; ++l) {
                    if (l >= static_cast<int>(sl.kv.size()) || sl.kv[l].empty())
                        continue;
                    const size_t dst0 = static_cast<size_t>(pos) * static_cast<size_t>(width);
                    if (dst0 + static_cast<size_t>(width) > sl.kv[l].size())
                        continue;
                    float *dst = sl.kv[l].data() + dst0;
                    if (kvL > 0) {
                        const size_t src = static_cast<size_t>(l) * static_cast<size_t>(kvL);
                        if (src < rec.L.size()) {
                            const int ncopy = std::min(
                                kvL, std::min(width, static_cast<int>(rec.L.size() - src)));
                            if (ncopy > 0)
                                std::memcpy(dst, rec.L.data() + src,
                                            static_cast<size_t>(ncopy) * sizeof(float));
                        }
                    }
                    if (qr > 0) {
                        const int off = std::min(kvL, width);
                        const int room = width - off;
                        const size_t src = static_cast<size_t>(l) * static_cast<size_t>(qr);
                        if (src < rec.R.size() && room > 0) {
                            const int ncopy = std::min(
                                qr, std::min(room, static_cast<int>(rec.R.size() - src)));
                            if (ncopy > 0)
                                std::memcpy(dst + off, rec.R.data() + src,
                                            static_cast<size_t>(ncopy) * sizeof(float));
                        }
                    }
                }
            }
            if (ID > 0) {
                size_t ioff = 0;
                for (int l = 0; l < nL; ++l) {
                    if (l >= static_cast<int>(sl.dsa_ikeys.size()) || sl.dsa_ikeys[l].empty())
                        continue;
                    std::vector<float> &ik = sl.dsa_ikeys[l];
                    const size_t koff = static_cast<size_t>(pos) * static_cast<size_t>(ID);
                    if (ioff + static_cast<size_t>(ID) <= rec.I.size() &&
                        koff + static_cast<size_t>(ID) <= ik.size())
                        std::memcpy(ik.data() + koff, rec.I.data() + ioff,
                                    static_cast<size_t>(ID) * sizeof(float));
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
    int kv_width() const { return std::max(cfg_.head_dim > 0 ? cfg_.head_dim : cfg_.mla.kv_lora, 1); }
    int n_heads() const { return std::max(cfg_.mla.n_heads > 0 ? cfg_.mla.n_heads : 1, 1); }
    int q_width() const { return n_heads() * kv_width(); }

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

    bool dsa_live(int l) const {
        return cfg_.dsa.topk > 0 && cfg_.dsa.n_heads > 0 && cfg_.dsa.head_dim > 0 &&
               l >= 0 && l < static_cast<int>(dsa_wq_.size()) && !dsa_wq_[l].empty() &&
               !dsa_wk_[l].empty();
    }

    void slot_embed(Slot &s, int id) {
        const int H = cfg_.hidden;
        const int M = mhc_mult();
        if (static_cast<int>(s.streams.size()) < M * H)
            s.streams.assign(static_cast<size_t>(M) * H, 0.f);
        float *h = s.streams.data();
        int tid = id;
        if (tid < 0 || tid >= cfg_.vocab)
            tid = 0;
        std::memcpy(h, embed_.data() + static_cast<size_t>(tid) * H,
                    static_cast<size_t>(H) * sizeof(float));
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

    void attn_one(Slot &s, float *hh, int at, int l, float *branch) {
        AccTimer t(t_attn_);
        const int H = cfg_.hidden;
        const int hd = kv_width();
        const int nh = n_heads();
        const int qw = nh * hd;
        const int qr = std::max(cfg_.mla.qk_rope, 0);
        const float theta = cfg_.mla.rope_theta > 0.f ? cfg_.mla.rope_theta : cfg_.rope_theta;
        std::vector<float> n(static_cast<size_t>(H)), y(static_cast<size_t>(H), 0.f);
        quant::rmsnorm(hh, in_n_[l].data(), n.data(), H, cfg_.rms_eps);

        // Low-rank Q: wq_a → q_norm → wq_b, else dense wq.
        std::vector<float> q(static_cast<size_t>(qw), 0.f);
        if (!wq_a_[l].empty() && !wq_b_[l].empty()) {
            const int ql = std::max(cfg_.mla.q_lora, wq_a_[l].O);
            std::vector<float> qa(static_cast<size_t>(std::max(ql, 1)), 0.f);
            wq_a_[l].gemm(qa.data(), n.data(), 1);
            if (!q_ln_[l].empty())
                quant::rmsnorm(qa.data(), q_ln_[l].data(), qa.data(),
                               std::min(ql, static_cast<int>(q_ln_[l].size())), cfg_.rms_eps);
            wq_b_[l].gemm(q.data(), qa.data(), 1);
        } else if (!wq_[l].empty()) {
            wq_[l].gemm(q.data(), n.data(), 1);
        }

        // MQA KV: wkv → kv_norm, cache [T, head_dim].
        std::vector<float> kv(static_cast<size_t>(hd), 0.f);
        if (!wkv_[l].empty())
            wkv_[l].gemm(kv.data(), n.data(), 1);
        if (!kv_ln_[l].empty())
            quant::rmsnorm(kv.data(), kv_ln_[l].data(), kv.data(),
                           std::min(hd, static_cast<int>(kv_ln_[l].size())), cfg_.rms_eps);
        if (qr > 0 && theta > 0.f) {
            const int rope_n = std::min(qr, hd);
            apply_rope(kv.data() + (hd - rope_n), rope_n, at, theta);
            for (int h = 0; h < nh; ++h)
                apply_rope(q.data() + static_cast<size_t>(h) * hd + (hd - rope_n), rope_n, at,
                           theta);
        }
        if (l < static_cast<int>(s.kv.size()) && !s.kv[l].empty() && at >= 0) {
            const size_t off = static_cast<size_t>(at) * static_cast<size_t>(hd);
            if (off + static_cast<size_t>(hd) <= s.kv[l].size())
                std::memcpy(s.kv[l].data() + off, kv.data(), static_cast<size_t>(hd) * sizeof(float));
        }

        std::vector<int> sel;
        if (dsa_live(l) && at >= 0) {
            const int ID = std::max(cfg_.dsa.head_dim, 1);
            const int IH = std::max(cfg_.dsa.n_heads, 1);
            std::vector<float> qn(static_cast<size_t>(std::max(cfg_.mla.q_lora, 1)), 0.f);
            if (!wq_a_[l].empty()) {
                wq_a_[l].gemm(qn.data(), n.data(), 1);
                if (!q_ln_[l].empty())
                    quant::rmsnorm(qn.data(), q_ln_[l].data(), qn.data(),
                                   std::min(cfg_.mla.q_lora, static_cast<int>(q_ln_[l].size())),
                                   cfg_.rms_eps);
            }
            std::vector<float> iq(static_cast<size_t>(IH) * ID, 0.f), hw(static_cast<size_t>(IH),
                                                                        0.f);
            dsa_wq_[l].gemm(iq.data(), qn.data(), 1);
            std::vector<float> &ikeys = s.dsa_ikeys[l];
            std::vector<float> &igates = s.dsa_igates[l];
            if (!ikeys.empty()) {
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
                float sc = 1.f / std::sqrt(static_cast<float>(std::max(IH, 1)));
                for (int i = 0; i < IH; ++i)
                    hw[static_cast<size_t>(i)] *= sc;
            }
            const int nsel = dsa_index_width(cfg_.dsa);
            sel.assign(static_cast<size_t>(std::max(nsel, 1)), -1);
            if (dsa_select_range(sel.data(), iq.data(), ikeys.empty() ? nullptr : ikeys.data(),
                                 igates.empty() ? nullptr : igates.data(), hw.data(),
                                 dsa_ape_[l].empty() ? nullptr : dsa_ape_[l].data(), nullptr,
                                 at + 1, cfg_.dsa, at, at + 1) < 0)
                dsa_select(sel.data(), iq.data(), ikeys.empty() ? nullptr : ikeys.data(),
                           igates.empty() ? nullptr : igates.data(), hw.data(),
                           dsa_ape_[l].empty() ? nullptr : dsa_ape_[l].data(), at + 1, cfg_.dsa);
        } else if (cfg_.sliding_window > 0 && at >= 0) {
            const int lo = std::max(0, at + 1 - cfg_.sliding_window);
            sel.resize(static_cast<size_t>(at - lo + 1));
            for (int i = lo; i <= at; ++i)
                sel[static_cast<size_t>(i - lo)] = i;
        } else if (at >= 0) {
            sel.resize(static_cast<size_t>(at + 1));
            for (int i = 0; i <= at; ++i)
                sel[static_cast<size_t>(i)] = i;
        }

        std::vector<float> ctx(static_cast<size_t>(qw), 0.f);
        const float scale = 1.f / std::sqrt(static_cast<float>(std::max(hd, 1)));
        const float *kbase =
            (l < static_cast<int>(s.kv.size()) && !s.kv[l].empty()) ? s.kv[l].data() : kv.data();
        const int past = at + 1;
        if (!sel.empty() && past > 0) {
            std::vector<float> sc(sel.size());
            for (int h = 0; h < nh; ++h) {
                const float *qh = q.data() + static_cast<size_t>(h) * hd;
                float mx = -1e30f;
                int nsc = 0;
                for (int idx : sel) {
                    if (idx < 0 || idx > at)
                        continue;
                    const float *kk = kbase + static_cast<size_t>(idx) * hd;
                    float acc = 0.f;
                    for (int d = 0; d < hd; ++d)
                        acc += qh[d] * kk[d];
                    sc[static_cast<size_t>(nsc)] = acc * scale;
                    if (h < static_cast<int>(sink_[l].size()))
                        sc[static_cast<size_t>(nsc)] -= sink_[l][static_cast<size_t>(h)];
                    if (sc[static_cast<size_t>(nsc)] > mx)
                        mx = sc[static_cast<size_t>(nsc)];
                    ++nsc;
                }
                float z = 0.f;
                for (int i = 0; i < nsc; ++i) {
                    sc[static_cast<size_t>(i)] = std::exp(sc[static_cast<size_t>(i)] - mx);
                    z += sc[static_cast<size_t>(i)];
                }
                if (z <= 0.f)
                    z = 1.f;
                float *oh = ctx.data() + static_cast<size_t>(h) * hd;
                int i = 0;
                for (int idx : sel) {
                    if (idx < 0 || idx > at)
                        continue;
                    const float *vv = kbase + static_cast<size_t>(idx) * hd;
                    const float a = sc[static_cast<size_t>(i++)] / z;
                    for (int d = 0; d < hd; ++d)
                        oh[d] += a * vv[d];
                }
            }
        }

        if (!wo_a_[l].empty() && !wo_b_[l].empty()) {
            const int oa = std::max(cfg_.o_groups * cfg_.o_lora, wo_a_[l].O);
            std::vector<float> mid(static_cast<size_t>(std::max(oa, 1)), 0.f);
            wo_a_[l].gemm(mid.data(), ctx.data(), 1);
            wo_b_[l].gemm(y.data(), mid.data(), 1);
        } else if (!wo_[l].empty()) {
            wo_[l].gemm(y.data(), ctx.data(), 1);
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

    void slot_alloc(Slot &s, int t_max) {
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int M = mhc_mult();
        const int hd = kv_width();
        const int tcap = std::max(t_max, 1);
        s.streams.assign(static_cast<size_t>(M) * H, 0.f);
        s.kv.assign(static_cast<size_t>(L), {});
        s.dsa_ikeys.assign(static_cast<size_t>(L), {});
        s.dsa_igates.assign(static_cast<size_t>(L), {});
        for (int l = 0; l < L; ++l) {
            s.kv[l].assign(static_cast<size_t>(tcap) * hd, 0.f);
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
    }

    void slot_ensure_t(Slot &s, int t_max) {
        const int L = cfg_.n_layers;
        const int hd = kv_width();
        const int tcap = std::max(t_max, 1);
        if (static_cast<int>(s.kv.size()) < L)
            s.kv.resize(static_cast<size_t>(L));
        if (static_cast<int>(s.dsa_ikeys.size()) < L)
            s.dsa_ikeys.resize(static_cast<size_t>(L));
        if (static_cast<int>(s.dsa_igates.size()) < L)
            s.dsa_igates.resize(static_cast<size_t>(L));
        const size_t need = static_cast<size_t>(tcap) * hd;
        for (int l = 0; l < L; ++l) {
            if (s.kv[l].size() < need)
                s.kv[l].resize(need, 0.f);
        }
        if (cfg_.dsa.topk > 0 && cfg_.dsa.head_dim > 0) {
            const size_t ineed = static_cast<size_t>(tcap) * cfg_.dsa.head_dim;
            for (int l = 0; l < L; ++l) {
                if (s.dsa_ikeys[l].size() < ineed)
                    s.dsa_ikeys[l].resize(ineed, 0.f);
                if (s.dsa_igates[l].size() < ineed)
                    s.dsa_igates[l].resize(ineed, 0.f);
            }
        }
    }

    Status slot_step(Slot &s, int token, std::string &err) {
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

    Status slot_step_n(Slot **ss, const int *tokens, int S, std::string &err) {
        if (S <= 0)
            return Status::Ok;
        for (int i = 0; i < S; ++i) {
            Status st = slot_step(*ss[i], tokens[i], err);
            if (st != Status::Ok)
                return st;
        }
        return Status::Ok;
    }

    Status slot_prefill(Slot &s, const std::vector<int> &ids, int start, std::string &err) {
        if (start < 0)
            start = 0;
        for (int i = start; i < static_cast<int>(ids.size()); ++i) {
            Status st = slot_step(s, ids[static_cast<size_t>(i)], err);
            if (st != Status::Ok)
                return st;
        }
        return Status::Ok;
    }

    bool ckpt_enabled() const { return ckpt_env_int("", 1) != 0; }
    int ckpt_min() const { return std::max(ckpt_env_int("_MIN", 512), 1); }
    int ckpt_slots() const { return std::max(ckpt_env_int("_SLOTS", 4), 1); }
    int ckpt_disk() const {
        const int d = ckpt_env_int("_DISK", 1);
        return d < 0 ? 0 : (d > 2 ? 2 : d);
    }
    bool ckpt_log() const {
        if (const char *s = std::getenv("MVLLM_PREFIX_CKPT_LOG"); s && s[0])
            return std::atoi(s) != 0;
        if (const char *s = std::getenv("V4_PREFIX_LOG"); s && s[0])
            return std::atoi(s) != 0;
        if (const char *s = std::getenv("V4_PREFIX_CKPT_LOG"); s && s[0])
            return std::atoi(s) != 0;
        return false;
    }

    uint64_t ckpt_fp() const {
        uint64_t h = 14695981039346656037ull;
        h = fnv1a64(h, static_cast<uint64_t>(cfg_.hidden));
        h = fnv1a64(h, static_cast<uint64_t>(cfg_.n_layers));
        h = fnv1a64(h, static_cast<uint64_t>(cfg_.moe.n_experts));
        h = fnv1a64(h, static_cast<uint64_t>(cfg_.moe.topk));
        h = fnv1a64(h, static_cast<uint64_t>(cfg_.vocab));
        h = fnv1a64(h, static_cast<uint64_t>(kv_width()));
        h = fnv1a64(h, static_cast<uint64_t>(cfg_.dsa.head_dim));
        h = fnv1a64(h, static_cast<uint64_t>(mhc_mult()));
        return h;
    }

    std::string ckpt_dir() const {
        if (rt_.model_dir.empty())
            return {};
        return rt_.model_dir + "/.coli_ckpt";
    }

    int ckpt_gateway_p(const std::vector<int> &prompt, const GenParams &gp) const {
        const int n = static_cast<int>(prompt.size());
        if (gp.prefix_reuse > 0)
            return std::min(gp.prefix_reuse, n);
        if (gp.prefix_bytes > 0)
            return std::min(gp.prefix_bytes, n);
        return 0;
    }

    int ckpt_capture_point(const std::vector<int> &prompt, const GenParams &gp) const {
        const int mn = ckpt_min();
        const int P = ckpt_gateway_p(prompt, gp);
        if (P >= mn)
            return P;
        if (!last_fresh_prompt_.empty()) {
            const int lcp = kv_common_prefix(last_fresh_prompt_, prompt);
            if (lcp >= mn)
                return lcp;
        }
        return 0;
    }

    PrefixCkpt ckpt_clone_from_slot(const Slot &s, int n) const {
        PrefixCkpt ck;
        const int L = cfg_.n_layers;
        const int hd = kv_width();
        const int ID = cfg_.dsa.head_dim;
        const int take = std::max(n, 0);
        ck.pos = take;
        if (take > 0 && take <= static_cast<int>(s.history.size()))
            ck.ids.assign(s.history.begin(), s.history.begin() + take);
        ck.streams = s.streams;
        ck.kv.resize(static_cast<size_t>(L));
        ck.dsa_ikeys.resize(static_cast<size_t>(L));
        ck.dsa_igates.resize(static_cast<size_t>(L));
        const size_t kvn = static_cast<size_t>(take) * static_cast<size_t>(std::max(hd, 0));
        const size_t dn = static_cast<size_t>(take) * static_cast<size_t>(std::max(ID, 0));
        for (int l = 0; l < L; ++l) {
            if (l < static_cast<int>(s.kv.size()) && s.kv[l].size() >= kvn)
                ck.kv[static_cast<size_t>(l)].assign(s.kv[l].begin(), s.kv[l].begin() +
                                                                        static_cast<std::ptrdiff_t>(kvn));
            if (ID > 0 && cfg_.dsa.topk > 0) {
                if (l < static_cast<int>(s.dsa_ikeys.size()) && s.dsa_ikeys[l].size() >= dn)
                    ck.dsa_ikeys[static_cast<size_t>(l)].assign(
                        s.dsa_ikeys[l].begin(),
                        s.dsa_ikeys[l].begin() + static_cast<std::ptrdiff_t>(dn));
                if (l < static_cast<int>(s.dsa_igates.size()) && s.dsa_igates[l].size() >= dn)
                    ck.dsa_igates[static_cast<size_t>(l)].assign(
                        s.dsa_igates[l].begin(),
                        s.dsa_igates[l].begin() + static_cast<std::ptrdiff_t>(dn));
            }
        }
        return ck;
    }

    void ckpt_apply(Slot &s, const PrefixCkpt &ck, int Tmax) {
        const int n = std::max(ck.pos, 0);
        slot_ensure_t(s, std::max(Tmax, n));
        const int H = cfg_.hidden;
        const int M = mhc_mult();
        const size_t need_st = static_cast<size_t>(std::max(M, 1)) * static_cast<size_t>(std::max(H, 1));
        if (s.streams.size() < need_st)
            s.streams.assign(need_st, 0.f);
        if (!ck.streams.empty()) {
            const size_t sc = std::min(s.streams.size(), ck.streams.size());
            std::memcpy(s.streams.data(), ck.streams.data(), sc * sizeof(float));
        }
        const int L = cfg_.n_layers;
        const int hd = kv_width();
        const int ID = cfg_.dsa.head_dim;
        const size_t kvn = static_cast<size_t>(n) * static_cast<size_t>(std::max(hd, 0));
        const size_t dn = static_cast<size_t>(n) * static_cast<size_t>(std::max(ID, 0));
        for (int l = 0; l < L; ++l) {
            if (l < static_cast<int>(ck.kv.size()) && l < static_cast<int>(s.kv.size()) &&
                ck.kv[static_cast<size_t>(l)].size() >= kvn && s.kv[l].size() >= kvn && kvn > 0)
                std::memcpy(s.kv[l].data(), ck.kv[static_cast<size_t>(l)].data(),
                            kvn * sizeof(float));
            if (ID > 0 && cfg_.dsa.topk > 0) {
                if (l < static_cast<int>(ck.dsa_ikeys.size()) &&
                    l < static_cast<int>(s.dsa_ikeys.size()) &&
                    ck.dsa_ikeys[static_cast<size_t>(l)].size() >= dn &&
                    s.dsa_ikeys[l].size() >= dn && dn > 0)
                    std::memcpy(s.dsa_ikeys[l].data(), ck.dsa_ikeys[static_cast<size_t>(l)].data(),
                                dn * sizeof(float));
                if (l < static_cast<int>(ck.dsa_igates.size()) &&
                    l < static_cast<int>(s.dsa_igates.size()) &&
                    ck.dsa_igates[static_cast<size_t>(l)].size() >= dn &&
                    s.dsa_igates[l].size() >= dn && dn > 0)
                    std::memcpy(s.dsa_igates[l].data(), ck.dsa_igates[static_cast<size_t>(l)].data(),
                                dn * sizeof(float));
            }
        }
        s.history = ck.ids;
        if (static_cast<int>(s.history.size()) > n)
            s.history.resize(static_cast<size_t>(n));
        s.pos = n;
        s.have = n > 0;
    }

    const PrefixCkpt *ckpt_lookup(const std::vector<int> &prompt) {
        const int mn = ckpt_min();
        const PrefixCkpt *best = nullptr;
        size_t best_i = 0;
        for (size_t i = 0; i < ckpts_.size(); ++i) {
            const PrefixCkpt &c = ckpts_[i];
            const int n = static_cast<int>(c.ids.size());
            if (n < mn || n > static_cast<int>(prompt.size()))
                continue;
            if (!std::equal(c.ids.begin(), c.ids.end(), prompt.begin()))
                continue;
            if (!best || n > static_cast<int>(best->ids.size())) {
                best = &c;
                best_i = i;
            }
        }
        if (!best)
            return nullptr;
        ckpts_[best_i].tick = ++ckpt_tick_;
        ++ckpt_hits_;
        if (ckpt_log())
            std::fprintf(stderr, "v4_ckpt hit prefix=%d\n", static_cast<int>(best->ids.size()));
        return &ckpts_[best_i];
    }

    void ckpt_insert(PrefixCkpt ck) {
        if (ck.ids.empty() || ck.pos <= 0)
            return;
        for (auto &ex : ckpts_) {
            if (ex.ids == ck.ids) {
                const bool keep_prefix = !ex.prompt_end || !ck.prompt_end;
                ex.kv = std::move(ck.kv);
                ex.dsa_ikeys = std::move(ck.dsa_ikeys);
                ex.dsa_igates = std::move(ck.dsa_igates);
                ex.streams = std::move(ck.streams);
                ex.pos = ck.pos;
                if (keep_prefix)
                    ex.prompt_end = false;
                else
                    ex.prompt_end = ck.prompt_end;
                ex.tick = ++ckpt_tick_;
                return;
            }
        }
        const int cap = ckpt_slots();
        while (static_cast<int>(ckpts_.size()) >= cap && !ckpts_.empty()) {
            size_t evict = 0;
            bool have_pe = false;
            uint64_t oldest = ~uint64_t{0};
            for (size_t i = 0; i < ckpts_.size(); ++i) {
                if (!ckpts_[i].prompt_end)
                    continue;
                if (!have_pe || ckpts_[i].tick < oldest) {
                    evict = i;
                    oldest = ckpts_[i].tick;
                    have_pe = true;
                }
            }
            if (!have_pe) {
                oldest = ~uint64_t{0};
                for (size_t i = 0; i < ckpts_.size(); ++i) {
                    if (ckpts_[i].tick < oldest) {
                        evict = i;
                        oldest = ckpts_[i].tick;
                    }
                }
            }
            ckpts_.erase(ckpts_.begin() + static_cast<std::ptrdiff_t>(evict));
        }
        ck.tick = ++ckpt_tick_;
        ckpts_.push_back(std::move(ck));
    }

    void ckpt_store(const std::vector<int> &ids, const Slot &slot, bool prompt_end) {
        if (!ckpt_enabled())
            return;
        const int n = static_cast<int>(ids.size());
        if (n < ckpt_min() || slot.pos < n)
            return;
        PrefixCkpt ck = ckpt_clone_from_slot(slot, n);
        ck.ids = ids;
        ck.pos = n;
        ck.prompt_end = prompt_end;
        if (ckpt_log())
            std::fprintf(stderr, "v4_ckpt store %s=%d\n", prompt_end ? "prompt_end" : "prefix", n);
        const int disk = ckpt_disk();
        const bool persist = disk == 2 || (disk == 1 && !prompt_end);
        if (persist) {
            PrefixCkpt disk_ck = ck;
            ckpt_insert(std::move(ck));
            ckpt_save_disk(disk_ck);
        } else {
            ckpt_insert(std::move(ck));
        }
    }

    bool ckpt_write_blob(std::ostream &o, const PrefixCkpt &ck) const {
        static const char kMagic[8] = {'D', 'S', 'V', '4', 'C', 'K', '1', '\0'};
        if (!wr_bytes(o, kMagic, 8))
            return false;
        if (!wr_pod<uint64_t>(o, ckpt_fp()))
            return false;
        const int32_t n = static_cast<int32_t>(ck.ids.size());
        if (!wr_pod<int32_t>(o, n))
            return false;
        if (n > 0 && !wr_bytes(o, ck.ids.data(), static_cast<size_t>(n) * sizeof(int)))
            return false;
        if (!wr_pod<int32_t>(o, static_cast<int32_t>(ck.pos)))
            return false;
        if (!wr_pod<int32_t>(o, ck.prompt_end ? 1 : 0))
            return false;
        const int32_t nL = cfg_.n_layers;
        const int32_t hd = kv_width();
        const int32_t idh = cfg_.dsa.head_dim;
        const int32_t sn = static_cast<int32_t>(ck.streams.size());
        if (!wr_pod<int32_t>(o, nL) || !wr_pod<int32_t>(o, hd) || !wr_pod<int32_t>(o, idh) ||
            !wr_pod<int32_t>(o, sn))
            return false;
        auto wr_vec = [&](const std::vector<float> &v) {
            const int32_t m = static_cast<int32_t>(v.size());
            if (!wr_pod<int32_t>(o, m))
                return false;
            return m <= 0 || wr_bytes(o, v.data(), static_cast<size_t>(m) * sizeof(float));
        };
        for (int l = 0; l < nL; ++l) {
            const std::vector<float> empty;
            const std::vector<float> &kv =
                l < static_cast<int>(ck.kv.size()) ? ck.kv[static_cast<size_t>(l)] : empty;
            const std::vector<float> &ik =
                l < static_cast<int>(ck.dsa_ikeys.size()) ? ck.dsa_ikeys[static_cast<size_t>(l)]
                                                          : empty;
            const std::vector<float> &ig =
                l < static_cast<int>(ck.dsa_igates.size()) ? ck.dsa_igates[static_cast<size_t>(l)]
                                                           : empty;
            if (!wr_vec(kv) || !wr_vec(ik) || !wr_vec(ig))
                return false;
        }
        return sn <= 0 || wr_bytes(o, ck.streams.data(), static_cast<size_t>(sn) * sizeof(float));
    }

    bool ckpt_read_blob(std::istream &in, PrefixCkpt &ck) const {
        char magic[8];
        if (!rd_bytes(in, magic, 8) || std::memcmp(magic, "DSV4CK1", 8) != 0)
            return false;
        uint64_t fp = 0;
        if (!rd_pod<uint64_t>(in, fp) || fp != ckpt_fp())
            return false;
        int32_t n = 0;
        if (!rd_pod<int32_t>(in, n) || n < 0 || n > (1 << 24))
            return false;
        ck.ids.resize(static_cast<size_t>(n));
        if (n > 0 && !rd_bytes(in, ck.ids.data(), static_cast<size_t>(n) * sizeof(int)))
            return false;
        int32_t pos = 0, pe = 0;
        if (!rd_pod<int32_t>(in, pos) || !rd_pod<int32_t>(in, pe))
            return false;
        ck.pos = pos;
        ck.prompt_end = pe != 0;
        int32_t nL = 0, hd = 0, idh = 0, sn = 0;
        if (!rd_pod<int32_t>(in, nL) || !rd_pod<int32_t>(in, hd) || !rd_pod<int32_t>(in, idh) ||
            !rd_pod<int32_t>(in, sn))
            return false;
        if (nL != cfg_.n_layers || hd != kv_width() || idh != cfg_.dsa.head_dim || sn < 0 ||
            sn > (1 << 24))
            return false;
        auto rd_vec = [&](std::vector<float> &v) {
            int32_t m = 0;
            if (!rd_pod<int32_t>(in, m) || m < 0 || m > (1 << 26))
                return false;
            v.resize(static_cast<size_t>(m));
            return m <= 0 || rd_bytes(in, v.data(), static_cast<size_t>(m) * sizeof(float));
        };
        ck.kv.assign(static_cast<size_t>(nL), {});
        ck.dsa_ikeys.assign(static_cast<size_t>(nL), {});
        ck.dsa_igates.assign(static_cast<size_t>(nL), {});
        for (int l = 0; l < nL; ++l) {
            if (!rd_vec(ck.kv[static_cast<size_t>(l)]) ||
                !rd_vec(ck.dsa_ikeys[static_cast<size_t>(l)]) ||
                !rd_vec(ck.dsa_igates[static_cast<size_t>(l)]))
                return false;
        }
        ck.streams.resize(static_cast<size_t>(sn));
        if (sn > 0 && !rd_bytes(in, ck.streams.data(), static_cast<size_t>(sn) * sizeof(float)))
            return false;
        if (ck.pos <= 0)
            ck.pos = n;
        return true;
    }

    uint64_t ckpt_ids_hash(const std::vector<int> &ids) const {
        uint64_t h = 14695981039346656037ull;
        return fnv1a64_bytes(h, ids.data(), ids.size() * sizeof(int));
    }

    void ckpt_save_disk(const PrefixCkpt &ck) {
        const std::string dir = ckpt_dir();
        if (dir.empty() || ck.ids.empty())
            return;
        if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST)
            return;
        const std::string name = "ck_" + hex64(ckpt_fp()) + "_" + std::to_string(ck.ids.size()) +
                                 "_" + hex64(ckpt_ids_hash(ck.ids)) + ".bin";
        const std::string path = dir + "/" + name;
        const std::string tmp = path + ".tmp";
        {
            std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
            if (!o || !ckpt_write_blob(o, ck)) {
                o.close();
                std::remove(tmp.c_str());
                return;
            }
        }
        if (std::rename(tmp.c_str(), path.c_str()) != 0) {
            std::remove(tmp.c_str());
            return;
        }
    }

    void ckpt_load_disk() {
        if (ckpt_disk_loaded_)
            return;
        ckpt_disk_loaded_ = true;
        if (!ckpt_enabled() || ckpt_disk() == 0)
            return;
        const std::string dir = ckpt_dir();
        if (dir.empty())
            return;
        DIR *d = ::opendir(dir.c_str());
        if (!d)
            return;
        while (dirent *ent = ::readdir(d)) {
            if (!ent->d_name[0] || ent->d_name[0] == '.')
                continue;
            const std::string path = dir + "/" + ent->d_name;
            std::ifstream in(path, std::ios::binary);
            if (!in)
                continue;
            PrefixCkpt ck;
            if (!ckpt_read_blob(in, ck))
                continue;
            ckpt_insert(std::move(ck));
        }
        ::closedir(d);
    }

    Status ckpt_run_prefill(Slot &s, const std::vector<int> &prompt, const GenParams &gp,
                            int kv_reuse, int Tmax, int &reuse, std::string &err) {
        reuse = std::max(kv_reuse, 0);
        if (ckpt_enabled()) {
            ckpt_load_disk();
            if (const PrefixCkpt *hit = ckpt_lookup(prompt)) {
                if (hit->pos > reuse && hit->pos <= static_cast<int>(prompt.size())) {
                    if (reuse == 0)
                        slot_alloc(s, Tmax);
                    PrefixCkpt ck = *hit;
                    ckpt_apply(s, ck, Tmax);
                    reuse = ck.pos;
                }
            }
        }
        if (reuse == 0)
            slot_alloc(s, Tmax);
        else
            slot_ensure_t(s, Tmax);

        const int snap = ckpt_enabled() ? ckpt_capture_point(prompt, gp) : 0;
        if (snap >= ckpt_min() && snap <= static_cast<int>(prompt.size())) {
            if (s.pos < snap) {
                const std::vector<int> pref(prompt.begin(), prompt.begin() + snap);
                Status st = slot_prefill(s, pref, s.pos, err);
                if (st != Status::Ok)
                    return st;
            }
            if (s.pos == snap) {
                const std::vector<int> pref(prompt.begin(), prompt.begin() + snap);
                ckpt_store(pref, s, false);
            }
        }
        Status st = slot_prefill(s, prompt, s.pos, err);
        if (st != Status::Ok)
            return st;
        if (ckpt_enabled() && static_cast<int>(prompt.size()) >= ckpt_min())
            ckpt_store(prompt, s, true);
        if (ckpt_enabled() && kv_reuse == 0)
            last_fresh_prompt_ = prompt;
        return Status::Ok;
    }

    int slot_sample(Slot &s, const GenParams &gp, uint64_t *rng, const uint8_t *allow) {
        const int H = cfg_.hidden;
        const int M = mhc_mult();
        std::vector<float> n(static_cast<size_t>(H)), logits(static_cast<size_t>(cfg_.vocab)),
            pooled(static_cast<size_t>(H));
        const float *head_in = s.streams.data();
        if (official_hc(0, false) || M > 1) {
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
        {
            AccTimer t(t_head_);
            lm_head_.gemm(logits.data(), n.data(), 1);
        }
        return sample_penalized(logits.data(), cfg_.vocab, gp, s.history.data(),
                               static_cast<int>(s.history.size()), rng, allow, nullptr,
                               &s.token_lps, &s.top_lps);
    }

    int64_t expert_bytes() const {
        return make_mx_geom(cfg_.hidden, cfg_.moe.intermediate > 0 ? cfg_.moe.intermediate : 32)
            .slot;
    }

    void fill_mx_expert(uint8_t *p, int layer, int eid) {
        const int I = cfg_.hidden;
        const int O = cfg_.moe.intermediate;
        auto g = make_mx_geom(I, O);
        std::vector<float> w1(static_cast<size_t>(O) * I, 0.02f);
        std::vector<float> w2(static_cast<size_t>(I) * O, 0.02f);
        std::vector<float> w3(static_cast<size_t>(O) * I, 0.02f);
        for (size_t i = 0; i < w1.size(); ++i)
            w1[i] = 0.02f * (((layer + eid + static_cast<int>(i)) & 7) - 3);
        quant::pack_mxfp4(w1.data(), O, I, p, p + g.w1p);
        quant::pack_mxfp4(w2.data(), I, O, p + g.w1p + g.w1s, p + g.w1p + g.w1s + g.w2p);
        quant::pack_mxfp4(w3.data(), O, I, p + g.w1p + g.w1s + g.w2p + g.w2s,
                          p + g.w1p + g.w1s + g.w2p + g.w2s + g.w1p);
    }

    Status write_synthetic_experts(const std::string &model_dir, std::string &err) {
        const int64_t ebytes = expert_bytes();
        const std::string epath = model_dir + "/.mvllm_dsv4_experts.bin";
        std::ofstream out(epath, std::ios::binary | std::ios::trunc);
        if (!out) {
            err = "cannot write synthetic dsv4 experts";
            return Status::IoError;
        }
        std::vector<uint8_t> blob(static_cast<size_t>(ebytes));
        for (int l = 0; l < cfg_.n_layers; ++l) {
            if (l < cfg_.first_dense)
                continue;
            for (int e = 0; e < cfg_.moe.n_experts; ++e) {
                fill_mx_expert(blob.data(), l, e);
                out.write(reinterpret_cast<const char *>(blob.data()),
                          static_cast<std::streamsize>(ebytes));
                if (!out) {
                    err = "dsv4 expert blob write failed";
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

    bool try_overlay_mat(const std::vector<io::StFile> &files, const std::vector<std::string> &names,
                         quant::QuantMat &dst, int bits, std::string &err) {
        for (const auto &n : names) {
            if (overlay_mat(files, n, dst, bits, err) == Status::Ok)
                return true;
        }
        return false;
    }

    bool try_overlay_f32(const std::vector<io::StFile> &files, const std::vector<std::string> &names,
                         std::vector<float> &dst, int expect, std::string &err) {
        for (const auto &n : names) {
            if (overlay_f32(files, n, dst, expect, err) == Status::Ok)
                return true;
        }
        return false;
    }

    bool find_expert(const std::vector<io::StFile> &files, const std::string &P, int layer,
                     int expert, const MxGeom &geom, ExpertLoc &loc) {
        const std::string L = std::to_string(layer);
        const std::string E = std::to_string(expert);
        const int64_t want[6] = {geom.w1p, geom.w1s, geom.w2p, geom.w2s, geom.w1p, geom.w1s};
        auto try_set = [&](const std::string names[6]) {
            return collect_pieces(files, names, want, loc);
        };

        std::string n_off[6] = {
            P + "layers." + L + ".ffn.experts." + E + ".w1.weight",
            P + "layers." + L + ".ffn.experts." + E + ".w1.scale",
            P + "layers." + L + ".ffn.experts." + E + ".w2.weight",
            P + "layers." + L + ".ffn.experts." + E + ".w2.scale",
            P + "layers." + L + ".ffn.experts." + E + ".w3.weight",
            P + "layers." + L + ".ffn.experts." + E + ".w3.scale",
        };
        if (try_set(n_off))
            return true;
        std::string n_pack[6] = {
            P + "layers." + L + ".ffn.experts." + E + ".w1.weight_packed",
            P + "layers." + L + ".ffn.experts." + E + ".w1.weight_scale",
            P + "layers." + L + ".ffn.experts." + E + ".w2.weight_packed",
            P + "layers." + L + ".ffn.experts." + E + ".w2.weight_scale",
            P + "layers." + L + ".ffn.experts." + E + ".w3.weight_packed",
            P + "layers." + L + ".ffn.experts." + E + ".w3.weight_scale",
        };
        if (try_set(n_pack))
            return true;
        std::string n_mlp[6] = {
            P + "layers." + L + ".mlp.experts." + E + ".w1.weight_packed",
            P + "layers." + L + ".mlp.experts." + E + ".w1.weight_scale",
            P + "layers." + L + ".mlp.experts." + E + ".w2.weight_packed",
            P + "layers." + L + ".mlp.experts." + E + ".w2.weight_scale",
            P + "layers." + L + ".mlp.experts." + E + ".w3.weight_packed",
            P + "layers." + L + ".mlp.experts." + E + ".w3.weight_scale",
        };
        if (try_set(n_mlp))
            return true;
        std::string n_hf[6] = {
            P + "layers." + L + ".mlp.experts." + E + ".gate_proj.weight",
            P + "layers." + L + ".mlp.experts." + E + ".gate_proj.weight_scale",
            P + "layers." + L + ".mlp.experts." + E + ".down_proj.weight",
            P + "layers." + L + ".mlp.experts." + E + ".down_proj.weight_scale",
            P + "layers." + L + ".mlp.experts." + E + ".up_proj.weight",
            P + "layers." + L + ".mlp.experts." + E + ".up_proj.weight_scale",
        };
        if (try_set(n_hf))
            return true;
        return false;
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
        if (has("embed.weight") || has("layers.0.attn_norm.weight") ||
            has("layers.0.ffn.experts.0.w1.weight")) {
            prefix_.clear();
        } else if (has("model.embed_tokens.weight") ||
                   has("model.layers.0.input_layernorm.weight") ||
                   has("model.layers.0.mlp.experts.0.gate_proj.weight")) {
            prefix_ = "model.";
        } else if (has("model.language_model.embed_tokens.weight")) {
            prefix_ = "model.language_model.";
        } else if (has("model.layers.0.mlp.experts.0.w1.weight_packed") ||
                   has("layers.0.mlp.experts.0.gate_proj.weight") ||
                   has("layers.0.ffn.experts.0.w1.weight_packed")) {
            prefix_.clear();
        } else {
            io::st_close_dir(files);
            return Status::NotFound;
        }

        const std::string P = prefix_;
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int V = cfg_.vocab;
        const int bits = rt_.dense_bits;
        const int hbits = rt_.head_bits;
        const int mbits = rt_.mla_bits;

        try_overlay_f32(files, {P + "embed.weight", P + "embed_tokens.weight", "embed.weight"},
                        embed_, V * H, err);
        try_overlay_f32(files, {P + "norm.weight", "norm.weight"}, norm_, H, err);
        if (!try_overlay_mat(files, {"lm_head.weight", "head.weight", P + "lm_head.weight",
                                     P + "head.weight"},
                             lm_head_, hbits, err) &&
            !embed_.empty())
            lm_head_.from_f32(embed_.data(), V, H, hbits);

        for (int i = 0; i < L; ++i) {
            const std::string ly = P + "layers." + std::to_string(i);
            try_overlay_f32(files,
                            {ly + ".attn_norm.weight", ly + ".input_layernorm.weight"},
                            in_n_[i], H, err);
            try_overlay_f32(files,
                            {ly + ".ffn_norm.weight", ly + ".post_attention_layernorm.weight"},
                            out_n_[i], H, err);
            try_overlay_mat(files,
                            {ly + ".attn.wq_a.weight", ly + ".attn.wq_a",
                             ly + ".self_attn.q_a_proj.weight"},
                            wq_a_[i], mbits, err);
            try_overlay_mat(files,
                            {ly + ".attn.wq_b.weight", ly + ".attn.wq_b",
                             ly + ".self_attn.q_b_proj.weight"},
                            wq_b_[i], mbits, err);
            try_overlay_mat(files,
                            {ly + ".attn.wkv.weight", ly + ".attn.wkv",
                             ly + ".self_attn.kv_proj.weight",
                             ly + ".self_attn.kv_a_proj_with_mqa.weight"},
                            wkv_[i], mbits, err);
            try_overlay_mat(files,
                            {ly + ".attn.wo_a.weight", ly + ".attn.wo_a",
                             ly + ".self_attn.o_a_proj.weight"},
                            wo_a_[i], hbits, err);
            try_overlay_mat(files,
                            {ly + ".attn.wo_b.weight", ly + ".attn.wo_b",
                             ly + ".self_attn.o_b_proj.weight", ly + ".self_attn.o_proj.weight"},
                            wo_b_[i], hbits, err);
            try_overlay_mat(files, {ly + ".self_attn.q_proj.weight"}, wq_[i], bits, err);
            try_overlay_mat(files, {ly + ".self_attn.o_proj.weight"}, wo_[i], hbits, err);
            try_overlay_f32(files,
                            {ly + ".attn.q_norm.weight", ly + ".self_attn.q_a_norm.weight",
                             ly + ".self_attn.q_a_layernorm.weight"},
                            q_ln_[i], cfg_.mla.q_lora, err);
            try_overlay_f32(files,
                            {ly + ".attn.kv_norm.weight", ly + ".self_attn.kv_norm.weight",
                             ly + ".self_attn.kv_a_layernorm.weight"},
                            kv_ln_[i], 0, err);
            try_overlay_f32(files, {ly + ".attn.attn_sink", ly + ".self_attn.sinks"}, sink_[i], 0,
                            err);
            try_overlay_mat(files,
                            {ly + ".attn.indexer.wq_b.weight", ly + ".attn.indexer.wq_b",
                             ly + ".self_attn.indexer.wq_b.weight"},
                            dsa_wq_[i], mbits, err);
            try_overlay_mat(files,
                            {ly + ".attn.indexer.wk.weight", ly + ".self_attn.indexer.wk.weight",
                             ly + ".attn.indexer.compressor.wkv.weight"},
                            dsa_wk_[i], mbits, err);
            try_overlay_mat(files,
                            {ly + ".attn.indexer.weights_proj.weight",
                             ly + ".self_attn.indexer.weights_proj.weight"},
                            dsa_wp_[i], mbits, err);
            try_overlay_f32(files, {ly + ".hc_attn_fn", ly + ".attn_hc.fn"}, hc_attn_fn_[i], 0, err);
            try_overlay_f32(files, {ly + ".hc_attn_base", ly + ".attn_hc.base"}, hc_attn_base_[i],
                            0, err);
            try_overlay_f32(files, {ly + ".hc_attn_scale", ly + ".attn_hc.scale"},
                            hc_attn_scale_[i], 0, err);
            try_overlay_f32(files, {ly + ".hc_ffn_fn", ly + ".ffn_hc.fn"}, hc_ffn_fn_[i], 0, err);
            try_overlay_f32(files, {ly + ".hc_ffn_base", ly + ".ffn_hc.base"}, hc_ffn_base_[i], 0,
                            err);
            try_overlay_f32(files, {ly + ".hc_ffn_scale", ly + ".ffn_hc.scale"}, hc_ffn_scale_[i],
                            0, err);
            {
                const int Mn = mhc_mult();
                try_overlay_f32(files, {ly + ".mhc.weight", ly + ".mhc.alpha"}, mhc_alpha_[i],
                                Mn * Mn, err);
            }
            if (i < cfg_.first_dense) {
                try_overlay_mat(files, {ly + ".mlp.gate_proj.weight", ly + ".ffn.gate_proj.weight"},
                                mlp_gate_[i], bits, err);
                try_overlay_mat(files, {ly + ".mlp.up_proj.weight", ly + ".ffn.up_proj.weight"},
                                mlp_up_[i], bits, err);
                try_overlay_mat(files, {ly + ".mlp.down_proj.weight", ly + ".ffn.down_proj.weight"},
                                mlp_down_[i], bits, err);
            } else {
                try_overlay_f32(files,
                                {ly + ".ffn.gate.weight", ly + ".mlp.gate.weight"},
                                router_[i], cfg_.moe.n_experts * H, err);
                try_overlay_f32(files,
                                {ly + ".ffn.gate.bias", ly + ".mlp.gate.e_score_correction_bias"},
                                router_bias_[i], cfg_.moe.n_experts, err);
                try_overlay_mat(files,
                                {ly + ".ffn.shared_experts.w1.weight", ly + ".ffn.shared_experts.w1",
                                 ly + ".mlp.shared_experts.gate_proj.weight"},
                                shared_gate_[i], bits, err);
                try_overlay_mat(files,
                                {ly + ".ffn.shared_experts.w3.weight", ly + ".ffn.shared_experts.w3",
                                 ly + ".mlp.shared_experts.up_proj.weight"},
                                shared_up_[i], bits, err);
                try_overlay_mat(files,
                                {ly + ".ffn.shared_experts.w2.weight", ly + ".ffn.shared_experts.w2",
                                 ly + ".mlp.shared_experts.down_proj.weight"},
                                shared_down_[i], bits, err);
            }
        }
        err.clear();

        const int probe_l = cfg_.first_dense < L ? cfg_.first_dense : 0;
        const auto geom = make_mx_geom(H, cfg_.moe.intermediate);
        ExpertLoc probe;
        if (!find_expert(files, P, probe_l, 0, geom, probe) &&
            !find_expert(files, "", probe_l, 0, geom, probe)) {
            io::st_close_dir(files);
            from_checkpoint_ = true;
            return write_synthetic_experts(model_dir, err);
        }

        int registered = 0;
        for (int i = cfg_.first_dense; i < L; ++i) {
            for (int e = 0; e < cfg_.moe.n_experts; ++e) {
                ExpertLoc loc;
                loc.key = {i, e};
                if (!find_expert(files, P, i, e, geom, loc) && !find_expert(files, "", i, e, geom, loc)) {
                    err = "dsv4 expert pieces missing at L" + std::to_string(i) + " E" +
                          std::to_string(e);
                    io::st_close_dir(files);
                    return Status::ParseError;
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
        const int I = cfg_.dense_intermediate > 0 ? cfg_.dense_intermediate : cfg_.moe.intermediate;
        std::vector<float> g(static_cast<size_t>(I)), u(static_cast<size_t>(I)),
            d(static_cast<size_t>(H));
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
        AccTimer t(t_emm_);
        if (C <= 0)
            return Status::Ok;
        const int H = cfg_.hidden;
        const int O = cfg_.moe.intermediate;
        const int K = std::max(cfg_.moe.topk, 0);
        std::vector<int> idx(static_cast<size_t>(C) * std::max(K, 1), -1);
        std::vector<float> wt(static_cast<size_t>(C) * std::max(K, 1), 0.f);
        std::string terr;
        for (int c = 0; c < C; ++c) {
            std::vector<float> scores(static_cast<size_t>(std::max(cfg_.moe.n_experts, 1))),
                choice(static_cast<size_t>(std::max(cfg_.moe.n_experts, 1)));
            if (!router_[layer].empty())
                quant::matmul_f32(scores.data(), xs + static_cast<size_t>(c) * H,
                                  router_[layer].data(), 1, H, cfg_.moe.n_experts);
            for (int i = 0; i < cfg_.moe.n_experts; ++i) {
                scores[i] = sqrt_softplus(scores[i]);
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
        const bool idot = rt_.idot && gpu::device() == Device::Cpu;
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
            if (n > 0 && v.data) {
                mxfp4_swiglu_expert(yb.data(), xb.data(), n, v.data, H, O, cfg_.moe.swiglu_limit,
                                    idot);
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
            for (int i = 0; i < H; ++i)
                h[i] += ac[i] * cfg_.moe.routed_scale;
            if (!shared_gate_[layer].empty()) {
                std::vector<float> sg(static_cast<size_t>(O)), su(static_cast<size_t>(O)),
                    sd(static_cast<size_t>(H));
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
        if (cfg_.mla.n_heads <= 0)
            cfg_.mla.n_heads = cfg_.n_q_heads > 0 ? cfg_.n_q_heads : 2;
        if (cfg_.head_dim <= 0)
            cfg_.head_dim = 16;
        if (cfg_.dense_intermediate <= 0)
            cfg_.dense_intermediate = std::max(H * 2, 32);
        if (cfg_.moe.intermediate <= 0)
            cfg_.moe.intermediate = 32;
        if (cfg_.moe.n_experts <= 0)
            cfg_.moe.n_experts = 1;
        if (cfg_.mla.q_lora <= 0)
            cfg_.mla.q_lora = std::min(H, 32);
        if (cfg_.mla.v_head <= 0)
            cfg_.mla.v_head = cfg_.head_dim;
        const int bits = rt_.dense_bits;
        const int hbits = rt_.head_bits;
        const int mbits = rt_.mla_bits;
        const int nh = n_heads();
        const int hd = kv_width();
        const int qw = nh * hd;
        const int ql = std::max(cfg_.mla.q_lora, 1);
        const int og = std::max(cfg_.o_groups, 1);
        const int ol = cfg_.o_lora > 0 ? cfg_.o_lora : 0;
        xavier(embed_, V, H, 3);
        ones(norm_, H);
        lm_head_ = qmat_xavier(V, H, 4, hbits);
        in_n_.resize(static_cast<size_t>(L));
        out_n_.resize(static_cast<size_t>(L));
        wq_a_.resize(static_cast<size_t>(L));
        wq_b_.resize(static_cast<size_t>(L));
        wq_.resize(static_cast<size_t>(L));
        wkv_.resize(static_cast<size_t>(L));
        wo_a_.resize(static_cast<size_t>(L));
        wo_b_.resize(static_cast<size_t>(L));
        wo_.resize(static_cast<size_t>(L));
        q_ln_.resize(static_cast<size_t>(L));
        kv_ln_.resize(static_cast<size_t>(L));
        sink_.resize(static_cast<size_t>(L));
        router_.resize(static_cast<size_t>(L));
        router_bias_.resize(static_cast<size_t>(L));
        mlp_gate_.resize(static_cast<size_t>(L));
        mlp_up_.resize(static_cast<size_t>(L));
        mlp_down_.resize(static_cast<size_t>(L));
        shared_gate_.resize(static_cast<size_t>(L));
        shared_up_.resize(static_cast<size_t>(L));
        shared_down_.resize(static_cast<size_t>(L));
        mhc_alpha_.resize(static_cast<size_t>(L));
        hc_attn_fn_.resize(static_cast<size_t>(L));
        hc_attn_base_.resize(static_cast<size_t>(L));
        hc_attn_scale_.resize(static_cast<size_t>(L));
        hc_ffn_fn_.resize(static_cast<size_t>(L));
        hc_ffn_base_.resize(static_cast<size_t>(L));
        hc_ffn_scale_.resize(static_cast<size_t>(L));
        dsa_wq_.resize(static_cast<size_t>(L));
        dsa_wk_.resize(static_cast<size_t>(L));
        dsa_wp_.resize(static_cast<size_t>(L));
        dsa_kg_.resize(static_cast<size_t>(L));
        dsa_knw_.resize(static_cast<size_t>(L));
        dsa_knb_.resize(static_cast<size_t>(L));
        dsa_ape_.resize(static_cast<size_t>(L));
        const int M = mhc_mult();
        for (int l = 0; l < L; ++l) {
            ones(in_n_[l], H);
            ones(out_n_[l], H);
            wq_a_[l] = qmat_xavier(ql, H, 400 + l, mbits);
            wq_b_[l] = qmat_xavier(qw, ql, 410 + l, mbits);
            wkv_[l] = qmat_xavier(hd, H, 420 + l, mbits);
            if (ol > 0) {
                wo_a_[l] = qmat_xavier(og * ol, qw, 430 + l, hbits);
                wo_b_[l] = qmat_xavier(H, og * ol, 440 + l, hbits);
            } else {
                wo_[l] = qmat_xavier(H, qw, 450 + l, hbits);
            }
            ones(q_ln_[l], ql);
            ones(kv_ln_[l], hd);
            sink_[l].assign(static_cast<size_t>(nh), 0.f);
            xavier(router_[l], std::max(cfg_.moe.n_experts, 1), H, 280 + l);
            router_bias_[l].assign(static_cast<size_t>(std::max(cfg_.moe.n_experts, 1)), 0.f);
            mlp_gate_[l] = qmat_xavier(cfg_.dense_intermediate, H, 290 + l, bits);
            mlp_up_[l] = qmat_xavier(cfg_.dense_intermediate, H, 300 + l, bits);
            mlp_down_[l] = qmat_xavier(H, cfg_.dense_intermediate, 310 + l, bits);
            shared_gate_[l] = qmat_xavier(cfg_.moe.intermediate, H, 320 + l, bits);
            shared_up_[l] = qmat_xavier(cfg_.moe.intermediate, H, 330 + l, bits);
            shared_down_[l] = qmat_xavier(H, cfg_.moe.intermediate, 340 + l, bits);
            mhc_alpha_[l].assign(static_cast<size_t>(M) * M, 1.f / static_cast<float>(M));
            if (cfg_.dsa.topk > 0 && cfg_.dsa.n_heads > 0 && cfg_.dsa.head_dim > 0) {
                const int IH = cfg_.dsa.n_heads;
                const int ID = cfg_.dsa.head_dim;
                dsa_wq_[l] = qmat_xavier(IH * ID, ql, 500 + l, mbits);
                dsa_wk_[l] = qmat_xavier(ID, H, 510 + l, mbits);
                dsa_wp_[l] = qmat_xavier(IH, H, 520 + l, mbits);
            }
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
    double t_attn_ = 0, t_emm_ = 0, t_kvb_ = 0, t_head_ = 0;
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
    std::vector<std::vector<float>> in_n_, out_n_, q_ln_, kv_ln_, sink_, router_, router_bias_,
        mhc_alpha_, hc_attn_fn_, hc_attn_base_, hc_attn_scale_, hc_ffn_fn_, hc_ffn_base_,
        hc_ffn_scale_, dsa_knw_, dsa_knb_, dsa_ape_;
    std::vector<quant::QuantMat> wq_a_, wq_b_, wq_, wkv_, wo_a_, wo_b_, wo_, mlp_gate_, mlp_up_,
        mlp_down_, shared_gate_, shared_up_, shared_down_, dsa_wq_, dsa_wk_, dsa_wp_, dsa_kg_;
    std::vector<std::vector<uint8_t>> experts_;
    Slot slots_[kMaxKvSlots];
    std::vector<PrefixCkpt> ckpts_;
    uint64_t ckpt_tick_ = 1;
    uint64_t ckpt_hits_ = 0;
    std::vector<int> last_fresh_prompt_;
    bool ckpt_disk_loaded_ = false;
};

std::unique_ptr<FamilyEngine> make_dsv4() { return std::make_unique<Dsv4Engine>(); }

} // namespace mvllm
