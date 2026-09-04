#include "family.hpp"
#include "../gpu/backend.hpp"
#include "../io/safetensors.hpp"
#include "../quant/quant.hpp"
#include "../quant/weight.hpp"

#include <algorithm>
#include <cmath>
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

} // namespace

class Glm53Engine final : public FamilyEngine {
public:
    Family family() const override { return Family::Glm53; }
    const ModelConfig &config() const override { return cfg_; }

    Status load(const std::string &model_dir, const RuntimeConfig &rt, std::string &err) override {
        rt_ = rt;
        Status st = load_model_config(model_dir, cfg_, err);
        if (st != Status::Ok)
            return st;
        if (cfg_.family != Family::Glm53) {
            err = "not a GLM-5.3 config";
            return Status::Unsupported;
        }
        if (cfg_.moe.swiglu_limit <= 0.f)
            cfg_.moe.swiglu_limit = 7.f;
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
            err = "glm53 not loaded";
            return Status::InvalidArgument;
        }
        if (prompt.empty()) {
            err = "empty prompt";
            return Status::InvalidArgument;
        }
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int M = cfg_.mhc.mult > 0 ? cfg_.mhc.mult : 1;
        std::vector<float> streams(static_cast<size_t>(M) * H, 0.f);
        float *h = streams.data();
        const int kd = std::max(cfg_.kda.head_dim, 1);
        const int kh = std::max(cfg_.kda.heads, 1);
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
            if (cfg_.dsa.topk > 0 && cfg_.dsa.head_dim > 0) {
                dsa_ikeys_[l].assign(static_cast<size_t>(Tmax) * cfg_.dsa.head_dim, 0.f);
                dsa_igates_[l].assign(static_cast<size_t>(Tmax) * cfg_.dsa.head_dim, 0.f);
            }
        }
        int pos = 0;

        auto embed_tok = [&](int id) {
            int tid = id;
            if (tid < 0 || tid >= cfg_.vocab)
                tid = 0;
            std::memcpy(h, embed_.data() + static_cast<size_t>(tid) * H, H * sizeof(float));
            for (int m = 1; m < M; ++m)
                std::memcpy(streams.data() + static_cast<size_t>(m) * H, h, H * sizeof(float));
        };

        auto attn_one = [&](float *hh, int at, int l) {
            if (M > 1 && !mhc_alpha_[l].empty()) {
                std::vector<float> st(static_cast<size_t>(M) * H);
                for (int m = 0; m < M; ++m)
                    std::memcpy(st.data() + static_cast<size_t>(m) * H, hh, H * sizeof(float));
                mhc_mix(st.data(), H, M, mhc_alpha_[l].data(), cfg_.mhc.iters, cfg_.mhc.eps);
                std::memcpy(hh, st.data(), H * sizeof(float));
            }
            std::vector<float> n(H), y(H, 0.f);
            quant::rmsnorm(hh, in_n_[l].data(), n.data(), H, cfg_.rms_eps);
            bool full = (l < static_cast<int>(cfg_.is_full.size())) ? cfg_.is_full[l] : 0;
            if (!full && cfg_.kda.heads > 0) {
                kda_step(n.data(), H, cfg_.kda, &wq_[l], &wk_[l], &wv_[l], &wb_[l], &wfa_[l],
                         &wfb_[l], wdt_[l].data(), alog_[l].data(), &wg_[l], &wo_[l], on_[l].data(),
                         S[l].data(), y.data(), cfg_.rms_eps,
                         l < static_cast<int>(conv_q_.size()) && !conv_q_[l].empty()
                             ? conv_q_[l].data()
                             : nullptr,
                         l < static_cast<int>(conv_k_.size()) && !conv_k_[l].empty()
                             ? conv_k_[l].data()
                             : nullptr,
                         l < static_cast<int>(conv_v_.size()) && !conv_v_[l].empty()
                             ? conv_v_[l].data()
                             : nullptr,
                         winq[l].data(), wink[l].data(), winv[l].data());
            } else {
                const int *sel = nullptr;
                int nsel = 0;
                std::vector<int> selbuf;
                if (dsa_live(l) && at >= 0) {
                    std::vector<float> qn(std::max(cfg_.mla.q_lora, 1));
                    if (!mla_qa_[l].empty()) {
                        mla_qa_[l].gemm(qn.data(), n.data(), 1);
                        if (!mla_qa_ln_[l].empty())
                            quant::rmsnorm(qn.data(), mla_qa_ln_[l].data(), qn.data(),
                                           cfg_.mla.q_lora, cfg_.rms_eps);
                    }
                    const int ID = std::max(cfg_.dsa.head_dim, 1);
                    const int IH = std::max(cfg_.dsa.n_heads, 1);
                    std::vector<float> iq(static_cast<size_t>(IH) * ID, 0.f), hw(IH, 0.f);
                    if (!dsa_wq_[l].empty())
                        dsa_wq_[l].gemm(iq.data(), qn.data(), 1);
                    if (!dsa_wk_[l].empty() && !dsa_ikeys_[l].empty()) {
                        std::vector<float> kraw(ID, 0.f);
                        dsa_wk_[l].gemm(kraw.data(), n.data(), 1);
                        layernorm(kraw.data(),
                                  dsa_knw_[l].empty() ? nullptr : dsa_knw_[l].data(),
                                  dsa_knb_[l].empty() ? nullptr : dsa_knb_[l].data(),
                                  dsa_ikeys_[l].data() + static_cast<size_t>(at) * ID, ID, 1e-5f);
                    }
                    if (!dsa_kg_[l].empty() && !dsa_igates_[l].empty())
                        dsa_kg_[l].gemm(dsa_igates_[l].data() + static_cast<size_t>(at) * ID,
                                        n.data(), 1);
                    if (!dsa_wp_[l].empty()) {
                        dsa_wp_[l].gemm(hw.data(), n.data(), 1);
                        float s = 1.f / std::sqrt(static_cast<float>(IH));
                        for (int i = 0; i < IH; ++i)
                            hw[i] *= s;
                    }
                    nsel = dsa_index_width(cfg_.dsa);
                    selbuf.assign(static_cast<size_t>(std::max(nsel, 1)), -1);
                    dsa_select(selbuf.data(), iq.data(), dsa_ikeys_[l].data(),
                               dsa_igates_[l].empty() ? nullptr : dsa_igates_[l].data(), hw.data(),
                               dsa_ape_[l].empty() ? nullptr : dsa_ape_[l].data(), at + 1,
                               cfg_.dsa);
                    sel = selbuf.data();
                }
                mla_step(n.data(), H, cfg_.mla, &mla_qa_[l],
                         mla_qa_ln_[l].empty() ? nullptr : mla_qa_ln_[l].data(), &mla_qb_[l],
                         &mla_kva_[l], mla_kva_ln_[l].empty() ? nullptr : mla_kva_ln_[l].data(),
                         &mla_kt_[l], &mla_v_[l], &mla_o_[l], &mla_g_[l],
                         mla_cache[l].empty() ? nullptr : mla_cache[l].data(), at, y.data(),
                         cfg_.rms_eps, sel, nsel);
            }
            for (int i = 0; i < H; ++i)
                hh[i] += y[i];
            if (M > 1 && !mhc_alpha_[l].empty()) {
                std::vector<float> st(static_cast<size_t>(M) * H);
                for (int m = 0; m < M; ++m)
                    std::memcpy(st.data() + static_cast<size_t>(m) * H, hh, H * sizeof(float));
                mhc_mix(st.data(), H, M, mhc_alpha_[l].data(), cfg_.mhc.iters, cfg_.mhc.eps);
                std::memcpy(hh, st.data(), H * sizeof(float));
            }
        };

        auto ffn_one = [&](float *hh, int l) -> Status {
            std::vector<float> n(H);
            quant::rmsnorm(hh, out_n_[l].data(), n.data(), H, cfg_.rms_eps);
            if (l < cfg_.first_dense) {
                dense_mlp(l, n.data(), hh);
                return Status::Ok;
            }
            return moe_layer(l, n.data(), hh, err);
        };

        auto step = [&](int token) -> Status {
            embed_tok(token);
            for (int l = 0; l < L; ++l) {
                attn_one(h, pos, l);
                Status st = ffn_one(h, l);
                if (st != Status::Ok)
                    return st;
            }
            ++pos;
            return Status::Ok;
        };

        const int chunk = rt_.prefill_chunk > 0 ? rt_.prefill_chunk : static_cast<int>(prompt.size());
        for (size_t i = 0; i < prompt.size();) {
            const int C = static_cast<int>(
                std::min(static_cast<size_t>(chunk), prompt.size() - i));
            std::vector<float> act(static_cast<size_t>(C) * H);
            for (int c = 0; c < C; ++c) {
                int tid = prompt[i + static_cast<size_t>(c)];
                if (tid < 0 || tid >= cfg_.vocab)
                    tid = 0;
                std::memcpy(act.data() + static_cast<size_t>(c) * H,
                            embed_.data() + static_cast<size_t>(tid) * H, H * sizeof(float));
            }
            for (int l = 0; l < L; ++l) {
                for (int c = 0; c < C; ++c)
                    attn_one(act.data() + static_cast<size_t>(c) * H, pos + c, l);
                if (l < cfg_.first_dense) {
                    for (int c = 0; c < C; ++c) {
                        std::vector<float> n(H);
                        quant::rmsnorm(act.data() + static_cast<size_t>(c) * H, out_n_[l].data(),
                                       n.data(), H, cfg_.rms_eps);
                        dense_mlp(l, n.data(), act.data() + static_cast<size_t>(c) * H);
                    }
                } else {
                    std::vector<float> norms(static_cast<size_t>(C) * H);
                    for (int c = 0; c < C; ++c)
                        quant::rmsnorm(act.data() + static_cast<size_t>(c) * H, out_n_[l].data(),
                                       norms.data() + static_cast<size_t>(c) * H, H, cfg_.rms_eps);
                    Status st = moe_layer_n(l, norms.data(), act.data(), C, err);
                    if (st != Status::Ok)
                        return st;
                }
            }
            std::memcpy(h, act.data() + static_cast<size_t>(C - 1) * H, H * sizeof(float));
            pos += C;
            i += static_cast<size_t>(C);
        }

        out.tokens.clear();
        out.prompt_tokens = static_cast<int>(prompt.size());
        for (int ntok = 0; ntok < gp.max_new_tokens; ++ntok) {
            std::vector<float> n(H), logits(cfg_.vocab);
            quant::rmsnorm(h, norm_.data(), n.data(), H, cfg_.rms_eps);
            lm_head_.gemm(logits.data(), n.data(), 1);
            int next = 0;
            float best = logits[0];
            for (int i = 1; i < cfg_.vocab; ++i) {
                if (logits[i] > best) {
                    best = logits[i];
                    next = i;
                }
            }
            out.tokens.push_back(next);
            if (is_stop_token(next, cfg_, gp.eos))
                break;
            Status st = step(next);
            if (st != Status::Ok)
                return st;
        }
        out.completion_tokens = static_cast<int>(out.tokens.size());
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

private:
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
            overlay_f32(files, P + "layers." + std::to_string(i) + ".self_attn.q_conv1d.weight",
                        conv_q_[i], 0, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".self_attn.k_conv1d.weight",
                        conv_k_[i], 0, err);
            overlay_f32(files, P + "layers." + std::to_string(i) + ".self_attn.v_conv1d.weight",
                        conv_v_[i], 0, err);
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
        for (int c = 0; c < C; ++c) {
            std::vector<float> scores(cfg_.moe.n_experts);
            quant::matmul_f32(scores.data(), xs + static_cast<size_t>(c) * H, router_[layer].data(),
                              1, H, cfg_.moe.n_experts);
            for (int i = 0; i < cfg_.moe.n_experts; ++i) {
                float b = i < static_cast<int>(router_bias_[layer].size()) ? router_bias_[layer][i]
                                                                          : 0.f;
                scores[i] = quant::sigmoid(scores[i]) + b;
            }
            moe_topk(scores.data(), cfg_.moe.n_experts, K, idx.data() + static_cast<size_t>(c) * K,
                     wt.data() + static_cast<size_t>(c) * K);
        }
        std::vector<int> uniq(static_cast<size_t>(std::max(cfg_.moe.n_experts, 1)));
        int nu = moe_union_ids(idx.data(), C, K, uniq.data(), static_cast<int>(uniq.size()));
        std::vector<ExpertKey> keys(static_cast<size_t>(nu));
        for (int i = 0; i < nu; ++i)
            keys[static_cast<size_t>(i)] = {layer, uniq[i]};
        if (rt_.pipe && nu > 0)
            store_.prefetch_tail(keys.data(), keys.size(), err);

        std::vector<float> acc(static_cast<size_t>(C) * H, 0.f);
        std::vector<float> xb(static_cast<size_t>(C) * H), yb(static_cast<size_t>(C) * H);
        std::vector<float> ww(static_cast<size_t>(C));
        std::vector<int> cmap(static_cast<size_t>(C));
        for (int ui = 0; ui < nu; ++ui) {
            const int eid = uniq[ui];
            ExpertView v{};
            Status st = store_.lookup({layer, eid}, v, err);
            if (st != Status::Ok)
                return st;
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
        }
        for (int c = 0; c < C; ++c) {
            const float *x = xs + static_cast<size_t>(c) * H;
            float *ac = acc.data() + static_cast<size_t>(c) * H;
            if (!shared_gate_[layer].empty()) {
                std::vector<float> sg(O), su(O), sd(H);
                shared_gate_[layer].gemm(sg.data(), x, 1);
                shared_up_[layer].gemm(su.data(), x, 1);
                for (int i = 0; i < O; ++i)
                    sg[i] = quant::clamped_swiglu(sg[i], su[i], cfg_.moe.swiglu_limit);
                shared_down_[layer].gemm(sd.data(), sg.data(), 1);
                for (int i = 0; i < H; ++i)
                    ac[i] += sd[i];
            }
            float *h = hs + static_cast<size_t>(c) * H;
            for (int i = 0; i < H; ++i)
                h[i] += ac[i] * cfg_.moe.routed_scale;
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
            wg_[l] = qmat_xavier(std::max(P, 1), H, 240 + l, bits);
            wfa_[l] = qmat_xavier(cfg_.kda.head_dim, H, 250 + l, bits);
            wfb_[l] = qmat_xavier(P, cfg_.kda.head_dim, 260 + l, bits);
            wdt_[l].assign(P, 0.f);
            alog_[l].assign(cfg_.kda.heads, 0.f);
            wb_[l] = qmat_xavier(P, H, 270 + l, bits);
            ones(on_[l], P);
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
                mla_qb_[l] = qmat_xavier(nh * qk, ql, 410 + l, mbits);
                mla_kva_[l] = qmat_xavier(kv, H, 420 + l, mbits);
                mla_kt_[l] = qmat_xavier(nh * kv, qk, 430 + l, mbits);
                mla_v_[l] = qmat_xavier(nh * vh, kv, 440 + l, mbits);
                mla_o_[l] = qmat_xavier(H, nh * vh, 450 + l, hbits);
                if (cfg_.mla.output_gate)
                    mla_g_[l] = qmat_xavier(nh * vh, H, 460 + l, hbits);
                ones(mla_qa_ln_[l], ql);
                ones(mla_kva_ln_[l], kv);
            }
        }
    }

    ModelConfig cfg_{};
    RuntimeConfig rt_{};
    ExpertStore store_;
    bool loaded_ = false;
    bool from_checkpoint_ = false;
    std::string prefix_;
    std::vector<float> embed_, norm_;
    quant::QuantMat lm_head_;
    std::vector<std::vector<float>> in_n_, out_n_, wdt_, alog_, on_, router_, router_bias_,
        mhc_alpha_, conv_q_, conv_k_, conv_v_;
    std::vector<quant::QuantMat> wq_, wk_, wv_, wo_, wg_, wfa_, wfb_, wb_, mlp_gate_, mlp_up_,
        mlp_down_, shared_gate_, shared_up_, shared_down_;
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
};

std::unique_ptr<FamilyEngine> make_glm53() { return std::make_unique<Glm53Engine>(); }

} // namespace mvllm
