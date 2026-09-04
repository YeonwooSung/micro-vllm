#include "family.hpp"
#include "../quant/quant.hpp"

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
void ones(std::vector<float> &w, int n) { w.assign(n, 1.f); }

} // namespace

class LlamaEngine final : public FamilyEngine {
public:
    Family family() const override { return Family::Llama; }
    const ModelConfig &config() const override { return cfg_; }

    Status load(const std::string &model_dir, const RuntimeConfig &rt, std::string &err) override {
        rt_ = rt;
        Status st = load_model_config(model_dir, cfg_, err);
        if (st != Status::Ok)
            return st;
        apply_family_defaults(cfg_);
        alloc_synthetic();
        loaded_ = true;
        return Status::Ok;
    }

    Status generate(const std::vector<int> &prompt, const GenParams &gp, GenResult &out,
                    std::string &err) override {
        if (!loaded_) {
            err = "llama not loaded";
            return Status::InvalidArgument;
        }
        if (prompt.empty()) {
            err = "empty prompt";
            return Status::InvalidArgument;
        }
        const int H = cfg_.hidden;
        const int I = cfg_.dense_intermediate;
        std::vector<float> h(H), n(H), q(H), g(I), u(I), d(H);
        auto embed = [&](int id) {
            int tid = id;
            if (tid < 0 || tid >= cfg_.vocab)
                tid = 0;
            std::memcpy(h.data(), embed_.data() + static_cast<size_t>(tid) * H, H * sizeof(float));
        };
        auto step = [&](int token) {
            embed(token);
            for (int l = 0; l < cfg_.n_layers; ++l) {
                quant::rmsnorm(h.data(), in_n_[l].data(), n.data(), H, cfg_.rms_eps);
                quant::matmul_f32(q.data(), n.data(), wq_[l].data(), 1, H, H);
                quant::matmul_f32(d.data(), q.data(), wo_[l].data(), 1, H, H);
                for (int i = 0; i < H; ++i)
                    h[i] += d[i];
                quant::rmsnorm(h.data(), out_n_[l].data(), n.data(), H, cfg_.rms_eps);
                quant::matmul_f32(g.data(), n.data(), gate_[l].data(), 1, H, I);
                quant::matmul_f32(u.data(), n.data(), up_[l].data(), 1, H, I);
                quant::silu_mul(g.data(), u.data(), I);
                quant::matmul_f32(d.data(), g.data(), down_[l].data(), 1, I, H);
                for (int i = 0; i < H; ++i)
                    h[i] += d[i];
            }
        };
        for (int t : prompt)
            step(t);
        out.tokens.clear();
        out.prompt_tokens = static_cast<int>(prompt.size());
        for (int k = 0; k < gp.max_new_tokens; ++k) {
            std::vector<float> logits(cfg_.vocab);
            quant::rmsnorm(h.data(), norm_.data(), n.data(), H, cfg_.rms_eps);
            quant::matmul_f32(logits.data(), n.data(), lm_head_.data(), 1, H, cfg_.vocab);
            int next = 0;
            float best = logits[0];
            for (int i = 1; i < cfg_.vocab; ++i) {
                if (logits[i] > best) {
                    best = logits[i];
                    next = i;
                }
            }
            out.tokens.push_back(next);
            if ((gp.eos >= 0 && next == gp.eos) || (cfg_.eos && next == cfg_.eos))
                break;
            step(next);
        }
        out.completion_tokens = static_cast<int>(out.tokens.size());
        return Status::Ok;
    }

    std::string describe() const override {
        std::ostringstream os;
        os << "llama  hidden=" << cfg_.hidden << " layers=" << cfg_.n_layers
           << " q_heads=" << cfg_.n_q_heads << " kv_heads=" << cfg_.n_kv_heads
           << " (CPU stand-in; CUDA demo is micro-vllm-cuda)";
        return os.str();
    }

private:
    void alloc_synthetic() {
        const int H = cfg_.hidden;
        const int L = cfg_.n_layers;
        const int V = cfg_.vocab;
        const int I = cfg_.dense_intermediate;
        xavier(embed_, V, H, 5);
        ones(norm_, H);
        xavier(lm_head_, V, H, 6);
        in_n_.resize(L);
        out_n_.resize(L);
        wq_.resize(L);
        wo_.resize(L);
        gate_.resize(L);
        up_.resize(L);
        down_.resize(L);
        for (int l = 0; l < L; ++l) {
            ones(in_n_[l], H);
            ones(out_n_[l], H);
            xavier(wq_[l], H, H, 400 + l);
            xavier(wo_[l], H, H, 410 + l);
            xavier(gate_[l], I, H, 420 + l);
            xavier(up_[l], I, H, 430 + l);
            xavier(down_[l], H, I, 440 + l);
        }
    }

    ModelConfig cfg_{};
    RuntimeConfig rt_{};
    bool loaded_ = false;
    std::vector<float> embed_, norm_, lm_head_;
    std::vector<std::vector<float>> in_n_, out_n_, wq_, wo_, gate_, up_, down_;
};

std::unique_ptr<FamilyEngine> make_llama() { return std::make_unique<LlamaEngine>(); }

} // namespace mvllm
