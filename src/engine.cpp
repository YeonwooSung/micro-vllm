#include "engine.hpp"

#include <sstream>

namespace mvllm {

Status Engine::load(const std::string &model_dir, const RuntimeConfig &rt, std::string &err) {
    rt_ = rt;
    rt_.model_dir = model_dir;
    model_dir_ = model_dir;
    Status st = load_model_config(model_dir, cfg_, err);
    if (st != Status::Ok) {
        Family sniffed = sniff_family(model_dir);
        if (sniffed != Family::H3)
            return st;
        cfg_ = ModelConfig{};
        cfg_.family = Family::H3;
        apply_family_defaults(cfg_);
        err.clear();
    }
    family_ = cfg_.family;
    impl_ = make_engine(family_);
    if (!impl_) {
        err = "no engine for family " + std::string(family_name(family_));
        return Status::Unsupported;
    }
    st = impl_->load(model_dir, rt_, err);
    if (st != Status::Ok)
        return st;
    cfg_ = impl_->config();
    family_ = cfg_.family;
    tok_.load(model_dir, err);
    err.clear();
    model_id_ = family_name(family_);
    if (!cfg_.architecture.empty())
        model_id_ = cfg_.architecture;
    return Status::Ok;
}

Status Engine::generate(const std::string &prompt, const GenParams &gp, GenResult &out,
                        std::string &err) {
    if (gp.apply_template && (family_ == Family::Glm53 || family_ == Family::KimiK3)) {
        ChatMessage m;
        m.role = "user";
        m.content = prompt;
        return generate_chat({m}, gp, out, err);
    }
    std::vector<int> ids;
    Status st = tok_.encode(prompt, ids);
    if (st != Status::Ok) {
        err = "tokenize failed";
        return st;
    }
    if (ids.empty())
        ids.push_back(cfg_.bos);
    st = generate_ids(ids, gp, out, err);
    if (st != Status::Ok)
        return st;
    std::string text;
    tok_.decode(out.tokens, text);
    out.text = text;
    return Status::Ok;
}

Status Engine::generate_chat(const std::vector<ChatMessage> &msgs, const GenParams &gp,
                             GenResult &out, std::string &err) {
    std::vector<int> ids;
    Status st = tok_.encode_chat(family_, msgs, gp.think, ids);
    if (st != Status::Ok) {
        err = "tokenize failed";
        return st;
    }
    if (family_ == Family::KimiK3 && cfg_.bos >= 0 &&
        (ids.empty() || ids.front() != cfg_.bos))
        ids.insert(ids.begin(), cfg_.bos);
    if (ids.empty())
        ids.push_back(cfg_.bos);
    GenParams g2 = gp;
    if (g2.eos < 0 && family_ == Family::KimiK3) {
        int eom = tok_.id_of("<|end_of_msg|>");
        if (eom >= 0)
            g2.eos = eom;
    }
    st = generate_ids(ids, g2, out, err);
    if (st != Status::Ok)
        return st;
    std::string text;
    tok_.decode(out.tokens, text);
    out.text = text;
    return Status::Ok;
}

Status Engine::generate_ids(const std::vector<int> &ids, const GenParams &gp, GenResult &out,
                            std::string &err) {
    if (!impl_) {
        err = "engine not loaded";
        return Status::InvalidArgument;
    }
    return impl_->generate(ids, gp, out, err);
}

Status Engine::generate_video(const H3GenParams &hp, H3GenResult &out, std::string &err) {
    if (!impl_) {
        err = "engine not loaded";
        return Status::InvalidArgument;
    }
    return impl_->generate_video(hp, out, err);
}

std::string Engine::info() const {
    std::ostringstream os;
    os << "micro-vllm  family=" << family_name(family_) << "  dir=" << model_dir_ << "\n";
    os << "  hidden=" << cfg_.hidden << " layers=" << cfg_.n_layers << " vocab=" << cfg_.vocab
       << "\n";
    if (cfg_.moe.n_experts)
        os << "  moe experts=" << cfg_.moe.n_experts << " topk=" << cfg_.moe.topk
           << " inter=" << cfg_.moe.intermediate << " latent=" << cfg_.moe.latent << "\n";
    if (cfg_.kda.heads)
        os << "  kda heads=" << cfg_.kda.heads << " dim=" << cfg_.kda.head_dim
           << " conv=" << cfg_.kda.conv_k << "\n";
    if (cfg_.mla.kv_lora)
        os << "  mla q_lora=" << cfg_.mla.q_lora << " kv_lora=" << cfg_.mla.kv_lora
           << " nope=" << cfg_.mla.qk_nope << "\n";
    if (cfg_.mhc.mult)
        os << "  mhc mult=" << cfg_.mhc.mult << " iters=" << cfg_.mhc.iters << "\n";
    if (family_ == Family::H3)
        os << "  h3 dit=" << cfg_.h3.dit_layers << " canvas=" << cfg_.h3.default_width << "x"
           << cfg_.h3.default_height << "\n";
    if (impl_)
        os << "  " << impl_->describe() << "\n";
    os << "  expert_gb=" << rt_.expert_gb << " bits=" << rt_.dense_bits
       << " device=" << static_cast<int>(rt_.device) << "\n";
    return os.str();
}

} // namespace mvllm
