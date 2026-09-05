#include "engine.hpp"
#include "gpu/backend.hpp"
#include "io/image.hpp"
#include "tok/decode_post.hpp"
#include "tok/k3_chat1.hpp"

#include <sstream>

namespace mvllm {

Status Engine::load(const std::string &model_dir, const RuntimeConfig &rt, std::string &err) {
    rt_ = rt;
    rt_.model_dir = model_dir;
    model_dir_ = model_dir;
    gpu::select(rt_.device);
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
    sessions_.configure(rt_.kv_slots);
    sched_.bind(this, &sessions_);
    return Status::Ok;
}

Status Engine::generate(const std::string &prompt, const GenParams &gp, GenResult &out,
                        std::string &err) {
    if (prompt.size() >= 8 && prompt.compare(0, 8, "K3CHAT1\n") == 0) {
        K3Chat1 parsed;
        if (!k3_chat1_parse(prompt, parsed, err))
            return Status::InvalidArgument;
        GenParams g2 = gp;
        g2.think = parsed.think;
        if (!parsed.tools.empty())
            g2.tools = parsed.tools;
        return generate_chat(parsed.msgs, g2, out, err);
    }
    GenParams g2 = gp;
    if (!g2.token_text) {
        g2.token_text = [this](int id) {
            std::string s;
            tok_.decode({id}, s);
            return s;
        };
    }
    if (g2.apply_template && (family_ == Family::Glm53 || family_ == Family::KimiK3)) {
        ChatMessage m;
        m.role = "user";
        m.content = prompt;
        return generate_chat({m}, g2, out, err);
    }
    std::vector<int> ids;
    Status st = tok_.encode(prompt, ids);
    if (st != Status::Ok) {
        err = "tokenize failed";
        return st;
    }
    if (ids.empty())
        ids.push_back(cfg_.bos);
    st = generate_ids(ids, g2, out, err);
    if (st != Status::Ok)
        return st;
    std::string text;
    tok_.decode(out.tokens, text);
    out.text = text;
    if (trim_stop(out.text, gp.stop))
        out.stopped_by_stop = true;
    std::string visible;
    split_assistant_text(family_, out.text, out.reasoning, visible);
    out.text = visible;
    out.reasoning_tokens = 0;
    if (!out.reasoning.empty()) {
        std::vector<int> rids;
        if (tok_.encode(out.reasoning, rids) == Status::Ok)
            out.reasoning_tokens = static_cast<int>(rids.size());
    }
    return Status::Ok;
}

Status Engine::generate_chat(const std::vector<ChatMessage> &msgs, const GenParams &gp,
                             GenResult &out, std::string &err) {
    std::vector<int> ids;
    Status st = tok_.encode_chat(family_, msgs, gp.think, ids, gp.reasoning_effort,
                                 gp.tools.empty() ? nullptr : &gp.tools);
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
    // GLM chat: stop before a new user/tool turn unless the client set stop.
    if (family_ == Family::Glm53 && g2.stop.empty()) {
        g2.stop.push_back("<|user|>");
        g2.stop.push_back("<|observation|>");
    }
    if (!g2.token_text) {
        g2.token_text = [this](int id) {
            std::string s;
            tok_.decode({id}, s);
            return s;
        };
    }
    std::vector<float> owned_rgb;
    if (!g2.image_rgb) {
        int iw = 0, ih = 0;
        std::string ierr;
        for (const auto &m : msgs) {
            for (const std::string &u : m.image_urls) {
                if (decode_image_url(u, owned_rgb, iw, ih, ierr) == Status::Ok && iw > 0 &&
                    ih > 0) {
                    g2.image_rgb = owned_rgb.data();
                    g2.image_w = iw;
                    g2.image_h = ih;
                    break;
                }
            }
            if (g2.image_rgb)
                break;
        }
    }
    if (g2.image_token < 0) {
        int img = tok_.id_of("<image>");
        if (img < 0)
            img = tok_.id_of("<|image|>");
        if (img < 0)
            img = cfg_.vision.image_token;
        if (img >= 0)
            g2.image_token = img;
    }
    if (g2.eos < 0 && family_ == Family::KimiK3) {
        int eom = tok_.id_of("<|end_of_msg|>");
        if (eom >= 0)
            g2.eos = eom;
    }
    if (!msgs.empty() && sessions_.n_slots() > 1)
        g2.cache_slot = sessions_.assign(msgs, g2.cache_slot);
    st = generate_ids(ids, g2, out, err);
    if (st != Status::Ok)
        return st;
    std::string text;
    tok_.decode(out.tokens, text);
    out.text = text;
    if (trim_stop(out.text, g2.stop))
        out.stopped_by_stop = true;
    std::string visible;
    split_assistant_text(family_, out.text, out.reasoning, visible);
    out.text = visible;
    out.reasoning_tokens = 0;
    if (!out.reasoning.empty()) {
        std::vector<int> rids;
        if (tok_.encode(out.reasoning, rids) == Status::Ok)
            out.reasoning_tokens = static_cast<int>(rids.size());
    }
    return Status::Ok;
}

Status Engine::generate_ids(const std::vector<int> &ids, const GenParams &gp, GenResult &out,
                            std::string &err) {
    if (!impl_) {
        err = "engine not loaded";
        return Status::InvalidArgument;
    }
    GenParams g2 = gp;
    if (!g2.token_text) {
        g2.token_text = [this](int id) {
            std::string s;
            tok_.decode({id}, s);
            return s;
        };
    }
    const int slot = g2.cache_slot >= 0 ? g2.cache_slot : 0;
    g2.prefix_reuse = sessions_.match(slot, ids);
    g2.cache_slot = slot;
    Status st = impl_->generate(ids, g2, out, err);
    if (st == Status::Ok) {
        std::vector<int> hist;
        hist.reserve(ids.size() + out.tokens.size());
        hist.insert(hist.end(), ids.begin(), ids.end());
        hist.insert(hist.end(), out.tokens.begin(), out.tokens.end());
        sessions_.commit(slot, hist);
    }
    return st;
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
       << " device=" << device_name(rt_.device) << " gpu=" << gpu::name()
       << " compiled=" << gpu::compiled() << "\n";
    return os.str();
}

} // namespace mvllm
