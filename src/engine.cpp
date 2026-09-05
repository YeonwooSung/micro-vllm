#include "engine.hpp"
#include "gpu/backend.hpp"
#include "io/image.hpp"
#include "tok/decode_post.hpp"
#include "tok/k3_chat1.hpp"
#include "tok/stop_set.hpp"

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
    if (!rt_.kv_path.empty()) {
        std::string perr;
        if (persist_open(rt_.kv_path, rt_.kv_persist_ver, perr) == Status::Ok)
            sessions_.commit(0, persist_hist_);
    }
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
    if (!g2.persist_path.empty()) {
        const int ver = g2.persist_ver ? g2.persist_ver : rt_.kv_persist_ver;
        Status pst = persist_open(g2.persist_path, ver, err);
        if (pst != Status::Ok)
            return pst;
    }

    std::vector<int> stops = g2.stop_ids;
    stops.insert(stops.end(), cfg_.eos_ids.begin(), cfg_.eos_ids.end());
    const int eos = g2.eos >= 0 ? g2.eos : cfg_.eos;
    int n_specials = cfg_.vocab;
    if (tok_.vocab_size() > n_specials)
        n_specials = tok_.vocab_size();
    if (n_specials < 1)
        n_specials = 1;
    std::vector<uint8_t> specials(static_cast<size_t>(n_specials), 0);
    static const char *kStopSpecials[] = {
        "<|endoftext|>", "<|eot_id|>",     "<|im_end|>", "<|end_of_msg|>",
        "<|user|>",      "<|observation|>", "</s>",      "<|eom_id|>",
    };
    for (const char *s : kStopSpecials) {
        const int id = tok_.id_of(s);
        if (id >= 0 && id < n_specials)
            specials[static_cast<size_t>(id)] = 1;
    }
    StopSet ss;
    ss.arm(stops.empty() ? nullptr : stops.data(), static_cast<int>(stops.size()), eos,
           specials.data(), n_specials, g2.eos_only);
    g2.stop_ids.assign(ss.data(), ss.data() + ss.size());
    if (g2.eos < 0 && eos >= 0)
        g2.eos = eos;

    int slot = g2.cache_slot >= 0 ? g2.cache_slot : 0;
    if (slot < 0 || slot >= kMaxKvSlots)
        slot = 0;
    if (prefixes_[slot].cap() <= 0) {
        int pcap = static_cast<int>(ids.size()) + g2.max_new_tokens + rt_.max_seq;
        if (pcap < 8)
            pcap = 8;
        prefixes_[slot].alloc(pcap);
        const std::vector<int> &prev = sessions_.history(slot);
        if (!prev.empty())
            prefixes_[slot].record(prev.data(), 0, static_cast<int>(prev.size()));
    }
    const int official = prefixes_[slot].reuse(ids.empty() ? nullptr : ids.data(),
                                               static_cast<int>(ids.size()));
    if (official > 0)
        g2.prefix_reuse = official;
    else if (g2.prefix_reuse < 0)
        g2.prefix_reuse = 0;
    g2.cache_slot = slot;

    Status st = impl_->generate(ids, g2, out, err);
    if (st == Status::Ok) {
        std::vector<int> hist;
        hist.reserve(ids.size() + out.tokens.size());
        hist.insert(hist.end(), ids.begin(), ids.end());
        hist.insert(hist.end(), out.tokens.begin(), out.tokens.end());
        sessions_.commit(slot, hist);
        const int n = static_cast<int>(hist.size());
        if (prefixes_[slot].cap() < n)
            prefixes_[slot].grow(n + 64, 0);
        if (n > 0)
            prefixes_[slot].record(hist.data(), 0, n);
        if (persist_open_) {
            Status ast = persist_append_tail(slot, hist, err);
            persist_hist_ = hist;
            if (ast != Status::Ok)
                err.clear();
        }
    }
    return st;
}

KvPersistConfig Engine::make_persist_cfg() const {
    KvPersistConfig pcfg;
    pcfg.n_layers = cfg_.n_layers > 0 ? cfg_.n_layers : 1;
    pcfg.kv_lora = cfg_.mla.kv_lora > 0 ? cfg_.mla.kv_lora : 0;
    pcfg.qk_rope = cfg_.mla.qk_rope > 0 ? cfg_.mla.qk_rope : 0;
    pcfg.index_hd = cfg_.dsa.head_dim > 0 ? cfg_.dsa.head_dim : 0;
    pcfg.vocab = cfg_.vocab > 0 ? cfg_.vocab : 0;
    pcfg.has_index.assign(static_cast<size_t>(pcfg.n_layers), 0);
    const int nfull = static_cast<int>(cfg_.is_full.size());
    for (int i = 0; i < pcfg.n_layers && i < nfull; ++i) {
        if (cfg_.is_full[static_cast<size_t>(i)])
            pcfg.has_index[static_cast<size_t>(i)] = 1;
    }
    return pcfg;
}

Status Engine::persist_open(const std::string &path, int ver, std::string &err) {
    persist_close();
    if (ver != 1 && ver != 2 && ver != 3) {
        err = "unsupported kv persist version";
        return Status::InvalidArgument;
    }
    const KvPersistConfig pcfg = make_persist_cfg();
    Status st = Status::InvalidArgument;
    if (ver == 1)
        st = persist_v1_.open(path, pcfg, err);
    else if (ver == 2)
        st = persist_v2_.open(path, pcfg, err);
    else {
        KvPersistV3Options opt;
        opt.codec = rt_.kv_tq_codec;
        opt.bits = rt_.kv_tq_bits;
        st = persist_v3_.open(path, pcfg, opt, err);
    }
    if (st != Status::Ok) {
        persist_close();
        return st;
    }
    persist_path_ = path;
    persist_ver_ = ver;
    persist_open_ = true;

    persist_hist_.clear();
    std::string lerr;
    if (ver == 1)
        persist_v1_.load(persist_hist_, nullptr, lerr);
    else if (ver == 2)
        persist_v2_.load(persist_hist_, nullptr, lerr);
    else
        persist_v3_.load(persist_hist_, nullptr, lerr);

    const int n = static_cast<int>(persist_hist_.size());
    int cap = n + rt_.max_seq;
    if (cap < 1)
        cap = 1;
    prefixes_[0].alloc(cap);
    if (n > 0)
        prefixes_[0].record(persist_hist_.data(), 0, n);
    return Status::Ok;
}

void Engine::persist_close() {
    persist_v1_.close();
    persist_v2_.close();
    persist_v3_.close();
    persist_hist_.clear();
    persist_path_.clear();
    persist_ver_ = 0;
    persist_open_ = false;
}

int Engine::persist_nrec() const {
    if (!persist_open_)
        return 0;
    if (persist_ver_ == 1)
        return persist_v1_.nrec();
    if (persist_ver_ == 2)
        return persist_v2_.nrec();
    if (persist_ver_ == 3)
        return persist_v3_.nrec();
    return 0;
}

const std::vector<int> &Engine::persist_hist() const { return persist_hist_; }

int Engine::prefix_reuse_len(int slot) const {
    if (slot < 0 || slot >= kMaxKvSlots)
        return 0;
    return prefixes_[slot].len();
}

Status Engine::persist_append_tail(int slot, const std::vector<int> &hist, std::string &err) {
    if (!persist_open_)
        return Status::Ok;
    const int nrec = persist_nrec();
    const int hist_n = static_cast<int>(hist.size());
    const int rec_n = hist_n - nrec;
    if (rec_n <= 0)
        return Status::Ok;
    const KvPersistConfig *pc = nullptr;
    if (persist_ver_ == 1)
        pc = &persist_v1_.config();
    else if (persist_ver_ == 2)
        pc = &persist_v2_.config();
    else
        pc = &persist_v3_.config();
    const size_t want_l = static_cast<size_t>(pc->n_layers) * static_cast<size_t>(pc->kv_lora);
    const size_t want_r = static_cast<size_t>(pc->n_layers) * static_cast<size_t>(pc->qk_rope);
    int n_index = 0;
    for (uint8_t h : pc->has_index) {
        if (h)
            ++n_index;
    }
    const size_t want_i = static_cast<size_t>(n_index) * static_cast<size_t>(pc->index_hd);
    std::vector<KvPersistRecord> recs(static_cast<size_t>(rec_n));
    for (int i = 0; i < rec_n; ++i) {
        recs[static_cast<size_t>(i)].token = hist[static_cast<size_t>(nrec + i)];
        recs[static_cast<size_t>(i)].L.assign(want_l, 0.f);
        recs[static_cast<size_t>(i)].R.assign(want_r, 0.f);
        recs[static_cast<size_t>(i)].I.assign(want_i, 0.f);
    }
    if (impl_)
        impl_->export_kv_rows(slot, nrec, rec_n, recs.data());
    if (persist_ver_ == 1)
        return persist_v1_.append(hist.data(), hist_n, recs.data(), rec_n, err);
    if (persist_ver_ == 2)
        return persist_v2_.append(hist.data(), hist_n, recs.data(), rec_n, err);
    return persist_v3_.append(hist.data(), hist_n, recs.data(), rec_n, err);
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
