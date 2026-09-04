#include "shard_probe.hpp"
#include "safetensors.hpp"
#include "../store/expert_store.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace mvllm {
namespace {

int parse_after(const std::string &name, const char *key) {
    auto p = name.find(key);
    if (p == std::string::npos)
        return -1;
    p += std::strlen(key);
    if (p >= name.size() || name[p] < '0' || name[p] > '9')
        return -1;
    return std::atoi(name.c_str() + static_cast<int>(p));
}

void tally_k3(const std::string &name, const std::string &prefix, int &layers, int &experts) {
    const std::string mid = prefix + "model.layers.";
    if (name.compare(0, mid.size(), mid) != 0)
        return;
    if (name.find(".block_sparse_moe.experts.") == std::string::npos)
        return;
    if (name.find("weight_packed") == std::string::npos)
        return;
    int l = parse_after(name, "layers.");
    int e = parse_after(name, "experts.");
    if (l + 1 > layers)
        layers = l + 1;
    if (e + 1 > experts)
        experts = e + 1;
}

void tally_glm(const std::string &name, const std::string &prefix, int &layers, int &experts) {
    const std::string mid = prefix + "layers.";
    if (name.compare(0, mid.size(), mid) != 0)
        return;
    if (name.find(".mlp.experts.") == std::string::npos)
        return;
    if (name.find("gate_proj.weight") == std::string::npos ||
        name.find("gate_proj.weight.qs") != std::string::npos)
        return;
    int l = parse_after(name, "layers.");
    int e = parse_after(name, "experts.");
    if (l + 1 > layers)
        layers = l + 1;
    if (e + 1 > experts)
        experts = e + 1;
}

void tally_h3(const std::string &name, int &layers) {
    if (name.compare(0, 7, "blocks.") != 0)
        return;
    if (name.find("attn.qkv_proj.weight") == std::string::npos)
        return;
    int b = parse_after(name, "blocks.");
    if (b + 1 > layers)
        layers = b + 1;
}

} // namespace

int64_t k3_expert_slot_bytes(int latent, int inter) {
    if (latent <= 0 || inter <= 0)
        return 0;
    int64_t w1p, w1s, w2p, w2s;
    if (latent >= 32 && inter >= 32 && latent % 32 == 0 && inter % 32 == 0) {
        w1p = static_cast<int64_t>(inter) * (latent / 2);
        w1s = static_cast<int64_t>(inter) * (latent / 32);
        w2p = static_cast<int64_t>(latent) * (inter / 2);
        w2s = static_cast<int64_t>(latent) * (inter / 32);
    } else {
        w1p = static_cast<int64_t>(inter) * ((latent + 1) / 2);
        w1s = static_cast<int64_t>(inter) * ((latent + 31) / 32);
        w2p = static_cast<int64_t>(latent) * ((inter + 1) / 2);
        w2s = static_cast<int64_t>(latent) * ((inter + 31) / 32);
    }
    return 2 * (w1p + w1s) + w2p + w2s;
}

int64_t glm53_expert_slot_bytes(int hidden, int inter) {
    if (hidden <= 0 || inter <= 0)
        return 0;
    int64_t pack_go, sc_go, pack_d, sc_d;
    if (hidden % 64 == 0 && inter % 64 == 0) {
        const int64_t pack = static_cast<int64_t>(inter) * hidden / 2;
        const int64_t sc =
            static_cast<int64_t>(inter) * hidden / 64 * static_cast<int64_t>(sizeof(float));
        pack_go = pack_d = pack;
        sc_go = sc_d = sc;
    } else {
        pack_go = static_cast<int64_t>(inter) * ((hidden + 1) / 2);
        sc_go = static_cast<int64_t>(inter) * ((hidden + 63) / 64) * static_cast<int64_t>(sizeof(float));
        pack_d = static_cast<int64_t>(hidden) * ((inter + 1) / 2);
        sc_d = static_cast<int64_t>(hidden) * ((inter + 63) / 64) * static_cast<int64_t>(sizeof(float));
    }
    return 2 * (pack_go + sc_go) + pack_d + sc_d;
}

Status probe_shards(const std::string &model_dir, const RuntimeConfig &rt, ShardReport &out,
                    std::string &err) {
    out = {};
    ModelConfig cfg;
    Status st = load_model_config(model_dir, cfg, err);
    if (st != Status::Ok) {
        Family sniffed = sniff_family(model_dir);
        if (sniffed != Family::H3)
            return st;
        cfg.family = Family::H3;
        apply_family_defaults(cfg);
        err.clear();
        st = Status::Ok;
    }
    apply_family_defaults(cfg);
    out.family = cfg.family;

    const char *k3_pref[] = {"language_model.", "", nullptr};
    const char *glm_pref[] = {"model.language_model.", "model.", "", nullptr};
    const char *h3_tails[] = {"/FL2VA/transformer", "/transformer", "/dit", "", nullptr};

    std::vector<io::StFile> files;
    if (cfg.family == Family::H3) {
        for (int i = 0; h3_tails[i]; ++i) {
            files.clear();
            std::string dir = model_dir + h3_tails[i];
            Status ost = io::st_open_dir(dir, files, err);
            if (ost == Status::ParseError)
                return ost;
            if (ost != Status::Ok || files.empty())
                continue;
            if (io::st_find_dir(files, "blocks.0.attn.qkv_proj.weight").tensor) {
                out.root = dir;
                out.prefix_ok = true;
                break;
            }
            io::st_close_dir(files);
        }
    } else {
        Status ost = io::st_open_dir(model_dir, files, err);
        if (ost != Status::Ok)
            return ost;
        out.root = model_dir;
    }

    out.n_files = static_cast<int>(files.size());
    for (const auto &f : files)
        out.n_tensors += static_cast<int>(f.tensors.size());

    if (files.empty()) {
        out.note = "no safetensors; synthetic pack path";
        out.shards_ok = true;
        err.clear();
        return Status::Ok;
    }

    int layers = 0, experts = 0;
    if (cfg.family == Family::KimiK3) {
        for (int i = 0; k3_pref[i]; ++i) {
            int L = 0, E = 0;
            for (const auto &f : files)
                for (const auto &t : f.tensors)
                    tally_k3(t.name, k3_pref[i], L, E);
            if (E > 0) {
                out.prefix = k3_pref[i];
                layers = L;
                experts = E;
                out.prefix_ok = true;
                break;
            }
        }
        int lat = cfg.moe.latent > 0 ? cfg.moe.latent : cfg.hidden;
        int inter = cfg.moe.intermediate > 0 ? cfg.moe.intermediate : 32;
        out.slot_bytes = k3_expert_slot_bytes(lat, inter);
    } else if (cfg.family == Family::Glm53) {
        for (int i = 0; glm_pref[i]; ++i) {
            int L = 0, E = 0;
            for (const auto &f : files)
                for (const auto &t : f.tensors)
                    tally_glm(t.name, glm_pref[i], L, E);
            if (E > 0) {
                out.prefix = glm_pref[i];
                layers = L;
                experts = E;
                out.prefix_ok = true;
                break;
            }
        }
        int H = cfg.hidden > 0 ? cfg.hidden : 32;
        int inter = cfg.moe.intermediate > 0 ? cfg.moe.intermediate : 32;
        out.slot_bytes = glm53_expert_slot_bytes(H, inter);
    } else if (cfg.family == Family::H3) {
        for (const auto &f : files)
            for (const auto &t : f.tensors)
                tally_h3(t.name, layers);
        out.prefix_ok = layers > 0;
        out.n_layers_seen = layers;
        out.shards_ok = out.prefix_ok;
        out.note = out.prefix_ok ? "h3 dit blocks" : "h3 tensors missing qkv_proj";
        io::st_close_dir(files);
        err.clear();
        return Status::Ok;
    }

    io::st_close_dir(files);
    out.n_layers_seen = layers;
    out.n_experts_seen = experts;
    const int nL = cfg.n_layers > 0 ? cfg.n_layers : std::max(layers, 1);
    const int nE = cfg.moe.n_experts > 0 ? cfg.moe.n_experts : std::max(experts, 1);
    const int64_t cap = static_cast<int64_t>(rt.expert_gb * 1024.0 * 1024.0 * 1024.0);
    const int64_t ebytes = out.slot_bytes > 0 ? out.slot_bytes : 4096;
    out.slots_per_layer = expert_store_slots_per_layer(nL, nE, ebytes, cap > 0 ? cap : ebytes);
    const int64_t aligned = (ebytes + 4095) & ~int64_t{4095};
    out.cache_bytes = aligned * static_cast<int64_t>(out.slots_per_layer) * nL;
    if (out.prefix_ok && experts > 0) {
        out.shards_ok = true;
        out.note = "shard table ok";
        if (cfg.moe.n_experts > 0 && experts != cfg.moe.n_experts)
            out.note += "; counted experts != config";
    } else {
        out.note = "safetensors present but no expert tensors for this family";
        out.shards_ok = true; // dense-only overlay still valid
    }
    err.clear();
    return Status::Ok;
}

std::string format_shard_report(const ShardReport &r) {
    std::ostringstream os;
    os << "smoke  family=" << family_name(r.family) << " prefix=" << (r.prefix.empty() ? "-" : r.prefix)
       << " files=" << r.n_files << " tensors=" << r.n_tensors << "\n";
    os << "  layers_seen=" << r.n_layers_seen << " experts_seen=" << r.n_experts_seen
       << " slot_bytes=" << r.slot_bytes << "\n";
    os << "  slots_per_layer=" << r.slots_per_layer << " cache_bytes=" << r.cache_bytes
       << " prefix_ok=" << (r.prefix_ok ? "yes" : "no") << "\n";
    os << "  note=" << r.note << "\n";
    return os.str();
}

} // namespace mvllm
