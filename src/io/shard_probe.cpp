#include "shard_probe.hpp"
#include "file_io.hpp"
#include "safetensors.hpp"
#include "../store/expert_store.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

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

std::string json_esc(const std::string &s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"')
            o += "\\\"";
        else if (c == '\\')
            o += "\\\\";
        else if (c == '\n')
            o += "\\n";
        else if (c == '\r')
            o += "\\r";
        else
            o += c;
    }
    return o;
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

void inventory_dir(const std::string &dir, const std::string &name, H3ComponentReport &c,
                   std::string &err) {
    c = {};
    c.name = name;
    c.path = dir;
    std::vector<io::StFile> files;
    Status st = io::st_open_dir(dir, files, err);
    if (st != Status::Ok || files.empty()) {
        if (!files.empty())
            io::st_close_dir(files);
        return;
    }
    c.present = true;
    c.n_files = static_cast<int>(files.size());
    for (const auto &f : files) {
        c.n_tensors += static_cast<int>(f.tensors.size());
        for (const auto &t : f.tensors)
            c.bytes += io::st_nbytes(t);
    }
    io::st_close_dir(files);
}

bool read_small_f32(const std::vector<io::StFile> &files, const char *name, int64_t *got,
                    std::string &err) {
    io::StHit h = io::st_find_dir(files, name);
    if (!h.tensor || !h.file)
        return false;
    int64_t n = 1;
    for (int64_t d : h.tensor->shape) {
        if (d <= 0)
            return false;
        n *= d;
    }
    if (n <= 0 || n > 65536)
        return false;
    std::vector<float> buf(static_cast<size_t>(n));
    if (io::st_read_f32(*h.file, *h.tensor, buf.data(), n, err) != Status::Ok)
        return false;
    if (got)
        *got = n * static_cast<int64_t>(sizeof(float));
    return true;
}

bool probe_expert_payload(const std::vector<io::StFile> &files, const std::string &prefix,
                          Family family, int layer, int64_t slot_bytes, int64_t *got,
                          std::string &err) {
    if (slot_bytes <= 0)
        return false;
    static const char *k3_mats[3] = {"w1", "w2", "w3"};
    static const char *k3_half[2] = {"packed", "scale"};
    static const char *glm_names[6] = {"gate_proj.weight", "gate_proj.weight.qs", "up_proj.weight",
                                       "up_proj.weight.qs", "down_proj.weight", "down_proj.weight.qs"};
    ExpertLoc loc;
    loc.key = {0, 0};
    if (family == Family::KimiK3) {
        for (int k = 0; k < 6; ++k) {
            const std::string n = prefix + "model.layers." + std::to_string(layer) +
                                  ".block_sparse_moe.experts.0." + k3_mats[k / 2] + ".weight_" +
                                  k3_half[k & 1];
            io::StHit hit = io::st_find_dir(files, n);
            if (!hit.tensor)
                return false;
            ExpertPiece ep;
            ep.path = hit.file->path;
            ep.offset = io::st_file_offset(*hit.file, *hit.tensor);
            ep.bytes = io::st_nbytes(*hit.tensor);
            loc.pieces.push_back(ep);
        }
    } else if (family == Family::Glm53) {
        const std::string base =
            prefix + "layers." + std::to_string(layer) + ".mlp.experts.0.";
        for (int k = 0; k < 6; ++k) {
            io::StHit hit = io::st_find_dir(files, base + glm_names[k]);
            if (!hit.tensor)
                return false;
            ExpertPiece ep;
            ep.path = hit.file->path;
            ep.offset = io::st_file_offset(*hit.file, *hit.tensor);
            ep.bytes = io::st_nbytes(*hit.tensor);
            loc.pieces.push_back(ep);
        }
    } else {
        return false;
    }
    int64_t sum = 0;
    for (const auto &p : loc.pieces)
        sum += p.bytes;
    loc.contig = true;
    for (size_t k = 1; k < loc.pieces.size(); ++k) {
        if (loc.pieces[k].path != loc.pieces[0].path ||
            loc.pieces[k].offset != loc.pieces[k - 1].offset + loc.pieces[k - 1].bytes)
            loc.contig = false;
    }
    if (loc.contig) {
        loc.path = loc.pieces[0].path;
        loc.offset = loc.pieces[0].offset;
        loc.bytes = sum;
        loc.pieces.clear();
    }
    ExpertStore store;
    std::string serr;
    if (store.open(1, 1, sum, sum + 8192, serr) != Status::Ok)
        return false;
    store.set_direct(true);
    if (store.register_expert(loc, serr) != Status::Ok) {
        store.close();
        return false;
    }
    ExpertView view;
    Status st = store.lookup(loc.key, view, serr);
    bool ok = st == Status::Ok && view.data && view.bytes == sum;
    if (ok && got)
        *got = view.bytes;
    store.release(view);
    store.close();
    (void)slot_bytes;
    return ok;
}

int first_expert_layer(const std::vector<io::StFile> &files, const std::string &prefix,
                       Family family) {
    int best = -1;
    for (const auto &f : files) {
        for (const auto &t : f.tensors) {
            if (family == Family::KimiK3) {
                if (t.name.find(prefix + "model.layers.") != 0)
                    continue;
                if (t.name.find(".block_sparse_moe.experts.0.w1.weight_packed") == std::string::npos)
                    continue;
                int l = parse_after(t.name, "layers.");
                if (l >= 0 && (best < 0 || l < best))
                    best = l;
            } else if (family == Family::Glm53) {
                if (t.name.find(prefix + "layers.") != 0)
                    continue;
                if (t.name.find(".mlp.experts.0.gate_proj.weight") == std::string::npos)
                    continue;
                if (t.name.find(".qs") != std::string::npos)
                    continue;
                int l = parse_after(t.name, "layers.");
                if (l >= 0 && (best < 0 || l < best))
                    best = l;
            }
        }
    }
    return best;
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
        out.synth = true;
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
        int64_t got = 0;
        if (read_small_f32(files, "rope.inv_freq", &got, err) ||
            read_small_f32(files, "time_embedder.proj_in.bias", &got, err)) {
            out.payload_ok = true;
            out.payload_bytes = got;
        } else if (out.prefix_ok) {
            io::StHit qkv = io::st_find_dir(files, "blocks.0.attn.qkv_proj.weight");
            if (qkv.tensor && qkv.file && qkv.file->fd >= 0) {
                const int64_t n = std::min(io::st_nbytes(*qkv.tensor), int64_t{64});
                std::vector<uint8_t> buf(static_cast<size_t>(std::max(n, int64_t{1})));
                if (n > 0 &&
                    io::pread_full(qkv.file->fd, buf.data(), static_cast<size_t>(n),
                                   io::st_file_offset(*qkv.file, *qkv.tensor), err) == Status::Ok) {
                    out.payload_ok = true;
                    out.payload_bytes = n;
                }
            }
        }
        io::st_close_dir(files);

        const char *comp_tails[][2] = {{"transformer", "/FL2VA/transformer"},
                                       {"video_vae", "/FL2VA/video_vae/source"},
                                       {"audio_vae", "/FL2VA/audio_vae"},
                                       {"text_encoder", "/FL2VA/text_encoder"},
                                       {"ref2va", "/Ref2VA/transformer"}};
        for (const auto &row : comp_tails) {
            H3ComponentReport c;
            std::string cerr;
            inventory_dir(model_dir + row[1], row[0], c, cerr);
            if (!c.present && std::strcmp(row[0], "transformer") == 0 && !out.root.empty())
                inventory_dir(out.root, row[0], c, cerr);
            out.components.push_back(c);
        }
        bool live = false;
        for (const auto &c : out.components)
            if (c.present && (c.n_files > 2 || c.bytes > (1ll << 20)))
                live = true;
        if (out.prefix_ok && out.payload_ok) {
            out.shards_ok = true;
            out.note = "h3 dit blocks";
            if (out.payload_ok)
                out.note += "; payload ok";
        } else if (!out.prefix_ok && !live) {
            out.shards_ok = true;
            out.synth = true;
            out.note = "h3 tensors missing qkv_proj; synthetic pack path";
        } else {
            out.shards_ok = false;
            out.note = out.prefix_ok ? "h3 payload read failed" : "h3 live dump missing qkv_proj";
        }
        err.clear();
        return Status::Ok;
    }

    out.n_layers_seen = layers;
    out.n_experts_seen = experts;
    const int nL = cfg.n_layers > 0 ? cfg.n_layers : std::max(layers, 1);
    const int nE = cfg.moe.n_experts > 0 ? cfg.moe.n_experts : std::max(experts, 1);
    const int64_t cap = static_cast<int64_t>(rt.expert_gb * 1024.0 * 1024.0 * 1024.0);
    const int64_t ebytes = out.slot_bytes > 0 ? out.slot_bytes : 4096;
    out.slots_per_layer = expert_store_slots_per_layer(nL, nE, ebytes, cap > 0 ? cap : ebytes);
    const int64_t aligned = (ebytes + 4095) & ~int64_t{4095};
    out.cache_bytes = aligned * static_cast<int64_t>(out.slots_per_layer) * nL;

    const int cfgE = cfg.moe.n_experts;
    if (out.prefix_ok && experts > 0) {
        if (cfgE > 0 && experts != cfgE) {
            out.shards_ok = false;
            out.note = "counted experts != config";
        } else {
            int layer = first_expert_layer(files, out.prefix, cfg.family);
            int64_t got = 0;
            if (layer >= 0 &&
                probe_expert_payload(files, out.prefix, cfg.family, layer, out.slot_bytes, &got,
                                     err)) {
                out.payload_ok = true;
                out.payload_bytes = got;
                out.shards_ok = true;
                out.note = "shard table ok; payload ok";
            } else {
                out.shards_ok = false;
                out.note = "expert payload read failed";
            }
        }
    } else if (cfgE > 0) {
        out.shards_ok = false;
        out.note = "safetensors present but no expert tensors for this family";
    } else {
        out.note = "safetensors present but no expert tensors for this family";
        out.shards_ok = true;
        out.synth = true;
    }
    io::st_close_dir(files);
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
    os << "  synth=" << (r.synth ? "yes" : "no") << " payload_ok=" << (r.payload_ok ? "yes" : "no")
       << " payload_bytes=" << r.payload_bytes << "\n";
    for (const auto &c : r.components) {
        os << "  " << c.name << " present=" << (c.present ? "yes" : "no") << " files=" << c.n_files
           << " tensors=" << c.n_tensors << " bytes=" << c.bytes;
        if (!c.path.empty())
            os << " path=" << c.path;
        os << "\n";
    }
    os << "  note=" << r.note << "\n";
    return os.str();
}

std::string format_shard_report_json(const ShardReport &r) {
    std::ostringstream os;
    os << "{\"family\":\"" << json_esc(family_name(r.family)) << '"'
       << ",\"prefix\":\"" << json_esc(r.prefix) << '"'
       << ",\"root\":\"" << json_esc(r.root) << '"'
       << ",\"note\":\"" << json_esc(r.note) << '"'
       << ",\"n_files\":" << r.n_files << ",\"n_tensors\":" << r.n_tensors
       << ",\"n_layers_seen\":" << r.n_layers_seen << ",\"n_experts_seen\":" << r.n_experts_seen
       << ",\"slot_bytes\":" << r.slot_bytes << ",\"slots_per_layer\":" << r.slots_per_layer
       << ",\"cache_bytes\":" << r.cache_bytes << ",\"payload_bytes\":" << r.payload_bytes
       << ",\"prefix_ok\":" << (r.prefix_ok ? "true" : "false")
       << ",\"shards_ok\":" << (r.shards_ok ? "true" : "false")
       << ",\"synth\":" << (r.synth ? "true" : "false")
       << ",\"payload_ok\":" << (r.payload_ok ? "true" : "false") << ",\"components\":[";
    for (size_t i = 0; i < r.components.size(); ++i) {
        const auto &c = r.components[i];
        if (i)
            os << ',';
        os << "{\"name\":\"" << json_esc(c.name) << "\",\"path\":\"" << json_esc(c.path)
           << "\",\"n_files\":" << c.n_files << ",\"n_tensors\":" << c.n_tensors
           << ",\"bytes\":" << c.bytes << ",\"present\":" << (c.present ? "true" : "false") << '}';
    }
    os << "]}";
    return os.str();
}

} // namespace mvllm
