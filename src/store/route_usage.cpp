#include "route_usage.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>

namespace mvllm {
namespace {

constexpr int kFormatVersion = 1;
constexpr uint32_t kIku1Magic = 0x31554B49u; // "IKU1"
constexpr int kTmpCap = 2100;

constexpr const char *kEngineNames[] = {
    "glm_moe_dsa", "kimi_k3", "olmoe", "inkling", "deepseek_v4", "qwen38",
};

uint32_t fnv1a32(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<unsigned char>(s[i]);
        h *= 16777619u;
    }
    return h;
}

uint32_t fnv1a32(const char *s) {
    return fnv1a32(s, s ? std::strlen(s) : 0);
}

std::string writer_label(uint32_t id) {
    for (const char *name : kEngineNames) {
        if (fnv1a32(name) == id)
            return name;
    }
    return "unknown engine " + std::to_string(id);
}

bool all_zero(const std::vector<std::vector<uint32_t>> &counts) {
    for (const auto &row : counts) {
        for (uint32_t c : row) {
            if (c)
                return false;
        }
    }
    return true;
}

} // namespace

uint32_t route_usage_hash(const std::string &engine) {
    return fnv1a32(engine.data(), engine.size());
}

Status RouteUsage::init(const std::string &engine, int n_layers, int n_experts, std::string &err) {
    counts_.clear();
    engine_.clear();
    engine_id_ = 0;
    n_layers_ = -1;
    n_experts_ = 0;

    if (n_layers < 0 || n_experts < 1) {
        err = "invalid route usage dimensions";
        return Status::InvalidArgument;
    }

    const size_t nrows = static_cast<size_t>(n_layers) + 1;
    counts_.assign(nrows, std::vector<uint32_t>(static_cast<size_t>(n_experts), 0));
    engine_ = engine;
    engine_id_ = route_usage_hash(engine);
    n_layers_ = n_layers;
    n_experts_ = n_experts;
    err.clear();
    return Status::Ok;
}

void RouteUsage::drop_row(int layer) {
    if (layer < 0 || layer > n_layers_)
        return;
    const size_t i = static_cast<size_t>(layer);
    if (i < counts_.size())
        counts_[i].clear();
}

uint32_t *RouteUsage::row(int layer) {
    if (layer < 0 || layer > n_layers_)
        return nullptr;
    const size_t i = static_cast<size_t>(layer);
    if (i >= counts_.size() || counts_[i].empty())
        return nullptr;
    return counts_[i].data();
}

const uint32_t *RouteUsage::row(int layer) const {
    if (layer < 0 || layer > n_layers_)
        return nullptr;
    const size_t i = static_cast<size_t>(layer);
    if (i >= counts_.size() || counts_[i].empty())
        return nullptr;
    return counts_[i].data();
}

bool RouteUsage::admit(int layer, int expert) const {
    return row(layer) != nullptr && expert >= 0 && expert < n_experts_;
}

void RouteUsage::count(int layer, const int *ids, int k) {
    uint32_t *r = row(layer);
    if (!r || !ids || k <= 0)
        return;
    for (int i = 0; i < k; ++i) {
        if (ids[i] >= 0 && ids[i] < n_experts_)
            ++r[ids[i]];
    }
}

uint32_t RouteUsage::get(int layer, int expert) const {
    const uint32_t *r = row(layer);
    if (!r || expert < 0 || expert >= n_experts_)
        return 0;
    return r[expert];
}

int64_t RouteUsage::load(const std::string &path, bool trusted, std::string &err) {
    if (n_layers_ < 0 || counts_.empty()) {
        err = "route usage not initialized";
        return -1;
    }
    if (path.empty()) {
        err = "empty usage path";
        return -1;
    }

    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (errno == ENOENT) {
            err.clear();
            return 0;
        }
        err = "cannot open usage file: " + path;
        return -1;
    }

    uint32_t magic = 0;
    if (std::fread(&magic, 4, 1, f) == 1 && magic == kIku1Magic) {
        std::fclose(f);
        err = "usage file is IKU1 (inkling dense); unsupported";
        return -1;
    }
    std::rewind(f);

    int64_t accepted = 0;
    int layer = 0;
    int field2 = 0;
    unsigned field3 = 0;
    while (std::fscanf(f, "%d %d %u", &layer, &field2, &field3) == 3) {
        if (layer == -1) {
            // -1 <n_layers> <n_experts> — dims never relax, even when trusted.
            if (field2 != n_layers_ || static_cast<int>(field3) != n_experts_) {
                err = "usage file is " + std::to_string(field2) + " layers x " +
                      std::to_string(field3) + " experts, this engine is " +
                      std::to_string(n_layers_) + " x " + std::to_string(n_experts_);
                std::fclose(f);
                return -1;
            }
            continue;
        }
        if (layer == -2) {
            // -2 <version> <engine_id> — version is parse geometry; identity may be trusted.
            if (field2 != kFormatVersion) {
                err = "usage file format version " + std::to_string(field2) +
                      ", this build reads " + std::to_string(kFormatVersion);
                std::fclose(f);
                return -1;
            }
            if (field3 != engine_id_ && !trusted) {
                err = "usage file written by " + writer_label(field3) + ", this engine is " +
                      (engine_.empty() ? "unnamed" : engine_);
                std::fclose(f);
                return -1;
            }
            continue;
        }
        if (layer < 0)
            continue;
        if (!admit(layer, field2))
            continue;
        counts_[static_cast<size_t>(layer)][static_cast<size_t>(field2)] += field3;
        ++accepted;
    }

    std::fclose(f);
    err.clear();
    return accepted;
}

bool RouteUsage::save(const std::string &path, std::string &err) const {
    if (n_layers_ < 0 || counts_.empty()) {
        err = "route usage not initialized";
        return false;
    }
    if (path.empty()) {
        err = "empty usage path";
        return false;
    }

    char tmp[kTmpCap];
    const int n = std::snprintf(tmp, sizeof(tmp), "%s.tmp", path.c_str());
    if (n < 0 || n >= static_cast<int>(sizeof(tmp))) {
        err = n < 0 ? "temp path encoding error" : "path too long";
        return false;
    }

    std::FILE *f = std::fopen(tmp, "w");
    if (!f) {
        err = "cannot open temp usage file";
        return false;
    }

    bool ok = true;
    if (!all_zero(counts_)) {
        if (std::fprintf(f, "-1 %d %d\n", n_layers_, n_experts_) < 0)
            ok = false;
        if (ok && std::fprintf(f, "-2 %d %u\n", kFormatVersion, engine_id_) < 0)
            ok = false;
        for (int layer = 0; ok && layer <= n_layers_; ++layer) {
            const uint32_t *r = row(layer);
            if (!r)
                continue;
            for (int e = 0; e < n_experts_; ++e) {
                if (!r[e])
                    continue;
                if (std::fprintf(f, "%d %d %u\n", layer, e, r[e]) < 0) {
                    ok = false;
                    break;
                }
            }
        }
    }
    if (std::ferror(f))
        ok = false;
    std::fclose(f);

    if (!ok) {
        std::remove(tmp);
        err = "write failed";
        return false;
    }
    if (std::rename(tmp, path.c_str()) != 0) {
        std::remove(tmp);
        err = "rename failed";
        return false;
    }
    err.clear();
    return true;
}

void RouteUsage::decay(double factor) {
    if (!(factor > 0.0 && factor <= 1.0) || factor == 1.0)
        return;
    for (auto &r : counts_) {
        for (uint32_t &c : r) {
            if (!c)
                continue;
            uint32_t next = static_cast<uint32_t>(static_cast<double>(c) * factor + 0.5);
            c = next == 0 ? 1u : next;
        }
    }
}

} // namespace mvllm
