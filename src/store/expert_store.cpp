#include "expert_store.hpp"

#include "../io/file_io.hpp"

#include <algorithm>
#include <limits>

namespace mvllm {

int expert_store_slots_per_layer(int n_layers, int n_experts, int64_t expert_bytes,
                                 int64_t capacity_bytes) {
    if (n_layers <= 0 || n_experts <= 0 || expert_bytes <= 0 || capacity_bytes < 0)
        return 0;
    const int64_t slot_bytes = (expert_bytes + 4095) & ~int64_t{4095};
    const int64_t denom = slot_bytes * static_cast<int64_t>(n_layers);
    int64_t slots = denom > 0 ? capacity_bytes / denom : 1;
    if (slots < 1)
        slots = 1;
    if (slots > n_experts)
        slots = n_experts;
    return static_cast<int>(slots);
}

Status ExpertStore::open(int n_layers, int n_experts, int64_t expert_bytes, int64_t capacity_bytes,
                         std::string &err) {
    close();
    if (n_layers <= 0 || n_experts <= 0 || expert_bytes <= 0 || capacity_bytes < 0) {
        err = "invalid expert store parameters";
        return Status::InvalidArgument;
    }
    if (expert_bytes > std::numeric_limits<int64_t>::max() / n_layers) {
        err = "expert capacity overflow";
        return Status::InvalidArgument;
    }
    if (static_cast<int64_t>(n_layers) > 2000000 / std::max(n_experts, 1)) {
        err = "expert table too large";
        return Status::InvalidArgument;
    }

    n_layers_ = n_layers;
    n_experts_ = n_experts;
    expert_bytes_ = expert_bytes;
    capacity_bytes_ = capacity_bytes;

    slots_per_layer_ = expert_store_slots_per_layer(n_layers, n_experts, expert_bytes,
                                                    capacity_bytes);

    clock_ = 1;
    stats_ = {};
    stats_.capacity_bytes = static_cast<uint64_t>(capacity_bytes_);

    meta_.assign(static_cast<size_t>(n_layers_) * static_cast<size_t>(n_experts_), Meta{});
    layers_.resize(static_cast<size_t>(n_layers_));
    for (Layer &layer : layers_) {
        layer.by_expert.assign(static_cast<size_t>(n_experts_), -1);
        layer.slots.resize(static_cast<size_t>(slots_per_layer_));
        for (Slot &slot : layer.slots) {
            slot.base = static_cast<uint8_t *>(io::aligned_alloc_pages(static_cast<size_t>(expert_bytes_)));
            if (!slot.base) {
                err = "aligned alloc failed";
                close();
                return Status::Oom;
            }
        }
    }

    open_ = true;
    return Status::Ok;
}

void ExpertStore::prefetch_one_async(ExpertKey key) {
    wait_prefetch(prefetch_err_);
    prefetch_err_.clear();
    prefetch_st_ = Status::Ok;
    prefetch_th_ = std::thread([this, key]() {
        std::string err;
        prefetch_st_ = prefetch(&key, 1, err);
        if (prefetch_st_ != Status::Ok)
            prefetch_err_ = err;
    });
}

Status ExpertStore::wait_prefetch(std::string &err) {
    if (prefetch_th_.joinable())
        prefetch_th_.join();
    if (prefetch_st_ != Status::Ok && err.empty())
        err = prefetch_err_;
    return prefetch_st_;
}

void ExpertStore::close() {
    {
        std::string ignored;
        wait_prefetch(ignored);
    }
    for (Layer &layer : layers_) {
        for (Slot &slot : layer.slots) {
            if (slot.base) {
                io::aligned_free_pages(slot.base);
                slot.base = nullptr;
            }
        }
    }

    std::vector<int> fds;
    fds.reserve(meta_.size() * 2);
    for (Meta &m : meta_) {
        if (m.fd >= 0)
            fds.push_back(m.fd);
        if (m.direct_fd >= 0)
            fds.push_back(m.direct_fd);
        for (Piece &p : m.pieces) {
            if (p.fd >= 0)
                fds.push_back(p.fd);
            if (p.direct_fd >= 0)
                fds.push_back(p.direct_fd);
            p.fd = -1;
            p.direct_fd = -1;
        }
        m.fd = -1;
        m.direct_fd = -1;
    }
    std::sort(fds.begin(), fds.end());
    fds.erase(std::unique(fds.begin(), fds.end()), fds.end());
    for (int fd : fds)
        io::close_fd(fd);

    layers_.clear();
    meta_.clear();
    n_layers_ = 0;
    n_experts_ = 0;
    expert_bytes_ = 0;
    capacity_bytes_ = 0;
    slots_per_layer_ = 0;
    clock_ = 1;
    stats_ = {};
    open_ = false;
}

Status ExpertStore::register_expert(const ExpertLoc &loc, std::string &err) {
    if (!open_) {
        err = "expert store not open";
        return Status::InvalidArgument;
    }
    const int layer = loc.key.layer;
    const int eid = loc.key.expert;
    if (layer < 0 || layer >= n_layers_ || eid < 0 || eid >= n_experts_) {
        err = "expert key out of range";
        return Status::InvalidArgument;
    }
    if (loc.pieces.empty() && loc.path.empty()) {
        err = "empty expert path";
        return Status::InvalidArgument;
    }

    auto open_path = [&](const std::string &path, int &fd, int &dfd) -> Status {
        for (const Meta &other : meta_) {
            if (other.fd >= 0 && other.path == path) {
                fd = other.fd;
                dfd = other.direct_fd;
                return Status::Ok;
            }
            for (const Piece &p : other.pieces) {
                if (p.fd >= 0 && other.path == path) {
                    fd = p.fd;
                    dfd = p.direct_fd;
                    return Status::Ok;
                }
            }
        }
        for (const Meta &other : meta_) {
            for (const Piece &p : other.pieces) {
                if (p.fd >= 0) {
                    // path not stored per-piece; fall through to open
                    break;
                }
            }
        }
        return io::open_readonly(path, fd, dfd, err);
    };

    Meta &m = meta_[static_cast<size_t>(layer) * static_cast<size_t>(n_experts_) +
                    static_cast<size_t>(eid)];

    if (!loc.pieces.empty()) {
        std::vector<Piece> pcs;
        pcs.reserve(loc.pieces.size());
        int64_t total = 0;
        for (const ExpertPiece &ep : loc.pieces) {
            if (ep.path.empty() || ep.bytes <= 0) {
                err = "invalid expert piece";
                return Status::InvalidArgument;
            }
            int fd = -1, dfd = -1;
            // Reuse an already-open fd for the same path among pieces we just opened.
            for (size_t i = 0; i < pcs.size(); ++i) {
                if (loc.pieces[i].path == ep.path && pcs[i].fd >= 0) {
                    fd = pcs[i].fd;
                    dfd = pcs[i].direct_fd;
                    break;
                }
            }
            if (fd < 0) {
                for (const Meta &other : meta_) {
                    if (other.fd >= 0 && other.path == ep.path) {
                        fd = other.fd;
                        dfd = other.direct_fd;
                        break;
                    }
                    for (const Piece &op : other.pieces) {
                        if (op.fd >= 0 && other.path == ep.path) {
                            fd = op.fd;
                            dfd = op.direct_fd;
                            break;
                        }
                    }
                    if (fd >= 0)
                        break;
                }
            }
            if (fd < 0) {
                Status st = io::open_readonly(ep.path, fd, dfd, err);
                if (st != Status::Ok)
                    return st;
            }
            Piece p;
            p.fd = fd;
            p.direct_fd = dfd;
            p.offset = ep.offset;
            p.bytes = ep.bytes;
            pcs.push_back(p);
            total += ep.bytes;
        }
        if (total > expert_bytes_) {
            err = "expert pieces exceed slot";
            return Status::InvalidArgument;
        }
        m.pieces = std::move(pcs);
        m.path = loc.pieces[0].path;
        m.offset = loc.pieces[0].offset;
        m.bytes = total;
        m.fd = m.pieces[0].fd;
        m.direct_fd = m.pieces[0].direct_fd;
        return Status::Ok;
    }

    const int64_t bytes = loc.bytes > 0 ? loc.bytes : expert_bytes_;
    if (bytes <= 0 || bytes > expert_bytes_) {
        err = "expert byte length exceeds slot";
        return Status::InvalidArgument;
    }
    int fd = -1, dfd = -1;
    Status st = open_path(loc.path, fd, dfd);
    if (st != Status::Ok)
        return st;
    m.path = loc.path;
    m.offset = loc.offset;
    m.bytes = bytes;
    m.fd = fd;
    m.direct_fd = dfd;
    m.pieces.clear();
    return Status::Ok;
}

Status ExpertStore::load_into(int layer, int eid, Slot &slot, std::string &err) {
    const Meta &m = meta_[static_cast<size_t>(layer) * static_cast<size_t>(n_experts_) +
                          static_cast<size_t>(eid)];
    if (m.fd < 0 && m.pieces.empty()) {
        err = "expert not registered";
        return Status::NotFound;
    }
    int64_t n = 0;
    Status st = Status::Ok;
    if (!m.pieces.empty()) {
        uint8_t *dst = slot.base;
        for (const Piece &p : m.pieces) {
            // Pieces are not 4 KiB-aligned in the slot; use the buffered fd.
            st = io::pread_full(p.fd, dst, static_cast<size_t>(p.bytes), p.offset, err);
            if (st != Status::Ok)
                return st;
            dst += p.bytes;
            n += p.bytes;
        }
    } else {
        n = m.bytes > 0 ? m.bytes : expert_bytes_;
        if (use_direct_ && m.direct_fd >= 0) {
            st = io::pread_direct(m.direct_fd, m.fd, slot.base, slot.base,
                                  static_cast<size_t>((expert_bytes_ + 4095) & ~int64_t{4095}),
                                  static_cast<size_t>(n), m.offset, err);
        } else {
            st = io::pread_full(m.fd, slot.base, static_cast<size_t>(n), m.offset, err);
        }
        if (st != Status::Ok)
            return st;
    }

    Layer &L = layers_[static_cast<size_t>(layer)];
    if (slot.valid && slot.eid >= 0 && slot.eid != eid) {
        if (slot.eid < n_experts_) L.by_expert[static_cast<size_t>(slot.eid)] = -1;
        ++stats_.evictions;
    }
    slot.eid = eid;
    slot.valid = true;
    slot.used = clock_++;
    L.by_expert[static_cast<size_t>(eid)] = static_cast<int>(&slot - L.slots.data());
    stats_.bytes_read += static_cast<uint64_t>(n);
    return Status::Ok;
}

ExpertStore::Slot *ExpertStore::find_slot(int layer, int eid) {
    if (layer < 0 || layer >= n_layers_ || eid < 0 || eid >= n_experts_) return nullptr;
    Layer &L = layers_[static_cast<size_t>(layer)];
    const int idx = L.by_expert[static_cast<size_t>(eid)];
    if (idx < 0 || idx >= static_cast<int>(L.slots.size())) return nullptr;
    Slot &slot = L.slots[static_cast<size_t>(idx)];
    if (!slot.valid || slot.eid != eid) return nullptr;
    return &slot;
}

ExpertStore::Slot *ExpertStore::evict_or_free(int layer) {
    Layer &L = layers_[static_cast<size_t>(layer)];
    Slot *free = nullptr;
    Slot *lru = nullptr;
    for (Slot &slot : L.slots) {
        if (slot.pins != 0) continue;
        if (!slot.valid) {
            if (!free) free = &slot;
        } else if (!lru || slot.used < lru->used) {
            lru = &slot;
        }
    }
    return free ? free : lru;
}

Status ExpertStore::lookup(ExpertKey key, ExpertView &view, std::string &err) {
    std::unique_lock<std::mutex> lock(mu_);
    view = {};
    ++stats_.requests;
    if (!open_) {
        err = "expert store not open";
        return Status::InvalidArgument;
    }
    if (key.layer < 0 || key.layer >= n_layers_ || key.expert < 0 || key.expert >= n_experts_) {
        err = "expert key out of range";
        return Status::InvalidArgument;
    }

    const Meta &m = meta_[static_cast<size_t>(key.layer) * static_cast<size_t>(n_experts_) +
                          static_cast<size_t>(key.expert)];
    if (m.fd < 0 && m.pieces.empty()) {
        err = "expert not registered";
        return Status::NotFound;
    }

    if (Slot *hit = find_slot(key.layer, key.expert)) {
        ++stats_.hits;
        ++hit->pins;
        hit->used = clock_++;
        view.key = key;
        view.data = hit->base;
        view.bytes = m.bytes > 0 ? m.bytes : expert_bytes_;
        view.lease = hit;
        return Status::Ok;
    }

    ++stats_.misses;
    Slot *slot = evict_or_free(key.layer);
    if (!slot) {
        err = "all expert slots pinned";
        return Status::Oom;
    }
    ++slot->pins; // sticky during unlocked I/O
    lock.unlock();
    const Status st = load_into(key.layer, key.expert, *slot, err);
    lock.lock();
    if (st != Status::Ok) {
        if (slot->pins > 0)
            --slot->pins;
        return st;
    }
    view.key = key;
    view.data = slot->base;
    view.bytes = m.bytes > 0 ? m.bytes : expert_bytes_;
    view.lease = slot;
    return Status::Ok;
}

void ExpertStore::release(ExpertView &view) {
    std::lock_guard<std::mutex> lock(mu_);
    if (view.lease) {
        Slot *slot = static_cast<Slot *>(view.lease);
        if (slot->pins > 0) --slot->pins;
    }
    view = {};
}

Status ExpertStore::prefetch(const ExpertKey *keys, size_t count, std::string &err) {
    if (count > 0 && !keys) {
        err = "null prefetch keys";
        return Status::InvalidArgument;
    }
    for (size_t i = 0; i < count; ++i) {
        const ExpertKey key = keys[i];
        std::unique_lock<std::mutex> lock(mu_);
        if (!open_) {
            err = "expert store not open";
            return Status::InvalidArgument;
        }
        if (key.layer < 0 || key.layer >= n_layers_ || key.expert < 0 || key.expert >= n_experts_)
            continue;
        const Meta &m = meta_[static_cast<size_t>(key.layer) * static_cast<size_t>(n_experts_) +
                              static_cast<size_t>(key.expert)];
        if (m.fd < 0)
            continue;
        if (Slot *hit = find_slot(key.layer, key.expert)) {
            ++stats_.prefetch_hits;
            hit->used = clock_++;
            continue;
        }
        Slot *slot = evict_or_free(key.layer);
        if (!slot)
            continue;
        ++slot->pins;
        lock.unlock();
        const Status st = load_into(key.layer, key.expert, *slot, err);
        lock.lock();
        if (slot->pins > 0)
            --slot->pins;
        if (st != Status::Ok)
            return st;
        ++stats_.prefetched;
    }
    return Status::Ok;
}

void ExpertStore::stats(ExpertStoreStats &out) const {
    out = stats_;
    uint64_t occupied = 0;
    for (const Layer &layer : layers_) {
        for (const Slot &slot : layer.slots) {
            if (slot.valid) ++occupied;
        }
    }
    out.resident_bytes = occupied * static_cast<uint64_t>(expert_bytes_);
    out.capacity_bytes = static_cast<uint64_t>(capacity_bytes_);
}

} // namespace mvllm
