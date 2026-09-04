#pragma once

#include "../core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace mvllm {

struct ExpertKey {
    int layer = 0;
    int expert = 0;
};

inline bool operator==(const ExpertKey &a, const ExpertKey &b) {
    return a.layer == b.layer && a.expert == b.expert;
}

struct ExpertPiece {
    std::string path;
    int64_t offset = 0;
    int64_t bytes = 0;
};

struct ExpertLoc {
    ExpertKey key;
    std::string path;
    int64_t offset = 0;
    int64_t bytes = 0;
    bool contig = true;
    // If non-empty, load_into scatter-reads these in order (GLM-5.3's six
    // U8+.qs tensors). Otherwise a single [path, offset, bytes] blob is used.
    std::vector<ExpertPiece> pieces;
};

struct ExpertView {
    ExpertKey key;
    const uint8_t *data = nullptr;
    int64_t bytes = 0;
    void *lease = nullptr;
};

struct ExpertStoreStats {
    uint64_t requests = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t prefetched = 0;
    uint64_t prefetch_hits = 0;
    uint64_t bytes_read = 0;
    uint64_t resident_bytes = 0;
    uint64_t capacity_bytes = 0;
    uint64_t evictions = 0;
};

// Per-layer LRU over file-backed expert blobs.
// lookup() must be paired with exactly one release() on success.
// prefetch() is advisory and never holds a lease.
class ExpertStore {
public:
    Status open(int n_layers, int n_experts, int64_t expert_bytes, int64_t capacity_bytes,
                std::string &err);
    int slots_per_layer() const { return slots_per_layer_; }
    int n_experts() const { return n_experts_; }
    void close();

    // Prefer O_DIRECT / F_NOCACHE when the OS opened a twin fd.
    void set_direct(bool on) { use_direct_ = on; }

    Status register_expert(const ExpertLoc &loc, std::string &err);

    Status lookup(ExpertKey key, ExpertView &view, std::string &err);
    void release(ExpertView &view);
    Status prefetch(const ExpertKey *keys, size_t count, std::string &err);

    // Load keys[1..] without pinning so expert 0's matmul can overlap I/O.
    Status prefetch_tail(const ExpertKey *keys, size_t count, std::string &err) {
        if (count <= 1)
            return Status::Ok;
        return prefetch(keys + 1, count - 1, err);
    }

    void stats(ExpertStoreStats &out) const;
    int64_t expert_bytes() const { return expert_bytes_; }
    int n_layers() const { return n_layers_; }

    ~ExpertStore() { close(); }
    ExpertStore() = default;
    ExpertStore(const ExpertStore &) = delete;
    ExpertStore &operator=(const ExpertStore &) = delete;

private:
    struct Slot {
        int eid = -1;
        int pins = 0;
        uint64_t used = 0;
        uint8_t *base = nullptr;
        bool valid = false;
    };
    struct Layer {
        std::vector<Slot> slots;
        std::vector<int> by_expert; // eid -> slot or -1
    };
    struct Piece {
        int fd = -1;
        int direct_fd = -1;
        int64_t offset = 0;
        int64_t bytes = 0;
    };
    struct Meta {
        std::string path;
        int64_t offset = 0;
        int64_t bytes = 0;
        int fd = -1;
        int direct_fd = -1;
        std::vector<Piece> pieces;
    };

    Status load_into(int layer, int eid, Slot &slot, std::string &err);
    Slot *find_slot(int layer, int eid);
    Slot *evict_or_free(int layer);

    int n_layers_ = 0;
    int n_experts_ = 0;
    int64_t expert_bytes_ = 0;
    int64_t capacity_bytes_ = 0;
    int slots_per_layer_ = 0;
    uint64_t clock_ = 1;
    std::vector<Layer> layers_;
    std::vector<Meta> meta_; // [layer * n_experts + eid]
    ExpertStoreStats stats_{};
    bool open_ = false;
    bool use_direct_ = true;
    std::mutex mu_;
};

// LRU slots per layer: budget against 4 KiB-aligned blobs, never more than n_experts.
int expert_store_slots_per_layer(int n_layers, int n_experts, int64_t expert_bytes,
                                 int64_t capacity_bytes);

} // namespace mvllm
