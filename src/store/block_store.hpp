#pragma once

#include "../core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {

struct BlockPiece {
    std::string path;
    int64_t offset = 0;
    int64_t bytes = 0;
};

// Two-slot (or N-slot) SSD streamer for H3 DiT blocks.
// acquire(i) returns a pointer that stays valid until release(i).
// prefetch(i+1) may overlap with compute on i.
class BlockStore {
public:
    Status open(int n_blocks, int64_t block_bytes, int n_slots, std::string &err);
    void set_direct(bool on) { use_direct_ = on; }
    void close();

    Status register_block(int index, const std::string &path, int64_t offset, int64_t bytes,
                          std::string &err);
    // Scatter-read the four DiT matrices (qkv/out/fc1/fc2) into one slot.
    Status register_block_pieces(int index, const std::vector<BlockPiece> &pieces,
                                 std::string &err);

    Status acquire(int index, const uint8_t **data, int64_t *bytes, std::string &err);
    void release(int index);
    Status prefetch(int index, std::string &err);

    int n_blocks() const { return n_blocks_; }
    int n_slots() const { return n_slots_; }
    uint64_t bytes_read() const { return bytes_read_; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }

    ~BlockStore() { close(); }
    BlockStore() = default;
    BlockStore(const BlockStore &) = delete;
    BlockStore &operator=(const BlockStore &) = delete;

private:
    struct Slot {
        int index = -1;
        int pins = 0;
        uint64_t used = 0;
        uint8_t *base = nullptr;
        int64_t bytes = 0;
        bool valid = false;
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

    Status load_into(int index, Slot &slot, std::string &err);
    Slot *find(int index);
    Slot *victim();

    int n_blocks_ = 0;
    int n_slots_ = 0;
    int64_t block_bytes_ = 0;
    uint64_t clock_ = 1;
    uint64_t bytes_read_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    std::vector<Slot> slots_;
    std::vector<Meta> meta_;
    bool open_ = false;
    bool use_direct_ = true;
};

} // namespace mvllm
