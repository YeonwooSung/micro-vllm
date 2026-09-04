#include "block_store.hpp"

#include "../io/file_io.hpp"

#include <algorithm>
#include <limits>

namespace mvllm {

Status BlockStore::open(int n_blocks, int64_t block_bytes, int n_slots, std::string &err) {
    close();
    if (n_blocks <= 0 || block_bytes <= 0 || n_slots <= 0) {
        err = "invalid block store parameters";
        return Status::InvalidArgument;
    }

    n_blocks_ = n_blocks;
    n_slots_ = n_slots;
    block_bytes_ = block_bytes;
    clock_ = 1;
    bytes_read_ = 0;
    hits_ = 0;
    misses_ = 0;

    meta_.assign(static_cast<size_t>(n_blocks_), Meta{});
    slots_.resize(static_cast<size_t>(n_slots_));
    for (Slot &slot : slots_) {
        slot.base = static_cast<uint8_t *>(io::aligned_alloc_pages(static_cast<size_t>(block_bytes_)));
        if (!slot.base) {
            err = "aligned alloc failed";
            close();
            return Status::Oom;
        }
    }

    open_ = true;
    return Status::Ok;
}

void BlockStore::close() {
    for (Slot &slot : slots_) {
        if (slot.base) {
            io::aligned_free_pages(slot.base);
            slot.base = nullptr;
        }
    }

    std::vector<int> fds;
    fds.reserve(meta_.size());
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
        m.pieces.clear();
    }
    std::sort(fds.begin(), fds.end());
    fds.erase(std::unique(fds.begin(), fds.end()), fds.end());
    for (int fd : fds) io::close_fd(fd);

    slots_.clear();
    meta_.clear();
    n_blocks_ = 0;
    n_slots_ = 0;
    block_bytes_ = 0;
    clock_ = 1;
    bytes_read_ = 0;
    hits_ = 0;
    misses_ = 0;
    open_ = false;
}

Status BlockStore::register_block(int index, const std::string &path, int64_t offset, int64_t bytes,
                                 std::string &err) {
    if (!open_) {
        err = "block store not open";
        return Status::InvalidArgument;
    }
    if (index < 0 || index >= n_blocks_) {
        err = "block index out of range";
        return Status::InvalidArgument;
    }
    if (path.empty()) {
        err = "empty block path";
        return Status::InvalidArgument;
    }
    const int64_t n = bytes > 0 ? bytes : block_bytes_;
    if (n <= 0 || n > block_bytes_) {
        err = "block byte length exceeds slot";
        return Status::InvalidArgument;
    }

    Meta &m = meta_[static_cast<size_t>(index)];
    const int old_fd = m.fd;

    int fd = -1;
    for (const Meta &other : meta_) {
        if (other.fd >= 0 && other.path == path) {
            fd = other.fd;
            break;
        }
    }
    int direct_fd = -1;
    if (fd < 0) {
        const Status st = io::open_readonly(path, fd, direct_fd, err);
        if (st != Status::Ok)
            return st;
    } else {
        for (const Meta &other : meta_) {
            if (other.fd == fd) {
                direct_fd = other.direct_fd;
                break;
            }
        }
    }

    m.path = path;
    m.offset = offset;
    m.bytes = n;
    m.fd = fd;
    m.direct_fd = direct_fd;

    if (old_fd >= 0 && old_fd != fd) {
        bool still_used = false;
        for (const Meta &other : meta_) {
            if (other.fd == old_fd) {
                still_used = true;
                break;
            }
        }
        if (!still_used) io::close_fd(old_fd);
    }
    return Status::Ok;
}

Status BlockStore::register_block_pieces(int index, const std::vector<BlockPiece> &pieces,
                                         std::string &err) {
    if (!open_) {
        err = "block store not open";
        return Status::InvalidArgument;
    }
    if (index < 0 || index >= n_blocks_) {
        err = "block index out of range";
        return Status::InvalidArgument;
    }
    if (pieces.empty()) {
        err = "empty block pieces";
        return Status::InvalidArgument;
    }
    int64_t total = 0;
    std::vector<Piece> pcs;
    pcs.reserve(pieces.size());
    for (const BlockPiece &bp : pieces) {
        if (bp.path.empty() || bp.bytes <= 0) {
            err = "invalid block piece";
            return Status::InvalidArgument;
        }
        int fd = -1, dfd = -1;
        for (size_t i = 0; i < pcs.size(); ++i) {
            if (pieces[i].path == bp.path && pcs[i].fd >= 0) {
                fd = pcs[i].fd;
                dfd = pcs[i].direct_fd;
                break;
            }
        }
        if (fd < 0) {
            for (const Meta &other : meta_) {
                if (other.fd >= 0 && other.path == bp.path) {
                    fd = other.fd;
                    dfd = other.direct_fd;
                    break;
                }
            }
        }
        if (fd < 0) {
            Status st = io::open_readonly(bp.path, fd, dfd, err);
            if (st != Status::Ok)
                return st;
        }
        Piece p;
        p.fd = fd;
        p.direct_fd = dfd;
        p.offset = bp.offset;
        p.bytes = bp.bytes;
        pcs.push_back(p);
        total += bp.bytes;
    }
    if (total > block_bytes_) {
        err = "block pieces exceed slot";
        return Status::InvalidArgument;
    }
    Meta &m = meta_[static_cast<size_t>(index)];
    m.pieces = std::move(pcs);
    m.path = pieces[0].path;
    m.offset = pieces[0].offset;
    m.bytes = total;
    m.fd = m.pieces[0].fd;
    m.direct_fd = m.pieces[0].direct_fd;
    return Status::Ok;
}

Status BlockStore::load_into(int index, Slot &slot, std::string &err) {
    const Meta &m = meta_[static_cast<size_t>(index)];
    if (m.fd < 0 && m.pieces.empty()) {
        err = "block not registered";
        return Status::NotFound;
    }
    int64_t n = 0;
    Status st = Status::Ok;
    if (!m.pieces.empty()) {
        uint8_t *dst = slot.base;
        for (const Piece &p : m.pieces) {
            st = io::pread_full(p.fd, dst, static_cast<size_t>(p.bytes), p.offset, err);
            if (st != Status::Ok)
                return st;
            dst += p.bytes;
            n += p.bytes;
        }
    } else {
        n = m.bytes > 0 ? m.bytes : block_bytes_;
        if (use_direct_ && m.direct_fd >= 0) {
            st = io::pread_direct(m.direct_fd, m.fd, slot.base, slot.base,
                                  static_cast<size_t>((block_bytes_ + 4095) & ~int64_t{4095}),
                                  static_cast<size_t>(n), m.offset, err);
        } else {
            st = io::pread_full(m.fd, slot.base, static_cast<size_t>(n), m.offset, err);
        }
        if (st != Status::Ok)
            return st;
    }
    slot.index = index;
    slot.valid = true;
    slot.bytes = n;
    slot.used = clock_++;
    bytes_read_ += static_cast<uint64_t>(n);
    return Status::Ok;
}

BlockStore::Slot *BlockStore::find(int index) {
    for (Slot &slot : slots_) {
        if (slot.valid && slot.index == index) return &slot;
    }
    return nullptr;
}

BlockStore::Slot *BlockStore::victim() {
    Slot *free = nullptr;
    Slot *lru = nullptr;
    for (Slot &slot : slots_) {
        if (slot.pins != 0) continue;
        if (!slot.valid) {
            if (!free) free = &slot;
        } else if (!lru || slot.used < lru->used) {
            lru = &slot;
        }
    }
    return free ? free : lru;
}

Status BlockStore::acquire(int index, const uint8_t **data, int64_t *bytes, std::string &err) {
    if (!open_) {
        err = "block store not open";
        return Status::InvalidArgument;
    }
    if (!data || !bytes) {
        err = "null acquire output";
        return Status::InvalidArgument;
    }
    if (index < 0 || index >= n_blocks_) {
        err = "block index out of range";
        return Status::InvalidArgument;
    }
    if (meta_[static_cast<size_t>(index)].fd < 0 &&
        meta_[static_cast<size_t>(index)].pieces.empty()) {
        err = "block not registered";
        return Status::NotFound;
    }

    if (Slot *hit = find(index)) {
        ++hits_;
        ++hit->pins;
        hit->used = clock_++;
        *data = hit->base;
        *bytes = hit->bytes;
        return Status::Ok;
    }

    ++misses_;
    Slot *slot = victim();
    if (!slot) {
        err = "all block slots pinned";
        return Status::Oom;
    }
    const Status st = load_into(index, *slot, err);
    if (st != Status::Ok) return st;
    ++slot->pins;
    *data = slot->base;
    *bytes = slot->bytes;
    return Status::Ok;
}

void BlockStore::release(int index) {
    if (Slot *slot = find(index)) {
        if (slot->pins > 0) --slot->pins;
    }
}

Status BlockStore::prefetch(int index, std::string &err) {
    if (!open_) {
        err = "block store not open";
        return Status::InvalidArgument;
    }
    if (index < 0 || index >= n_blocks_) {
        err = "block index out of range";
        return Status::InvalidArgument;
    }
    if (meta_[static_cast<size_t>(index)].fd < 0 &&
        meta_[static_cast<size_t>(index)].pieces.empty()) {
        err = "block not registered";
        return Status::NotFound;
    }
    if (Slot *hit = find(index)) {
        hit->used = clock_++;
        return Status::Ok;
    }
    Slot *slot = victim();
    if (!slot) return Status::Ok;
    return load_into(index, *slot, err);
}

} // namespace mvllm
