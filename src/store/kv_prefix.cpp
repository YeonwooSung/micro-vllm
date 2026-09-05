#include "kv_prefix.hpp"

#include <cstring>
#include <new>

namespace mvllm {

KvPrefix::~KvPrefix() { free_mem(); }

void KvPrefix::free_mem() {
    fed_.clear();
    fed_.shrink_to_fit();
    cap_ = 0;
    len_ = 0;
    tainted_ = false;
}

bool KvPrefix::alloc(int cap) {
    free_mem();
    if (cap <= 0)
        return false;
    try {
        fed_.assign(static_cast<size_t>(cap), 0);
    } catch (const std::bad_alloc &) {
        free_mem();
        return false;
    }
    cap_ = cap;
    return true;
}

bool KvPrefix::grow(int cap, int keep) {
    if (cap <= 0)
        return false;
    std::vector<int> grown;
    try {
        grown.assign(static_cast<size_t>(cap), 0);
    } catch (const std::bad_alloc &) {
        free_mem();
        return false;
    }
    if (keep > len_)
        keep = len_;
    if (keep > cap)
        keep = cap;
    if (keep > 0 && !fed_.empty())
        std::memcpy(grown.data(), fed_.data(), static_cast<size_t>(keep) * sizeof(int));
    fed_.swap(grown);
    cap_ = cap;
    len_ = keep > 0 ? keep : 0;
    return true;
}

void KvPrefix::clear() {
    len_ = 0;
    tainted_ = false;
}

void KvPrefix::record(const int *ids, int pos0, int n) {
    if (fed_.empty() || !ids || n <= 0 || pos0 < 0)
        return;
    if (pos0 >= cap_ || n > cap_ - pos0) {
        len_ = 0;
        return;
    }
    std::memcpy(fed_.data() + pos0, ids, static_cast<size_t>(n) * sizeof(int));
    if (pos0 + n > len_)
        len_ = pos0 + n;
}

void KvPrefix::taint() { tainted_ = true; }

int KvPrefix::reuse(const int *ids, int n) const {
    if (fed_.empty() || !ids)
        return 0;
    if (tainted_ || len_ <= 0 || len_ >= n)
        return 0;
    if (std::memcmp(fed_.data(), ids, static_cast<size_t>(len_) * sizeof(int)) != 0)
        return 0;
    return len_;
}

const int *KvPrefix::fed() const { return fed_.empty() ? nullptr : fed_.data(); }

} // namespace mvllm
