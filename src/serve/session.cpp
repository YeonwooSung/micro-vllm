#include "session.hpp"

#include <algorithm>
#include <mutex>

namespace mvllm {
namespace {

// FNV-1a 64 (offset basis / prime). Mixes role, content, tool_name, and any
// non-empty reasoning / xtml_type / image_urls / tool_calls through the first
// user turn into a stable conversation key. Empty extras are skipped so
// text-only prefixes keep the same digest as role+content+tool_name.
constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

void fnv1a_bytes(uint64_t &h, const void *data, size_t n) {
    const auto *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= kFnvPrime;
    }
}

void fnv1a_u64(uint64_t &h, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; ++i)
        b[i] = static_cast<unsigned char>(v >> (8 * i));
    fnv1a_bytes(h, b, sizeof(b));
}

void fnv1a_str(uint64_t &h, const std::string &s) {
    fnv1a_u64(h, static_cast<uint64_t>(s.size()));
    fnv1a_bytes(h, s.data(), s.size());
}

void fnv1a_tool_call(uint64_t &h, const K3ToolCall &call) {
    fnv1a_str(h, call.name);
    fnv1a_u64(h, static_cast<uint64_t>(static_cast<int64_t>(call.index)));
    fnv1a_str(h, call.json);
    for (const K3ToolArg &a : call.args) {
        fnv1a_str(h, a.key);
        fnv1a_str(h, a.type);
        fnv1a_str(h, a.value);
    }
}

uint64_t conversation_key_digest(const std::vector<ChatMessage> &msgs) {
    uint64_t h = kFnvOffset;
    for (const ChatMessage &m : msgs) {
        fnv1a_str(h, m.role);
        fnv1a_str(h, m.content);
        fnv1a_str(h, m.tool_name);
        if (!m.reasoning.empty())
            fnv1a_str(h, m.reasoning);
        if (!m.xtml_type.empty())
            fnv1a_str(h, m.xtml_type);
        for (const std::string &url : m.image_urls)
            fnv1a_str(h, url);
        for (const K3ToolCall &call : m.tool_calls)
            fnv1a_tool_call(h, call);
        if (m.role == "user")
            break;
    }
    return h;
}

} // namespace

int kv_common_prefix(const std::vector<int> &a, const std::vector<int> &b) {
    const int n = static_cast<int>(std::min(a.size(), b.size()));
    int i = 0;
    for (; i < n; ++i)
        if (a[static_cast<size_t>(i)] != b[static_cast<size_t>(i)])
            break;
    return i;
}

int conversation_cache_slot(const std::vector<ChatMessage> &msgs, int n_slots, int explicit_slot) {
    if (n_slots <= 1)
        return 0;
    if (explicit_slot >= 0 && explicit_slot < n_slots)
        return explicit_slot;
    if (msgs.empty())
        return 0;
    return static_cast<int>(conversation_key_digest(msgs) % static_cast<uint64_t>(n_slots));
}

void SessionStore::configure(int n_slots) {
    if (n_slots < 1)
        n_slots = 1;
    if (n_slots > kMaxKvSlots)
        n_slots = kMaxKvSlots;
    std::lock_guard<std::mutex> lock(mu_);
    n_ = n_slots;
    hist_.assign(static_cast<size_t>(n_), {});
    busy_.assign(static_cast<size_t>(n_), 0);
}

bool SessionStore::valid_slot(int slot) const {
    std::lock_guard<std::mutex> lock(mu_);
    return slot >= 0 && slot < n_;
}

int SessionStore::assign(const std::vector<ChatMessage> &msgs, int explicit_slot) const {
    int n = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        n = n_;
    }
    return conversation_cache_slot(msgs, n, explicit_slot);
}

int SessionStore::match(int slot, const std::vector<int> &ids) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot < 0 || slot >= n_ || static_cast<size_t>(slot) >= hist_.size())
        return 0;
    return kv_common_prefix(hist_[static_cast<size_t>(slot)], ids);
}

const std::vector<int> &SessionStore::history(int slot) const {
    static const std::vector<int> empty;
    std::lock_guard<std::mutex> lock(mu_);
    if (slot < 0 || slot >= n_ || static_cast<size_t>(slot) >= hist_.size())
        return empty;
    return hist_[static_cast<size_t>(slot)];
}

int SessionStore::history_len(int slot) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot < 0 || slot >= n_ || static_cast<size_t>(slot) >= hist_.size())
        return 0;
    return static_cast<int>(hist_[static_cast<size_t>(slot)].size());
}

int SessionStore::history_total() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return 0;
    int total = 0;
    for (int i = 0; i < n_; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx < hist_.size())
            total += static_cast<int>(hist_[idx].size());
    }
    return total;
}

bool SessionStore::has_history(int slot) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot < 0 || slot >= n_ || static_cast<size_t>(slot) >= hist_.size())
        return false;
    return !hist_[static_cast<size_t>(slot)].empty();
}

void SessionStore::commit(int slot, const std::vector<int> &ids) {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot < 0 || slot >= n_ || static_cast<size_t>(slot) >= hist_.size())
        return;
    hist_[static_cast<size_t>(slot)] = ids;
}

void SessionStore::reset(int slot) {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot < 0 || slot >= n_)
        return;
    if (static_cast<size_t>(slot) < hist_.size())
        hist_[static_cast<size_t>(slot)].clear();
    if (static_cast<size_t>(slot) < busy_.size())
        busy_[static_cast<size_t>(slot)] = 0;
}

void SessionStore::reset_all() {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return;
    for (int i = 0; i < n_; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx < hist_.size())
            hist_[idx].clear();
        if (idx < busy_.size())
            busy_[idx] = 0;
    }
}

bool SessionStore::busy(int slot) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot < 0 || slot >= n_ || static_cast<size_t>(slot) >= busy_.size())
        return true;
    return busy_[static_cast<size_t>(slot)] != 0;
}

int SessionStore::busy_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return 0;
    int count = 0;
    for (int i = 0; i < n_; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx < busy_.size() && busy_[idx] != 0)
            ++count;
    }
    return count;
}

bool SessionStore::all_free() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return true;
    for (int i = 0; i < n_; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx < busy_.size() && busy_[idx] != 0)
            return false;
    }
    return true;
}

bool SessionStore::all_busy() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return false;
    for (int i = 0; i < n_; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx >= busy_.size() || busy_[idx] == 0)
            return false;
    }
    return true;
}

bool SessionStore::any_busy() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return false;
    for (int i = 0; i < n_; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx < busy_.size() && busy_[idx] != 0)
            return true;
    }
    return false;
}

bool SessionStore::any_free() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return false;
    for (int i = 0; i < n_; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx >= busy_.size() || busy_[idx] == 0)
            return true;
    }
    return false;
}

int SessionStore::free_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return 0;
    int busy = 0;
    for (int i = 0; i < n_; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx < busy_.size() && busy_[idx] != 0)
            ++busy;
    }
    const int free = n_ - busy;
    return free < 0 ? 0 : free;
}

int SessionStore::first_free() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return -1;
    for (int i = 0; i < n_; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx >= busy_.size() || busy_[idx] == 0)
            return i;
    }
    return -1;
}

int SessionStore::last_free() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return -1;
    for (int i = n_ - 1; i >= 0; --i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx >= busy_.size() || busy_[idx] == 0)
            return i;
    }
    return -1;
}

int SessionStore::first_busy() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return -1;
    for (int i = 0; i < n_; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx < busy_.size() && busy_[idx] != 0)
            return i;
    }
    return -1;
}

int SessionStore::last_busy() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return -1;
    for (int i = n_ - 1; i >= 0; --i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx < busy_.size() && busy_[idx] != 0)
            return i;
    }
    return -1;
}

int SessionStore::try_acquire_any() {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return -1;
    for (int i = 0; i < n_; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx >= busy_.size())
            continue;
        if (busy_[idx] == 0) {
            busy_[idx] = 1;
            return i;
        }
    }
    return -1;
}

int SessionStore::try_acquire_last() {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return -1;
    for (int i = n_ - 1; i >= 0; --i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx >= busy_.size())
            continue;
        if (busy_[idx] == 0) {
            busy_[idx] = 1;
            return i;
        }
    }
    return -1;
}

bool SessionStore::try_acquire(int slot) {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot < 0 || slot >= n_ || static_cast<size_t>(slot) >= busy_.size())
        return false;
    if (busy_[static_cast<size_t>(slot)])
        return false;
    busy_[static_cast<size_t>(slot)] = 1;
    return true;
}

void SessionStore::release(int slot) {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot < 0 || slot >= n_ || static_cast<size_t>(slot) >= busy_.size())
        return;
    busy_[static_cast<size_t>(slot)] = 0;
}

void SessionStore::release_all() {
    std::lock_guard<std::mutex> lock(mu_);
    if (n_ == 0)
        return;
    for (int i = 0; i < n_; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (idx < busy_.size())
            busy_[idx] = 0;
    }
}

} // namespace mvllm
