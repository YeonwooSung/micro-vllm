#pragma once

#include "../tok/tokenizer.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace mvllm {

constexpr int kMaxKvSlots = 16;

// Longest common prefix length of two token sequences.
int kv_common_prefix(const std::vector<int> &a, const std::vector<int> &b);

// Official conversation_cache_slot: hash system…first-user, then % n_slots.
// explicit_slot in [0, n) wins. n_slots<=1 → 0.
int conversation_cache_slot(const std::vector<ChatMessage> &msgs, int n_slots,
                            int explicit_slot = -1);

// Token-history table for up to 16 KV slots (official mux).
class SessionStore {
public:
    void configure(int n_slots);
    int n_slots() const { return n_; }

    int assign(const std::vector<ChatMessage> &msgs, int explicit_slot = -1) const;
    int match(int slot, const std::vector<int> &ids) const;
    const std::vector<int> &history(int slot) const;
    void commit(int slot, const std::vector<int> &ids);
    void reset(int slot);

    bool busy(int slot) const;
    bool try_acquire(int slot);
    void release(int slot);

private:
    int n_ = 1;
    std::vector<std::vector<int>> hist_;
    std::vector<uint8_t> busy_;
    mutable std::mutex mu_;
};

} // namespace mvllm
