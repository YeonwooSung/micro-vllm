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
    // True if slot is in [0, n_). False if n_ == 0 or slot out of range.
    bool valid_slot(int slot) const;

    int assign(const std::vector<ChatMessage> &msgs, int explicit_slot = -1) const;
    int match(int slot, const std::vector<int> &ids) const;
    const std::vector<int> &history(int slot) const;
    // Token count of slot history. 0 if slot out of range.
    int history_len(int slot) const;
    // Sum of history token counts over valid slots. 0 if n_ == 0.
    int history_total() const;
    // True if slot history is non-empty. False if slot out of range or empty.
    bool has_history(int slot) const;
    void commit(int slot, const std::vector<int> &ids);
    void reset(int slot);
    // Official \x02RESET: clear every slot history and drop busy flags.
    void reset_all();

    bool busy(int slot) const;
    // How many slots currently have busy != 0. 0 if n_ == 0.
    int busy_count() const;
    // True if no slot is busy. True if n_ == 0.
    bool all_free() const;
    // True if every slot in [0, n_) is busy. False if n_ == 0.
    bool all_busy() const;
    // True if at least one slot is busy. False if n_ == 0.
    bool any_busy() const;
    // True if at least one slot is free. False if n_ == 0.
    bool any_free() const;
    // n_ - busy_count, not below 0. 0 if n_ == 0.
    int free_count() const;
    // Lowest slot in [0, n_) with busy == 0. -1 if none (or n_ == 0).
    int first_free() const;
    // Highest slot in [0, n_) with busy == 0. -1 if none (or n_ == 0).
    int last_free() const;
    // Lowest slot in [0, n_) with busy != 0. -1 if none (or n_ == 0).
    int first_busy() const;
    // Highest slot in [0, n_) with busy != 0. -1 if none (or n_ == 0).
    int last_busy() const;
    // Acquire the lowest free slot. Returns the slot, or -1 if none.
    int try_acquire_any();
    // Acquire the highest free slot. Returns the slot, or -1 if none.
    int try_acquire_last();
    bool try_acquire(int slot);
    void release(int slot);
    // Clear every busy flag. History is unchanged.
    void release_all();

private:
    int n_ = 1;
    std::vector<std::vector<int>> hist_;
    std::vector<uint8_t> busy_;
    mutable std::mutex mu_;
};

} // namespace mvllm
