#pragma once

#include <cstdint>

namespace mvllm {

constexpr int kStopSetMax = 64;

class StopSet {
public:
    // specials may be null (skip step 3). n_specials is length of specials[] (vocab).
    void arm(const int *stop_ids, int n_stop, int eos, const uint8_t *specials, int n_specials,
             bool eos_only);
    bool contains(int token) const;
    int size() const { return n_; }
    bool empty() const { return n_ == 0; }
    void clear() { n_ = 0; nsp_ = 0; }
    const int *data() const { return ids_; } // first size() ids
    int specials_added() const { return nsp_; } // how many came from the special set

private:
    bool try_add(int id);

    int ids_[kStopSetMax]{};
    int n_ = 0;
    int nsp_ = 0;
};

} // namespace mvllm
