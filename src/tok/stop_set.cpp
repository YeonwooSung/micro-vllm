#include "stop_set.hpp"

namespace mvllm {

bool StopSet::contains(int token) const {
    for (int i = 0; i < n_; ++i)
        if (ids_[i] == token)
            return true;
    return false;
}

bool StopSet::try_add(int id) {
    if (n_ >= kStopSetMax || contains(id))
        return false;
    ids_[n_++] = id;
    return true;
}

void StopSet::arm(const int *stop_ids, int n_stop, int eos, const uint8_t *specials, int n_specials,
                  bool eos_only) {
    n_ = 0;
    nsp_ = 0;

    if (stop_ids != nullptr && n_stop > 0) {
        for (int i = 0; i < n_stop && n_ < kStopSetMax; ++i)
            try_add(stop_ids[i]);
    }

    if (eos >= 0)
        try_add(eos);

    if (specials != nullptr && n_specials > 0) {
        for (int id = 0; id < n_specials && n_ < kStopSetMax; ++id) {
            if (specials[id] && try_add(id))
                ++nsp_;
        }
    }

    // Batched SERVE_BATCH filter: keep only eos. eos < 0 yields an empty set.
    if (eos_only) {
        int kept = 0;
        if (eos >= 0) {
            for (int i = 0; i < n_; ++i)
                if (ids_[i] == eos)
                    ids_[kept++] = ids_[i];
        }
        n_ = kept;
        nsp_ = 0;
    }
}

} // namespace mvllm
