#pragma once

#include <vector>

namespace mvllm {

// Token ids the current KV state was built from. reuse() is all-or-nothing.
class KvPrefix {
public:
    KvPrefix() = default;
    ~KvPrefix();

    bool alloc(int cap);          // restart; failure leaves reuse disabled
    bool grow(int cap, int keep); // preserve first keep positions
    void clear();
    void free_mem();
    void record(const int *ids, int pos0, int n);
    void taint();
    int reuse(const int *ids, int n) const;
    int len() const { return len_; }
    int cap() const { return cap_; }
    bool tainted() const { return tainted_; }
    bool empty() const { return len_ == 0; }
    bool full() const { return cap_ > 0 && len_ >= cap_; }
    const int *fed() const;

private:
    std::vector<int> fed_;
    int len_ = 0;
    int cap_ = 0;
    bool tainted_ = false;
};

} // namespace mvllm
