#pragma once

#include "../core/types.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace mvllm {

struct KvPersistConfig {
    int n_layers = 0;
    int kv_lora = 0;
    int qk_rope = 0;
    int index_hd = 0;
    int vocab = 0;
    std::vector<uint8_t> has_index; // size n_layers; empty means none
};

struct KvPersistRecord {
    int token = 0;
    // Packed [n_layers, kv_lora] then [n_layers, qk_rope]; disk stores L/R per layer.
    std::vector<float> L;
    std::vector<float> R;
    // Concatenated DSA rows in layer order for has_index==1.
    std::vector<float> I;
};

// COLIKV1 F32 on-disk KV cache. Little-endian host. Not thread-safe.
class KvPersist {
public:
    // Missing file → empty (nrec=0). Wrong magic / header geometry → ignore (nrec=0).
    Status open(const std::string &path, const KvPersistConfig &cfg, std::string &err);
    void close();

    int nrec() const { return disk_nrec_; }
    const KvPersistConfig &config() const { return cfg_; }
    int64_t record_bytes() const { return rec_bytes_; }

    // Load complete records. hist is cleared; rows is cleared when non-null.
    int load(std::vector<int> &hist, std::vector<KvPersistRecord> *rows, std::string &err);

    // Append hist[nrec() .. hist_n). recs are the new rows only; rec_n == hist_n - nrec().
    Status append(const int *hist, int hist_n, const KvPersistRecord *recs, int rec_n,
                  std::string &err);

    Status truncate(int nrec, std::string &err); // never grows nrec
    Status reset(std::string &err);

    KvPersist() = default;
    KvPersist(const KvPersist &) = delete;
    KvPersist &operator=(const KvPersist &) = delete;
    KvPersist(KvPersist &&) = delete;
    KvPersist &operator=(KvPersist &&) = delete;
    ~KvPersist();

private:
    Status normalize_config(const KvPersistConfig &in, std::string &err);
    Status create_empty(std::string &err);
    Status ensure_writable(std::string &err);
    Status flush_sync(std::string &err) const;
    Status seek_abs(int64_t off, std::string &err) const;
    Status write_nrec(int nrec, std::string &err);
    bool read_prefix(int32_t hdr[8]) const;
    bool header_matches(const int32_t hdr[8]) const;
    int physical_nrec() const;
    bool record_ok(const KvPersistRecord &rec, std::string &err) const;
    void pack_record(uint8_t *dst, int token, const KvPersistRecord &rec) const;
    void unpack_record(const uint8_t *src, KvPersistRecord &rec) const;

    std::string path_;
    KvPersistConfig cfg_;
    std::FILE *fp_ = nullptr;
    int disk_nrec_ = 0;
    int n_index_layers_ = 0;
    int64_t rec_bytes_ = 0;
    bool open_ = false;
    bool stale_ = false; // existing file ignored; rewrite on first write
};

} // namespace mvllm
