#pragma once

#include "kv_persist.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace mvllm {

struct KvPersistV3Options {
    int codec = 0; // 0 PolarQuant, 1 rotated int4
    int bits = 4;
};

// COLIKV3 PolarQuant / rotated-int4 on-disk KV cache. Little-endian host. Not thread-safe.
// Public records are F32; L/R are quantized on append and dequantized on load. DSA I stays F32.
class KvPersistV3 {
public:
    // Missing file → empty (nrec=0). Wrong magic / format_tag / geometry → ignore (nrec=0).
    // Invalid codec/bits → InvalidArgument.
    Status open(const std::string &path, const KvPersistConfig &cfg, const KvPersistV3Options &opt,
                std::string &err);
    void close();

    int nrec() const { return disk_nrec_; }
    const KvPersistConfig &config() const { return cfg_; }
    int64_t record_bytes() const { return rec_bytes_; }
    int codec() const { return codec_; }
    int bits() const { return bits_; }

    // Load complete records. hist is cleared; rows is cleared when non-null.
    int load(std::vector<int> &hist, std::vector<KvPersistRecord> *rows, std::string &err);

    // Append hist[nrec() .. hist_n). recs are the new rows only; rec_n == hist_n - nrec().
    Status append(const int *hist, int hist_n, const KvPersistRecord *recs, int rec_n,
                  std::string &err);

    Status truncate(int nrec, std::string &err); // never grows nrec
    Status reset(std::string &err);

    KvPersistV3() = default;
    KvPersistV3(const KvPersistV3 &) = delete;
    KvPersistV3 &operator=(const KvPersistV3 &) = delete;
    KvPersistV3(KvPersistV3 &&) = delete;
    KvPersistV3 &operator=(KvPersistV3 &&) = delete;
    ~KvPersistV3();

private:
    Status normalize_config(const KvPersistConfig &in, const KvPersistV3Options &opt,
                            std::string &err);
    Status create_empty(std::string &err);
    Status ensure_writable(std::string &err);
    Status flush_sync(std::string &err) const;
    Status seek_abs(int64_t off, std::string &err) const;
    Status write_nrec(int nrec, std::string &err);
    bool read_prefix(int32_t hdr[8]) const;
    bool header_matches(const int32_t hdr[8]) const;
    int physical_nrec() const;
    bool record_ok(const KvPersistRecord &rec, std::string &err) const;
    int packed_row_bytes(int n) const;
    int32_t format_tag() const;
    void pack_row(uint8_t *&dst, const float *src, int n) const;
    void unpack_row(const uint8_t *&src, float *dst, int n) const;
    void pack_record(uint8_t *dst, int token, const KvPersistRecord &rec) const;
    void unpack_record(const uint8_t *src, KvPersistRecord &rec) const;

    std::string path_;
    KvPersistConfig cfg_;
    std::FILE *fp_ = nullptr;
    int disk_nrec_ = 0;
    int n_index_layers_ = 0;
    int codec_ = 0;
    int bits_ = 4;
    int64_t rec_bytes_ = 0;
    bool open_ = false;
    bool stale_ = false; // existing file ignored; rewrite on first write
};

} // namespace mvllm
