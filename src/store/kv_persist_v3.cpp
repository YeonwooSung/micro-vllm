#include "kv_persist_v3.hpp"

#include "../quant/kv_tq.hpp"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>

#include <unistd.h>

namespace mvllm {
namespace {

constexpr char kMagic[8] = {'C', 'O', 'L', 'I', 'K', 'V', '3', '\0'};
constexpr int64_t kPrefixBytes = 8 + 8 * 4;
constexpr int64_t kNrecOffset = 8 + 6 * 4;
constexpr float kRadiusMax = 3.4e38f;

bool write_full(std::FILE *fp, const void *src, size_t n) {
    return std::fwrite(src, 1, n, fp) == n;
}

bool read_full(std::FILE *fp, void *dst, size_t n) {
    return std::fread(dst, 1, n, fp) == n;
}

void copy_floats(void *dst, const void *src, int n) {
    if (n > 0)
        std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
}

bool tq_width_ok(int n) {
    return n == 0 || (n >= 2 && n <= kKvTqMaxN && (n & (n - 1)) == 0);
}

// Official kv_tq_sanitize: non-finite or negative radius → inert 0.
float sanitize_radius(float r) {
    return (r >= 0.f && r < kRadiusMax) ? r : 0.f;
}

bool rec_bytes_for(int n_layers, int lb, int rb, int n_index, int index_hd, int64_t &out,
                   std::string &err) {
    if (lb < 0 || rb < 0) {
        err = "kv persist record overflow";
        return false;
    }
    const int64_t layers = n_layers;
    const int64_t layer = static_cast<int64_t>(lb) + static_cast<int64_t>(rb) + 8;
    if (layers > 0 && layer > std::numeric_limits<int64_t>::max() / layers) {
        err = "kv persist record overflow";
        return false;
    }
    int64_t rec = 4 + layers * layer;
    if (index_hd > 0 && n_index > 0) {
        if (n_index > std::numeric_limits<int64_t>::max() / index_hd) {
            err = "kv persist record overflow";
            return false;
        }
        const int64_t idx = static_cast<int64_t>(n_index) * static_cast<int64_t>(index_hd);
        if (idx > (std::numeric_limits<int64_t>::max() - rec) / 4) {
            err = "kv persist record overflow";
            return false;
        }
        rec += idx * 4;
    }
    if (static_cast<uint64_t>(rec) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        err = "kv persist record too large";
        return false;
    }
    out = rec;
    return true;
}

} // namespace

int KvPersistV3::packed_row_bytes(int n) const {
    if (n <= 0)
        return 0;
    return codec_ == 0 ? kv_tq_row_bytes(n, bits_) : kv_q4_row_bytes(n);
}

int32_t KvPersistV3::format_tag() const { return (codec_ << 8) | (bits_ & 0xFF); }

Status KvPersistV3::normalize_config(const KvPersistConfig &in, const KvPersistV3Options &opt,
                                     std::string &err) {
    if (opt.codec != 0 && opt.codec != 1) {
        err = "invalid kv persist v3 codec";
        return Status::InvalidArgument;
    }
    if (opt.bits < 1 || opt.bits > 255) {
        err = "invalid kv persist v3 bits";
        return Status::InvalidArgument;
    }
    if (in.n_layers <= 0 || in.kv_lora < 0 || in.qk_rope < 0 || in.index_hd < 0 || in.vocab < 0) {
        err = "invalid kv persist config";
        return Status::InvalidArgument;
    }
    if (!tq_width_ok(in.kv_lora) || !tq_width_ok(in.qk_rope)) {
        err = "kv persist v3 kv_lora/qk_rope must be 0 or a power of two";
        return Status::InvalidArgument;
    }
    if (!in.has_index.empty() && static_cast<int>(in.has_index.size()) != in.n_layers) {
        err = "has_index size must be n_layers";
        return Status::InvalidArgument;
    }

    codec_ = opt.codec;
    bits_ = opt.bits;
    cfg_ = in;
    cfg_.has_index.assign(static_cast<size_t>(in.n_layers), 0);
    n_index_layers_ = 0;
    if (in.index_hd > 0 && !in.has_index.empty()) {
        for (int i = 0; i < in.n_layers; ++i) {
            if (in.has_index[static_cast<size_t>(i)]) {
                cfg_.has_index[static_cast<size_t>(i)] = 1;
                ++n_index_layers_;
            }
        }
    }

    const int lb = packed_row_bytes(cfg_.kv_lora);
    const int rb = packed_row_bytes(cfg_.qk_rope);
    if (!rec_bytes_for(cfg_.n_layers, lb, rb, n_index_layers_, cfg_.index_hd, rec_bytes_, err)) {
        return Status::InvalidArgument;
    }
    return Status::Ok;
}

void KvPersistV3::close() {
    if (fp_) {
        std::fclose(fp_);
        fp_ = nullptr;
    }
    path_.clear();
    cfg_ = {};
    disk_nrec_ = 0;
    n_index_layers_ = 0;
    codec_ = 0;
    bits_ = 4;
    rec_bytes_ = 0;
    open_ = false;
    stale_ = false;
}

KvPersistV3::~KvPersistV3() { close(); }

Status KvPersistV3::flush_sync(std::string &err) const {
    if (!fp_) {
        err = "kv persist not open";
        return Status::InvalidArgument;
    }
    if (std::fflush(fp_) != 0) {
        err = std::string("fflush: ") + std::strerror(errno);
        return Status::IoError;
    }
    if (::fsync(::fileno(fp_)) != 0) {
        err = std::string("fsync: ") + std::strerror(errno);
        return Status::IoError;
    }
    return Status::Ok;
}

Status KvPersistV3::seek_abs(int64_t off, std::string &err) const {
    if (!fp_ || off < 0) {
        err = "kv persist seek failed";
        return Status::IoError;
    }
    if (::fseeko(fp_, static_cast<off_t>(off), SEEK_SET) != 0) {
        err = std::string("fseeko: ") + std::strerror(errno);
        return Status::IoError;
    }
    return Status::Ok;
}

bool KvPersistV3::header_matches(const int32_t hdr[8]) const {
    return hdr[0] == cfg_.n_layers && hdr[1] == cfg_.kv_lora && hdr[2] == cfg_.qk_rope &&
           hdr[3] == cfg_.index_hd && hdr[4] == n_index_layers_ && hdr[5] == cfg_.vocab &&
           hdr[7] == format_tag();
}

bool KvPersistV3::read_prefix(int32_t hdr[8]) const {
    char magic[8];
    if (!read_full(fp_, magic, 8) || std::memcmp(magic, kMagic, 8) != 0)
        return false;
    return read_full(fp_, hdr, sizeof(int32_t) * 8);
}

int KvPersistV3::physical_nrec() const {
    if (!fp_ || rec_bytes_ <= 0)
        return 0;
    const off_t cur = ::ftello(fp_);
    if (cur < 0)
        return 0;
    if (::fseeko(fp_, 0, SEEK_END) != 0) {
        ::fseeko(fp_, cur, SEEK_SET);
        return 0;
    }
    const off_t end = ::ftello(fp_);
    ::fseeko(fp_, cur, SEEK_SET);
    if (end < kPrefixBytes)
        return 0;
    const int64_t n = (static_cast<int64_t>(end) - kPrefixBytes) / rec_bytes_;
    if (n > std::numeric_limits<int>::max())
        return std::numeric_limits<int>::max();
    return static_cast<int>(n);
}

Status KvPersistV3::write_nrec(int nrec, std::string &err) {
    const int32_t nr = nrec;
    const Status st = seek_abs(kNrecOffset, err);
    if (st != Status::Ok)
        return st;
    if (!write_full(fp_, &nr, sizeof(nr))) {
        err = "write kv persist nrec failed";
        return Status::IoError;
    }
    return flush_sync(err);
}

Status KvPersistV3::create_empty(std::string &err) {
    if (fp_) {
        std::fclose(fp_);
        fp_ = nullptr;
    }

    std::FILE *fp = std::fopen(path_.c_str(), "wb");
    if (!fp) {
        err = "open(" + path_ + "): " + std::strerror(errno);
        return Status::IoError;
    }

    int32_t hdr[8] = {};
    hdr[0] = cfg_.n_layers;
    hdr[1] = cfg_.kv_lora;
    hdr[2] = cfg_.qk_rope;
    hdr[3] = cfg_.index_hd;
    hdr[4] = n_index_layers_;
    hdr[5] = cfg_.vocab;
    hdr[6] = 0;
    hdr[7] = format_tag();

    if (!write_full(fp, kMagic, 8) || !write_full(fp, hdr, sizeof(hdr))) {
        err = "write kv persist header failed";
        std::fclose(fp);
        return Status::IoError;
    }
    if (std::fflush(fp) != 0) {
        err = std::string("fflush: ") + std::strerror(errno);
        std::fclose(fp);
        return Status::IoError;
    }
    if (::fsync(::fileno(fp)) != 0) {
        err = std::string("fsync: ") + std::strerror(errno);
        std::fclose(fp);
        return Status::IoError;
    }
    std::fclose(fp);

    fp = std::fopen(path_.c_str(), "r+b");
    if (!fp) {
        err = "reopen(" + path_ + "): " + std::strerror(errno);
        return Status::IoError;
    }
    fp_ = fp;
    disk_nrec_ = 0;
    stale_ = false;
    return Status::Ok;
}

Status KvPersistV3::ensure_writable(std::string &err) {
    if (!open_) {
        err = "kv persist not open";
        return Status::InvalidArgument;
    }
    if (fp_ && !stale_)
        return Status::Ok;
    return create_empty(err);
}

Status KvPersistV3::open(const std::string &path, const KvPersistConfig &cfg,
                        const KvPersistV3Options &opt, std::string &err) {
    close();
    if (path.empty()) {
        err = "empty kv persist path";
        return Status::InvalidArgument;
    }
    const Status st = normalize_config(cfg, opt, err);
    if (st != Status::Ok)
        return st;

    path_ = path;
    std::FILE *fp = std::fopen(path.c_str(), "r+b");
    if (!fp) {
        if (errno != ENOENT) {
            err = "open(" + path + "): " + std::strerror(errno);
            close();
            return Status::IoError;
        }
        const Status cst = create_empty(err);
        if (cst != Status::Ok) {
            close();
            return cst;
        }
        open_ = true;
        return Status::Ok;
    }

    fp_ = fp;
    int32_t hdr[8] = {};
    if (!read_prefix(hdr) || !header_matches(hdr)) {
        std::fclose(fp_);
        fp_ = nullptr;
        disk_nrec_ = 0;
        stale_ = true;
        open_ = true;
        return Status::Ok;
    }

    int nrec = hdr[6];
    if (nrec < 1)
        nrec = 0;
    const int phys = physical_nrec();
    if (nrec > phys)
        nrec = phys;
    disk_nrec_ = nrec;
    stale_ = false;
    open_ = true;
    return Status::Ok;
}

bool KvPersistV3::record_ok(const KvPersistRecord &rec, std::string &err) const {
    const size_t want_l = static_cast<size_t>(cfg_.n_layers) * static_cast<size_t>(cfg_.kv_lora);
    const size_t want_r = static_cast<size_t>(cfg_.n_layers) * static_cast<size_t>(cfg_.qk_rope);
    const size_t want_i = static_cast<size_t>(n_index_layers_) * static_cast<size_t>(cfg_.index_hd);
    if (rec.L.size() != want_l || rec.R.size() != want_r || rec.I.size() != want_i) {
        err = "kv persist record size mismatch";
        return false;
    }
    return true;
}

void KvPersistV3::pack_row(uint8_t *&dst, const float *src, int n) const {
    const int rb = packed_row_bytes(n);
    float radius = 0.f;
    if (n > 0) {
        if (codec_ == 0)
            radius = kv_tq_quant_row(src, dst, n, bits_);
        else
            radius = kv_q4_quant_row(src, dst, n);
    }
    dst += static_cast<size_t>(rb);
    std::memcpy(dst, &radius, sizeof(radius));
    dst += sizeof(radius);
}

void KvPersistV3::unpack_row(const uint8_t *&src, float *dst, int n) const {
    const int rb = packed_row_bytes(n);
    const uint8_t *packed = src;
    src += static_cast<size_t>(rb);
    float radius = 0.f;
    std::memcpy(&radius, src, sizeof(radius));
    src += sizeof(radius);
    radius = sanitize_radius(radius);
    if (n <= 0)
        return;
    if (codec_ == 0)
        kv_tq_dequant_row(packed, radius, dst, n, bits_);
    else
        kv_q4_dequant_row(packed, radius, dst, n);
}

void KvPersistV3::pack_record(uint8_t *dst, int token, const KvPersistRecord &rec) const {
    const int32_t tk = token;
    std::memcpy(dst, &tk, sizeof(tk));
    dst += sizeof(tk);

    const int nl = cfg_.n_layers;
    const int kl = cfg_.kv_lora;
    const int qr = cfg_.qk_rope;
    const int ih = cfg_.index_hd;
    for (int i = 0; i < nl; ++i) {
        pack_row(dst, rec.L.data() + static_cast<size_t>(i) * static_cast<size_t>(kl), kl);
        pack_row(dst, rec.R.data() + static_cast<size_t>(i) * static_cast<size_t>(qr), qr);
    }
    size_t ioff = 0;
    if (ih > 0) {
        for (int i = 0; i < nl; ++i) {
            if (cfg_.has_index[static_cast<size_t>(i)]) {
                copy_floats(dst, rec.I.data() + ioff, ih);
                dst += static_cast<size_t>(ih) * sizeof(float);
                ioff += static_cast<size_t>(ih);
            }
        }
    }
}

void KvPersistV3::unpack_record(const uint8_t *src, KvPersistRecord &rec) const {
    int32_t tk = 0;
    std::memcpy(&tk, src, sizeof(tk));
    src += sizeof(tk);
    rec.token = tk;

    const int nl = cfg_.n_layers;
    const int kl = cfg_.kv_lora;
    const int qr = cfg_.qk_rope;
    const int ih = cfg_.index_hd;
    rec.L.assign(static_cast<size_t>(nl) * static_cast<size_t>(kl), 0.f);
    rec.R.assign(static_cast<size_t>(nl) * static_cast<size_t>(qr), 0.f);
    rec.I.assign(static_cast<size_t>(n_index_layers_) * static_cast<size_t>(ih), 0.f);

    for (int i = 0; i < nl; ++i) {
        unpack_row(src, rec.L.data() + static_cast<size_t>(i) * static_cast<size_t>(kl), kl);
        unpack_row(src, rec.R.data() + static_cast<size_t>(i) * static_cast<size_t>(qr), qr);
    }
    size_t ioff = 0;
    if (ih > 0) {
        for (int i = 0; i < nl; ++i) {
            if (cfg_.has_index[static_cast<size_t>(i)]) {
                copy_floats(rec.I.data() + ioff, src, ih);
                src += static_cast<size_t>(ih) * sizeof(float);
                ioff += static_cast<size_t>(ih);
            }
        }
    }
}

int KvPersistV3::load(std::vector<int> &hist, std::vector<KvPersistRecord> *rows, std::string &err) {
    hist.clear();
    if (rows)
        rows->clear();
    if (!open_ || !fp_ || stale_)
        return 0;

    if (seek_abs(0, err) != Status::Ok) {
        err.clear();
        return 0;
    }
    int32_t hdr[8] = {};
    if (!read_prefix(hdr) || !header_matches(hdr))
        return 0;

    int nrec = hdr[6];
    if (nrec < 1)
        return 0;
    const int phys = physical_nrec();
    if (nrec > phys)
        nrec = phys;
    if (nrec < 1)
        return 0;

    if (seek_abs(kPrefixBytes, err) != Status::Ok) {
        err.clear();
        return 0;
    }

    hist.reserve(static_cast<size_t>(nrec));
    if (rows)
        rows->reserve(static_cast<size_t>(nrec));

    std::vector<uint8_t> buf(static_cast<size_t>(rec_bytes_));
    int got = 0;
    for (int p = 0; p < nrec; ++p) {
        if (!read_full(fp_, buf.data(), static_cast<size_t>(rec_bytes_))) {
            std::clearerr(fp_);
            break;
        }
        KvPersistRecord rec;
        unpack_record(buf.data(), rec);
        hist.push_back(rec.token);
        if (rows)
            rows->push_back(std::move(rec));
        ++got;
    }
    disk_nrec_ = got;
    return got;
}

Status KvPersistV3::append(const int *hist, int hist_n, const KvPersistRecord *recs, int rec_n,
                           std::string &err) {
    if (!open_) {
        err = "kv persist not open";
        return Status::InvalidArgument;
    }
    if (hist_n < 0 || rec_n < 0) {
        err = "invalid kv persist append length";
        return Status::InvalidArgument;
    }
    if (hist_n <= disk_nrec_)
        return Status::Ok;

    if (rec_n != hist_n - disk_nrec_) {
        err = "append rec_n must equal hist_n - nrec()";
        return Status::InvalidArgument;
    }
    if (!hist || !recs) {
        err = "null kv persist append input";
        return Status::InvalidArgument;
    }
    if (rec_n > std::numeric_limits<int>::max() - disk_nrec_) {
        err = "kv persist nrec overflow";
        return Status::InvalidArgument;
    }
    for (int i = 0; i < rec_n; ++i) {
        if (!record_ok(recs[i], err))
            return Status::InvalidArgument;
    }

    const Status est = ensure_writable(err);
    if (est != Status::Ok)
        return est;

    const int64_t off = kPrefixBytes + static_cast<int64_t>(disk_nrec_) * rec_bytes_;
    const Status sst = seek_abs(off, err);
    if (sst != Status::Ok)
        return sst;

    std::vector<uint8_t> buf(static_cast<size_t>(rec_bytes_));
    int written = 0;
    for (int i = 0; i < rec_n; ++i) {
        pack_record(buf.data(), hist[disk_nrec_ + i], recs[i]);
        if (!write_full(fp_, buf.data(), static_cast<size_t>(rec_bytes_))) {
            std::clearerr(fp_);
            err = "write kv persist record failed";
            return Status::IoError;
        }
        ++written;
    }

    const Status fst = flush_sync(err);
    if (fst != Status::Ok)
        return fst;

    const int nr = disk_nrec_ + written;
    const Status nst = write_nrec(nr, err);
    if (nst != Status::Ok)
        return nst;

    disk_nrec_ = nr;
    return Status::Ok;
}

Status KvPersistV3::truncate(int nrec, std::string &err) {
    if (!open_) {
        err = "kv persist not open";
        return Status::InvalidArgument;
    }
    if (nrec < 0) {
        err = "invalid kv persist nrec";
        return Status::InvalidArgument;
    }
    if (nrec > disk_nrec_)
        nrec = disk_nrec_;

    if (!fp_ || stale_)
        return create_empty(err);

    if (nrec == disk_nrec_)
        return Status::Ok;

    const Status st = write_nrec(nrec, err);
    if (st != Status::Ok)
        return st;
    disk_nrec_ = nrec;
    return Status::Ok;
}

Status KvPersistV3::reset(std::string &err) { return truncate(0, err); }

} // namespace mvllm
