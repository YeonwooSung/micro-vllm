#include "safetensors.hpp"
#include "file_io.hpp"

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

namespace mvllm {
namespace io {
namespace {

uint64_t read_le_u64(const uint8_t *p) {
    return static_cast<uint64_t>(p[0]) | (static_cast<uint64_t>(p[1]) << 8) |
           (static_cast<uint64_t>(p[2]) << 16) | (static_cast<uint64_t>(p[3]) << 24) |
           (static_cast<uint64_t>(p[4]) << 32) | (static_cast<uint64_t>(p[5]) << 40) |
           (static_cast<uint64_t>(p[6]) << 48) | (static_cast<uint64_t>(p[7]) << 56);
}

bool ends_with(const char *name, const char *suffix) {
    const size_t n = std::strlen(name);
    const size_t s = std::strlen(suffix);
    return n >= s && std::memcmp(name + (n - s), suffix, s) == 0;
}

std::string join_path(const std::string &dir, const char *name) {
    if (dir.empty()) {
        return std::string(name);
    }
    if (dir.back() == '/') {
        return dir + name;
    }
    return dir + '/' + name;
}

} // namespace

Status st_open(const std::string &path, StFile &out, std::string &err) {
    st_close(out);
    Status s = open_readonly(path, out.fd, out.direct_fd, err);
    if (s != Status::Ok) {
        return s;
    }
    out.path = path;

    uint8_t hs[8];
    s = pread_full(out.fd, hs, sizeof(hs), 0, err);
    if (s != Status::Ok) {
        st_close(out);
        return s;
    }
    out.header_size = read_le_u64(hs);
    out.data_origin = 8ull + out.header_size;

    // Official safetensors limit is 100 MB to avoid zip-bomb headers.
    constexpr uint64_t kMaxHeader = 100ull * 1000ull * 1000ull;
    if (out.header_size == 0 || out.header_size > kMaxHeader) {
        err = "invalid safetensors header_size";
        st_close(out);
        return Status::ParseError;
    }

    const int64_t sz = file_size(out.fd);
    if (sz >= 0 && out.data_origin > static_cast<uint64_t>(sz)) {
        err = "safetensors header exceeds file size";
        st_close(out);
        return Status::ParseError;
    }

    std::string header;
    header.resize(static_cast<size_t>(out.header_size));
    s = pread_full(out.fd, header.data(), header.size(), 8, err);
    if (s != Status::Ok) {
        st_close(out);
        return s;
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(header);
    } catch (const nlohmann::json::exception &e) {
        err = std::string("safetensors header: ") + e.what();
        st_close(out);
        return Status::ParseError;
    }
    if (!j.is_object()) {
        err = "safetensors header is not an object";
        st_close(out);
        return Status::ParseError;
    }

    try {
        for (auto it = j.begin(); it != j.end(); ++it) {
            const std::string name = it.key();
            if (name == "__metadata__") {
                continue;
            }
            const nlohmann::json &v = it.value();
            if (!v.is_object()) {
                err = "tensor entry is not an object: " + name;
                st_close(out);
                return Status::ParseError;
            }
            if (!v.contains("dtype") || !v.at("dtype").is_string()) {
                err = "missing dtype: " + name;
                st_close(out);
                return Status::ParseError;
            }
            if (!v.contains("shape") || !v.at("shape").is_array()) {
                err = "missing shape: " + name;
                st_close(out);
                return Status::ParseError;
            }
            if (!v.contains("data_offsets") || !v.at("data_offsets").is_array() ||
                v.at("data_offsets").size() != 2) {
                err = "bad data_offsets: " + name;
                st_close(out);
                return Status::ParseError;
            }

            StTensor t;
            t.name = name;
            t.dtype = v.at("dtype").get<std::string>();
            t.shape.reserve(v.at("shape").size());
            for (const auto &d : v.at("shape")) {
                t.shape.push_back(d.get<int64_t>());
            }
            t.begin = v.at("data_offsets").at(0).get<uint64_t>();
            t.end = v.at("data_offsets").at(1).get<uint64_t>();
            if (t.end < t.begin) {
                err = "data_offsets inverted: " + name;
                st_close(out);
                return Status::ParseError;
            }
            out.index[t.name] = out.tensors.size();
            out.tensors.push_back(std::move(t));
        }
    } catch (const nlohmann::json::exception &e) {
        err = std::string("safetensors header: ") + e.what();
        st_close(out);
        return Status::ParseError;
    }
    return Status::Ok;
}

void st_close(StFile &f) {
    close_fd(f.fd);
    close_fd(f.direct_fd);
    f.fd = -1;
    f.direct_fd = -1;
    f.header_size = 0;
    f.data_origin = 0;
    f.tensors.clear();
    f.index.clear();
    f.path.clear();
}

const StTensor *st_find(const StFile &f, const std::string &name) {
    auto it = f.index.find(name);
    if (it == f.index.end() || it->second >= f.tensors.size()) {
        return nullptr;
    }
    return &f.tensors[it->second];
}

Status st_read(const StFile &f, const StTensor &t, void *dst, std::string &err) {
    if (t.end < t.begin) {
        err = "invalid tensor offsets: " + t.name;
        return Status::InvalidArgument;
    }
    const uint64_t n = t.end - t.begin;
    if (n == 0) {
        return Status::Ok;
    }
    if (dst == nullptr) {
        err = "st_read: null dest";
        return Status::InvalidArgument;
    }
    if (f.fd < 0) {
        err = "st_read: file not open";
        return Status::IoError;
    }
    return pread_full(f.fd, dst, static_cast<size_t>(n), st_file_offset(f, t), err);
}

Status st_open_dir(const std::string &dir, std::vector<StFile> &out, std::string &err) {
    st_close_dir(out);
    DIR *d = ::opendir(dir.c_str());
    if (d == nullptr) {
        err = "opendir(" + dir + "): " + std::strerror(errno);
        return Status::IoError;
    }

    std::vector<std::string> paths;
    while (true) {
        errno = 0;
        struct dirent *ent = ::readdir(d);
        if (ent == nullptr) {
            if (errno != 0) {
                err = "readdir(" + dir + "): " + std::strerror(errno);
                ::closedir(d);
                return Status::IoError;
            }
            break;
        }
        const char *name = ent->d_name;
        if (std::strcmp(name, ".") == 0 || std::strcmp(name, "..") == 0) {
            continue;
        }
        if (!ends_with(name, ".safetensors")) {
            continue;
        }
        std::string full = join_path(dir, name);
        struct stat st {};
        if (::stat(full.c_str(), &st) != 0) {
            continue;
        }
        if (S_ISDIR(st.st_mode)) {
            continue;
        }
        paths.push_back(std::move(full));
    }
    ::closedir(d);

    std::sort(paths.begin(), paths.end());
    out.reserve(paths.size());
    for (const auto &p : paths) {
        StFile f;
        Status s = st_open(p, f, err);
        if (s != Status::Ok) {
            st_close_dir(out);
            return s;
        }
        out.push_back(std::move(f));
    }
    return Status::Ok;
}

void st_close_dir(std::vector<StFile> &files) {
    for (auto &f : files) {
        st_close(f);
    }
    files.clear();
}

Status st_load_index(const std::string &path, std::unordered_map<std::string, std::string> &weight_map,
                     std::string &err) {
    weight_map.clear();
    int fd = -1;
    int direct_fd = -1;
    Status s = open_readonly(path, fd, direct_fd, err);
    if (s != Status::Ok) {
        return s;
    }
    const int64_t sz = file_size(fd);
    if (sz < 0) {
        err = "fstat(" + path + "): " + std::strerror(errno);
        close_fd(fd);
        close_fd(direct_fd);
        return Status::IoError;
    }
    if (sz == 0) {
        err = "empty index file: " + path;
        close_fd(fd);
        close_fd(direct_fd);
        return Status::ParseError;
    }
    std::string text;
    text.resize(static_cast<size_t>(sz));
    s = pread_full(fd, text.data(), text.size(), 0, err);
    close_fd(fd);
    close_fd(direct_fd);
    if (s != Status::Ok) {
        return s;
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception &e) {
        err = std::string("index json: ") + e.what();
        return Status::ParseError;
    }
    if (!j.is_object() || !j.contains("weight_map") || !j.at("weight_map").is_object()) {
        err = "index missing weight_map object";
        return Status::ParseError;
    }
    try {
        const nlohmann::json &wm = j.at("weight_map");
        for (auto it = wm.begin(); it != wm.end(); ++it) {
            if (!it.value().is_string()) {
                err = "weight_map value is not a string: " + it.key();
                weight_map.clear();
                return Status::ParseError;
            }
            weight_map.emplace(it.key(), it.value().get<std::string>());
        }
    } catch (const nlohmann::json::exception &e) {
        weight_map.clear();
        err = std::string("weight_map: ") + e.what();
        return Status::ParseError;
    }
    return Status::Ok;
}

StHit st_find_dir(const std::vector<StFile> &files, const std::string &name) {
    for (const auto &f : files) {
        const StTensor *t = st_find(f, name);
        if (t)
            return {&f, t};
    }
    return {};
}

Status st_read_f32(const StFile &f, const StTensor &t, float *dst, int64_t cap, std::string &err) {
    if (!dst || cap <= 0) {
        err = "st_read_f32: bad dest";
        return Status::InvalidArgument;
    }
    int64_t numel = 1;
    for (int64_t d : t.shape) {
        if (d <= 0) {
            err = "st_read_f32: empty shape " + t.name;
            return Status::ParseError;
        }
        numel *= d;
    }
    if (numel > cap) {
        err = "st_read_f32: " + t.name + " does not fit";
        return Status::InvalidArgument;
    }
    const int64_t nbytes = st_nbytes(t);
    if (t.dtype == "F32" || t.dtype == "F32_") {
        if (nbytes != numel * 4) {
            err = "st_read_f32: F32 size mismatch " + t.name;
            return Status::ParseError;
        }
        return st_read(f, t, dst, err);
    }
    if (t.dtype == "BF16") {
        if (nbytes != numel * 2) {
            err = "st_read_f32: BF16 size mismatch " + t.name;
            return Status::ParseError;
        }
        std::vector<uint16_t> raw(static_cast<size_t>(numel));
        Status st = st_read(f, t, raw.data(), err);
        if (st != Status::Ok)
            return st;
        for (int64_t i = 0; i < numel; ++i) {
            uint32_t bits = static_cast<uint32_t>(raw[static_cast<size_t>(i)]) << 16;
            float v;
            std::memcpy(&v, &bits, sizeof(v));
            dst[i] = v;
        }
        return Status::Ok;
    }
    if (t.dtype == "F16") {
        err = "st_read_f32: F16 not supported for " + t.name;
        return Status::Unsupported;
    }
    err = "st_read_f32: not a float tensor " + t.name + " (" + t.dtype + ")";
    return Status::Unsupported;
}

} // namespace io
} // namespace mvllm
