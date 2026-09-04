#pragma once

#include "../core/types.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mvllm {
namespace io {

struct StTensor {
    std::string name;
    std::string dtype; // "BF16", "F32", "U8", "I8", "F16"
    std::vector<int64_t> shape;
    uint64_t begin = 0; // offset in the data section
    uint64_t end = 0;
};

struct StFile {
    std::string path;
    int fd = -1;
    int direct_fd = -1;
    uint64_t header_size = 0;
    uint64_t data_origin = 0; // 8 + header_size
    std::vector<StTensor> tensors;
    std::unordered_map<std::string, size_t> index;
};

Status st_open(const std::string &path, StFile &out, std::string &err);
void st_close(StFile &f);

const StTensor *st_find(const StFile &f, const std::string &name);

// Read tensor bytes into dst (must be end-begin bytes).
Status st_read(const StFile &f, const StTensor &t, void *dst, std::string &err);

// Absolute file offset of tensor data.
inline int64_t st_file_offset(const StFile &f, const StTensor &t) {
    return static_cast<int64_t>(f.data_origin + t.begin);
}

// Load every *.safetensors (and model-*-of-*.safetensors) in a directory.
Status st_open_dir(const std::string &dir, std::vector<StFile> &out, std::string &err);
void st_close_dir(std::vector<StFile> &files);

// HuggingFace index.json helper: weight_map[name] -> shard filename.
Status st_load_index(const std::string &path,
                     std::unordered_map<std::string, std::string> &weight_map, std::string &err);

struct StHit {
    const StFile *file = nullptr;
    const StTensor *tensor = nullptr;
};

StHit st_find_dir(const std::vector<StFile> &files, const std::string &name);

inline int64_t st_nbytes(const StTensor &t) { return static_cast<int64_t>(t.end - t.begin); }

// Expand BF16/F32 tensor into dst[0..numel). cap is the float count.
Status st_read_f32(const StFile &f, const StTensor &t, float *dst, int64_t cap, std::string &err);

} // namespace io
} // namespace mvllm
