#pragma once

#include "../core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {
namespace io {

// 4 KiB-aligned allocation for O_DIRECT windows.
void *aligned_alloc_pages(size_t bytes);
void aligned_free_pages(void *p);

// pread the exact length. Returns Status::IoError on short read / errno.
Status pread_full(int fd, void *dst, size_t bytes, int64_t offset, std::string &err);

// Open path read-only. If want_direct and the OS supports it, also try O_DIRECT
// (Linux). On macOS / unsupported hosts, *direct_fd stays -1.
Status open_readonly(const std::string &path, int &fd, int &direct_fd, std::string &err);
void close_fd(int fd);

// Darwin: F_NOCACHE so SSD streaming does not pollute the page cache.
// Linux / others: no-op (O_DIRECT is the equivalent, via direct_fd).
void set_uncached(int fd);

// Aligned-window read for O_DIRECT. `aligned_dst` must be 4 KiB-aligned and
// large enough for the padded window. Copies the [offset, offset+bytes) slice
// into `dst` (may alias aligned_dst). Falls back to buffered `fd` for the
// head/tail that cannot be direct-read.
Status pread_direct(int direct_fd, int fd, void *dst, void *aligned_scratch, size_t scratch_bytes,
                    size_t bytes, int64_t offset, std::string &err);

// File-backed mmap (read-only). length 0 maps the whole file.
struct Map {
    void *ptr = nullptr;
    size_t bytes = 0;
    int fd = -1;
};
Status mmap_file(const std::string &path, Map &out, std::string &err);
void munmap_file(Map &m);

int64_t file_size(int fd);

} // namespace io
} // namespace mvllm
