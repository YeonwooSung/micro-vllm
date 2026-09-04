#include "file_io.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mvllm {
namespace io {

void *aligned_alloc_pages(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    constexpr size_t kAlign = 4096;
    size_t n = (bytes + kAlign - 1) & ~(kAlign - 1);
    void *p = nullptr;
    if (posix_memalign(&p, kAlign, n) != 0) {
        return nullptr;
    }
    return p;
}

void aligned_free_pages(void *p) { std::free(p); }

Status pread_full(int fd, void *dst, size_t bytes, int64_t offset, std::string &err) {
    if (bytes == 0) {
        return Status::Ok;
    }
    if (fd < 0 || dst == nullptr || offset < 0) {
        err = "pread_full: invalid argument";
        return Status::InvalidArgument;
    }
    auto *p = static_cast<uint8_t *>(dst);
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = ::pread(fd, p + done, bytes - done,
                            static_cast<off_t>(offset) + static_cast<off_t>(done));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            err = std::string("pread: ") + std::strerror(errno);
            return Status::IoError;
        }
        if (n == 0) {
            err = "pread: short read";
            return Status::IoError;
        }
        done += static_cast<size_t>(n);
    }
    return Status::Ok;
}

Status open_readonly(const std::string &path, int &fd, int &direct_fd, std::string &err) {
    fd = -1;
    direct_fd = -1;
    int regular = ::open(path.c_str(), O_RDONLY);
    if (regular < 0) {
        err = "open(" + path + "): " + std::strerror(errno);
        return Status::IoError;
    }
    fd = regular;
#if defined(__linux__) && defined(O_DIRECT)
    int dfd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (dfd >= 0) {
        direct_fd = dfd;
    }
#endif
#if defined(__APPLE__)
    // Bypass the UBC so a 40 GiB DiT / 1.5 TiB expert corpus is not cached twice.
    set_uncached(fd);
#endif
    return Status::Ok;
}

void set_uncached(int fd) {
    if (fd < 0)
        return;
#if defined(__APPLE__) && defined(F_NOCACHE)
    (void)fcntl(fd, F_NOCACHE, 1);
#endif
    (void)fd;
}

Status pread_direct(int direct_fd, int fd, void *dst, void *aligned_scratch, size_t scratch_bytes,
                    size_t bytes, int64_t offset, std::string &err) {
    if (bytes == 0)
        return Status::Ok;
    if (direct_fd < 0 || !aligned_scratch) {
        return pread_full(fd, dst, bytes, offset, err);
    }
    constexpr int64_t kAlign = 4096;
    const int64_t a0 = offset & ~(kAlign - 1);
    const int64_t pad = offset - a0;
    const int64_t want = pad + static_cast<int64_t>(bytes);
    int64_t dlen = (want + kAlign - 1) & ~(kAlign - 1);
    if (static_cast<size_t>(dlen) > scratch_bytes) {
        return pread_full(fd, dst, bytes, offset, err);
    }
    struct stat sb {};
    if (::fstat(direct_fd, &sb) == 0 && a0 + dlen > sb.st_size) {
        dlen = (sb.st_size - a0) & ~(kAlign - 1);
    }
    if (dlen > 0) {
        Status st = pread_full(direct_fd, aligned_scratch, static_cast<size_t>(dlen), a0, err);
        if (st != Status::Ok)
            return st;
    }
    if (dlen < want) {
        Status st = pread_full(fd, static_cast<uint8_t *>(aligned_scratch) + dlen,
                               static_cast<size_t>(want - dlen), a0 + dlen, err);
        if (st != Status::Ok)
            return st;
    }
    if (dst != static_cast<uint8_t *>(aligned_scratch) + pad) {
        std::memmove(dst, static_cast<uint8_t *>(aligned_scratch) + pad, bytes);
    }
    return Status::Ok;
}

void close_fd(int fd) {
    if (fd >= 0) {
        ::close(fd);
    }
}

int64_t file_size(int fd) {
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        return -1;
    }
    return static_cast<int64_t>(st.st_size);
}

Status mmap_file(const std::string &path, Map &out, std::string &err) {
    out = Map{};
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "open(" + path + "): " + std::strerror(errno);
        return Status::IoError;
    }
    int64_t sz = file_size(fd);
    if (sz < 0) {
        err = "fstat(" + path + "): " + std::strerror(errno);
        ::close(fd);
        return Status::IoError;
    }
    const size_t bytes = static_cast<size_t>(sz);
    if (bytes == 0) {
        out.ptr = nullptr;
        out.bytes = 0;
        out.fd = fd;
        return Status::Ok;
    }
    void *p = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        err = "mmap(" + path + "): " + std::strerror(errno);
        ::close(fd);
        return Status::IoError;
    }
    out.ptr = p;
    out.bytes = bytes;
    out.fd = fd;
    return Status::Ok;
}

void munmap_file(Map &m) {
    if (m.ptr != nullptr && m.ptr != MAP_FAILED && m.bytes > 0) {
        ::munmap(m.ptr, m.bytes);
    }
    close_fd(m.fd);
    m.ptr = nullptr;
    m.bytes = 0;
    m.fd = -1;
}

} // namespace io
} // namespace mvllm
