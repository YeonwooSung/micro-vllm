#include "file_io.hpp"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

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

std::string mime_type(const std::string &path) {
    const auto dot = path.rfind('.');
    if (dot == std::string::npos || dot + 1 >= path.size()) {
        return "application/octet-stream";
    }
    std::string ext = path.substr(dot);
    for (char &c : ext) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    if (ext == ".html" || ext == ".htm")
        return "text/html; charset=utf-8";
    if (ext == ".css")
        return "text/css";
    if (ext == ".js")
        return "application/javascript";
    if (ext == ".json")
        return "application/json";
    if (ext == ".svg")
        return "image/svg+xml";
    if (ext == ".png")
        return "image/png";
    if (ext == ".jpg" || ext == ".jpeg")
        return "image/jpeg";
    if (ext == ".gif")
        return "image/gif";
    if (ext == ".webp")
        return "image/webp";
    if (ext == ".ico")
        return "image/x-icon";
    if (ext == ".woff2")
        return "font/woff2";
    if (ext == ".txt")
        return "text/plain; charset=utf-8";
    if (ext == ".wasm")
        return "application/wasm";
    return "application/octet-stream";
}

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

std::string url_unquote(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            const int hi = hex_nibble(s[i + 1]);
            const int lo = hex_nibble(s[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(s[i]);
    }
    return out;
}

static bool realpath_into(const std::string &in, std::string &out) {
    char buf[PATH_MAX];
    if (::realpath(in.c_str(), buf) == nullptr) {
        return false;
    }
    out.assign(buf);
    return true;
}

static bool is_dir(const std::string &path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static bool is_reg(const std::string &path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// Resolved path must equal root or sit under root+"/".
static bool under_root(const std::string &root, const std::string &path) {
    if (path == root) {
        return true;
    }
    std::string prefix = root;
    if (prefix.empty() || prefix.back() != '/') {
        prefix += '/';
    }
    return path.size() >= prefix.size() && path.compare(0, prefix.size(), prefix) == 0;
}

bool static_resolve(const std::string &root, const std::string &url_path, std::string &out_path,
                    std::string &content_type) {
    if (root.empty()) {
        return false;
    }
    std::string root_real;
    if (!realpath_into(root, root_real) || !is_dir(root_real)) {
        return false;
    }

    std::string rel = url_unquote(url_path);
    size_t nslash = 0;
    while (nslash < rel.size() && rel[nslash] == '/') {
        ++nslash;
    }
    rel.erase(0, nslash);
    if (rel.empty()) {
        rel = "index.html";
    }

    std::string joined = root_real;
    if (joined.back() != '/') {
        joined += '/';
    }
    joined += rel;

    std::string file_real;
    const bool hit = realpath_into(joined, file_real) && under_root(root_real, file_real) &&
                     is_reg(file_real);
    if (!hit) {
        // SPA: "/" or extension-less routes fall back to index.html.
        if (url_path != "/" && rel.find('.') != std::string::npos) {
            return false;
        }
        joined = root_real;
        if (joined.back() != '/') {
            joined += '/';
        }
        joined += "index.html";
        if (!realpath_into(joined, file_real) || !under_root(root_real, file_real) ||
            !is_reg(file_real)) {
            return false;
        }
    }
    out_path = file_real;
    content_type = mime_type(file_real);
    return true;
}

} // namespace io
} // namespace mvllm
