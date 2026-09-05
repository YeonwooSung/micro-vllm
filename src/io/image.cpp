#include "image.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <zlib.h>

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#endif

namespace mvllm {
namespace {

int align_up(int v, int factor) {
    if (factor <= 0)
        return v;
    return ((v + factor - 1) / factor) * factor;
}

uint32_t be32(const uint8_t *p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}

int paeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    if (pa <= pb && pa <= pc)
        return a;
    if (pb <= pc)
        return b;
    return c;
}

#if defined(__APPLE__)
Status decode_apple(const uint8_t *data, size_t n, std::vector<float> &rgb, int &width, int &height,
                    std::string &err) {
    CFDataRef cf = CFDataCreate(kCFAllocatorDefault, data, static_cast<CFIndex>(n));
    if (!cf) {
        err = "imageio: out of memory";
        return Status::Oom;
    }
    CGImageSourceRef src = CGImageSourceCreateWithData(cf, nullptr);
    CFRelease(cf);
    if (!src) {
        err = "imageio: cannot open";
        return Status::ParseError;
    }
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);
    if (!img) {
        err = "imageio: cannot decode";
        return Status::ParseError;
    }
    width = static_cast<int>(CGImageGetWidth(img));
    height = static_cast<int>(CGImageGetHeight(img));
    if (width < 1 || height < 1) {
        CGImageRelease(img);
        err = "imageio: empty image";
        return Status::ParseError;
    }
    rgb.assign(static_cast<size_t>(width) * height * 3, 0.f);
    std::vector<uint8_t> buf(static_cast<size_t>(width) * height * 4, 0);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(buf.data(), static_cast<size_t>(width),
                                             static_cast<size_t>(height), 8,
                                             static_cast<size_t>(width) * 4, cs,
                                             kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(cs);
    if (!ctx) {
        CGImageRelease(img);
        err = "imageio: cannot rasterize";
        return Status::ParseError;
    }
    CGContextDrawImage(ctx, CGRectMake(0, 0, width, height), img);
    const uint8_t *px = buf.data();
    const size_t stride = static_cast<size_t>(width) * 4;
    for (int y = 0; y < height; ++y) {
        const uint8_t *row = px + static_cast<size_t>(y) * stride;
        float *dst = rgb.data() + static_cast<size_t>(y) * width * 3;
        for (int x = 0; x < width; ++x) {
            dst[x * 3 + 0] = row[x * 4 + 0] / 255.f;
            dst[x * 3 + 1] = row[x * 4 + 1] / 255.f;
            dst[x * 3 + 2] = row[x * 4 + 2] / 255.f;
        }
    }
    CGContextRelease(ctx);
    CGImageRelease(img);
    return Status::Ok;
}
#endif

Status decode_png(const uint8_t *data, size_t n, std::vector<float> &rgb, int &width, int &height,
                  std::string &err) {
    static const uint8_t sig[8] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
    if (n < 8 || std::memcmp(data, sig, 8) != 0) {
        err = "not png";
        return Status::ParseError;
    }
    width = 0;
    height = 0;
    int depth = 0, ctype = 0, interlace = 0;
    std::vector<uint8_t> idat, plte, trns;
    size_t i = 8;
    bool seen_ihdr = false, seen_iend = false;
    while (i + 12 <= n) {
        uint32_t len = be32(data + i);
        if (i + 12 + len > n) {
            err = "truncated png chunk";
            return Status::ParseError;
        }
        const uint8_t *tag = data + i + 4;
        const uint8_t *pay = data + i + 8;
        if (std::memcmp(tag, "IHDR", 4) == 0) {
            if (len < 13) {
                err = "bad png ihdr";
                return Status::ParseError;
            }
            width = static_cast<int>(be32(pay));
            height = static_cast<int>(be32(pay + 4));
            depth = pay[8];
            ctype = pay[9];
            interlace = pay[12];
            seen_ihdr = true;
        } else if (std::memcmp(tag, "PLTE", 4) == 0) {
            plte.assign(pay, pay + len);
        } else if (std::memcmp(tag, "tRNS", 4) == 0) {
            trns.assign(pay, pay + len);
        } else if (std::memcmp(tag, "IDAT", 4) == 0) {
            idat.insert(idat.end(), pay, pay + len);
        } else if (std::memcmp(tag, "IEND", 4) == 0) {
            seen_iend = true;
            break;
        }
        i += 12 + len;
    }
    if (!seen_ihdr || !seen_iend || width < 1 || height < 1 || idat.empty()) {
        err = "incomplete png";
        return Status::ParseError;
    }
    if (interlace != 0) {
        err = "interlaced png unsupported";
        return Status::Unsupported;
    }
    if (depth != 8 && depth != 16) {
        err = "png bit depth unsupported";
        return Status::Unsupported;
    }
    int ch = 0;
    switch (ctype) {
    case 0:
        ch = 1;
        break;
    case 2:
        ch = 3;
        break;
    case 3:
        ch = 1;
        break;
    case 4:
        ch = 2;
        break;
    case 6:
        ch = 4;
        break;
    default:
        err = "png color type unsupported";
        return Status::Unsupported;
    }
    const int bpp = (depth / 8) * ch;
    const size_t raw_need = static_cast<size_t>(height) * (1 + static_cast<size_t>(width) * bpp);
    std::vector<uint8_t> raw(raw_need + 64);
    z_stream zs{};
    zs.next_in = idat.data();
    zs.avail_in = static_cast<uInt>(idat.size());
    zs.next_out = raw.data();
    zs.avail_out = static_cast<uInt>(raw.size());
    if (inflateInit(&zs) != Z_OK) {
        err = "png inflate init failed";
        return Status::ParseError;
    }
    int zr = inflate(&zs, Z_FINISH);
    size_t got = zs.total_out;
    inflateEnd(&zs);
    if (zr != Z_STREAM_END || got < raw_need) {
        err = "png inflate failed";
        return Status::ParseError;
    }
    std::vector<uint8_t> recon(static_cast<size_t>(height) * width * ch);
    const size_t stride = static_cast<size_t>(width) * bpp;
    std::vector<uint8_t> prev(stride, 0), cur(stride, 0);
    const uint8_t *src = raw.data();
    for (int y = 0; y < height; ++y) {
        int filt = *src++;
        std::memcpy(cur.data(), src, stride);
        src += stride;
        for (size_t x = 0; x < stride; ++x) {
            int a = x >= static_cast<size_t>(bpp) ? cur[x - bpp] : 0;
            int b = prev[x];
            int c = x >= static_cast<size_t>(bpp) ? prev[x - bpp] : 0;
            int v = cur[x];
            switch (filt) {
            case 1:
                v = (v + a) & 255;
                break;
            case 2:
                v = (v + b) & 255;
                break;
            case 3:
                v = (v + ((a + b) / 2)) & 255;
                break;
            case 4:
                v = (v + paeth(a, b, c)) & 255;
                break;
            case 0:
                break;
            default:
                err = "png bad filter";
                return Status::ParseError;
            }
            cur[x] = static_cast<uint8_t>(v);
        }
        if (depth == 8) {
            std::memcpy(recon.data() + static_cast<size_t>(y) * width * ch, cur.data(),
                        static_cast<size_t>(width) * ch);
        } else {
            for (int x = 0; x < width * ch; ++x)
                recon[static_cast<size_t>(y) * width * ch + x] = cur[static_cast<size_t>(x) * 2];
        }
        prev.swap(cur);
        std::fill(cur.begin(), cur.end(), 0);
    }
    rgb.assign(static_cast<size_t>(width) * height * 3, 0.f);
    auto put = [&](int y, int x, float r, float g, float b) {
        float *d = rgb.data() + (static_cast<size_t>(y) * width + x) * 3;
        d[0] = r;
        d[1] = g;
        d[2] = b;
    };
    for (int y = 0; y < height; ++y) {
        const uint8_t *row = recon.data() + static_cast<size_t>(y) * width * ch;
        for (int x = 0; x < width; ++x) {
            const uint8_t *p = row + x * ch;
            if (ctype == 0) {
                float g = p[0] / 255.f;
                put(y, x, g, g, g);
            } else if (ctype == 2) {
                put(y, x, p[0] / 255.f, p[1] / 255.f, p[2] / 255.f);
            } else if (ctype == 3) {
                int idx = p[0];
                if (static_cast<size_t>(idx) * 3 + 2 >= plte.size()) {
                    err = "png palette index";
                    return Status::ParseError;
                }
                put(y, x, plte[idx * 3] / 255.f, plte[idx * 3 + 1] / 255.f,
                    plte[idx * 3 + 2] / 255.f);
                (void)trns;
            } else if (ctype == 4) {
                float g = p[0] / 255.f;
                put(y, x, g, g, g);
            } else {
                put(y, x, p[0] / 255.f, p[1] / 255.f, p[2] / 255.f);
            }
        }
    }
    return Status::Ok;
}

// Baseline sequential JPEG (SOF0), 8-bit, 1 or 3 components.
struct JpegBits {
    const uint8_t *p = nullptr;
    const uint8_t *end = nullptr;
    uint32_t buf = 0;
    int nbits = 0;
    bool err = false;
    int stuffed() {
        if (p >= end) {
            err = true;
            return 0;
        }
        int b = *p++;
        if (b == 0xff) {
            if (p >= end) {
                err = true;
                return 0;
            }
            int n = *p;
            if (n == 0x00) {
                ++p;
                return 0xff;
            }
            err = true;
            return 0;
        }
        return b;
    }
    int get(int n) {
        while (nbits < n) {
            buf = (buf << 8) | static_cast<uint32_t>(stuffed());
            nbits += 8;
            if (err)
                return 0;
        }
        nbits -= n;
        return static_cast<int>((buf >> nbits) & ((1u << n) - 1));
    }
};

struct JpegHuff {
    int minc[17]{}, maxc[17]{}, valp[17]{};
    uint8_t vals[256]{};
    int decode(JpegBits &b) const {
        int code = 0;
        for (int i = 1; i <= 16; ++i) {
            code = (code << 1) | b.get(1);
            if (b.err)
                return 0;
            if (minc[i] <= maxc[i] && code <= maxc[i])
                return vals[valp[i] + code - minc[i]];
        }
        b.err = true;
        return 0;
    }
};

void jpeg_build_huff(JpegHuff &h, const uint8_t bits[16], const uint8_t *v, int nv) {
    std::memset(&h, 0, sizeof(h));
    int k = 0, code = 0;
    for (int i = 1; i <= 16; ++i) {
        h.minc[i] = code;
        h.valp[i] = k;
        for (int j = 0; j < bits[i - 1] && k < nv && k < 256; ++j)
            h.vals[k++] = v[h.valp[i] + j];
        h.maxc[i] = bits[i - 1] ? (code + bits[i - 1] - 1) : -1;
        code = (code + bits[i - 1]) << 1;
        if (!bits[i - 1])
            h.minc[i] = 0x7fffffff;
    }
}

void jpeg_idct(float *b) {
    float t[64];
    const float pi = 3.14159265358979323846f;
    auto row = [&](const float *x, float *y) {
        for (int i = 0; i < 8; ++i) {
            float acc = x[0] * 0.707106781f;
            for (int k = 1; k < 8; ++k)
                acc += x[k] * std::cos(pi * (2 * i + 1) * k / 16.f);
            y[i] = acc * 0.5f;
        }
    };
    for (int r = 0; r < 8; ++r)
        row(b + r * 8, t + r * 8);
    for (int c = 0; c < 8; ++c) {
        float col[8], out[8];
        for (int r = 0; r < 8; ++r)
            col[r] = t[r * 8 + c];
        row(col, out);
        for (int r = 0; r < 8; ++r)
            b[r * 8 + c] = out[r];
    }
}

static const int kZig[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
                             12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
                             35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
                             58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

Status decode_jpeg(const uint8_t *data, size_t n, std::vector<float> &rgb, int &width, int &height,
                   std::string &err) {
    if (n < 4 || data[0] != 0xff || data[1] != 0xd8) {
        err = "not jpeg";
        return Status::ParseError;
    }
    int qt[4][64]{};
    JpegHuff dc_h[4], ac_h[4];
    int hsf[3] = {1, 1, 1}, vsf[3] = {1, 1, 1}, qtid[3]{}, cid[3]{};
    int ncomp = 0, precision = 8;
    width = 0;
    height = 0;
    size_t i = 2;
    auto marker = [&]() -> int {
        while (i < n && data[i] != 0xff)
            ++i;
        while (i < n && data[i] == 0xff)
            ++i;
        if (i >= n)
            return -1;
        return data[i++];
    };
    auto seglen = [&]() -> int {
        if (i + 2 > n)
            return -1;
        int L = (data[i] << 8) | data[i + 1];
        i += 2;
        return L;
    };
    bool have_sof = false;
    int rst_int = 0;
    while (i < n) {
        int m = marker();
        if (m < 0)
            break;
        if (m == 0xd9)
            break;
        if (m == 0xda)
            break;
        if (m >= 0xd0 && m <= 0xd7)
            continue;
        int L = seglen();
        if (L < 2 || i + static_cast<size_t>(L - 2) > n) {
            err = "bad jpeg marker";
            return Status::ParseError;
        }
        const uint8_t *p = data + i;
        int plen = L - 2;
        i += static_cast<size_t>(plen);
        if (m == 0xdb) {
            int o = 0;
            while (o < plen) {
                int pq = p[o] >> 4, id = p[o] & 15;
                ++o;
                if (id > 3 || (pq != 0 && pq != 1)) {
                    err = "bad dqt";
                    return Status::ParseError;
                }
                for (int k = 0; k < 64; ++k) {
                    int v = pq ? ((p[o] << 8) | p[o + 1]) : p[o];
                    o += pq ? 2 : 1;
                    qt[id][kZig[k]] = v;
                }
            }
        } else if (m == 0xc4) {
            int o = 0;
            while (o < plen) {
                int tc = p[o] >> 4, th = p[o] & 15;
                ++o;
                if (tc > 1 || th > 3) {
                    err = "bad dht";
                    return Status::ParseError;
                }
                uint8_t bits[16];
                int nv = 0;
                for (int b = 0; b < 16; ++b) {
                    bits[b] = p[o++];
                    nv += bits[b];
                }
                if (o + nv > plen) {
                    err = "short dht";
                    return Status::ParseError;
                }
                jpeg_build_huff(tc ? ac_h[th] : dc_h[th], bits, p + o, nv);
                o += nv;
            }
        } else if (m == 0xc0) {
            if (plen < 6) {
                err = "short sof";
                return Status::ParseError;
            }
            precision = p[0];
            height = (p[1] << 8) | p[2];
            width = (p[3] << 8) | p[4];
            ncomp = p[5];
            if (precision != 8 || ncomp < 1 || ncomp > 3 || width < 1 || height < 1) {
                err = "unsupported jpeg sof";
                return Status::Unsupported;
            }
            if (plen < 6 + ncomp * 3) {
                err = "short sof components";
                return Status::ParseError;
            }
            for (int c = 0; c < ncomp; ++c) {
                cid[c] = p[6 + c * 3];
                hsf[c] = p[7 + c * 3] >> 4;
                vsf[c] = p[7 + c * 3] & 15;
                qtid[c] = p[8 + c * 3];
                if (hsf[c] < 1 || vsf[c] < 1 || qtid[c] > 3) {
                    err = "bad jpeg sampling";
                    return Status::ParseError;
                }
            }
            have_sof = true;
        } else if (m == 0xc2) {
            err = "progressive jpeg unsupported";
            return Status::Unsupported;
        } else if (m == 0xdd && plen >= 2) {
            rst_int = (p[0] << 8) | p[1];
        }
        if (m == 0xda) {
            i -= static_cast<size_t>(plen);
            break;
        }
    }
    // Re-find SOS (loop above breaks after consuming SOS length when m==0xda
    // at the start of the next iteration). Rewind to SOS payload.
    // We exited when m==0xda before seglen. i points at SOS length.
    if (!have_sof) {
        err = "jpeg missing sof";
        return Status::ParseError;
    }
    // Parse SOS
    if (i + 2 > n) {
        err = "jpeg missing sos";
        return Status::ParseError;
    }
    {
        int L = (data[i] << 8) | data[i + 1];
        i += 2;
        if (L < 6 || i + static_cast<size_t>(L - 2) > n) {
            err = "bad sos";
            return Status::ParseError;
        }
    }
    int ns = data[i++];
    if (ns < 1 || ns > 3) {
        err = "bad sos ns";
        return Status::ParseError;
    }
    int dcid[3]{}, acid[3]{}, scan_comp[3]{};
    for (int k = 0; k < ns; ++k) {
        int id = data[i++];
        int ta = data[i++];
        int found = -1;
        for (int c = 0; c < ncomp; ++c)
            if (cid[c] == id)
                found = c;
        if (found < 0) {
            err = "sos unknown component";
            return Status::ParseError;
        }
        scan_comp[k] = found;
        dcid[found] = ta >> 4;
        acid[found] = ta & 15;
    }
    i += 3; // Ss Se AhAl
    int maxh = 1, maxv = 1;
    for (int c = 0; c < ncomp; ++c) {
        maxh = std::max(maxh, hsf[c]);
        maxv = std::max(maxv, vsf[c]);
    }
    const int mcu_w = 8 * maxh;
    const int mcu_h = 8 * maxv;
    const int nx = (width + mcu_w - 1) / mcu_w;
    const int ny = (height + mcu_h - 1) / mcu_h;
    std::vector<float> plane[3];
    int pw[3], ph[3];
    for (int c = 0; c < ncomp; ++c) {
        pw[c] = nx * 8 * hsf[c];
        ph[c] = ny * 8 * vsf[c];
        plane[c].assign(static_cast<size_t>(pw[c]) * ph[c], 0.f);
    }
    JpegBits bits;
    bits.p = data + i;
    bits.end = data + n;
    int pred[3] = {0, 0, 0};
    int mcu_count = 0;
    for (int my = 0; my < ny && !bits.err; ++my) {
        for (int mx = 0; mx < nx && !bits.err; ++mx) {
            if (rst_int > 0 && mcu_count > 0 && (mcu_count % rst_int) == 0) {
                bits.buf = 0;
                bits.nbits = 0;
                pred[0] = pred[1] = pred[2] = 0;
                if (bits.p < bits.end && bits.p[0] == 0xff && bits.p + 1 < bits.end &&
                    bits.p[1] >= 0xd0 && bits.p[1] <= 0xd7)
                    bits.p += 2;
            }
            for (int ci = 0; ci < ns && !bits.err; ++ci) {
                int c = scan_comp[ci];
                int nbh = hsf[c], nbv = vsf[c];
                for (int by = 0; by < nbv; ++by) {
                    for (int bx = 0; bx < nbh; ++bx) {
                        float blk[64]{};
                        int t = dc_h[dcid[c]].decode(bits);
                        int diff = 0;
                        if (t > 0) {
                            int v = bits.get(t);
                            diff = (v < (1 << (t - 1))) ? v - ((1 << t) - 1) : v;
                        }
                        pred[c] += diff;
                        blk[0] = static_cast<float>(pred[c]) * static_cast<float>(qt[qtid[c]][0]);
                        int k = 1;
                        while (k < 64) {
                            int rs = ac_h[acid[c]].decode(bits);
                            if (bits.err)
                                break;
                            int r = rs >> 4, s = rs & 15;
                            if (s == 0) {
                                if (r == 0)
                                    break;
                                k += 16;
                                continue;
                            }
                            k += r;
                            if (k >= 64)
                                break;
                            int v = bits.get(s);
                            int ac = (v < (1 << (s - 1))) ? v - ((1 << s) - 1) : v;
                            blk[kZig[k]] = static_cast<float>(ac) * static_cast<float>(qt[qtid[c]][kZig[k]]);
                            ++k;
                        }
                        jpeg_idct(blk);
                        int ox = mx * 8 * hsf[c] + bx * 8;
                        int oy = my * 8 * vsf[c] + by * 8;
                        for (int yy = 0; yy < 8; ++yy)
                            for (int xx = 0; xx < 8; ++xx)
                                plane[c][static_cast<size_t>(oy + yy) * pw[c] + (ox + xx)] =
                                    blk[yy * 8 + xx];
                    }
                }
            }
            ++mcu_count;
        }
    }
    if (bits.err && mcu_count == 0) {
        err = "jpeg entropy error";
        return Status::ParseError;
    }
    rgb.assign(static_cast<size_t>(width) * height * 3, 0.f);
    auto sample = [&](int c, int x, int y) {
        int sx = x * hsf[c] / maxh;
        int sy = y * vsf[c] / maxv;
        if (sx >= pw[c])
            sx = pw[c] - 1;
        if (sy >= ph[c])
            sy = ph[c] - 1;
        return plane[c][static_cast<size_t>(sy) * pw[c] + sx];
    };
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float Y = sample(0, x, y) + 128.f;
            float Cb = ncomp > 1 ? sample(1, x, y) : 0.f;
            float Cr = ncomp > 2 ? sample(2, x, y) : 0.f;
            float R = Y + 1.402f * Cr;
            float G = Y - 0.344136f * Cb - 0.714136f * Cr;
            float B = Y + 1.772f * Cb;
            auto clip01 = [](float v) {
                v /= 255.f;
                if (v < 0.f)
                    v = 0.f;
                if (v > 1.f)
                    v = 1.f;
                return v;
            };
            float *d = rgb.data() + (static_cast<size_t>(y) * width + x) * 3;
            if (ncomp == 1) {
                float g = clip01(Y);
                d[0] = d[1] = d[2] = g;
            } else {
                d[0] = clip01(R);
                d[1] = clip01(G);
                d[2] = clip01(B);
            }
        }
    }
    return Status::Ok;
}

} // namespace

void glm_smart_resize(int height, int width, int *out_h, int *out_w, int patch, int merge,
                      int temporal, int min_tokens, int max_tokens) {
    if (!out_h || !out_w)
        return;
    if (height < 1)
        height = 1;
    if (width < 1)
        width = 1;
    const int factor = std::max(patch * merge, 1);
    const int temp = temporal > 0 ? temporal : 1;
    const int ppt = temp * factor * factor;
    const int64_t min_px = static_cast<int64_t>(std::max(min_tokens, 1)) * ppt;
    const int64_t max_px = static_cast<int64_t>(std::max(max_tokens, 1)) * ppt;
    int ah = align_up(height, factor);
    int aw = align_up(width, factor);
    int64_t budget = static_cast<int64_t>(temp) * ah * aw;
    if (budget < min_px) {
        float scale = std::sqrt(static_cast<float>(min_px) /
                                static_cast<float>(temp * height * width));
        ah = align_up(std::max(1, static_cast<int>(std::ceil(height * scale))), factor);
        aw = align_up(std::max(1, static_cast<int>(std::ceil(width * scale))), factor);
        budget = static_cast<int64_t>(temp) * ah * aw;
    }
    if (budget > max_px) {
        int lo = 1, hi = height, bh = factor, bw = factor;
        while (lo <= hi) {
            int ch = (lo + hi) / 2;
            int cw = std::max(1, width * ch / height);
            int cand_h = align_up(ch, factor);
            int cand_w = align_up(cw, factor);
            if (static_cast<int64_t>(temp) * cand_h * cand_w <= max_px) {
                bh = cand_h;
                bw = cand_w;
                lo = ch + 1;
            } else {
                hi = ch - 1;
            }
        }
        ah = bh;
        aw = bw;
    }
    *out_h = ah;
    *out_w = aw;
}

Status decode_base64(const std::string &s, std::vector<uint8_t> &out, std::string &err) {
    static const int8_t tab[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    out.clear();
    int val = 0, valb = -8;
    for (unsigned char c : s) {
        if (c == '=' || std::isspace(c))
            continue;
        int8_t d = tab[c];
        if (d < 0) {
            err = "invalid base64";
            return Status::ParseError;
        }
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<uint8_t>((val >> valb) & 0xff));
            valb -= 8;
        }
    }
    return Status::Ok;
}

Status decode_image_bytes(const uint8_t *data, size_t n, std::vector<float> &rgb, int &width,
                          int &height, std::string &err) {
    rgb.clear();
    width = 0;
    height = 0;
    if (!data || n < 8) {
        err = "image too small";
        return Status::ParseError;
    }
    if (n >= 3 && data[0] == 'P' && data[1] == '6') {
        size_t i = 2;
        auto skip = [&]() {
            while (i < n && std::isspace(data[i]))
                ++i;
            if (i < n && data[i] == '#') {
                while (i < n && data[i] != '\n')
                    ++i;
            }
            while (i < n && std::isspace(data[i]))
                ++i;
        };
        auto num = [&](int &v) -> bool {
            skip();
            if (i >= n || !std::isdigit(data[i]))
                return false;
            v = 0;
            while (i < n && std::isdigit(data[i]))
                v = v * 10 + (data[i++] - '0');
            return v > 0;
        };
        int maxv = 0;
        if (!num(width) || !num(height) || !num(maxv)) {
            err = "bad ppm header";
            return Status::ParseError;
        }
        skip();
        const size_t need = static_cast<size_t>(width) * height * 3;
        if (i + need > n) {
            err = "truncated ppm";
            return Status::ParseError;
        }
        rgb.resize(need);
        const float s = maxv > 0 ? 1.f / static_cast<float>(maxv) : 1.f / 255.f;
        for (size_t p = 0; p < need; ++p)
            rgb[p] = data[i + p] * s;
        return Status::Ok;
    }
    if (n >= 54 && data[0] == 'B' && data[1] == 'M') {
        auto u16 = [&](size_t o) {
            return static_cast<int>(data[o] | (data[o + 1] << 8));
        };
        auto i32 = [&](size_t o) {
            return static_cast<int>(data[o] | (data[o + 1] << 8) | (data[o + 2] << 16) |
                                    (data[o + 3] << 24));
        };
        const int off = i32(10);
        width = i32(18);
        int raw_h = i32(22);
        height = raw_h < 0 ? -raw_h : raw_h;
        const int bpp = u16(28);
        const int comp = i32(30);
        if (bpp != 24 || comp != 0 || width <= 0 || height <= 0 || off < 54) {
            err = "unsupported bmp";
            return Status::Unsupported;
        }
        const int rowb = ((width * 3 + 3) / 4) * 4;
        rgb.assign(static_cast<size_t>(width) * height * 3, 0.f);
        const bool bottom = raw_h > 0;
        for (int y = 0; y < height; ++y) {
            int src_y = bottom ? (height - 1 - y) : y;
            const uint8_t *row = data + off + static_cast<size_t>(src_y) * rowb;
            if (row + width * 3 > data + n) {
                err = "truncated bmp";
                return Status::ParseError;
            }
            float *dst = rgb.data() + static_cast<size_t>(y) * width * 3;
            for (int x = 0; x < width; ++x) {
                dst[x * 3 + 0] = row[x * 3 + 2] / 255.f;
                dst[x * 3 + 1] = row[x * 3 + 1] / 255.f;
                dst[x * 3 + 2] = row[x * 3 + 0] / 255.f;
            }
        }
        return Status::Ok;
    }
    if (n >= 8 && data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4e && data[3] == 0x47) {
#if defined(__APPLE__)
        if (decode_apple(data, n, rgb, width, height, err) == Status::Ok)
            return Status::Ok;
#endif
        return decode_png(data, n, rgb, width, height, err);
    }
    if (n >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff) {
#if defined(__APPLE__)
        if (decode_apple(data, n, rgb, width, height, err) == Status::Ok)
            return Status::Ok;
#endif
        return decode_jpeg(data, n, rgb, width, height, err);
    }
#if defined(__APPLE__)
    if (decode_apple(data, n, rgb, width, height, err) == Status::Ok)
        return Status::Ok;
#endif
    err = "unsupported image format (need PPM P6, 24-bit BMP, PNG, or JPEG)";
    return Status::Unsupported;
}

Status decode_image_url(const std::string &url, std::vector<float> &rgb, int &width, int &height,
                        std::string &err) {
    if (url.empty()) {
        err = "image_url.url must be a non-empty string";
        return Status::InvalidArgument;
    }
    if (url.compare(0, 7, "http://") == 0 || url.compare(0, 8, "https://") == 0) {
        err = "remote image URLs are not fetched; send a base64 data: URI or a local path";
        return Status::Unsupported;
    }
    std::vector<uint8_t> bytes;
    if (url.compare(0, 5, "data:") == 0) {
        size_t comma = url.find(',');
        if (comma == std::string::npos) {
            err = "malformed data: URI";
            return Status::ParseError;
        }
        if (url.find("base64", 0) == std::string::npos || url.find("base64") > comma) {
            err = "only base64 data: URIs are supported";
            return Status::Unsupported;
        }
        Status st = decode_base64(url.substr(comma + 1), bytes, err);
        if (st != Status::Ok)
            return st;
    } else {
        std::string path = url;
        if (url.compare(0, 7, "file://") == 0)
            path = url.substr(7);
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            err = "cannot read image " + path;
            return Status::IoError;
        }
        in.seekg(0, std::ios::end);
        const std::streamoff n = in.tellg();
        in.seekg(0);
        if (n <= 0) {
            err = "empty image file";
            return Status::ParseError;
        }
        bytes.resize(static_cast<size_t>(n));
        in.read(reinterpret_cast<char *>(bytes.data()), n);
    }
    return decode_image_bytes(bytes.data(), bytes.size(), rgb, width, height, err);
}

} // namespace mvllm
