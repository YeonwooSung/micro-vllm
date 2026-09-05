#include "h3_resize.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <new>

namespace mvllm {
namespace {

struct Tap {
    int i0 = 0;
    int i1 = 0;
    float w0 = 1.f;
    float w1 = 0.f;
};

bool mul_size(size_t a, size_t b, size_t &out) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
        return false;
    out = a * b;
    return true;
}

bool plane_elems(int frames, int w, int h, size_t &elems) {
    size_t area = 0;
    size_t pix = 0;
    if (!mul_size(static_cast<size_t>(w), static_cast<size_t>(h), area))
        return false;
    if (!mul_size(area, 3, pix))
        return false;
    return mul_size(pix, static_cast<size_t>(frames), elems);
}

Status check_geom(const void *input, int frames, int in_w, int in_h, int out_w, int out_h,
                  size_t &in_elems, size_t &out_elems, std::string &err) {
    if (!input || frames < 1 || in_w < 1 || in_h < 1 || out_w < 1 || out_h < 1) {
        err = "h3_resize: invalid argument";
        return Status::InvalidArgument;
    }
    if (!plane_elems(frames, in_w, in_h, in_elems) || !plane_elems(frames, out_w, out_h, out_elems)) {
        err = "h3_resize: size overflow";
        return Status::InvalidArgument;
    }
    return Status::Ok;
}

// Pixel-center map, then clamp to [0, in-1] (vImageEdgeExtend).
void make_taps(int in_n, int out_n, std::vector<Tap> &taps) {
    taps.resize(static_cast<size_t>(out_n));
    const double scale = static_cast<double>(in_n) / static_cast<double>(out_n);
    const double maxc = static_cast<double>(in_n - 1);
    for (int i = 0; i < out_n; ++i) {
        double s = (static_cast<double>(i) + 0.5) * scale - 0.5;
        if (s < 0.0)
            s = 0.0;
        if (s > maxc)
            s = maxc;
        int i0 = static_cast<int>(s);
        int i1 = i0 + 1;
        if (i1 >= in_n)
            i1 = in_n - 1;
        const float f = static_cast<float>(s - static_cast<double>(i0));
        taps[static_cast<size_t>(i)] = {i0, i1, 1.f - f, f};
    }
}

uint8_t sat_u8(float v) {
    if (!(v > 0.f))
        return 0;
    if (v >= 255.f)
        return 255;
    return static_cast<uint8_t>(std::lround(v));
}

void resize_frame_u8(const uint8_t *src, int in_w, const std::vector<Tap> &xs,
                     const std::vector<Tap> &ys, uint8_t *dst, int out_w, int out_h) {
    const size_t in_row = static_cast<size_t>(in_w) * 3;
    const size_t out_row = static_cast<size_t>(out_w) * 3;
    for (int y = 0; y < out_h; ++y) {
        const Tap &yt = ys[static_cast<size_t>(y)];
        const uint8_t *row0 = src + static_cast<size_t>(yt.i0) * in_row;
        const uint8_t *row1 = src + static_cast<size_t>(yt.i1) * in_row;
        uint8_t *drow = dst + static_cast<size_t>(y) * out_row;
        for (int x = 0; x < out_w; ++x) {
            const Tap &xt = xs[static_cast<size_t>(x)];
            const uint8_t *p00 = row0 + static_cast<size_t>(xt.i0) * 3;
            const uint8_t *p10 = row0 + static_cast<size_t>(xt.i1) * 3;
            const uint8_t *p01 = row1 + static_cast<size_t>(xt.i0) * 3;
            const uint8_t *p11 = row1 + static_cast<size_t>(xt.i1) * 3;
            for (int c = 0; c < 3; ++c) {
                const float v = yt.w0 * (xt.w0 * static_cast<float>(p00[c]) +
                                         xt.w1 * static_cast<float>(p10[c])) +
                                yt.w1 * (xt.w0 * static_cast<float>(p01[c]) +
                                         xt.w1 * static_cast<float>(p11[c]));
                drow[static_cast<size_t>(x) * 3 + static_cast<size_t>(c)] = sat_u8(v);
            }
        }
    }
}

void resize_frame_f32(const float *src, int in_w, const std::vector<Tap> &xs,
                      const std::vector<Tap> &ys, float *dst, int out_w, int out_h) {
    const size_t in_row = static_cast<size_t>(in_w) * 3;
    const size_t out_row = static_cast<size_t>(out_w) * 3;
    for (int y = 0; y < out_h; ++y) {
        const Tap &yt = ys[static_cast<size_t>(y)];
        const float *row0 = src + static_cast<size_t>(yt.i0) * in_row;
        const float *row1 = src + static_cast<size_t>(yt.i1) * in_row;
        float *drow = dst + static_cast<size_t>(y) * out_row;
        for (int x = 0; x < out_w; ++x) {
            const Tap &xt = xs[static_cast<size_t>(x)];
            const float *p00 = row0 + static_cast<size_t>(xt.i0) * 3;
            const float *p10 = row0 + static_cast<size_t>(xt.i1) * 3;
            const float *p01 = row1 + static_cast<size_t>(xt.i0) * 3;
            const float *p11 = row1 + static_cast<size_t>(xt.i1) * 3;
            for (int c = 0; c < 3; ++c) {
                drow[static_cast<size_t>(x) * 3 + static_cast<size_t>(c)] =
                    yt.w0 * (xt.w0 * p00[c] + xt.w1 * p10[c]) +
                    yt.w1 * (xt.w0 * p01[c] + xt.w1 * p11[c]);
            }
        }
    }
}

} // namespace

Status h3_resize_rgb24(const uint8_t *input, int frames, int in_w, int in_h, int out_w, int out_h,
                       std::vector<uint8_t> &out, std::string &err) {
    size_t in_elems = 0;
    size_t out_elems = 0;
    Status st = check_geom(input, frames, in_w, in_h, out_w, out_h, in_elems, out_elems, err);
    if (st != Status::Ok)
        return st;
    try {
        if (in_w == out_w && in_h == out_h) {
            out.assign(input, input + in_elems);
            return Status::Ok;
        }
        out.resize(out_elems);
    } catch (const std::bad_alloc &) {
        err = "h3_resize: out of memory";
        return Status::Oom;
    }
    std::vector<Tap> xs;
    std::vector<Tap> ys;
    make_taps(in_w, out_w, xs);
    make_taps(in_h, out_h, ys);
    const size_t in_frame = in_elems / static_cast<size_t>(frames);
    const size_t out_frame = out_elems / static_cast<size_t>(frames);
    for (int f = 0; f < frames; ++f) {
        resize_frame_u8(input + static_cast<size_t>(f) * in_frame, in_w, xs, ys,
                        out.data() + static_cast<size_t>(f) * out_frame, out_w, out_h);
    }
    return Status::Ok;
}

Status h3_resize_rgb_f32(const float *input, int frames, int in_w, int in_h, int out_w, int out_h,
                         std::vector<float> &out, std::string &err) {
    size_t in_elems = 0;
    size_t out_elems = 0;
    Status st = check_geom(input, frames, in_w, in_h, out_w, out_h, in_elems, out_elems, err);
    if (st != Status::Ok)
        return st;
    try {
        if (in_w == out_w && in_h == out_h) {
            out.assign(input, input + in_elems);
            return Status::Ok;
        }
        out.resize(out_elems);
    } catch (const std::bad_alloc &) {
        err = "h3_resize: out of memory";
        return Status::Oom;
    }
    std::vector<Tap> xs;
    std::vector<Tap> ys;
    make_taps(in_w, out_w, xs);
    make_taps(in_h, out_h, ys);
    const size_t in_frame = in_elems / static_cast<size_t>(frames);
    const size_t out_frame = out_elems / static_cast<size_t>(frames);
    for (int f = 0; f < frames; ++f) {
        resize_frame_f32(input + static_cast<size_t>(f) * in_frame, in_w, xs, ys,
                         out.data() + static_cast<size_t>(f) * out_frame, out_w, out_h);
    }
    return Status::Ok;
}

} // namespace mvllm
