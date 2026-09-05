#include "h3_canvas.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace mvllm {
namespace {

constexpr double kCanvasBase = 768.0;
constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
constexpr uint64_t kPcgMult = 6364136223846793005ull;
constexpr uint64_t kGolden = 0x9e3779b97f4a7c15ull;

int snap_multiple(double value) {
    return static_cast<int>(std::nearbyint(value / kH3CanvasMultiple) * kH3CanvasMultiple);
}

int at_least_multiple(int value) {
    return value < kH3CanvasMultiple ? kH3CanvasMultiple : value;
}

} // namespace

bool h3_adapt_canvas(int width, int height, int *adapted_w, int *adapted_h) {
    if (width <= 0 || height <= 0 || !adapted_w || !adapted_h)
        return false;

    const double ratio = static_cast<double>(width) / static_cast<double>(height);
    double nominal_w = 0.0;
    double nominal_h = 0.0;
    if (ratio >= 1.0) {
        nominal_w = kCanvasBase * ratio;
        nominal_h = kCanvasBase;
    } else {
        nominal_w = kCanvasBase;
        nominal_h = kCanvasBase / ratio;
    }

    const double pixels = nominal_w * nominal_h;
    if (pixels > static_cast<double>(kH3MaxPixels)) {
        const double scale = std::sqrt(static_cast<double>(kH3MaxPixels) / pixels);
        nominal_w *= scale;
        nominal_h *= scale;
    }

    *adapted_w = at_least_multiple(snap_multiple(nominal_w));
    *adapted_h = at_least_multiple(snap_multiple(nominal_h));
    return true;
}

bool h3_reference_image_canvas(int width, int height, int target_width, int target_height,
                               int max_short_edge, int *adapted_w, int *adapted_h) {
    if (width < 1 || height < 1 || target_width < 1 || target_height < 1 || max_short_edge < 0 ||
        !adapted_w || !adapted_h)
        return false;

    double scale = 1.0;
    if (max_short_edge) {
        const int short_edge = std::min(width, height);
        scale = std::min(1.0, static_cast<double>(max_short_edge) / static_cast<double>(short_edge));
    } else {
        const double target_area = static_cast<double>(target_width) * static_cast<double>(target_height);
        const double source_area = static_cast<double>(width) * static_cast<double>(height);
        scale = std::min(1.0, std::sqrt(target_area / source_area));
    }

    double out_w = std::nearbyint(static_cast<double>(width) * scale / kH3CanvasMultiple) *
                   kH3CanvasMultiple;
    double out_h = std::nearbyint(static_cast<double>(height) * scale / kH3CanvasMultiple) *
                   kH3CanvasMultiple;
    if (out_w < kH3CanvasMultiple)
        out_w = kH3CanvasMultiple;
    if (out_h < kH3CanvasMultiple)
        out_h = kH3CanvasMultiple;
    if (out_w > static_cast<double>(std::numeric_limits<int>::max()) ||
        out_h > static_cast<double>(std::numeric_limits<int>::max()))
        return false;

    *adapted_w = static_cast<int>(out_w);
    *adapted_h = static_cast<int>(out_h);
    return true;
}

bool h3_reference_video_canvas(int width, int height, int *adapted_w, int *adapted_h) {
    if (width < 1 || height < 1 || !adapted_w || !adapted_h ||
        !h3_adapt_canvas(width, height, adapted_w, adapted_h))
        return false;

    const double source_area = static_cast<double>(width) * static_cast<double>(height);
    const double target_area = static_cast<double>(*adapted_w) * static_cast<double>(*adapted_h);
    if (source_area < target_area) {
        *adapted_w = at_least_multiple(snap_multiple(static_cast<double>(width)));
        *adapted_h = at_least_multiple(snap_multiple(static_cast<double>(height)));
    }
    return true;
}

uint32_t h3_rng_u32(H3Rng &rng) {
    const uint64_t old = rng.state;
    rng.state = old * kPcgMult + rng.increment;
    const uint32_t shifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
    const uint32_t rotation = static_cast<uint32_t>(old >> 59u);
    return (shifted >> rotation) | (shifted << ((-static_cast<int32_t>(rotation)) & 31));
}

void h3_rng_seed(H3Rng &rng, uint64_t seed) {
    rng = H3Rng{};
    rng.increment = (seed << 1u) | 1u;
    (void)h3_rng_u32(rng);
    rng.state += seed ^ kGolden;
    (void)h3_rng_u32(rng);
}

float h3_rng_normal(H3Rng &rng) {
    if (rng.has_spare) {
        rng.has_spare = 0;
        return rng.spare;
    }
    const double u1 = (static_cast<double>(h3_rng_u32(rng)) + 1.0) / 4294967297.0;
    const double u2 = (static_cast<double>(h3_rng_u32(rng)) + 0.5) / 4294967296.0;
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double angle = kTwoPi * u2;
    rng.spare = static_cast<float>(radius * std::sin(angle));
    rng.has_spare = 1;
    return static_cast<float>(radius * std::cos(angle));
}

void h3_rng_fill_normal(H3Rng &rng, float *values, int count) {
    if (!values || count <= 0)
        return;
    for (int i = 0; i < count; ++i)
        values[i] = h3_rng_normal(rng);
}

} // namespace mvllm
