#include "h3_adaln.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace mvllm {
namespace {

float silu_one(float x) {
    return x * (1.f / (1.f + std::exp(-x)));
}

bool valid_linear(const float *x, const float *w, const float *y, int rows, int in, int out) {
    return x && w && y && rows >= 1 && in >= 1 && out >= 1;
}

} // namespace

int h3_adaln_out(int hidden) {
    if (hidden < 1)
        return 0;
    return kH3AdalnModalities * kH3AdalnSlots * hidden;
}

void h3_silu(float *x, int n) {
    if (!x || n < 1)
        return;
    for (int i = 0; i < n; ++i)
        x[i] = silu_one(x[i]);
}

void h3_linear(float *y, const float *x, const float *w, const float *b, int rows, int in,
               int out) {
    if (!valid_linear(x, w, y, rows, in, out))
        return;
    const std::size_t in_n = static_cast<std::size_t>(in);
    const std::size_t out_n = static_cast<std::size_t>(out);
    for (int r = 0; r < rows; ++r) {
        const float *xr = x + static_cast<std::size_t>(r) * in_n;
        float *yr = y + static_cast<std::size_t>(r) * out_n;
        for (int o = 0; o < out; ++o) {
            float acc = b ? b[o] : 0.f;
            const float *wo = w + static_cast<std::size_t>(o) * in_n;
            for (int i = 0; i < in; ++i)
                acc += wo[i] * xr[i];
            yr[o] = acc;
        }
    }
}

bool h3_time_embed(const float *features, int rows, int time_input, const float *w_in,
                   const float *b_in, int time_hidden, const float *w_out, const float *b_out,
                   int time_dim, float *temb) {
    if (!features || !w_in || !w_out || !temb || rows < 1 || time_input < 1 || time_hidden < 1 ||
        time_dim < 1)
        return false;
    std::vector<float> hidden(static_cast<std::size_t>(rows) * static_cast<std::size_t>(time_hidden));
    h3_linear(hidden.data(), features, w_in, b_in, rows, time_input, time_hidden);
    for (int r = 0; r < rows; ++r)
        h3_silu(hidden.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(time_hidden),
                time_hidden);
    h3_linear(temb, hidden.data(), w_out, b_out, rows, time_hidden, time_dim);
    for (int r = 0; r < rows; ++r)
        h3_silu(temb + static_cast<std::size_t>(r) * static_cast<std::size_t>(time_dim), time_dim);
    return true;
}

bool h3_adaln_mod(const float *temb, int rows, int time_dim, const float *w_adaln,
                  const float *b_adaln, int adaln_out, float *mod) {
    if (!temb || !w_adaln || !mod || rows < 1 || time_dim < 1 || adaln_out < 1)
        return false;
    h3_linear(mod, temb, w_adaln, b_adaln, rows, time_dim, adaln_out);
    return true;
}

} // namespace mvllm
