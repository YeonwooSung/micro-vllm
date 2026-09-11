#include "vk_ops.hpp"

#include "../quant/quant.hpp"

#include <vector>

#if defined(MVLLM_WITH_VULKAN)
#include <vulkan/vulkan.h>
#endif

namespace mvllm {
namespace vk_ops {
namespace {

bool g_inited = false;
bool g_vk = false;
#if defined(MVLLM_WITH_VULKAN)
VkInstance g_inst = VK_NULL_HANDLE;
#endif

} // namespace

bool init() {
#if defined(MVLLM_WITH_VULKAN)
    if (!g_inst) {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "micro-vllm";
        app.apiVersion = VK_API_VERSION_1_0;
        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app;
        if (vkCreateInstance(&ci, nullptr, &g_inst) == VK_SUCCESS)
            g_vk = true;
    }
#endif
    g_inited = true;
    return true;
}

void shutdown() {
#if defined(MVLLM_WITH_VULKAN)
    if (g_inst) {
        vkDestroyInstance(g_inst, nullptr);
        g_inst = VK_NULL_HANDLE;
    }
    g_vk = false;
#endif
    g_inited = false;
}

bool available() { return g_inited; }

const char *backend_name() { return g_vk ? "vulkan" : "cpu"; }

bool rmsnorm(float *y, const float *x, const float *w, int nrows, int D, float eps) {
    if (!y || !x || nrows < 1 || D < 1)
        return false;
    for (int r = 0; r < nrows; ++r)
        quant::rmsnorm(x + static_cast<size_t>(r) * D, w, y + static_cast<size_t>(r) * D, D, eps);
    return true;
}

bool add(float *y, const float *a, size_t n) {
    if (!y || !a)
        return false;
    for (size_t i = 0; i < n; ++i)
        y[i] += a[i];
    return true;
}

bool silu_mul(float *g, const float *u, size_t n) {
    if (!g || !u)
        return false;
    quant::silu_mul(g, u, static_cast<int>(n));
    return true;
}

bool gemm_f32(float *y, const float *x, const float *w, int S, int I, int O) {
    if (!y || !x || !w || S < 1 || I < 1 || O < 1)
        return false;
    quant::matmul_f32(y, x, w, S, I, O);
    return true;
}

bool layer_residual(float *x, const float *attn, const float *post_ln, float *nrm, int D,
                    float eps) {
    if (!x || !attn || !post_ln || !nrm || D < 1)
        return false;
    for (int i = 0; i < D; ++i)
        x[i] += attn[i];
    quant::rmsnorm(x, post_ln, nrm, D, eps);
    return true;
}

bool moe_block_f32(int nb, int D, int Iinter, const float *const *g, const float *const *u,
                   const float *const *d, const float *xg, const int *xoff, const int *nr,
                   const int *rows, const float *rw, float *out, int S) {
    if (nb < 1 || D < 1 || !out || !xg || !xoff || !nr)
        return false;
    int base = 0;
    std::vector<float> gate, up, hh;
    for (int e = 0; e < nb; ++e) {
        const int nre = nr[e];
        if (nre <= 0)
            continue;
        if (Iinter < 1 || !g || !u || !d || !g[e] || !u[e] || !d[e]) {
            base += nre;
            continue;
        }
        const float *xe = xg + static_cast<size_t>(xoff[e]) * static_cast<size_t>(D);
        gate.assign(static_cast<size_t>(nre) * static_cast<size_t>(Iinter), 0.f);
        up.assign(static_cast<size_t>(nre) * static_cast<size_t>(Iinter), 0.f);
        hh.assign(static_cast<size_t>(nre) * static_cast<size_t>(D), 0.f);
        quant::matmul_f32(gate.data(), xe, g[e], nre, D, Iinter);
        quant::matmul_f32(up.data(), xe, u[e], nre, D, Iinter);
        quant::silu_mul(gate.data(), up.data(), nre * Iinter);
        quant::matmul_f32(hh.data(), gate.data(), d[e], nre, Iinter, D);
        if (rows && rw) {
            for (int r = 0; r < nre; ++r) {
                const int dest = rows[base + r];
                if (dest < 0 || (S > 0 && dest >= S))
                    continue;
                const float w = rw[base + r];
                const float *hr = hh.data() + static_cast<size_t>(r) * static_cast<size_t>(D);
                float *orow = out + static_cast<size_t>(dest) * static_cast<size_t>(D);
                for (int i = 0; i < D; ++i)
                    orow[i] += w * hr[i];
            }
        }
        base += nre;
    }
    return true;
}

} // namespace vk_ops
} // namespace mvllm
