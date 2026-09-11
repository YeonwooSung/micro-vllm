#include "vk_ops.hpp"

#include "../quant/quant.hpp"

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

} // namespace vk_ops
} // namespace mvllm
