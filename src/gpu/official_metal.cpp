#include "official_metal.hpp"

#include <cstdio>
#include <cstring>

#if defined(MVLLM_OFFICIAL_METAL_HOST)
extern "C" {
#include "backend_metal.h"
#include "h3_gpu.h"
}
#endif

namespace mvllm {
namespace official_metal {
namespace {

bool g_coli = false;
bool g_h3 = false;
char g_st[256] = "off";
#if defined(MVLLM_OFFICIAL_METAL_HOST)
h3_gpu *g_h3g = nullptr;
#endif

} // namespace

bool init() {
#if !defined(MVLLM_OFFICIAL_METAL_HOST)
    std::snprintf(g_st, sizeof(g_st), "off");
    return true;
#else
    g_coli = coli_metal_init() != 0;
    char err[256] = {};
#if defined(MVLLM_VENDOR_METAL_DIR)
    const char *sp = MVLLM_VENDOR_METAL_DIR "/h3_shaders.metal";
#else
    const char *sp = "src/gpu/vendor/h3_shaders.metal";
#endif
    g_h3g = h3_gpu_create(sp, err, sizeof(err));
    g_h3 = g_h3g != nullptr;
    if (g_coli && g_h3)
        std::snprintf(g_st, sizeof(g_st), "coli+h3");
    else if (g_coli)
        std::snprintf(g_st, sizeof(g_st), "coli");
    else if (g_h3)
        std::snprintf(g_st, sizeof(g_st), "h3");
    else
        std::snprintf(g_st, sizeof(g_st), "err: %s", err[0] ? err : "init");
    return true;
#endif
}

void shutdown() {
#if defined(MVLLM_OFFICIAL_METAL_HOST)
    if (g_h3g) {
        h3_gpu_free(g_h3g);
        g_h3g = nullptr;
    }
    coli_metal_shutdown();
#endif
    g_coli = false;
    g_h3 = false;
    std::snprintf(g_st, sizeof(g_st), "off");
}

bool coli_available() { return g_coli; }

bool h3_available() { return g_h3; }

const char *status() { return g_st; }

} // namespace official_metal
} // namespace mvllm
