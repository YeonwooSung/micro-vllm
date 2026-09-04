#include "backend.hpp"

#include "../model/family.hpp"
#include "../quant/quant.hpp"

#include <vector>

namespace mvllm {
namespace gpu {
namespace {

int packed_stride(int I) { return (I + 1) / 2; }
int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct K3Geom {
    int64_t w1p = 0, w1s = 0, w2p = 0, w2s = 0;
};

K3Geom k3_geom(int latent, int inter) {
    K3Geom g{};
    if (latent >= 32 && inter >= 32 && latent % 32 == 0 && inter % 32 == 0) {
        g.w1p = static_cast<int64_t>(inter) * (latent / 2);
        g.w1s = static_cast<int64_t>(inter) * (latent / 32);
        g.w2p = static_cast<int64_t>(latent) * (inter / 2);
        g.w2s = static_cast<int64_t>(latent) * (inter / 32);
    } else {
        g.w1p = static_cast<int64_t>(inter) * packed_stride(latent);
        g.w1s = static_cast<int64_t>(inter) * ceil_div(latent, 32);
        g.w2p = static_cast<int64_t>(latent) * packed_stride(inter);
        g.w2s = static_cast<int64_t>(latent) * ceil_div(inter, 32);
    }
    return g;
}

struct GlmGeom {
    int64_t pack_go = 0, sc_go = 0, pack_d = 0, sc_d = 0;
};

GlmGeom glm_geom(int hidden, int inter) {
    GlmGeom g{};
    if (hidden > 0 && inter > 0 && hidden % 64 == 0 && inter % 64 == 0) {
        const int64_t pack = static_cast<int64_t>(inter) * hidden / 2;
        const int64_t sc =
            static_cast<int64_t>(inter) * hidden / 64 * static_cast<int64_t>(sizeof(float));
        g.pack_go = pack;
        g.sc_go = sc;
        g.pack_d = pack;
        g.sc_d = sc;
        return g;
    }
    g.pack_go = static_cast<int64_t>(inter) * packed_stride(hidden);
    g.sc_go = static_cast<int64_t>(inter) * ceil_div(hidden, 64) * static_cast<int64_t>(sizeof(float));
    g.pack_d = static_cast<int64_t>(hidden) * packed_stride(inter);
    g.sc_d = static_cast<int64_t>(hidden) * ceil_div(inter, 64) * static_cast<int64_t>(sizeof(float));
    return g;
}

class CpuBackend final : public Backend {
public:
    Device device() const override { return Device::Cpu; }
    const char *name() const override { return "cpu"; }

    void gemm_f32(float *y, const float *x, const float *w, int S, int I, int O) override {
        quant::matmul_f32(y, x, w, S, I, O);
    }
    void gemm_int4_g64(float *y, const float *x, const uint8_t *packed, const float *scales, int S,
                       int I, int O) override {
        quant::matmul_int4_g64(y, x, packed, scales, S, I, O);
    }
    void gemm_mxfp4(float *y, const float *x, const uint8_t *packed, const uint8_t *scales, int S,
                    int I, int O, bool idot) override {
        if (idot)
            quant::matmul_mxfp4_i8(y, x, packed, scales, S, I, O);
        else
            quant::matmul_mxfp4(y, x, packed, scales, S, I, O);
    }
    void dit_block(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                   int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                   int tokens, float eps) override {
        h3_dit_block_cpu(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                         head_dim, x, tokens, eps);
    }
};

CpuBackend g_cpu;
Backend *g_active = &g_cpu;

} // namespace

Backend &active() { return *g_active; }

Backend &select(Device want) {
    if (want == Device::Metal) {
#if defined(MVLLM_WITH_METAL)
        static Backend *metal = make_metal_backend();
        if (metal) {
            g_active = metal;
            return *g_active;
        }
#endif
    }
    if (want == Device::Cuda) {
#if defined(MVLLM_WITH_CUDA_GEMM)
        static Backend *cuda = make_cuda_backend();
        if (cuda) {
            g_active = cuda;
            return *g_active;
        }
#endif
    }
    g_active = &g_cpu;
    return *g_active;
}

Device device() { return active().device(); }
const char *name() { return active().name(); }

const char *compiled() {
#if defined(MVLLM_WITH_METAL) && defined(MVLLM_WITH_CUDA_GEMM)
    return "cpu,metal,cuda";
#elif defined(MVLLM_WITH_METAL)
    return "cpu,metal";
#elif defined(MVLLM_WITH_CUDA_GEMM)
    return "cpu,cuda";
#else
    return "cpu";
#endif
}

bool gpu_ready() {
    Device d = device();
    return d == Device::Metal || d == Device::Cuda;
}

void gemm_f32(float *y, const float *x, const float *w, int S, int I, int O) {
    active().gemm_f32(y, x, w, S, I, O);
}
void gemm_int4_g64(float *y, const float *x, const uint8_t *packed, const float *scales, int S,
                   int I, int O) {
    active().gemm_int4_g64(y, x, packed, scales, S, I, O);
}
void gemm_mxfp4(float *y, const float *x, const uint8_t *packed, const uint8_t *scales, int S,
                int I, int O, bool idot) {
    active().gemm_mxfp4(y, x, packed, scales, S, I, O, idot);
}

void k3_expert(float *y, const float *x, int S, const uint8_t *blob, int I, int O, float b1,
               float b2, bool idot) {
    if (!y || !x || !blob || S <= 0 || I <= 0 || O <= 0)
        return;
    const K3Geom g = k3_geom(I, O);
    const uint8_t *w1p = blob;
    const uint8_t *w1s = w1p + g.w1p;
    const uint8_t *w2p = w1s + g.w1s;
    const uint8_t *w2s = w2p + g.w2p;
    const uint8_t *w3p = w2s + g.w2s;
    const uint8_t *w3s = w3p + g.w1p;
    std::vector<float> gate(static_cast<size_t>(S) * O), up(static_cast<size_t>(S) * O);
    active().gemm_mxfp4(gate.data(), x, w1p, w1s, S, I, O, idot);
    active().gemm_mxfp4(up.data(), x, w3p, w3s, S, I, O, idot);
    for (int s = 0; s < S; ++s) {
        float *gg = gate.data() + static_cast<size_t>(s) * O;
        const float *uu = up.data() + static_cast<size_t>(s) * O;
        for (int i = 0; i < O; ++i)
            gg[i] = quant::situ_glu(gg[i], uu[i], b1, b2);
    }
    active().gemm_mxfp4(y, gate.data(), w2p, w2s, S, O, I, idot);
}

void glm_expert(float *y, const float *x, int S, const uint8_t *blob, int H, int O, float limit) {
    if (!y || !x || !blob || S <= 0 || H <= 0 || O <= 0)
        return;
    const GlmGeom g = glm_geom(H, O);
    const uint8_t *gp = blob;
    const float *gs = reinterpret_cast<const float *>(gp + g.pack_go);
    const uint8_t *up = gp + g.pack_go + g.sc_go;
    const float *us = reinterpret_cast<const float *>(up + g.pack_go);
    const uint8_t *dp = up + g.pack_go + g.sc_go;
    const float *ds = reinterpret_cast<const float *>(dp + g.pack_d);
    std::vector<float> gate(static_cast<size_t>(S) * O), u(static_cast<size_t>(S) * O);
    active().gemm_int4_g64(gate.data(), x, gp, gs, S, H, O);
    active().gemm_int4_g64(u.data(), x, up, us, S, H, O);
    for (int s = 0; s < S; ++s) {
        float *gg = gate.data() + static_cast<size_t>(s) * O;
        const float *uu = u.data() + static_cast<size_t>(s) * O;
        for (int i = 0; i < O; ++i)
            gg[i] = quant::clamped_swiglu(gg[i], uu[i], limit);
    }
    active().gemm_int4_g64(y, gate.data(), dp, ds, S, O, H);
}

void dit_block(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
               int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
               int tokens, float eps) {
    active().dit_block(blob, qkv_bytes, out_bytes, fc1_bytes, fc2_bytes, hidden, inner, ffn,
                       head_dim, x, tokens, eps);
}

} // namespace gpu
} // namespace mvllm
