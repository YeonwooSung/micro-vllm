#pragma once

#include "../core/types.hpp"

#include <cstdint>

namespace mvllm {
namespace gpu {

class Backend {
public:
    virtual ~Backend() = default;
    virtual Device device() const = 0;
    virtual const char *name() const = 0;
    virtual void gemm_f32(float *y, const float *x, const float *w, int S, int I, int O) = 0;
    virtual void gemm_int4_g64(float *y, const float *x, const uint8_t *packed, const float *scales,
                               int S, int I, int O) = 0;
    virtual void gemm_mxfp4(float *y, const float *x, const uint8_t *packed, const uint8_t *scales,
                            int S, int I, int O, bool idot) = 0;
    virtual void dit_block(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes,
                           int64_t fc1_bytes, int64_t fc2_bytes, int hidden, int inner, int ffn,
                           int head_dim, float *x, int tokens, float eps) = 0;
};

// Process-wide backend. select() may fall back to CPU if the GPU is missing.
Backend &active();
Backend &select(Device want);
Device device();
const char *name();
const char *compiled();
bool gpu_ready();

void gemm_f32(float *y, const float *x, const float *w, int S, int I, int O);
void gemm_int4_g64(float *y, const float *x, const uint8_t *packed, const float *scales, int S,
                   int I, int O);
void gemm_mxfp4(float *y, const float *x, const uint8_t *packed, const uint8_t *scales, int S,
                int I, int O, bool idot);

// K3 MXFP4 SiTU-GLU expert: y[S,I] from x[S,I] and the six-piece blob.
void k3_expert(float *y, const float *x, int S, const uint8_t *blob, int I, int O, float b1,
               float b2, bool idot);
// GLM int4-g64 clamped-SwiGLU expert: y[S,H] from x[S,H].
void glm_expert(float *y, const float *x, int S, const uint8_t *blob, int H, int O, float limit);

void dit_block(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
               int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
               int tokens, float eps);

#if defined(MVLLM_WITH_METAL)
Backend *make_metal_backend();
#endif
#if defined(MVLLM_WITH_CUDA_GEMM)
Backend *make_cuda_backend();
#endif

} // namespace gpu
} // namespace mvllm
