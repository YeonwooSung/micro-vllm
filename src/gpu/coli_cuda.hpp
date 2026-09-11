#pragma once

#include <cstddef>
#include <cstdint>

namespace mvllm {
namespace coli_cuda {

// Generic K3/GLM CUDA-tier API. Names follow official backend_cuda.
// Always compiled: ops run on CPU until a CUDA backend is linked.
// Device GEMM lives in an optional .cu when MVLLM_GPU_CUDA and nvcc exist.
// This header is the CPU-capable API.
// Returns false on bad args; never throws.

// weight_at decodes fmt 0–4 only. fmt 6/7/8 have their own matmul paths.
inline bool weight_at_supported(int fmt) {
    return fmt == 0 || fmt == 1 || fmt == 2 || fmt == 3 || fmt == 4;
}

constexpr int kMaxDevices = 16;

struct Tensor;

bool init(const int *devices, int count);
void shutdown();
bool available();
int available_device_count();
int device_count();
int device_at(int index);
bool mem_info(int device, size_t *free_bytes, size_t *total_bytes);
bool device_integrated(int device);
void stats(int device, size_t *tensor_count, size_t *tensor_bytes);
void group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows, double *h2d_ms,
                 double *kernel_ms, double *d2h_ms);
void group_stats_device(int device, uint64_t *calls, uint64_t *experts, uint64_t *rows,
                        double *h2d_ms, double *kernel_ms, double *d2h_ms);

// Must be published before fmt=6 / fmt=8 uploads. Host keeps no private copy.
bool e8_set_grid(const void *grid);   // 256 x 4 bytes
bool fp8_set_lut(const float *lut);   // 256 f32

// fmt: 0=f32, 1=int8-row, 2=int4 nibble + per-row scale, 3=int2,
//      4=int4 grouped (gs along I), 6=E8/IQ3, 7=MXFP4 (use matmul_mxfp4),
//      8=fp8-e4m3 + 128x128 block scale.
bool tensor_upload(Tensor **t, const void *weights, const float *scales, int fmt, int I, int O,
                   int device, int gs);
bool tensor_upload_g(Tensor **t, const void *weights, const float *scales, int fmt, int I, int O,
                     int device, int gs);
void tensor_free(Tensor *t);
size_t tensor_bytes(const Tensor *t);
int tensor_device(const Tensor *t);
int tensor_fmt(const Tensor *t);
int tensor_rows(const Tensor *t);
int tensor_cols(const Tensor *t);
bool tensor_update(Tensor *t, const void *weights, const float *scales);

// y[S,O] = x[S,I] @ dequant(W[O,I])^T. First call may upload *tensor.
bool matmul(Tensor **t, float *y, const float *x, const void *weights, const float *scales, int fmt,
            int S, int I, int O, int device, int gs);
bool matmul_mxfp4(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s, int S, int I,
                  int O);

// y = down(silu(gate(x)) * up(x)). Tensors already resident, same device.
bool expert_mlp(Tensor *gate, Tensor *up, Tensor *down, float *y, const float *x, int S);
bool shared_mlp_w4a16(Tensor *gate, Tensor *up, Tensor *down, float *y, const float *x, int S);

// Packed group: x/y hold sum(rows) consecutive [D] rows in call order.
bool expert_group(Tensor *const *gates, Tensor *const *ups, Tensor *const *downs, const int *rows,
                  int count, float *y, const float *x);
bool expert_group_issue(Tensor *const *gates, Tensor *const *ups, Tensor *const *downs,
                        const int *rows, int count, const float *x);
const float *expert_group_take(int device);
bool expert_group_pinned(Tensor *const *gates, Tensor *const *ups, Tensor *const *downs,
                         const int *rows, int count, float *y, const float *x, int pin_small_batch);

// Decode MLA absorb. kv_b is [H*(Q+V), K]. q is [H*(Q+R)].
// latent [T,K], rope [T,R]. ctx [H*V].
bool attention_absorb(Tensor *kv_b, float *ctx, const float *q, const float *latent,
                      const float *rope, int H, int Q, int R, int V, int K, int T, float scale);
bool attention_absorb_batch(Tensor *kv_b, float *ctx, const float *q, const float *latent,
                            const float *rope, int S, int H, int Q, int R, int V, int K, int T,
                            float scale);
bool attention_project_batch(Tensor *kv_b, Tensor *o_proj, float *out, const float *q,
                             const float *latent, const float *rope, int S, int H, int Q, int R,
                             int V, int K, int T, float scale);
bool attention_project_ragged(Tensor *kv_b, Tensor *o_proj, float *out, const float *q,
                              const void *const *keys, const float *const *latent,
                              const float *const *rope, const int *lengths, int S, int H, int Q,
                              int R, int V, int K, int max_t, float scale);

float *pipe_scratch(int device, int slot, size_t bytes);
void *pipe_alloc(int device, size_t bytes);
void pipe_free(int device, void *p);
bool pipe_upload(int device, void *dst, const void *src, size_t bytes);
bool pipe_download(int device, const void *src, void *dst, size_t bytes);
bool pipe_rmsnorm(int device, float *y, const float *x, const float *w, int S, int D, float eps);
bool pipe_rope(int device, float *v, const int *pos, int rows, int stride, int offset, int R,
               int heads, float theta);
bool pipe_silu_mul(int device, float *gate, const float *up, size_t n);
bool pipe_add(int device, float *x, const float *t, size_t n);
bool pipe_rows_add(int device, float *x, const float *partial, const int *rows, int nrows, int D);
bool pipe_gemm(Tensor *t, float *y, const float *x, int S);
bool pipe_router(int device, const float *x, const void *rw, const void *rb, int D, int E,
                 int Ksel, float topp, int norm_topk, float routed_scale, int *idx, float *w,
                 int *keff);
bool pipe_sync(int device);

} // namespace coli_cuda
} // namespace mvllm
