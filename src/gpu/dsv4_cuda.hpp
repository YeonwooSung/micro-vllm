#pragma once

#include <cstdint>
#include <cstddef>

namespace mvllm {
namespace dsv4_cuda {

// Host-side DSV4 GPU-tier API. Names follow official backend_cuda_dsv4.
// CPU API is always on; src/gpu/dsv4_cuda.cu is optional nvcc
// (MVLLM_GPU_CUDA). Official backend_cuda_dsv4.cu is not pasted.
// Returns false on bad args; never throws.

struct Tensor;
struct Activation;
struct KvCache;
struct ExpertSet;
struct Graph;

struct AttentionWeights {
    Tensor *attn_norm = nullptr;
    Tensor *q_a = nullptr;
    Tensor *qkv = nullptr;
    Tensor *q_norm = nullptr;
    Tensor *q_b = nullptr;
    Tensor *wkv = nullptr;
    Tensor *kv_norm = nullptr;
    Tensor *sink = nullptr;
    Tensor *wo_a = nullptr;
    Tensor *wo_b = nullptr;
    Tensor *compress_wkv = nullptr;
    Tensor *compress_wgate = nullptr;
    Tensor *compress_ape = nullptr;
    Tensor *compress_norm = nullptr;
};

enum class Dtype : int { F32 = 0, BF16 = 1, FP8 = 2, FP8BF16 = 3, FP4 = 4 };

bool init(const int *devices, int count);
void shutdown();
bool available();
const char *backend_name();
bool backend_arch_ok(int device);
long long mem_free_mb(int device);

// Weight uploads. scale is e8m0 tiles: FP8 [ceil(O/128), ceil(I/128)],
// FP4 [O, ceil(I/32)]. FP4 weights are packed e2m1 nibbles [O, ceil(I/2)].
bool upload_fp8(Tensor **t, const uint8_t *w, const uint8_t *scale, int O, int I, int device);
bool upload_fp8_bf16(Tensor **t, const uint8_t *w, const uint8_t *scale, int O, int I, int device);
bool upload_fp4(Tensor **t, const uint8_t *w, const uint8_t *scale, int O, int I, int device);
bool upload_bf16(Tensor **t, const uint16_t *w, int O, int I, int device);
bool upload_f32(Tensor **t, const float *w, int O, int I, int device);
bool tensor_refill_fp4(Tensor *t, const uint8_t *w, const uint8_t *scale, int O, int I, int sync);
void tensor_free(Tensor *t);
long long tensor_bytes(const Tensor *t);
int tensor_device(const Tensor *t);
Dtype tensor_dtype(const Tensor *t);
int tensor_rows(const Tensor *t);
int tensor_cols(const Tensor *t);

// y[O] = dequant(W[O,I]) x[I]. No activation QDQ on the host path.
bool matvec(Tensor *t, float *y, const float *x);
bool matmul_batch(Tensor *t, const Activation *input, int tokens, Activation *output);
bool matmul_bf16_batch(Tensor *t, const float *x, int tokens, float *y);
bool matvec_grouped(Tensor *t, float *y, const float *x, int groups);

// Sparse window attention (prefill). vals is [value_rows, dim]: rows
// [0,comp_base) are the window, the rest are compressed. meta[3t+0]=window
// row offset, meta[3t+1]=window rows, meta[3t+2]=visible compressed prefix.
// out is [tokens, heads*dim].
bool sparse_attn_batch(int device, const float *q, const float *vals, const float *sinks,
                       const int *meta, int value_rows, int comp_base, int heads, int dim,
                       int tokens, float scale, float *out);
bool sparse_attn_batch_cached(int device, int layer, const float *q, const float *chunk,
                              int chunk_start, const float *sinks, const int *meta, int abs_base,
                              int comp_limit, int heads, int dim, int tokens, float scale,
                              float *out);
bool sparse_attn_batch_cached_idx(int device, int layer, const float *q, const float *chunk,
                                  int chunk_start, const float *sinks, const int *meta,
                                  const int *sel, int selstride, int abs_base, int comp_limit,
                                  int heads, int dim, int tokens, float scale, float *out);

// scores[t*count+c] = sum_h relu(q[t,h]·k[c]) * head_w[t,h] for c < counts[t].
bool indexer_score_batch(int device, const float *queries, const float *keys, const float *head_w,
                         const int *counts, int tokens, int heads, int dim, int count,
                         float *scores);

// Bitwise host replica of block-scaled e4m3 W @ x (no act QDQ).
bool fp8_ref_matmul(int device, const uint8_t *w, const float *bscale, int rows, int cols,
                    int packed_rows8, const float *x, int tokens, float *y);

bool stream_drain(int device);
bool kv_ring_append(int device, int layer, const float *rows, int start_pos, int count, int window,
                    int dim);
bool kv_comp_append(int device, int layer, const float *rows, int start_idx, int count, int dim);

bool head_argmax(Tensor *t, const float *x, int *id, float *value);
bool final_argmax(const Activation *residual, Tensor *fn, Tensor *scale, Tensor *base, Tensor *norm,
                  Tensor *head, int M, int H, float eps, float pre_eps, int *id, float *value);

// y = sum_e w_e * down_e(swiglu(gate_e(x), up_e(x), limit)). Shared tensors
// in moe() are added after the routed group.
bool expert_group(Tensor *const *gate, Tensor *const *up, Tensor *const *down, const float *weights,
                  int count, float limit, float *y, const float *x);
bool expert_fp8(Tensor *gate, Tensor *up, Tensor *down, float limit, float *y, const float *x);
bool moe(Tensor *const *gate, Tensor *const *up, Tensor *const *down, const float *weights,
         int count, Tensor *shared_gate, Tensor *shared_up, Tensor *shared_down, float limit,
         float *y, const float *x);

bool qkv(Tensor *q_a, Tensor *q_norm, Tensor *q_b, Tensor *kv, float eps, float *q_out,
         float *kv_out, const float *x);
bool wo(Tensor *wo_a, Tensor *wo_b, int groups, float *out, const float *context);

Activation *activation_create(int device, long long elements);
void activation_free(Activation *a);
bool activation_upload(Activation *a, const float *x, long long elements);
bool activation_download(float *x, const Activation *a, long long elements);
bool activation_copy(Activation *dst, const Activation *src, long long elements);
bool activation_copy_range(Activation *dst, long long dst_offset, const Activation *src,
                           long long src_offset, long long elements);
bool activation_sync(const Activation *a);
int activation_device(const Activation *a);
long long activation_elements(const Activation *a);

bool decode_state_set(int device, int token, int position);
void profiler_start();
void profiler_stop();
bool graph_begin(int device);
Graph *graph_end(int device);
bool graph_end_pair(int primary, int peer, Graph **primary_graph, Graph **peer_graph);
bool graph_launch(Graph *graph);
void graph_free(Graph *graph);

// state = post[M] || comb[M*M]. input = collapsed [H]. residual = [M*H].
bool mhc_pre(const Activation *residual, Tensor *fn, Tensor *scale, Tensor *base, int M, int H,
             float rms_eps, float pre_eps, float sink_eps, float post_mult, int sink_iters,
             Activation *state, Activation *input);
bool mhc_pre_norm(const Activation *residual, Tensor *fn, Tensor *scale, Tensor *base, Tensor *norm,
                  int M, int H, float rms_eps, float pre_eps, float sink_eps, float post_mult,
                  int sink_iters, float norm_eps, Activation *state, Activation *input);
bool mhc_pre_norm_batch(const Activation *residual, Tensor *fn, Tensor *scale, Tensor *base,
                        Tensor *norm, int tokens, int H, Activation *state, Activation *input);
bool mhc_pre_batch(const Activation *residual, Tensor *fn, Tensor *scale, Tensor *base, int tokens,
                   int H, Activation *state, Activation *input);
bool mhc_post(const Activation *x, const Activation *residual, const Activation *state, int M,
              int H, Activation *out);
bool mhc_post_pre(const Activation *x, const Activation *residual, Activation *state, int M, int H,
                  Activation *out, Tensor *fn, Tensor *scale, Tensor *base, float rms_eps,
                  float pre_eps, float sink_eps, float post_mult, int sink_iters,
                  Activation *input);
bool mhc_post_pre_norm(const Activation *x, const Activation *residual, Activation *state, int M,
                       int H, Activation *out, Tensor *fn, Tensor *scale, Tensor *base,
                       Tensor *norm, float rms_eps, float pre_eps, float sink_eps, float post_mult,
                       int sink_iters, float norm_eps, Activation *input);
bool mhc_post_pre_norm_batch(const Activation *x, const Activation *residual, Activation *state,
                             int tokens, int H, Activation *out, Tensor *fn, Tensor *scale,
                             Tensor *base, Tensor *norm, Activation *input);
bool mhc_post_batch(const Activation *x, const Activation *residual, const Activation *state,
                    int tokens, int H, Activation *out);

bool attention_first(const Activation *input, Tensor *attn_norm, Tensor *q_a, Tensor *q_norm,
                     Tensor *q_b, Tensor *wkv, Tensor *kv_norm, Tensor *sink, Tensor *wo_a,
                     Tensor *wo_b, int heads, int head_dim, int qk_rope, int groups, float eps,
                     Activation *output);

KvCache *kv_create(int device, int window, int head_dim, int max_tokens, int rope_pairs,
                   const float *rope_cos, const float *rope_sin, const float *compress_cos,
                   const float *compress_sin);
void kv_free(KvCache *cache);

bool attention_window(const Activation *input, Tensor *attn_norm, Tensor *q_a, Tensor *q_norm,
                      Tensor *q_b, Tensor *wkv, Tensor *kv_norm, Tensor *sink, Tensor *wo_a,
                      Tensor *wo_b, Tensor *compress_wkv, Tensor *compress_wgate,
                      Tensor *compress_ape, Tensor *compress_norm, int compress_ratio, int heads,
                      int head_dim, int qk_rope, int groups, int pos, float eps, KvCache *cache,
                      Activation *output);
bool attention_sparse_batch(const Activation *input, Tensor *attn_norm, Tensor *qkv, Tensor *q_norm,
                            Tensor *q_b, Tensor *kv_norm, Tensor *sink, int heads, int head_dim,
                            int start_pos, int tokens, float eps, KvCache *cache,
                            Activation *context);
bool attention_output_batch(const Activation *context, Tensor *wo_a, Tensor *wo_b, int groups,
                            int tokens, Activation *output);
bool attention_window_tp2(const Activation *input, Activation *peer_input,
                          const AttentionWeights *primary, const AttentionWeights *peer,
                          int compress_ratio, int heads, int head_dim, int qk_rope, int groups,
                          int pos, float eps, KvCache *cache, KvCache *peer_cache,
                          Activation *output, Activation *peer_output);

// Top-6. score = sqrt(softplus(Wx)); mix from score; * routed_scale.
// fixed_ids != null uses those 6 ids instead of top-k.
bool route(const Activation *input, Tensor *gate, Tensor *bias, const int *fixed_ids,
           float routed_scale, int ids[6], float weights[6]);
bool rmsnorm(Activation *x, Tensor *weight, float eps, int elements);
bool moe_activation(Tensor *const *gate, Tensor *const *up, Tensor *const *down,
                    const float *weights, int count, Tensor *shared_gate, Tensor *shared_up,
                    Tensor *shared_down, float limit, const Activation *input, Activation *output);

ExpertSet *expert_set_create(Tensor *const *gate, Tensor *const *up, Tensor *const *down, int count,
                             Tensor *shared_gate, Tensor *shared_up, Tensor *shared_down);
ExpertSet *expert_bank_create(int count, int hidden, int intermediate, int device,
                              Tensor *shared_gate, Tensor *shared_up, Tensor *shared_down);
bool expert_bank_upload(ExpertSet *set, int expert, const uint8_t *gate_weight,
                        const uint8_t *gate_scale, const uint8_t *up_weight,
                        const uint8_t *up_scale, const uint8_t *down_weight,
                        const uint8_t *down_scale, Tensor **gate, Tensor **up, Tensor **down);
bool expert_bank_set_shared(ExpertSet *set, Tensor *sg, Tensor *su, Tensor *sd);
bool expert_bank_upload_aux(ExpertSet *set, int expert, const uint8_t *gw, const uint8_t *gs,
                            const uint8_t *uw, const uint8_t *us, const uint8_t *dw,
                            const uint8_t *ds, Tensor **gate, Tensor **up, Tensor **down);
bool expert_bank_upload_tp2(ExpertSet *set, int expert, int rank, const uint8_t *gate_weight,
                            const uint8_t *gate_scale, const uint8_t *up_weight,
                            const uint8_t *up_scale, const uint8_t *down_weight,
                            const uint8_t *down_scale);
void expert_set_free(ExpertSet *set);
bool expert_set_upload_hash(ExpertSet *set, const int64_t *map, int vocab, int topk);

bool route_moe(const Activation *input, Tensor *gate, Tensor *bias, int token, float routed_scale,
               ExpertSet *experts, float limit, Activation *output);
bool route_moe_batch(const Activation *input, Tensor *gate, Tensor *bias, const int *tokens,
                     int count, float routed_scale, ExpertSet *experts, float limit,
                     Activation *output);
bool route_top6_batch(const Activation *input, Tensor *gate, Tensor *bias, int count,
                      float routed_scale, int *ids, float *weights);
bool route_moe_ids_batch(const Activation *input, const int *ids, const float *weights, int count,
                         ExpertSet *experts, float limit, Activation *output);
bool route_moe_ep2(const Activation *input, Tensor *gate, Tensor *bias,
                   const Activation *peer_input, Tensor *peer_gate, Tensor *peer_bias, int token,
                   float routed_scale, ExpertSet *local, ExpertSet *peer, float limit,
                   Activation *output, Activation *peer_output);

} // namespace dsv4_cuda
} // namespace mvllm
