#include "cuda_to_hip.h"
#include "kernels.cuh"
#include "legacy/llama_dims.hpp"
#include <iostream>

// RoPE sin/cos for (pos, freq), freq = 0..ROPE_FREQS-1. Filled once on first rope call.
static float *d_rope_cos = nullptr;
static float *d_rope_sin = nullptr;

__global__ void initRopeTablesKernel(float *rope_cos, float *rope_sin)
{
    int pos = blockIdx.x;
    int freq = threadIdx.x;
    if (pos >= MAX_SEQ_LEN || freq >= ROPE_FREQS)
    {
        return;
    }
    int double_i = 2 * freq;
    float theta = 1.0 / (pow(500000.0, ((float)double_i / HEAD_DIM)));
    float angle = pos * theta;
    rope_cos[pos * ROPE_FREQS + freq] = cos(angle);
    rope_sin[pos * ROPE_FREQS + freq] = sin(angle);
}

static void ensureRopeTables()
{
    if (d_rope_cos != nullptr)
    {
        return;
    }
    const size_t n = static_cast<size_t>(MAX_SEQ_LEN) * ROPE_FREQS;
    cudaMalloc(&d_rope_cos, n * sizeof(float));
    cudaMalloc(&d_rope_sin, n * sizeof(float));
    initRopeTablesKernel<<<MAX_SEQ_LEN, ROPE_FREQS>>>(d_rope_cos, d_rope_sin);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// prefill / shared

// gpu_input_tokens - N tokens
// gpu_input_embeds - N * sizeof(__nv_bfloat16) * 2048
// embed_tokens - (100000+smth, 2048)
// num_input_tokens - N (just N, not N tokens)
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    if (workIndex < num_input_tokens * 2048)
    {
        gpu_input_embeds[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
        gpu_input_embeds[workIndex + 1024] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
    }
}

void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    // even though embedding is 2048, I can only dispatch 1024 because it's max threads per block on my gpu
    embeddingGatherKernel<<<num_input_tokens, 1024>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens, num_input_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights, int num_tokens)
{
    __shared__ float rms_vector[1024];
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    if (workIndex < num_tokens * 2048)
    {
        rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + 1024] * (float)input[workIndex + 1024];
        __syncthreads();
        // tree reduction
        for (int i = 1; i < 1024; i = i * 2)
        {
            if (threadIdx.x % (i * 2) == 0)
            {
                rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + i];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5);
        }
        __syncthreads();
        // <(^-^)>
        output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);
        output[workIndex + 1024] = (__nv_bfloat16)(((float)input[workIndex + 1024] / rms_vector[0]) * (float)norm_weights[threadIdx.x + 1024]);
    }
}

// (N, 2048) -> (N, 2048)
void rmsNorm(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights, int num_tokens)
{
    rmsNormKernel<<<num_tokens, 1024>>>(input, output, norm_weights, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void ropeKernel(__nv_bfloat16 *input, int num_tokens, int proj_dim, const float *rope_cos, const float *rope_sin)
{
    int pair = threadIdx.x;
    int pos = blockIdx.x;
    // 2*pair+1 must stay inside this token row (proj_dim elements)
    if (pos < num_tokens && pos < MAX_SEQ_LEN && 2 * pair + 1 < proj_dim)
    {
        int freq = pair % ROPE_FREQS;
        float cos_a = rope_cos[pos * ROPE_FREQS + freq];
        float sin_a = rope_sin[pos * ROPE_FREQS + freq];
        __nv_bfloat16 prev_2i = input[2 * pair + pos * proj_dim];
        __nv_bfloat16 prev_2i_1 = input[2 * pair + 1 + pos * proj_dim];
        input[2 * pair + pos * proj_dim] = (__nv_bfloat16)((float)prev_2i * cos_a - (float)prev_2i_1 * sin_a);
        input[2 * pair + 1 + pos * proj_dim] = (__nv_bfloat16)((float)prev_2i * sin_a + (float)prev_2i_1 * cos_a);
    }
}

// proj_dim: q_proj 2048, k_proj 512
// num_threads: I want to use it for both q_proj and k_proj so need to parameterize num_threads (1024 for q_proj and 512 for k_proj)
void rope(__nv_bfloat16 *input, int num_tokens, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ensureRopeTables();
    ropeKernel<<<num_tokens, num_threads>>>(input, num_tokens, proj_dim, d_rope_cos, d_rope_sin);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void causalMaskKernel(__nv_bfloat16 *input, int num_tokens)
{
    if (threadIdx.x + blockIdx.x * blockDim.x >= num_tokens * num_tokens * NUM_Q_HEADS)
    {
        return;
    }

    int column = threadIdx.x;
    int row = blockIdx.x % num_tokens;
    if (column > row)
    {
        input[blockIdx.x * num_tokens + threadIdx.x] = -HUGE_VALF;
    }
}

void causalMask(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Causal mask kernel not launched";
        return;
    }

    causalMaskKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void softmaxKernel(__nv_bfloat16 *input, int num_tokens)
{
    // softmaxxing per head
    // might waste a lot of memory by hardcoding the size here but can't use num_tokens directly
    __shared__ float row[1024]; // row[0] will contain max value after the loop
    __shared__ float max_val;
    // find max of the row to subtract it for numerical stability
    int workIndex = blockIdx.x * num_tokens + threadIdx.x;
    __nv_bfloat16 token = input[workIndex];
    row[threadIdx.x] = (float)token;
    __syncthreads();

    for (int i = 1; i < num_tokens; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens)
        {
            row[threadIdx.x] = fmaxf(row[threadIdx.x], row[threadIdx.x + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        max_val = row[0]; // so I don't need to allocate another shared value for max_val
    }
    __syncthreads();

    // turn into exp
    row[threadIdx.x] = expf((float)token - max_val);
    __syncthreads();

    // now I can compute the numerical stable sum, similar pattern - tree reduction
    // reusing row memory
    for (int i = 1; i < num_tokens; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens)
        {
            row[threadIdx.x] = row[threadIdx.x] + row[threadIdx.x + i];
        }
        __syncthreads();
    }

    input[workIndex] = (__nv_bfloat16)(expf((float)token - max_val) / row[0]);
}

// input are masked attention scores (NUM_Q_HEADS, num_tok, num_tok)
void softmax(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Softmax kernel not launched";
        return;
    }

    softmaxKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void residualKernel(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input[workIndex] = input[workIndex] + input_embeds[workIndex];
    input[workIndex + 1024] = input[workIndex + 1024] + input_embeds[workIndex + 1024];
}

// (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
void residualAdd(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds, int num_tokens)
{
    residualKernel<<<num_tokens, 1024>>>(input, input_embeds);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void siluKernel(__nv_bfloat16 *a, __nv_bfloat16 *b)
{
    int workIndex = threadIdx.x + blockIdx.x * 8192;
    for (int i = 0; i < 8192; i += 1024)
    {
        a[workIndex + i] = (__nv_bfloat16)((float)a[workIndex + i] * (1 / (1 + expf(-(float)a[workIndex + i]))) * (float)b[workIndex + i]);
    }
}

// in-place, overwriting a
void silu(__nv_bfloat16 *a, __nv_bfloat16 *b, int num_tokens)
{
    siluKernel<<<num_tokens, 1024>>>(a, b);
}

// decode
__global__ void embeddingGatherKernelDecode(int *gpu_last_tokens, int num_tokens, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens)
{
    int input_token = gpu_last_tokens[blockIdx.x];
    int workIndex = blockIdx.x * 2048 + threadIdx.x;
    if (workIndex < num_tokens * 2048)
    {
        output[workIndex] = embed_tokens[input_token * 2048 + threadIdx.x];
        output[workIndex + 1024] = embed_tokens[input_token * 2048 + threadIdx.x + 1024];
    }
}

void embeddingGatherDecode(int *gpu_last_tokens, int num_tokens, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens)
{
    // even though embedding is 2048, I can only dispatch 1024 because it's max threads per block on my gpu
    embeddingGatherKernelDecode<<<num_tokens, 1024>>>(gpu_last_tokens, num_tokens, output, embed_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

__global__ void ropeKernelDecode(__nv_bfloat16 *input, int position_in_sequence, int proj_dim, const float *rope_cos, const float *rope_sin)
{
    int pair = threadIdx.x;
    // 2*pair+1 must stay inside this token row (proj_dim elements)
    if (2 * pair + 1 < proj_dim && position_in_sequence >= 0 && position_in_sequence < MAX_SEQ_LEN)
    {
        int freq = pair % ROPE_FREQS;
        float cos_a = rope_cos[position_in_sequence * ROPE_FREQS + freq];
        float sin_a = rope_sin[position_in_sequence * ROPE_FREQS + freq];
        __nv_bfloat16 prev_2i = input[2 * pair];
        __nv_bfloat16 prev_2i_1 = input[2 * pair + 1];
        input[2 * pair] = (__nv_bfloat16)((float)prev_2i * cos_a - (float)prev_2i_1 * sin_a);
        input[2 * pair + 1] = (__nv_bfloat16)((float)prev_2i * sin_a + (float)prev_2i_1 * cos_a);
    }
}

// proj_dim: q_proj 2048, k_proj 512
// num_threads: I want to use it for both q_proj and k_proj so need to parameterize num_threads (1024 for q_proj and 512 for k_proj)
void ropeDecode(__nv_bfloat16 *input, int position_in_sequence, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ensureRopeTables();
    ropeKernelDecode<<<1, num_threads>>>(input, position_in_sequence, proj_dim, d_rope_cos, d_rope_sin);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// seq_len increases by 1 with every new token
__global__ void softmaxKernelDecode(__nv_bfloat16 *input, int seq_len)
{
    // softmaxxing per head
    // might waste a lot of memory by hardcoding the size here but can't use num_tokens directly
    __shared__ float row[1024]; // row[0] will contain max value after the loop
    __shared__ float max_val;
    // find max of the row to subtract it for numerical stability
    int workIndex = blockIdx.x * MAX_SEQ_LEN + threadIdx.x;
    __nv_bfloat16 token = input[workIndex];
    row[threadIdx.x] = (float)token;
    __syncthreads();

    for (int i = 1; i < seq_len; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < seq_len)
        {
            row[threadIdx.x] = fmaxf(row[threadIdx.x], row[threadIdx.x + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        max_val = row[0]; // so I don't need to allocate another shared value for max_val
    }
    __syncthreads();

    // turn into exp
    row[threadIdx.x] = expf((float)token - max_val);
    __syncthreads();

    // now I can compute the numerical stable sum, similar pattern - tree reduction
    // reusing row memory
    for (int i = 1; i < seq_len; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < seq_len)
        {
            row[threadIdx.x] = row[threadIdx.x] + row[threadIdx.x + i];
        }
        __syncthreads();
    }

    input[workIndex] = (__nv_bfloat16)(expf((float)token - max_val) / row[0]);
}

// input are masked attention scores (NUM_Q_HEADS, seq_len)
void softmaxDecode(__nv_bfloat16 *input, int seq_len)
{
    if (seq_len > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Softmax kernel not launched";
        return;
    }

    softmaxKernelDecode<<<NUM_Q_HEADS, seq_len>>>(input, seq_len);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}

// inside a single particular thread that processes a single position of particular Q head for a particular sequence, for particular layer
__global__ void pagedAttentionKernel(int layer, int num_active_slots, __nv_bfloat16 *q_proj, __nv_bfloat16 *kv_cache, int *block_table_gpu, int *gpu_seq_lens, int *gpu_active_slots, __nv_bfloat16 *output)
{
    __shared__ float dot_products[2];
    int active_slot = blockIdx.x; // active_slot == seq_id
    int slot = gpu_active_slots[active_slot];
    int q_head_id = blockIdx.y;
    int thread_id = threadIdx.x;
    int kv_head_idx = q_head_id / GQA_Q_TO_K_RATIO;
    __nv_bfloat16 q = q_proj[active_slot * EMBEDDING_LENGTH + q_head_id * HEAD_DIM + thread_id];
    int seq_len = gpu_seq_lens[active_slot];
    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // for online softmax https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
    float current_max = -INFINITY;
    float acc = 0.0f;
    float d = 0.0f; // denominator, same name as in paper above

    for (int logical_block_idx = 0; logical_block_idx < num_blocks; ++logical_block_idx)
    {
        int physical_block = block_table_gpu[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
        int tokens_in_block = min(seq_len - logical_block_idx * BLOCK_SIZE, BLOCK_SIZE);
        for (int token = 0; token < tokens_in_block; ++token)
        {
            __nv_bfloat16 *k = (__nv_bfloat16 *)((char *)kv_cache + physical_block * BLOCK_BYTES + token * KV_DIM * sizeof(__nv_bfloat16) + kv_head_idx * HEAD_DIM * sizeof(__nv_bfloat16) + thread_id * sizeof(__nv_bfloat16));
            __nv_bfloat16 *v = (__nv_bfloat16 *)((char *)kv_cache + physical_block * BLOCK_BYTES + V_OFFSET + token * KV_DIM * sizeof(__nv_bfloat16) + kv_head_idx * HEAD_DIM * sizeof(__nv_bfloat16) + thread_id * sizeof(__nv_bfloat16));
            float qk = (float)q * (float)*k;
            // tree reduction within current warp, thread 0 gets sum of all 32 elements within warp
            // could be done with __syncthreads but accessing memory of other threads in warp is op
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 16);
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 8);
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 4);
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 2);
            qk += __shfl_down_sync(WARP_FULL_MASK, qk, 1);
            if (thread_id == 0)
            {
                dot_products[0] = qk;
            }
            if (thread_id == 32)
            {
                dot_products[1] = qk;
            }
            __syncthreads();
            if (thread_id == 0)
            {
                dot_products[0] = (dot_products[0] + dot_products[1]) / SQRT_HEAD_DIM;
            }
            __syncthreads();
            float dot_product = dot_products[0];
            // online softmax
            float new_max = current_max;
            if (dot_product > current_max)
            {
                new_max = dot_product;
            }
            float correction_factor = expf(current_max - new_max);
            current_max = new_max;
            float exp_score = expf(dot_product - current_max);
            d = d * correction_factor + exp_score;
            acc = acc * correction_factor + exp_score * (float)*v;
        }
    }
    output[active_slot * EMBEDDING_LENGTH + q_head_id * HEAD_DIM + thread_id] = acc / d;
}

void pagedAttention(int layer, int num_active_slots, __nv_bfloat16 *q_proj, __nv_bfloat16 *kv_cache, int *block_table_gpu, int *gpu_seq_lens, int *gpu_active_slots, __nv_bfloat16 *output)
{
    pagedAttentionKernel<<<dim3(num_active_slots, NUM_Q_HEADS), HEAD_DIM>>>(layer, num_active_slots, q_proj, kv_cache, block_table_gpu, gpu_seq_lens, gpu_active_slots, output);
}

// one block per row; each thread grid-strides over vocab and reduces in shared memory
__global__ void argmaxRowsKernel(const __nv_bfloat16 *logits, int *out_ids, int n, int vocab)
{
    int row = blockIdx.x;
    if (row >= n)
    {
        return;
    }
    const __nv_bfloat16 *row_logits = logits + (size_t)row * (size_t)vocab;
    float best_val = -INFINITY;
    int best_idx = 0;
    for (int col = threadIdx.x; col < vocab; col += blockDim.x)
    {
        float v = (float)row_logits[col];
        if (v > best_val)
        {
            best_val = v;
            best_idx = col;
        }
    }
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    s_val[threadIdx.x] = best_val;
    s_idx[threadIdx.x] = best_idx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            float other_val = s_val[threadIdx.x + stride];
            int other_idx = s_idx[threadIdx.x + stride];
            if (other_val > s_val[threadIdx.x] || (other_val == s_val[threadIdx.x] && other_idx < s_idx[threadIdx.x]))
            {
                s_val[threadIdx.x] = other_val;
                s_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        out_ids[row] = s_idx[0];
    }
}

void argmaxRows(const __nv_bfloat16 *logits, int *out_ids, int n, int vocab)
{
    if (n <= 0 || vocab <= 0)
    {
        return;
    }
    argmaxRowsKernel<<<n, 256>>>(logits, out_ids, n, vocab);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}