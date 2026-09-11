#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "cuda_to_hip.h"
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"
#include "kernels.cuh"
#include "llama_dims.hpp"

using json = nlohmann::json;

constexpr int B_TO_MB = 1024 * 1024;
constexpr int B_TO_GB = 1024 * 1024 * 1024;
constexpr int END_OF_TEXT_TOKEN_ID = 128001; // <|end_of_text|>
constexpr int EOT_ID_TOKEN_ID = 128009;      // <|eot_id|>
int g_max_prompt = DEFAULT_MAX_PROMPT;


int checkGPUStatus()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
    return 0;
}

struct Weights
{
    __nv_bfloat16 *embed_tokens;
    __nv_bfloat16 *input_layernorm[N_LAYERS];
    __nv_bfloat16 *mlp_gate_proj[N_LAYERS];
    __nv_bfloat16 *mlp_up_proj[N_LAYERS];
    __nv_bfloat16 *mlp_down_proj[N_LAYERS];
    __nv_bfloat16 *post_attn_layernorms[N_LAYERS];
    __nv_bfloat16 *w_k[N_LAYERS];
    __nv_bfloat16 *w_o[N_LAYERS];
    __nv_bfloat16 *w_q[N_LAYERS];
    __nv_bfloat16 *w_v[N_LAYERS];
    __nv_bfloat16 *norm;
};

int loadWeights(Weights &weights, const std::string &path) {
    if (checkGPUStatus() != 0) {
        return 1;
    }

    // read safetensors (read as read_binary)
    std::ifstream safetensors_file(path, std::ios_base::binary);
    if (!safetensors_file.is_open()) {
        // if failed to open, print an error message and return 1
        std::cout << "Can't open " << path << " file\n";
        safetensors_file.close();
        return 1;
    }

    // read safetensors header size
    uint64_t header_size;
    safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);

    // read safetnesors header
    std::string header;
    header.resize(header_size);
    safetensors_file.read(header.data(), header_size);

    // Read offsets of every layer (tensor) to know where every layer starts and ends in the memory.
    //
    // The header is a JSON that contains about all the tensors inside the file.
    // JSON is just a group of pairs <key, value>, where key is a unique string with a tensor name and value is another JSON object,
    // containing info about this tensor.
    // Every key in this JSON is a name of the tensor, except a single key which is called __metadata__,
    // probably for some additonal info when necessary.
    // Every value is a JSON containing three keys - dtype, shape and offsets. dtype says what data type the tensor is stored in.
    // shape says the dimensions of a tensor and offsets say where the tensor is stored, within the tensors data section.
    // Every shape is a list of ints of unknown length and every offsets value is a vector of exactly two ints.
    // First element says where the tensor begins and last element says where the tensor ends.
    std::unordered_map<std::string, uint64_t> offsets;
    json header_json = json::parse(header);
    uint64_t max_offset = 0;
    std::vector<char> seen_layers(256, 0);
    int file_layer_count = 0;
    for (auto &[key, value] : header_json.items()) {
        if (key == "__metadata__") {
            continue;
        }
        uint64_t offset_end = value["data_offsets"].at(1).get<uint64_t>();
        if (offset_end > max_offset) {
            max_offset = offset_end;
        }
        offsets[key] = value["data_offsets"].at(0).get<uint64_t>();
        if (key.compare(0, 13, "model.layers.") == 0) {
            try {
                int n = std::stoi(key.substr(13));
                if (n >= 0 && n < (int)seen_layers.size() && !seen_layers[n]) {
                    seen_layers[n] = 1;
                    ++file_layer_count;
                }
            } catch (...) {
            }
        }
    }
    if (file_layer_count != N_LAYERS) {
        std::cerr << "warning: safetensors has " << file_layer_count << " layers, expected " << N_LAYERS << "\n";
    }
    std::cout << "dims layers=" << N_LAYERS << " hidden=" << EMBEDDING_LENGTH << " mlp=" << HIDDEN_DIM
              << " q=" << NUM_Q_HEADS << " kv=" << NUM_K_HEADS << " vocab=" << VOCAB_SIZE << "\n";

    void *model_weights;
    cudaMalloc(&model_weights, max_offset); // max_offset tells where the model weights end in the memory

    std::vector<char> model_weights_cpu;
    model_weights_cpu.resize(max_offset);
    safetensors_file.read(model_weights_cpu.data(), max_offset);

    cudaMemcpy(model_weights, model_weights_cpu.data(), max_offset, cudaMemcpyHostToDevice);
    safetensors_file.close();

    // BASICALLY A HELPER STRUCT TO HAVE AN EASY ACCESS TO ANY MODEL WEIGHTS ON GPU
    weights.embed_tokens = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.embed_tokens.weight"));
    weights.norm = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.norm.weight"));
    for (int i = 0; i < N_LAYERS; ++i) {
        weights.input_layernorm[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".input_layernorm.weight"));
        weights.mlp_down_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.down_proj.weight"));
        weights.mlp_gate_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"));
        weights.mlp_up_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".mlp.up_proj.weight"));
        weights.post_attn_layernorms[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"));
        weights.w_k[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.k_proj.weight"));
        weights.w_o[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight"));
        weights.w_q[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight"));
        weights.w_v[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".self_attn.v_proj.weight"));
    }
    return 0;
}

void syncBlockTableSlot(int slot, const std::vector<int> &block_table, int *block_table_gpu)
{
    const int off = llama_block_index(slot, 0, 0);
    const int n = llama_slot_table_elems();
    cudaMemcpy(block_table_gpu + off, block_table.data() + off, n * sizeof(int), cudaMemcpyHostToDevice);
}

void releaseSlot(int slot, std::vector<bool> &is_slot_free, std::vector<int> &block_table, std::vector<int> &free_blocks, int *block_table_gpu)
{
    is_slot_free[slot] = true;
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        for (int logical_block_idx = 0; logical_block_idx < MAX_BLOCKS_PER_SEQ; ++logical_block_idx)
        {
            int block_idx = llama_block_index(slot, layer, logical_block_idx);
            if (block_table[block_idx] != -1)
            {
                free_blocks.push_back(block_table[block_idx]);
                block_table[block_idx] = -1;
            }
        }
    }
    syncBlockTableSlot(slot, block_table, block_table_gpu);
}

void set_prompt_len(int slot, int n, std::vector<int> &current_prompt_len, std::vector<int> &prompt_lengths)
{
    current_prompt_len[slot] = n;
    prompt_lengths[slot] = n;
}

int max_live_prompt_len(const std::vector<int> &current_prompt_len, const std::vector<bool> &is_slot_free)
{
    int packed = 0;
    for (int slot = 0; slot < (int)current_prompt_len.size(); ++slot)
    {
        if (!is_slot_free[slot] && current_prompt_len[slot] > packed)
        {
            packed = current_prompt_len[slot];
        }
    }
    return packed;
}

struct PrefillCtx
{
    std::queue<std::vector<int>> &queue;
    std::vector<bool> &is_slot_free;
    int *gpu_input_tokens;
    nv_bfloat16 *input_embeddings;
    Weights &weights;
    nv_bfloat16 *hidden_state;
    nv_bfloat16 *rms_norms;
    nv_bfloat16 *buf_2048_1;
    cublasHandle_t cublas_handle;
    float &q_proj_alpha;
    float &q_proj_beta;
    float &k_proj_alpha;
    float &k_proj_beta;
    float &v_proj_alpha;
    float &v_proj_beta;
    nv_bfloat16 *prefill_attn_scores;
    float &attn_alpha;
    float &attn_beta;
    float &attn_scores_v_alpha;
    float &attn_scores_v_beta;
    nv_bfloat16 *buf_2048_2;
    float &o_proj_alpha;
    float &o_proj_beta;
    float &gate_alpha;
    float &gate_beta;
    nv_bfloat16 *gate;
    float &up_alpha;
    float &up_beta;
    nv_bfloat16 *up;
    float &down_alpha;
    float &down_beta;
    float &embed_alpha;
    float &embed_beta;
    nv_bfloat16 *embed_proj;
    std::vector<std::vector<int>> &generated_tokens;
    std::vector<int> &last_generated_tokens;
    std::vector<int> &current_prompt_len;
    std::vector<int> &prompt_lengths;
    int &input_tokens_size;
    __nv_bfloat16 *k_proj_temp_buf;
    __nv_bfloat16 *v_proj_temp_buf;
    std::vector<int> &block_table;
    int *block_table_gpu;
    std::vector<int> &free_blocks;
    __nv_bfloat16 *kv_cache;
};

void prefill(PrefillCtx &ctx, int slot)
{
    std::queue<std::vector<int>> &queue = ctx.queue;
    std::vector<bool> &is_slot_free = ctx.is_slot_free;
    int *gpu_input_tokens = ctx.gpu_input_tokens;
    nv_bfloat16 *input_embeddings = ctx.input_embeddings;
    Weights &weights = ctx.weights;
    nv_bfloat16 *hidden_state = ctx.hidden_state;
    nv_bfloat16 *rms_norms = ctx.rms_norms;
    nv_bfloat16 *buf_2048_1 = ctx.buf_2048_1;
    cublasHandle_t cublas_handle = ctx.cublas_handle;
    float &q_proj_alpha = ctx.q_proj_alpha;
    float &q_proj_beta = ctx.q_proj_beta;
    float &k_proj_alpha = ctx.k_proj_alpha;
    float &k_proj_beta = ctx.k_proj_beta;
    float &v_proj_alpha = ctx.v_proj_alpha;
    float &v_proj_beta = ctx.v_proj_beta;
    nv_bfloat16 *prefill_attn_scores = ctx.prefill_attn_scores;
    float &attn_alpha = ctx.attn_alpha;
    float &attn_beta = ctx.attn_beta;
    float &attn_scores_v_alpha = ctx.attn_scores_v_alpha;
    float &attn_scores_v_beta = ctx.attn_scores_v_beta;
    nv_bfloat16 *buf_2048_2 = ctx.buf_2048_2;
    float &o_proj_alpha = ctx.o_proj_alpha;
    float &o_proj_beta = ctx.o_proj_beta;
    float &gate_alpha = ctx.gate_alpha;
    float &gate_beta = ctx.gate_beta;
    nv_bfloat16 *gate = ctx.gate;
    float &up_alpha = ctx.up_alpha;
    float &up_beta = ctx.up_beta;
    nv_bfloat16 *up = ctx.up;
    float &down_alpha = ctx.down_alpha;
    float &down_beta = ctx.down_beta;
    float &embed_alpha = ctx.embed_alpha;
    float &embed_beta = ctx.embed_beta;
    nv_bfloat16 *embed_proj = ctx.embed_proj;
    std::vector<std::vector<int>> &generated_tokens = ctx.generated_tokens;
    std::vector<int> &last_generated_tokens = ctx.last_generated_tokens;
    std::vector<int> &current_prompt_len = ctx.current_prompt_len;
    std::vector<int> &prompt_lengths = ctx.prompt_lengths;
    int &input_tokens_size = ctx.input_tokens_size;
    __nv_bfloat16 *k_proj_temp_buf = ctx.k_proj_temp_buf;
    __nv_bfloat16 *v_proj_temp_buf = ctx.v_proj_temp_buf;
    std::vector<int> &block_table = ctx.block_table;
    int *block_table_gpu = ctx.block_table_gpu;
    std::vector<int> &free_blocks = ctx.free_blocks;
    __nv_bfloat16 *kv_cache = ctx.kv_cache;
    nv_bfloat16 *q_proj = nullptr;
    nv_bfloat16 *attn_scores_v = nullptr;
    nv_bfloat16 *o_proj = nullptr;
    nv_bfloat16 *down = nullptr;

    if (queue.empty())
    {
        return;
    }
    std::vector<int> prompt = queue.front();
    queue.pop();
    if ((int)prompt.size() > g_max_prompt)
    {
        return;
    }

    int prompt_len = (int)prompt.size();
    // packed prefill width for this slot's single-prompt buffer
    input_tokens_size = prompt_len;
    is_slot_free[slot] = false;

    cudaMemcpy(gpu_input_tokens, prompt.data(), prompt_len * sizeof(int), cudaMemcpyHostToDevice);
    embeddingGather(gpu_input_tokens, input_embeddings, weights.embed_tokens, prompt_len);

    cudaMemcpy(hidden_state,
               input_embeddings,
               prompt_len * EMBEDDING_LENGTH * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToDevice);
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], prompt_len);

        // Q = inputs * wq^T; my matrices are row-major, cublas expects column-major
        // it perceives my matrices as transposed
        // there's a trick where C = A * B == C^T = B^T * A^T
        // so in my scenario cublas sees now: Q = inputs^T * wq^T^T = inputs ^T * wq
        // so I need to do: Q^T = wq ^T * inputs
        // the beauty is that we don't need to transpose Q^T back to Q
        // because cublas sees the output as column-major
        // so it's in fact transposed
        // final dim (num_tok, EMBEDDING_LENGTH)
        q_proj = buf_2048_1;
        cublasStatus_t q_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    EMBEDDING_LENGTH,
                                                    prompt_len,
                                                    EMBEDDING_LENGTH,
                                                    &q_proj_alpha,
                                                    weights.w_q[layer],
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    rms_norms,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &q_proj_beta,
                                                    q_proj,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // input = (num_tokens, EMBEDDING_LENGTH), weights = (KV_DIM, EMBEDDING_LENGTH)
        // after trick: (KV_DIM, EMBEDDING_LENGTH) * (EMBEDDING_LENGTH, num_tokens) -> (KV_DIM, num_tokens), which really is (num_tok, KV_DIM)
        // lda: EMBEDDING_LENGTH, ldb: EMBEDDING_LENGTH, ldc: KV_DIM
        cublasStatus_t k_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    KV_DIM,
                                                    prompt_len,
                                                    EMBEDDING_LENGTH,
                                                    &k_proj_alpha,
                                                    weights.w_k[layer],
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    rms_norms,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &k_proj_beta,
                                                    k_proj_temp_buf,
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // same as K projection
        cublasStatus_t v_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    KV_DIM,
                                                    prompt_len,
                                                    EMBEDDING_LENGTH,
                                                    &v_proj_alpha,
                                                    weights.w_v[layer],
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    rms_norms,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &v_proj_beta,
                                                    v_proj_temp_buf,
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // RoPE now

        rope(q_proj, prompt_len, EMBEDDING_LENGTH);
        rope(k_proj_temp_buf, prompt_len, KV_DIM);

        // PagedAttention - scatter K and V into blocks
        // slot - index within batch
        // layer - index of layer
        // ceil(prompt_len/BLOCK_SIZE) = number of blocks needed to allocate in block table
        for (int token_idx = 0; token_idx < prompt_len; token_idx += BLOCK_SIZE)
        {
            int num_tokens_to_copy = prompt_len - token_idx;
            if (num_tokens_to_copy > BLOCK_SIZE)
            {
                num_tokens_to_copy = BLOCK_SIZE;
            }
            // read index of physical block from logical block_table
            // if -1, then need to allocate the new block
            // pop from free_blocks
            // write its value to block_table on the same position we read from
            // compute address of this block table in kv_cache
            // write tokens to it
            int block_idx = token_idx / BLOCK_SIZE;
            int block = block_table[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + block_idx];
            if (block == -1)
            {
                int physical_block_idx = free_blocks.back();
                free_blocks.pop_back();
                block = physical_block_idx;
                block_table[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + block_idx] = block;
            }
            else
            {
                assert(false && "block must be -1 during prefill - what happened?");
                // probably in prefill this doesn't make a lot of sense? but will matter in decode
            }

            // store K
            __nv_bfloat16 *k_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES);
            __nv_bfloat16 *k_proj_ptr = k_proj_temp_buf + token_idx * KV_DIM;
            cudaMemcpy(k_cache_ptr, k_proj_ptr, num_tokens_to_copy * KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

            // store V
            __nv_bfloat16 *v_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES + V_OFFSET);
            __nv_bfloat16 *v_proj_ptr = v_proj_temp_buf + token_idx * KV_DIM;
            cudaMemcpy(v_cache_ptr, v_proj_ptr, num_tokens_to_copy * KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
        }

        // attention scores
        // per head, 64 elements each
        // so total 32 heads
        // Q (num_tok, 2048)
        // K (num_tok, 512)
        // GQA grouping reuses 1 K head per 4 consecutive Q heads
        // Q_head (num_tok, 64)
        // K_head (num_tok, 64)
        // attn_score_head = Q_head * K_head^T / sqrt(64)
        // so: head output dims (num_tok, num_tok)
        // total output (32, num_tok, num_tok)
        for (int i = 0; i < NUM_Q_HEADS; ++i)
        {
            int k_head_idx = i / GQA_Q_TO_K_RATIO;
            __nv_bfloat16 *q_head = q_proj + i * HEAD_DIM;
            __nv_bfloat16 *k_head = k_proj_temp_buf + k_head_idx * HEAD_DIM;
            __nv_bfloat16 *attn_score_head = prefill_attn_scores + prompt_len * prompt_len * i;

            cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                            CUBLAS_OP_T,
                                                            CUBLAS_OP_N,
                                                            prompt_len,
                                                            prompt_len,
                                                            HEAD_DIM,
                                                            &attn_alpha,
                                                            k_head,
                                                            CUDA_R_16BF,
                                                            KV_DIM,
                                                            q_head,
                                                            CUDA_R_16BF,
                                                            EMBEDDING_LENGTH,
                                                            &attn_beta,
                                                            attn_score_head,
                                                            CUDA_R_16BF,
                                                            prompt_len,
                                                            CUBLAS_COMPUTE_32F,
                                                            CUBLAS_GEMM_DEFAULT);
        }

        causalMask(prefill_attn_scores, prompt_len);

        softmax(prefill_attn_scores, prompt_len);

        // attn scores * V
        // (32, num_tok, num_tok) * (num_tok, 512)
        // GQA - 4 Q heads share 1 V head
        // attn_scores dim (32, num_tok, num_tok)
        // attn_scores head dim (num_tok, num_tok)
        // V dim (num_tok, 512)
        // NUM_V_HEADS is 8 -> 512 / 8 = 64
        // V_head dim (num_tok, 64)
        // output head dim: scores head * V head -> (num_tok, num_tok) * (num_tok, 64) = (num_tok, 64)
        // in total 32 output heads: so (num_tok, 64 * 32) = (num_tok, 2048)
        attn_scores_v = buf_2048_1;
        for (int i = 0; i < NUM_Q_HEADS; ++i)
        {
            int v_head_idx = i / GQA_ATTN_SCORES_TO_V_RATIO;
            // i * prompt_under_prefill.size() * prompt_under_prefill.size(),  because attn scores is (32, num_tok, num_tok)
            __nv_bfloat16 *attn_scores_head = prefill_attn_scores + i * prompt_len * prompt_len;
            __nv_bfloat16 *v_head = v_proj_temp_buf + v_head_idx * HEAD_DIM;
            __nv_bfloat16 *output_attn_scores_head = attn_scores_v + i * HEAD_DIM;

            cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                            CUBLAS_OP_N,
                                                            CUBLAS_OP_N,
                                                            HEAD_DIM,
                                                            prompt_len,
                                                            prompt_len,
                                                            &attn_scores_v_alpha,
                                                            v_head,
                                                            CUDA_R_16BF,
                                                            KV_DIM,
                                                            attn_scores_head,
                                                            CUDA_R_16BF,
                                                            prompt_len,
                                                            &attn_scores_v_beta,
                                                            output_attn_scores_head,
                                                            CUDA_R_16BF,
                                                            EMBEDDING_LENGTH,
                                                            CUBLAS_COMPUTE_32F,
                                                            CUBLAS_GEMM_DEFAULT);
        }

        // output projection, it will be an input for MLP blocks
        // attn_scores_v * w_o^T
        // (num_tok, 2048) * (2048, 2048) -> (num_tok, 2048)
        // same as Q projection, so copy paste
        o_proj = buf_2048_2;
        cublasStatus_t o_proj_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    EMBEDDING_LENGTH,
                                                    prompt_len,
                                                    EMBEDDING_LENGTH,
                                                    &o_proj_alpha,
                                                    weights.w_o[layer],
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    attn_scores_v,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &o_proj_beta,
                                                    o_proj,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);

        // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
        residualAdd(hidden_state, o_proj, prompt_len);
        // post attention RMS Norm
        rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer], prompt_len);

        // SwiGLU time - just MLP + SiLU
        // gate = hidden_state (rms-normed) * mlp_gate_proj ^ T
        // HIDDEN_DIM = 8192
        // (num_tok, 2048) * (2048, 8192) -> (num_tok, 8192)
        // my data is row major so transpose trick
        // gate ^T = (mlp_gate_proj ^ T)^T * hidden_state^T
        // gate ^T = mlp_gate_proj * hidden_state^T
        // (num_tok, 8192)^T = (8192, 2048) * (2048, num_tok)
        // but data is perceived as column major so I need to transpose mlp_gate_proj
        // to make it work
        // m 8192 n num_tok k 2048 lda 2048 ldb 2048 ldc 8192
        cublasStatus_t gate_status = cublasGemmEx(cublas_handle,
                                                  CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  HIDDEN_DIM,
                                                  prompt_len,
                                                  EMBEDDING_LENGTH,
                                                  &gate_alpha,
                                                  weights.mlp_gate_proj[layer],
                                                  CUDA_R_16BF,
                                                  EMBEDDING_LENGTH,
                                                  rms_norms,
                                                  CUDA_R_16BF,
                                                  EMBEDDING_LENGTH,
                                                  &gate_beta,
                                                  gate,
                                                  CUDA_R_16BF,
                                                  HIDDEN_DIM,
                                                  CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT);

        // up, the same dims as gate
        cublasStatus_t up_status = cublasGemmEx(cublas_handle,
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                HIDDEN_DIM,
                                                prompt_len,
                                                EMBEDDING_LENGTH,
                                                &up_alpha,
                                                weights.mlp_up_proj[layer],
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                rms_norms,
                                                CUDA_R_16BF,
                                                EMBEDDING_LENGTH,
                                                &up_beta,
                                                up,
                                                CUDA_R_16BF,
                                                HIDDEN_DIM,
                                                CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT);

        // SiLU
        // after_silu = SiLU(gate) * up (element-wise multication)
        // after_silu = gate * (1 / (1 + e^(-gate))) * up
        // gate is dim (num_tok, 8192), up too
        silu(gate, up, prompt_len); // gate = after_silu now

        // down projection
        // output = post-silu * down_proj^T
        // dims: (num_tok, 8192) * (2048, 8192) ^ T = (num_tok, 8192) * (8192, 2048) = (num_tok, 2048)
        // output^T = (down_proj^T)^T * post-silu^T
        // output^T = down_proj * post-silu^T
        // cublas sees them already as transposed so only down_proj I need to transpose
        // dims = (2048, 8192) * (8192, num_tok) = (2048, num_tok)
        // m: 2048 n: num_tok, k: 8192
        // lda: 8192, ldb: 8192, ldc: 2048
        down = buf_2048_2;
        cublasStatus_t down_status = cublasGemmEx(cublas_handle,
                                                  CUBLAS_OP_T,
                                                  CUBLAS_OP_N,
                                                  EMBEDDING_LENGTH,
                                                  prompt_len,
                                                  HIDDEN_DIM,
                                                  &down_alpha,
                                                  weights.mlp_down_proj[layer],
                                                  CUDA_R_16BF,
                                                  HIDDEN_DIM,
                                                  gate,
                                                  CUDA_R_16BF,
                                                  HIDDEN_DIM,
                                                  &down_beta,
                                                  down,
                                                  CUDA_R_16BF,
                                                  EMBEDDING_LENGTH,
                                                  CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT);

        // (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
        residualAdd(hidden_state, down, prompt_len);
    }
    rmsNorm(hidden_state, rms_norms, weights.norm, prompt_len);

    // logits = rms_norms * weights.embed_tokens^T
    // dim rms_norms: (num_tok, 2048), dim embed_tokens: (128256, 2048)
    // logits dim = (num_tok, 2048) * (2048, 128256) = (num_tok, 128256) => m = num_tok, n = 128256, k = 2048
    // I leave this comment above because it shows a bug in my thinking
    // because I use the cublas trick, logits are transposed so m and n should be swapped
    // so m 128256, n num_tok
    // data is row major so we treat it as transposed and use the trick
    // logits^T = ((weights.embed_tokens^T)^T * rms_norms^T
    // logits^T = weights.embed_tokens * rms_norms^T
    // so we need to transpose embed_tokens, because rms_norms already
    // appears to cublas as transposed
    // lda = 2048, ldb = 2048, ldc = 128256

    cublasStatus_t embed_status = cublasGemmEx(cublas_handle,
                                               CUBLAS_OP_T,
                                               CUBLAS_OP_N,
                                               VOCAB_SIZE,
                                               prompt_len,
                                               EMBEDDING_LENGTH,
                                               &embed_alpha,
                                               weights.embed_tokens,
                                               CUDA_R_16BF,
                                               EMBEDDING_LENGTH,
                                               rms_norms,
                                               CUDA_R_16BF,
                                               EMBEDDING_LENGTH,
                                               &embed_beta,
                                               embed_proj,
                                               CUDA_R_16BF,
                                               VOCAB_SIZE,
                                               CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT);

    argmaxRows(embed_proj + (prompt_len - 1) * VOCAB_SIZE, gpu_input_tokens, 1, VOCAB_SIZE);
    int max_token_idx = 0;
    cudaMemcpy(&max_token_idx, gpu_input_tokens, sizeof(int), cudaMemcpyDeviceToHost);
#ifdef DEBUG
    std::vector<nv_bfloat16> last_logits(VOCAB_SIZE);
    cudaMemcpy(last_logits.data(), embed_proj + (prompt_len - 1) * VOCAB_SIZE, sizeof(__nv_bfloat16) * VOCAB_SIZE, cudaMemcpyDeviceToHost);
    std::cout << "Output token: " << (float)last_logits[max_token_idx] << ", token index: " << std::to_string(max_token_idx) << std::endl;
#endif

    generated_tokens[slot].push_back(max_token_idx);
    last_generated_tokens[slot] = max_token_idx;
    set_prompt_len(slot, prompt_len, current_prompt_len, prompt_lengths);
    if (input_tokens_size > g_max_prompt)
    {
        input_tokens_size = g_max_prompt;
    }
    assert(input_tokens_size <= g_max_prompt);

    // synchronize state of block_table with block_table_gpu
    syncBlockTableSlot(slot, block_table, block_table_gpu);
}


void printUsage(const char *argv0)
{
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "  --model PATH, -m PATH   safetensors path (default: model.safetensors)\n"
              << "  --n N, --max-tokens N   max new tokens (default: 20)\n"
              << "  --prompt-ids 1,2,3      prompt token ids (comma-separated; repeatable)\n"
              << "  --max-prompt N          max prompt tokens (default: 512)\n"
              << "  --kv-gb F               KV cache size in GiB (default: 2)\n"
              << "  --batch N, --batch-size N  concurrent sequences (default: 2)\n"
              << "  --help, -h              show this help\n";
}

std::vector<int> parsePromptIds(const std::string &s)
{
    std::vector<int> ids;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (item.empty())
        {
            continue;
        }
        ids.push_back(std::stoi(item));
    }
    return ids;
}

int main(int argc, char *argv[]) {
    std::string model_path = "model.safetensors";
    int max_new_tokens = 20;
    double kv_gb = 2.0;
    int batch_cli = 2;
    bool has_prompt_ids = false;
    std::vector<std::vector<int>> cli_prompt_list;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto take_value = [&](const std::string &flag) -> const char * {
            if (i + 1 >= argc)
            {
                std::cerr << flag << " requires a value\n";
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--model" || arg == "-m")
        {
            const char *v = take_value(arg);
            if (v)
            {
                model_path = v;
            }
        }
        else if (arg == "--n" || arg == "--max-tokens")
        {
            const char *v = take_value(arg);
            if (v)
            {
                try
                {
                    max_new_tokens = std::stoi(v);
                }
                catch (...)
                {
                    std::cerr << "Invalid " << arg << " value: " << v << "\n";
                }
            }
        }
        else if (arg == "--prompt-ids")
        {
            const char *v = take_value(arg);
            if (v)
            {
                try
                {
                    cli_prompt_list.push_back(parsePromptIds(v));
                    has_prompt_ids = true;
                }
                catch (...)
                {
                    std::cerr << "Invalid --prompt-ids value: " << v << "\n";
                }
            }
        }
        else if (arg == "--max-prompt")
        {
            const char *v = take_value(arg);
            if (v)
            {
                try
                {
                    g_max_prompt = std::max(1, std::stoi(v));
                }
                catch (...)
                {
                    std::cerr << "Invalid --max-prompt value: " << v << "\n";
                }
            }
        }
        else if (arg == "--kv-gb")
        {
            const char *v = take_value(arg);
            if (v)
            {
                try
                {
                    kv_gb = std::stod(v);
                }
                catch (...)
                {
                    std::cerr << "Invalid --kv-gb value: " << v << "\n";
                }
            }
        }
        else if (arg == "--batch" || arg == "--batch-size")
        {
            const char *v = take_value(arg);
            if (v)
            {
                try
                {
                    batch_cli = std::stoi(v);
                }
                catch (...)
                {
                    std::cerr << "Invalid " << arg << " value: " << v << "\n";
                }
            }
        }
        else
        {
            std::cerr << "Unknown flag: " << arg << "\n";
        }
    }

    const int batch = std::max(1, batch_cli);
    const int buf = std::max(g_max_prompt, batch);

    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS init failed, status: " << status << "\n";
        return 1;
    }

    Weights weights{};
    if (loadWeights(weights, model_path) != 0)
    {
        return 1;
    }

    // Allocator for pagedattn
    // To prevent the memory fragmentation, we allocate a single large buffer for K and V cache and then manage it ourselves.
    size_t kv_bytes = static_cast<size_t>(kv_gb * 1024.0 * 1024.0 * 1024.0);
    int num_blocks = static_cast<int>(kv_bytes / BLOCK_BYTES);
    __nv_bfloat16 *kv_cache;
    cudaMalloc(&kv_cache, kv_bytes);
    std::vector<int> free_blocks(num_blocks);
    std::iota(free_blocks.begin(), free_blocks.end(), 0);
    std::vector<int> block_table(batch * N_LAYERS * MAX_BLOCKS_PER_SEQ, -1);
    int *block_table_gpu;
    cudaMalloc(&block_table_gpu, batch * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int));

    std::queue<std::vector<int>> queue;
    if (has_prompt_ids)
    {
        for (const auto &ids : cli_prompt_list)
        {
            if (!ids.empty())
            {
                queue.push(ids);
            }
        }
    }
    else
    {
        // PROMPT 0 (What is 2+2?) - length 17
        queue.push({128000, 128006, 882, 128007, 271, 3923, 374, 220, 17, 10, 17, 30, 128009, 128006, 78191, 128007, 271});

        // PROMPT 1 (Name a color.) - length 14
        queue.push({128000, 128006, 882, 128007, 271, 678, 264, 1933, 13, 128009, 128006, 78191, 128007, 271});

        // PROMPT 2 (Say hello.) - length 13
        queue.push({128000, 128006, 882, 128007, 271, 46864, 24748, 13, 128009, 128006, 78191, 128007, 271});

        // PROMPT 3 (Capital of France?) - length 14
        queue.push({128000, 128006, 882, 128007, 271, 64693, 315, 9822, 30, 128009, 128006, 78191, 128007, 271});
    }

    // BATCH
    std::vector<bool> is_slot_free(batch, true); // set to false when slot taken, set to true when free

    std::vector<std::vector<int>> generated_tokens(batch);
    std::vector<int> last_generated_tokens(batch);
    std::vector<int> current_prompt_len(batch, 0);
    std::vector<int> prompt_lengths(batch, 0);

    // needed to provide contiguous data for decode
    std::vector<int> active_slots;
    std::vector<int> active_tokens;

    int *gpu_active_slots;
    cudaMalloc(&gpu_active_slots, batch * sizeof(int));
    int *gpu_seq_lens;
    cudaMalloc(&gpu_seq_lens, batch * sizeof(int));

    // input_tokens_size / prompt_lengths refresh when a slot takes a new prompt
    int input_tokens_size = 0;

    int *gpu_input_tokens;
    cudaMalloc(&gpu_input_tokens, g_max_prompt * sizeof(int));
    __nv_bfloat16 *input_embeddings;
    cudaMalloc(&input_embeddings, g_max_prompt * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);

    __nv_bfloat16 *hidden_state;
    cudaMalloc(&hidden_state, buf * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);

    __nv_bfloat16 *rms_norms;
    cudaMalloc(&rms_norms, buf * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);

    __nv_bfloat16 *buf_2048_1; // shared between q_proj and attn_scores_v
    cudaMalloc(&buf_2048_1, buf * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    __nv_bfloat16 *q_proj;
    float q_proj_alpha = 1.0f;
    float q_proj_beta = 0.0f;

    // K and V cache
    __nv_bfloat16 *k_proj_temp_buf;
    cudaMalloc(&k_proj_temp_buf, g_max_prompt * KV_DIM * sizeof(__nv_bfloat16));

    __nv_bfloat16 *v_proj_temp_buf;
    cudaMalloc(&v_proj_temp_buf, g_max_prompt * KV_DIM * sizeof(__nv_bfloat16));

    float k_proj_alpha = 1.0f;
    float k_proj_beta = 0.0f;

    float v_proj_alpha = 1.0f;
    float v_proj_beta = 0.0f;

    __nv_bfloat16 *prefill_attn_scores;
    cudaMalloc(&prefill_attn_scores, g_max_prompt * g_max_prompt * sizeof(__nv_bfloat16) * NUM_Q_HEADS);
    float attn_alpha = 1.0f / 8.0f;
    float attn_beta = 0.0f;

    float attn_scores_v_alpha = 1.0f;
    float attn_scores_v_beta = 0.0f;

    __nv_bfloat16 *buf_2048_2; // shared between o_proj and down
    cudaMalloc(&buf_2048_2, buf * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
    __nv_bfloat16 *o_proj;
    float o_proj_alpha = 1.0f;
    float o_proj_beta = 0.0f;

    __nv_bfloat16 *gate;
    cudaMalloc(&gate, buf * sizeof(__nv_bfloat16) * HIDDEN_DIM);
    float gate_alpha = 1.0f;
    float gate_beta = 0.0f;

    __nv_bfloat16 *up;
    cudaMalloc(&up, buf * sizeof(__nv_bfloat16) * HIDDEN_DIM);
    float up_alpha = 1.0f;
    float up_beta = 0.0f;

    __nv_bfloat16 *down;
    float down_alpha = 1.0f;
    float down_beta = 0.0f;

    __nv_bfloat16 *embed_proj;
    cudaMalloc(&embed_proj, sizeof(__nv_bfloat16) * buf * VOCAB_SIZE);
    float embed_alpha = 1.0f;
    float embed_beta = 0.0f;

    // decode-only allocation
    int *gpu_last_tokens;
    cudaMalloc(&gpu_last_tokens, batch * sizeof(int));

    // reused temporary buffers for K and V cache computation during decode
    __nv_bfloat16 *k_proj_batched_buffer;
    cudaMalloc(&k_proj_batched_buffer, batch * sizeof(__nv_bfloat16) * KV_DIM);

    __nv_bfloat16 *v_proj_batched_buffer;
    cudaMalloc(&v_proj_batched_buffer, batch * sizeof(__nv_bfloat16) * KV_DIM);

    PrefillCtx ctx{
        queue,
        is_slot_free,
        gpu_input_tokens,
        input_embeddings,
        weights,
        hidden_state,
        rms_norms,
        buf_2048_1,
        cublas_handle,
        q_proj_alpha,
        q_proj_beta,
        k_proj_alpha,
        k_proj_beta,
        v_proj_alpha,
        v_proj_beta,
        prefill_attn_scores,
        attn_alpha,
        attn_beta,
        attn_scores_v_alpha,
        attn_scores_v_beta,
        buf_2048_2,
        o_proj_alpha,
        o_proj_beta,
        gate_alpha,
        gate_beta,
        gate,
        up_alpha,
        up_beta,
        up,
        down_alpha,
        down_beta,
        embed_alpha,
        embed_beta,
        embed_proj,
        generated_tokens,
        last_generated_tokens,
        current_prompt_len,
        prompt_lengths,
        input_tokens_size,
        k_proj_temp_buf,
        v_proj_temp_buf,
        block_table,
        block_table_gpu,
        free_blocks,
        kv_cache,
    };

    for (int slot = 0; slot < is_slot_free.size() && !queue.empty(); ++slot)
    {
        if (!is_slot_free[slot])
        {
            continue; // slot taken, skip
        }
        prefill(ctx, slot);
    }

    // INFERENCE STARTS HERE! =]
    // I have the same amount of embeddings as input tokens
    // it's just every embedding is EMBEDDING_LENGTH length bf16 vector
    // retrieved from model weights based on token's value

    // PREFILL

    // DECODE
    // since now I operate always on index 0 for all values and for current_position_token for new K and V

    while (true) // exit condition irrelevant for now, since it's an inference server that's supposed to run foreveeer!!!
    {
        active_slots.clear();
        active_tokens.clear();
        for (int slot = 0; slot < batch; ++slot)
        {
            if (is_slot_free[slot])
            {
                if (queue.empty())
                {
                    continue;
                }
                generated_tokens[slot].clear();
                prefill(ctx, slot);
                if (is_slot_free[slot])
                {
                    continue;
                }
            }
            if ((int)generated_tokens[slot].size() >= max_new_tokens)
            {
                releaseSlot(slot, is_slot_free, block_table, free_blocks, block_table_gpu);
                if (queue.empty())
                {
                    continue;
                }
                generated_tokens[slot].clear();
                prefill(ctx, slot);
                if (is_slot_free[slot] || (int)generated_tokens[slot].size() >= max_new_tokens)
                {
                    continue;
                }
            }
            active_slots.push_back(slot);
            active_tokens.push_back(last_generated_tokens[slot]);
        }
        int num_active_slots = active_slots.size();
        input_tokens_size = max_live_prompt_len(current_prompt_len, is_slot_free);
        if (num_active_slots == 0)
        {
            if (queue.empty())
            {
                break;
            }
            continue;
        }

        // copy useful data to gpu
        cudaMemcpy(gpu_last_tokens, active_tokens.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_active_slots, active_slots.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
        std::vector<int> seq_lens(num_active_slots);
        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = active_slots[slot];
            seq_lens[slot] = current_prompt_len[active_slot] + 1;
        }
        cudaMemcpy(gpu_seq_lens, seq_lens.data(), seq_lens.size() * sizeof(int), cudaMemcpyHostToDevice);

        embeddingGatherDecode(gpu_last_tokens, num_active_slots, hidden_state, weights.embed_tokens);
        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer], num_active_slots);
            q_proj = buf_2048_1;
            // q proj (num_prompts, 2048)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_active_slots, // n
                         EMBEDDING_LENGTH, // k
                         &q_proj_alpha,
                         weights.w_q[layer], // A
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // lda
                         rms_norms,        // B
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldb
                         &q_proj_beta,
                         q_proj, // C
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldc
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
            // k proj (1, 512), writing output to next position in current layer's K cache
            // K proj = rms_norms (num_prompt, 2048) * W_k (512, 2048)
            // W_k is actually stored as 512, 2048 (out features, in features)
            // so that's why we need to transpose it
            // all the data is stored in row major and cublas reads it as column major
            // so all the data appears as transposed
            // so data actually apppears as (2048, num_prompt) * (2048, 512)
            // the output of matmul will also be produced as transposed, so we can say that
            // in our mental model we talk about K_proj^T
            // and to get K_proj^T we can do transposition trick and write the cublas call as
            // W_k^T * rms_nroms
            // so we end up with: K_proj^T = W_k^T (512, 2048) * rms_norms (2048, num_prompt)
            // result dim is K_proj^T = (512, num_prompt)
            // but it's transposed, so in fact we get correct output dimension (num_prompt, 512)
            // Decode K/V land in the batched buffer, then scatter into paged KV because sequences have different lengths.
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         KV_DIM,           // m = 512
                         num_active_slots, // n = num prompts
                         EMBEDDING_LENGTH, // k = 2048
                         &k_proj_alpha,
                         weights.w_k[layer], // A
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // lda 2048, because W_k is in memory as 512, 2048
                         // so the gap between subsequent elements is 2048
                         rms_norms, // B
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH, // ldb, same reason for rms_norms
                         &k_proj_beta,
                         k_proj_batched_buffer,
                         CUDA_R_16BF,
                         KV_DIM, // ldc = 512
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // same
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         KV_DIM,
                         num_active_slots,
                         EMBEDDING_LENGTH,
                         &v_proj_alpha,
                         weights.w_v[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &v_proj_beta,
                         v_proj_batched_buffer,
                         CUDA_R_16BF,
                         KV_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            for (int slot = 0; slot < num_active_slots; ++slot)
            {
                int active_slot = active_slots[slot];
                ropeDecode(&q_proj[slot * EMBEDDING_LENGTH], current_prompt_len[active_slot], EMBEDDING_LENGTH);
                ropeDecode(k_proj_batched_buffer + slot * KV_DIM, current_prompt_len[active_slot], KV_DIM);
            }

            // PagedAttn scatter k and v from a temp buffer, like in the prefill
            for (int slot = 0; slot < num_active_slots; ++slot)
            {
                int active_slot = active_slots[slot];
                int seq_len = current_prompt_len[active_slot]; // + generated tokens?
                int logical_block_idx = seq_len / BLOCK_SIZE;
                int token_in_block_idx = seq_len % BLOCK_SIZE;
                int block = block_table[active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
                if (token_in_block_idx == 0)
                {
                    int physical_block_idx = free_blocks.back();
                    free_blocks.pop_back();
                    block = physical_block_idx;
                    block_table[active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx] = block;
                }
                __nv_bfloat16 *k_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES + token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16));
                __nv_bfloat16 *k_proj_ptr = k_proj_batched_buffer + slot * KV_DIM;
                cudaMemcpy(k_cache_ptr, k_proj_ptr, KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

                __nv_bfloat16 *v_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES + V_OFFSET + token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16));
                __nv_bfloat16 *v_proj_ptr = v_proj_batched_buffer + slot * KV_DIM;
                cudaMemcpy(v_cache_ptr, v_proj_ptr, KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
            }

            // synchronize block table on cpu with block table on gpu (for attention)
            for (int slot = 0; slot < num_active_slots; ++slot)
            {
                syncBlockTableSlot(active_slots[slot], block_table, block_table_gpu);
            }

            pagedAttention(layer, num_active_slots, q_proj, kv_cache, block_table_gpu, gpu_seq_lens, gpu_active_slots, buf_2048_1);

            o_proj = buf_2048_2;
            // (1, 2048) * (2048, 2048) -> (1, 2048)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_active_slots, // n
                         EMBEDDING_LENGTH, // k
                         &o_proj_alpha,
                         weights.w_o[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         buf_2048_1,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &o_proj_beta,
                         o_proj,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            residualAdd(hidden_state, o_proj, num_active_slots);

            rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer], num_active_slots);

            // MLP
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         HIDDEN_DIM,       // m
                         num_active_slots, // n
                         EMBEDDING_LENGTH, // k
                         &gate_alpha,
                         weights.mlp_gate_proj[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &gate_beta,
                         gate,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // (1, 2048) * (2048, 8192) -> (1, 8192)
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         HIDDEN_DIM,       // m
                         num_active_slots, // n
                         EMBEDDING_LENGTH, // k
                         &up_alpha,
                         weights.mlp_up_proj[layer],
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         rms_norms,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         &up_beta,
                         up,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            silu(gate, up, num_active_slots);

            down = buf_2048_2;
            cublasGemmEx(cublas_handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         EMBEDDING_LENGTH, // m
                         num_active_slots, // n
                         HIDDEN_DIM,       // k
                         &down_alpha,
                         weights.mlp_down_proj[layer],
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         gate,
                         CUDA_R_16BF,
                         HIDDEN_DIM,
                         &down_beta,
                         down,
                         CUDA_R_16BF,
                         EMBEDDING_LENGTH,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            residualAdd(hidden_state, down, num_active_slots);
        }

        rmsNorm(hidden_state, rms_norms, weights.norm, num_active_slots);

        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     VOCAB_SIZE,       // m
                     num_active_slots, // n
                     EMBEDDING_LENGTH, // k
                     &embed_alpha,
                     weights.embed_tokens,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     rms_norms,
                     CUDA_R_16BF,
                     EMBEDDING_LENGTH,
                     &embed_beta,
                     embed_proj,
                     CUDA_R_16BF,
                     VOCAB_SIZE,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);

        argmaxRows(embed_proj, gpu_last_tokens, num_active_slots, VOCAB_SIZE);
        std::vector<int> sampled_ids(num_active_slots);
        cudaMemcpy(sampled_ids.data(), gpu_last_tokens, num_active_slots * sizeof(int), cudaMemcpyDeviceToHost);
#ifdef DEBUG
        std::vector<__nv_bfloat16> embed_proj_cpu(num_active_slots * VOCAB_SIZE);
        cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * num_active_slots * VOCAB_SIZE, cudaMemcpyDeviceToHost);
#endif

        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = active_slots[slot];
            int max_token_idx = sampled_ids[slot];
#ifdef DEBUG
            float max_token = (float)embed_proj_cpu[slot * VOCAB_SIZE + max_token_idx];
            std::cout << "Output token: " << max_token << ", token index: " << std::to_string(max_token_idx) << std::endl;
#endif
            if (max_token_idx == END_OF_TEXT_TOKEN_ID || max_token_idx == EOT_ID_TOKEN_ID || current_prompt_len[active_slot] == MAX_SEQ_LEN - 1)
            {
                releaseSlot(active_slot, is_slot_free, block_table, free_blocks, block_table_gpu);
            }
            else
            {
                last_generated_tokens[active_slot] = max_token_idx;
                generated_tokens[active_slot].push_back(max_token_idx);
                set_prompt_len(active_slot, current_prompt_len[active_slot] + 1, current_prompt_len, prompt_lengths);
                if ((int)generated_tokens[active_slot].size() >= max_new_tokens)
                {
                    releaseSlot(active_slot, is_slot_free, block_table, free_blocks, block_table_gpu);
                    if (!queue.empty())
                    {
                        generated_tokens[active_slot].clear();
                        prefill(ctx, active_slot);
                    }
                }
            }
        }
    }
    std::cout << "\nOk bye!\n";
    cublasDestroy(cublas_handle);
    cudaDeviceSynchronize();
    return 0;
}
