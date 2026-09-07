#pragma once

// Llama 3.2 1B defaults. Override at compile time:
//   -DLLAMA_N_LAYERS=... -DLLAMA_MAX_SEQ_LEN=... -DLLAMA_BLOCK_SIZE=...

#ifndef LLAMA_N_LAYERS
#define LLAMA_N_LAYERS 16
#endif

#ifndef LLAMA_MAX_SEQ_LEN
#define LLAMA_MAX_SEQ_LEN 2048
#endif

#ifndef LLAMA_BLOCK_SIZE
#define LLAMA_BLOCK_SIZE 16
#endif

#ifndef LLAMA_EMBEDDING_LENGTH
#define LLAMA_EMBEDDING_LENGTH 2048
#endif

#ifndef LLAMA_HIDDEN_DIM
#define LLAMA_HIDDEN_DIM 8192
#endif

#ifndef LLAMA_VOCAB_SIZE
#define LLAMA_VOCAB_SIZE 128256
#endif

#ifndef LLAMA_NUM_Q_HEADS
#define LLAMA_NUM_Q_HEADS 32
#endif

#ifndef LLAMA_NUM_K_HEADS
#define LLAMA_NUM_K_HEADS 8
#endif

#ifndef LLAMA_NUM_V_HEADS
#define LLAMA_NUM_V_HEADS 8
#endif

#ifndef LLAMA_HEAD_DIM
#define LLAMA_HEAD_DIM 64
#endif

#ifndef LLAMA_DEFAULT_MAX_PROMPT
#define LLAMA_DEFAULT_MAX_PROMPT 512
#endif

constexpr int N_LAYERS = LLAMA_N_LAYERS;
constexpr int EMBEDDING_LENGTH = LLAMA_EMBEDDING_LENGTH;
constexpr int HIDDEN_DIM = LLAMA_HIDDEN_DIM;
constexpr int VOCAB_SIZE = LLAMA_VOCAB_SIZE;
constexpr int KV_DIM = LLAMA_NUM_K_HEADS * LLAMA_HEAD_DIM;
constexpr int HEAD_DIM = LLAMA_HEAD_DIM;
constexpr float SQRT_HEAD_DIM = 8;
constexpr int NUM_Q_HEADS = LLAMA_NUM_Q_HEADS;
constexpr int NUM_K_HEADS = LLAMA_NUM_K_HEADS;
constexpr int NUM_V_HEADS = LLAMA_NUM_V_HEADS;
constexpr int GQA_Q_TO_K_RATIO = NUM_Q_HEADS / NUM_K_HEADS;
constexpr int GQA_ATTN_SCORES_TO_V_RATIO = NUM_Q_HEADS / NUM_V_HEADS;
constexpr int MAX_SEQ_LEN = LLAMA_MAX_SEQ_LEN;
constexpr int BLOCK_SIZE = LLAMA_BLOCK_SIZE;
constexpr int BF16_BYTES = 2;
constexpr int V_OFFSET = BLOCK_SIZE * KV_DIM * BF16_BYTES;
constexpr int BLOCK_BYTES = V_OFFSET * 2; // K and V
constexpr int MAX_BLOCKS_PER_SEQ = MAX_SEQ_LEN / BLOCK_SIZE;
constexpr int ROPE_FREQS = HEAD_DIM / 2;
constexpr int DEFAULT_MAX_PROMPT = LLAMA_DEFAULT_MAX_PROMPT;

inline int llama_clamp_n(int n, int lo) { return n < lo ? lo : n; }

inline int llama_block_table_elems(int nseq) {
    return llama_clamp_n(nseq, 0) * N_LAYERS * MAX_BLOCKS_PER_SEQ;
}

inline int llama_slot_table_elems() { return N_LAYERS * MAX_BLOCKS_PER_SEQ; }

inline int llama_block_index(int slot, int layer, int logical) {
    return slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical;
}

static_assert(HEAD_DIM % 2 == 0, "HEAD_DIM must be even for RoPE pairs");
static_assert(BLOCK_SIZE > 0, "BLOCK_SIZE must be positive");
static_assert(MAX_SEQ_LEN % BLOCK_SIZE == 0, "MAX_SEQ_LEN must be a multiple of BLOCK_SIZE");
