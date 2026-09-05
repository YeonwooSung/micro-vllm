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

constexpr int N_LAYERS = LLAMA_N_LAYERS;
constexpr int EMBEDDING_LENGTH = 2048;
constexpr int KV_DIM = 512;
constexpr int HEAD_DIM = 64;
constexpr float SQRT_HEAD_DIM = 8;
constexpr int NUM_Q_HEADS = 32;
constexpr int GQA_Q_TO_K_RATIO = 4;
constexpr int MAX_SEQ_LEN = LLAMA_MAX_SEQ_LEN;
constexpr int BLOCK_SIZE = LLAMA_BLOCK_SIZE;
constexpr int BF16_BYTES = 2;
constexpr int V_OFFSET = BLOCK_SIZE * KV_DIM * BF16_BYTES;
constexpr int BLOCK_BYTES = V_OFFSET * 2; // K and V
constexpr int MAX_BLOCKS_PER_SEQ = MAX_SEQ_LEN / BLOCK_SIZE;
constexpr int ROPE_FREQS = HEAD_DIM / 2;

static_assert(HEAD_DIM % 2 == 0, "HEAD_DIM must be even for RoPE pairs");
static_assert(BLOCK_SIZE > 0, "BLOCK_SIZE must be positive");
static_assert(MAX_SEQ_LEN % BLOCK_SIZE == 0, "MAX_SEQ_LEN must be a multiple of BLOCK_SIZE");
