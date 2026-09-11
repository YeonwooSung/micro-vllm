# micro-vllm

C++17 host engine for consumer desktops and Macs. NVMe, RAM, and optional
Metal/CUDA are one hierarchy: dense weights stay resident, routed experts and
H3 DiT blocks stream from disk. The original Llama 3.2 1B CUDA loop is still
`micro-vllm-cuda`.

Families: **Kimi K3**, **GLM-5.3** (HF / GLM-5.2-style names also probe),
**DeepSeek V4 Flash**, **MiniMax-H3**, **Llama** (CPU stand-in).

Internals: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Build

```bash
# Host engine + tests (default). Metal expert/DiT/KDA on Apple.
cmake -B build -DMVLLM_HOST_ONLY=ON
cmake --build build -j
./build/mvllm_tests
```

Optional CUDA expert GEMM in the host engine (needs `nvcc`):

```bash
cmake -B build -DMVLLM_HOST_ONLY=ON -DMVLLM_GPU_CUDA=ON
```

Llama CUDA demo (separate binary):

```bash
cmake -B build -DMVLLM_HOST_ONLY=OFF -DMVLLM_CUDA=ON
cmake --build build -j
./build/micro-vllm-cuda --model model.safetensors --batch 2 --max-prompt 512
```

HIP: `-DUSE_HIP=ON -DMVLLM_HOST_ONLY=OFF`.

## Commands

```bash
./build/micro-vllm info     --model DIR
./build/micro-vllm smoke    --model DIR          # shard headers + one expert/DiT pread
./build/micro-vllm generate --model DIR --prompt "..." --n 32
./build/micro-vllm serve    --model DIR --port 8000
./build/micro-vllm mux      --model DIR          # stdin framed protocol
./build/micro-vllm video    --model DIR --prompt "a red fox" -o out.mp4
```

`smoke` without `--model` uses the first existing `MVLLM_DUMP_*` / `COLI_DUMP_*` dir.

The family is sniffed from `config.json` (`model_type` / `architectures`)
and then the directory name (`kimi`/`k3`, `glm`, `deepseek-v4`/`dsv4`,
`minimax`/`h3`, `llama`). GLM-5.2-style names load as **glm53**. There is
no `--family` flag.

`--device cpu|metal|cuda` (or `MVLLM_DEVICE`) selects the compute backend.
Default is `cpu` so tests stay deterministic.

- K3 / GLM: Metal residual + post-LN (`layer_decode`) when the Metal
  backend is live; otherwise `vk_ops::layer_residual`. S=1 decode may fuse
  KDA (`layer_decode_kda`) or absorbed MLA (`layer_decode_mla`) into that
  tail. GLM routed int4 uses `moe_block` with clamped-SwiGLU. K3 F32
  dense/shared uses SiTU; routed MXFP4 uses `moe_block` fmt 7 (SiTU)
  before `coli_cuda` / host.
- H3: Metal AdaLN residual (`h3gpu=`), plus VAE / vision / audio blocks
  when the op matches (`int8=` / `nax=`).
- CUDA (`-DMVLLM_GPU_CUDA=ON`): K3/GLM expert GEMM via `coli_cuda`; DSV4
  route / mHC / sparse attn via `dsv4_cuda` when shapes match.

`info` tags: `coli=cpu|cuda|off`, `metal=cpu|metal|off`, `vk=cpu|vulkan|off`,
`tier=cpu|cuda|off` (DSV4), `h3gpu=…`, `mtp=off|loaded|markov|fwd`,
`ckpt=N hits=M`.

Also: `--expert-gb N`, `--kv-slots N` (serve/mux, 1–16).

Common generate flags: `--n` / `--max-tokens` (default 16), `--chat`,
`--think` / `--no-think`, `--raw` / `--no-chat`, `--temp`, `--top-p`,
`--top-k`, `--min-p`, `--seed`, `--stop` / `--stop-id` (repeatable),
`--grammar`, `--json`, `--image PATH` (GLM ViT), `--tools PATH`,
`--tool-choice`, `--persist` / `--kv`, `--cache-slot`,
repeat/freq/presence penalties.

## Kimi K3

HF snapshot or colibri pack. Official MXFP4 expert shards load as-is
(`k3_repack.py` is optional).

```bash
./build/micro-vllm info     --model /path/to/kimi-k3
./build/micro-vllm generate --model /path/to/kimi-k3 --prompt "hi" --n 32 --chat
./build/micro-vllm serve    --model /path/to/kimi-k3 --port 8000
```

- Chat is XTML (`<|open|>` / `<|sep|>` / `<|close|>`) when those specials exist,
  else `<|im_start|>role`.
- Routed experts: six U8 tensors
  `block_sparse_moe.experts.E.w{1,2,3}.weight_{packed,scale}` (e2m1 + ue8m0).
- LRU RAM: `MVLLM_EXPERT_GB` / `K3_EXPERT_GB` or `--expert-gb`. Dense bits:
  `MVLLM_BITS` / `K3_BITS`.
- `info` includes `coli=` and `metal=`. `--device metal` runs residual +
  post-LN, optional fused KDA/MLA, F32 SiTU dense/shared, and routed
  MXFP4 via `moe_block` fmt 7. Fallback is host / `coli_cuda`.

Empty / fixture dirs (`fixtures/kimi_k3_tiny`) use a synthetic expert pack.

## GLM-5.3 (and 5.2-style names)

Official Flash checkpoints are FP8. Convert routed experts to int4-g64 first:

```bash
python3 python/convert_glm53.py --indir /path/GLM-5.3-Flash --outdir /path/glm53_i4
./build/micro-vllm smoke    --model /path/glm53_i4
./build/micro-vllm generate --model /path/glm53_i4 --prompt "hi" --n 32 --chat --think
```

`--no-think` closes the reasoning block. `--image PATH` runs the ViT when the
image token is in the prompt.

- Config accepts `text_config`, `n_routed_experts`, `Glm5ForConditionalGeneration`,
  and GLM-5.2-style `model.` / `model.language_model.` prefixes.
- Chat: `[gMASK]<sop><|user|>…<|assistant|><think>`.
- Experts on disk: `gate/up/down_proj.weight` + `.qs` scales.
- Env: `GLM53_EXPERT_GB`, `GLM53_BITS`. `info` includes `coli=` and `metal=`.
- `--device metal` runs residual + post-LN, fused KDA/MLA when it fits,
  and routed int4 `moe_block` with clamped-SwiGLU (including `swiglu_limit>0`).
- GLM-5.2 HF trees (same `glm*` / `text_config` markers) use this engine.

## DeepSeek V4 Flash

Official tree: `deepseek-ai/DeepSeek-V4-Flash-0731` (43 layers, hidden 4096,
256 routed MXFP4 + 1 shared, top-6, MLA + DSA + mHC). Missing shards fall back
to a synthetic pack so tiny dirs still `generate`.

```bash
./build/micro-vllm info     --model /path/to/DeepSeek-V4-Flash
./build/micro-vllm generate --model /path/to/DeepSeek-V4-Flash --prompt "hi" --n 32
```

- Expert names: `layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}` (also
  `layers.N.mlp.experts.E`).
- Prefix checkpoints: `V4_PREFIX_CKPT` (default on), disk under
  `<model>/.coli_ckpt/`. Aliases `MVLLM_PREFIX_CKPT*`. `info` shows
  `ckpt=N hits=M` and `tier=cpu|cuda|off`.
- Prefill chunks: `COLI_PREFILL_CHUNK` / `V4_PREFILL_CHUNK` (layer-major).
- Draft: `V4_DRAFT` / `MVLLM_V4_DRAFT` (prompt bigram). `V4_MTP` /
  `MVLLM_V4_MTP` defaults draft depth to 3. Loaded `main_proj` / `wq_a` /
  confidence (`mtp=fwd`) draft with a rolling hidden, then verify the
  prefix in one pass after the main sample agrees on the first token.
  Markov `[V,rank]` is `mtp=markov`. `info` has `mtp=` / `draft=` /
  `dacc=` / `vk=`.
- With CUDA: top-6 `route`, mHC pre, and sparse attn (zero sinks) go
  through `dsv4_cuda`; otherwise the host kernels. Indexer stays on host.
- Dump env: `MVLLM_DUMP_DSV4` / `COLI_DUMP_DSV4`.

## MiniMax-H3 (video / audio DiT)

Not an LLM. Use `video` or `POST /v1/videos/generations`.

```bash
./build/micro-vllm video --model /path/to/MiniMax-H3 \
    --prompt "a red fox" -o out.mp4 \
    --width 864 --height 480 --frames 56 --seed 1 \
    [--audio in.wav] [--ref-image img.png]... \
    [--first-frame a.png] [--last-frame b.png]
```

- Layout: `FL2VA/{transformer,video_vae,audio_vae,text_encoder}` or the model root.
- DiT blocks stream four BF16 matrices through a 2-slot SSD window.
- `--device metal` runs AdaLN residual, VAE transformer blocks, vision
  LN/SDPA/GELU, and audio pre-blocks when the tensors match (`info` has
  `h3gpu=` / `int8=` / `nax=`).
- WAV/MP4 via ffmpeg when present; otherwise a written WAV/raw path still works in tests.

## Llama

Host `micro-vllm` is a small CPU stand-in (GQA). The full paged-attn loop is
`micro-vllm-cuda`:

```bash
./build/micro-vllm-cuda --model model.safetensors --n 20 --batch 2 \
    --max-prompt 512 --prompt-ids 128000,128006,882
```

`--prompt-ids` is repeatable (queue). Dims default to Llama 3.2 1B; override
with `-DLLAMA_N_LAYERS=…` etc. Needs CUDA/HIP; not built in `MVLLM_HOST_ONLY`.

## HTTP

`POST /v1/chat/completions` for LLM families. `POST /v1/videos/generations` for H3.

Also: `GET /health` `/ready` `/version` `/metrics` `/v1/models`,
`POST /tokenize` `/detokenize` `/count`, Anthropic `POST /v1/messages`
(`x-api-key` or Bearer), `POST /v1/messages/count_tokens`.

## Live dumps (optional)

```bash
export MVLLM_DUMP_K3=/path/to/kimi-k3          # or COLI_DUMP_K3
export MVLLM_DUMP_GLM53=/path/to/glm53_i4      # or COLI_DUMP_GLM53
export MVLLM_DUMP_H3=/path/to/minimax-h3       # or COLI_DUMP_H3
export MVLLM_DUMP_DSV4=/path/to/dsv4           # or COLI_DUMP_DSV4
./build/micro-vllm smoke
./build/mvllm_tests                            # sniff + probe; skip if unset
```

Smoke walks shard headers and `pread`s one expert or a small H3 vector. It does
not `Engine::load` a full TB overlay.

## Env (short)

| Variable | Role |
|----------|------|
| `MVLLM_DEVICE` | `cpu` / `metal` / `cuda` |
| `MVLLM_EXPERT_GB` | Expert LRU (also `K3_EXPERT_GB` / `GLM53_EXPERT_GB`) |
| `MVLLM_BITS` / `MVLLM_HEAD_BITS` / `MVLLM_MLA_BITS` | Dense / head / MLA quant |
| `MVLLM_PORT` | Serve port |
| `V4_PREFIX_CKPT` | DSV4 prefix checkpoints (default on) |
| `MVLLM_KV` / `COLI_KV` | Persistent KV path |
