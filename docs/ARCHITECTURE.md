# micro-vllm architecture

Consumer-hardware inference server. The original CUDA Llama 3.2 1B engine
stays as `micro-vllm-cuda`. The new `micro-vllm` binary is a family-aware
host engine: disk/RAM/VRAM weight placement, CPU kernels, optional CUDA/HIP,
and an OpenAI-compatible HTTP server. Expert GEMM (K3 MXFP4, GLM int4-g64)
and the H3 DiT residual run on Metal or CUDA when `--device metal|cuda` /
`MVLLM_DEVICE` is set; otherwise the CPU kernels. The Llama CUDA demo stays
in `src/kernels.cu` and is not used by this path.

## What the three source projects taught us

**colibri** (Kimi K3, GLM-5.3): a 2.8T / 321B MoE does not need to *fit*.
Dense attention + shared experts stay resident (int4/int8). Routed experts
live on NVMe and are staged by a per-layer LRU. The overlap that matters is
same-layer: compute expert `j` while `pread`ing `j+1` (`K3_PIPE`), not a
magical “prefetch layer L+1”. I/O is part of the engine (`pread`, `O_DIRECT`
/ Darwin `F_NOCACHE`).

**h3.c** (MiniMax-H3): this is a video/audio DiT, not an LLM. The Mac path
is Metal + unified memory. The desktop-memory trick is SSD streaming: keep
two DiT blocks resident and read `i+1` while the GPU runs `i`.

**micro-vllm (this repo)**: a working Llama GQA + PagedAttention + continuous
batch CUDA loop. We keep that path and hang the new families off a registry.

## Tiers

```
VRAM  — hot experts / DiT block / KV pages   (optional)
RAM   — dense weights, expert LRU, KDA state
NVMe  — routed experts, DiT blocks, shards
```

Insufficient fast memory changes speed, not router semantics.

## Families

| Family    | Kind        | Attention                         | Sparse weights      | Quant            |
|-----------|-------------|-----------------------------------|---------------------|------------------|
| llama     | LLM         | GQA + RoPE + paged KV             | none                | BF16 resident    |
| kimi_k3   | LLM MoE     | KDA + gated MLA (NoPE) + AttnRes  | MXFP4 experts       | dense int4/int8  |
| glm53     | LLM MoE+ViT | KDA + absorbed MLA/DSA + mHC      | int4-g64 experts    | dense load-time  |
| h3        | Video DiT   | 50 residual DiT blocks            | whole blocks        | BF16 + int8 FC   |

## Module map

```
src/core     types, ModelConfig, RuntimeConfig
src/io       aligned I/O, O_DIRECT, safetensors
src/quant    MXFP4, int4-g64, int8-row, SiTU-GLU, clamped SwiGLU
src/store    ExpertStore (MoE LRU) + BlockStore + COLIKV1 + .coli_usage
src/tok      whitespace / HF tokenizer.json / raw ids
src/model    family registry + llama / kimi_k3 / glm53 / h3 (+ canvas, DiT schedule)
src/serve    OpenAI HTTP + mux TOOL/TOPK/HITS/EMAP telemetry frames
src/legacy   original CUDA Llama demo
```

## Serving

```
./micro-vllm info --model <dir>
./micro-vllm generate --model <dir> --prompt "..." --n 32
./micro-vllm serve --model <dir> --port 8000
```

`POST /v1/chat/completions` for LLM families.
`POST /v1/videos/generations` for H3.

## What is wired vs still missing

MiniMax-H3 DiT shards (`FL2VA/transformer` or the model root):

- Probe `FL2VA/transformer`, `transformer`, `dit`, then the model dir.
- Each block streams four BF16 matrices, same names as h3.c:
  `blocks.N.attn.qkv_proj.weight`, `attn.out_proj.weight`,
  `mlp.fc1.weight`, `mlp.fc2.weight`.
- Contiguous runs collapse to one `pread`; otherwise BlockStore scatter-reads
  the four pieces into a 2-slot SSD window (acquire/pin then prefetch next).
- Geometry is taken from block 0 (`hidden`, `inner = qkv_rows/3`, `ffn`).
- No matching tensors → 4 KiB synthetic pack (tiny fixtures).
- `generate_video` runs a CPU DiT residual (QKV attention + SwiGLU MLP)
  on the streamed BF16 matrices when sizes match, then decodes a
  `[24,T,H/16,W/16]` latent to RGB (synthetic mix, or transformer
  blocks when `FL2VA/video_vae` is present).

KDA short-conv: `kda_step` applies a causal depthwise conv (`conv_k` taps,
usually 4) after the Q/K/V projections, then SiLU. Windows persist across
tokens. Missing conv weights keep the previous identity path.

Kimi K3 MXFP4 shards (HF snapshot or `k3_repack.py` container):

- Prefix probe: `language_model.` if `model.layers.0...` is missing.
- Routed experts stay on disk as six U8 tensors
  `block_sparse_moe.experts.E.w{1,2,3}.weight_{packed,scale}`
  (e2m1 + ue8m0). Contiguous runs collapse to one `pread`.
- Geometry matches colibri when `latent` and `moe_inter` are multiples of 32:
  `w1p = inter*(latent/2)`, `w1s = inter*(latent/32)`.
- Dense overlay: embed, norms, KDA, router, latent down/up, shared experts.
- No matching expert tensors → synthetic pack (tiny fixtures).

GLM-5.3 container (colibri `convert_glm53.py` / official HF names):

- Config accepts `text_config`, `n_routed_experts`, `num_experts_per_tok`,
  `n_shared_experts`, `mlp_layer_types`, `hc_sinkhorn_iters`.
- Prefix probe: `model.language_model.` then `model.`.
- Routed experts stay on disk as six U8+`.qs` tensors
  (`gate/up/down_proj.weight` + scales). Contiguous runs collapse to one
  `pread`; otherwise ExpertStore scatter-reads the six pieces into a slot.
- Dense tensors (embed, norms, KDA, router, shared experts) are read as
  BF16/F32 at load. Missing names keep the synthetic fallback so tiny
  fixtures still run.
- No `*.safetensors` → previous synthetic pack path.

Wired (host tests cover the store path):

- K3/GLM53 pack experts to `model_dir/.mvllm_*_experts.bin`, `register_expert`,
  then `lookup`/`release` on every routed expert. RAM `experts_` is cleared.
- After `lookup(j)` a helper thread `pread`s expert `j+1` while the current
  GEMM runs. Store mutex is dropped during `pread`. The whole tail is not
  prefetched up front (that thrashed a small LRU).
- LRU budget is `MVLLM_EXPERT_GB`; slots capped at `n_experts`.
- Darwin `F_NOCACHE`, Linux `O_DIRECT` + aligned window read.
- H3: `acquire(b)` pins first, then `prefetch(b+1)` into the other slot.
- MXFP4 IDOT (`matmul_mxfp4_i8`, NEON/AVX2). K3 uses it when `I % 32 == 0`.

MLA (full-attn layers): absorbed NoPE. Cache stride is `kv_lora+qk_rope`
(K3 `qk_rope=64`; GLM `0`). `score_j = (W_kᵀ q) · c_j + q_rot · R_j`,
`out = W_v (Σ a_j c_j)`. `kv_b_proj` is folded at load (`mla_absorb_kvb`).
`mla_bits` quantizes q/kv; `head_bits` quantizes o/g. Missing absorbed KV
keeps the dense Q/O stand-in.

KDA `o_norm` is the official shared `[head_dim]` scale (not `[heads, head_dim]`).
GLM mHC keeps M residual streams (stream 0 is the layer view; others stay
mixed) and overlays `mhc.weight` / `hyper_connection.weight` when present.

`micro-vllm smoke` fails on unreadable/corrupt shards. An empty dir is still
the synthetic-pack success path.

Metal / CUDA expert GEMM and H3 DiT (host engine, not `kernels.cu`):

- `src/gpu/` dispatches `gemm_f32` / `gemm_int4_g64` / `gemm_mxfp4` and one
  DiT residual (BF16→F32, RMSNorm, QKV attention, SwiGLU).
- Apple builds compile `metal.mm` (`MVLLM_METAL`, default ON). CUDA expert
  kernels live in `cuda_gemm.cu` (`MVLLM_GPU_CUDA`, default OFF).
- `Engine::load` calls `gpu::select(rt.device)` and falls back to CPU if the
  GPU is missing. Default device stays CPU so host tests stay deterministic.
- K3/GLM union-MoE batches tokens that share an expert into one GEMM.

`micro-vllm smoke --model DIR` walks shard headers and, when expert / DiT
tensors exist, `pread`s one expert slot (or a small H3 vector). Counted
experts must match `config.json`. Empty / fixture dirs stay on the synthetic
pack path. Live dumps: set `MVLLM_DUMP_K3` / `MVLLM_DUMP_GLM53` / `MVLLM_DUMP_H3`
and run smoke against those trees. Official GLM FP8 experts need
`python/convert_glm53.py` first. K3 official MXFP4 loads without `k3_repack.py`.

Not absorbed yet: none from the host-migration plan
(`/health` running/queued, smoke `format_shard_report_json`, Anthropic
SSE `usage` also in).

Tokenizer: rank-BPE when `merges` is empty (Kimi tiktoken), cl100k BPE otherwise.
A K3 HF dir with only `tiktoken.model` (+ optional `tokenizer_config.json`)
loads without a synthesized `tokenizer.json`.
Kimi pretok sniffs `\\p{Han}`. GLM chat is `[gMASK]<sop><|user|>…<|assistant|><think></think>`.
K3 chat is segmented XTML (`<|open|>` / `<|sep|>` / `<|close|>` / `<|end_of_msg|>`)
when those specials exist; otherwise `<|im_start|>role`. HTTP
`/v1/chat/completions` applies the family template. GLM chat defaults to
think-on (`<|assistant|><think>` + Reasoning Effort); `--no-think` /
`enable_thinking=false` closes the block. `generate --prompt` is raw.
`eos_token_id` arrays (config + `generation_config.json`) are honored.

Prefill is layer-major: a chunk of tokens walks each layer, then union-MoE loads
each routed expert once for the chunk (`prefill=layer`). Decode stays C=1.

KV prefix reuse (`KvPrefix`) is all-or-nothing: the next prompt must strictly
extend the recorded fed ids, else prefill from scratch. Tainted state (ids
cannot describe the input) never reuses. Equal or shorter prompts cannot rewind.

`logprob_target` is official double softmax log p(token). Ragged decode
rows are `kv_row(base, pos, width)` per sequence. UTF-8 is official
`utf8_next`/`utf8_put` (invalid lead = one byte). Logit dump is
`[LOGITS] id:%.6f` top-5 and `LOGITGAP` top1/top2.

H3 layout `signature[5]` is `(text_len, latent_t, latent_h, latent_w, audio_t)`;
segment names are `text`/`cond`/`ref_img`/`ref_audio`/`audio`/`video`.

DSA (GLM full-attn): k-pool indexer + top-k slots into absorbed MLA. Missing
indexer tensors keep dense MLA.

H3 host canvas (h3.c `h3_adapt_canvas`): snap to a 32-multiple, 768 short-edge
nominal, cap `768*1344`. Ref2VA images are down-only; ref video never enlarges.
PCG + Box-Muller seeds DiT noise. Independent video/audio sigma grids
(`h3_schedule_build` / serving linear base), timestep row maps (shared row when
`1-σ_v == 1-σ_a`, plus condition rows at 0.999/1.0), sinusoidal 256-d time
features, AdaLN `row_map` by segment kind, and official RES multistep
(`h3_res_step`; Euler when `next==0` or no previous denoised).

K3/GLM COLIKV1: crash-safe F32 KV file (`COLIKV1\\0` + header + per-token L/R
rows, optional DSA index). `nrec` is fsynced last. Mismatched geometry is
ignored (empty reopen). Host F32 only. MLA latent rows can be stored as
e4m3 (`kv_fp8_*`, per-row amax/448, optional group scale) or PolarQuant /
rotated int4 (`kv_tq_*` / `kv_q4_*`). COLIKV2 stores e4m3 L/R + per-row
scale (`KvPersistV2`). COLIKV3 stores PolarQuant or rotated-int4 L/R +
radius (`KvPersistV3`; `format_tag = codec<<8 | bits`).

Mux sideband (colibri serve_protocol): `TOOL` counted frames (zero-byte
declares the sideband), `TOPK` hextext, `HITS` bit-hex, `EMAP` `(tier<<6)|heat`,
plus `HWINFO`/`TIERS`/`PERF`/`ENTROPY` formatters. Buffer reader
`mux_parse_command` accepts SUBMIT/STOP/CANCEL/IMAGE (NeedMore vs BadFrame).
SUBMIT extras are `logprobs=` / `ids=` (`mux_submit_ext`); `ids=1` payloads
are ASCII token ids (`mux_ids_parse`).

Native act QDQ: UE8M0 block scales with E4M3fn (NaN→0x7f) or E2M1.
BF16 is high-16 nearest-even; Hadamard-BF16 is unnormalized FWHT then
`bf16_round(v/sqrt(n))`.

`.coli_usage`: sparse `layer expert count` with `-1`/`-2` headers (FNV-1a
engine id). All-zero history is a zero-byte file. Atomic tmp+rename.
`ROUTE_TRACE` lines are `<call> <row> <layer> <id>:<gate.4f> …`.

H3 INT8 linear (CPU): one F32 scale per output channel on W, one per row on
X, `y = (w_sc[o]*x_sc[s])*dot_i32`.

H3 AdaLN time embed (official two-SiLU MLP): `SiLU(W_out SiLU(W_in x+b)+b)`
then per-block `W_adaln @ temb + b`. Serving reuse mask keeps step 0, the
last step, and every `reuse_interval` (`h3_dit_reuse_schedule`; optional
`0,3,6,…` list). Token reduction pair-pools target video along W
(`h3_token_reduce_*`; default blocks 4:30, early 10:40).

MoE pick: unused-scan top-k; NaN scores never win (`moe_router_pick`
falls back to slot index). Nucleus sampling uses the official max-heap
partial top-p (`nuc_dist_build`); greedy ignores ban. Stop set: config
ids + eos + tokenizer specials, cap 64; `eos_only` keeps just eos
(batched-serve tool-call safety). GBNF forced draft walks while exactly
one next token/byte is legal (`gbnf_forced_*`).

Host `hw_probe` / `rss_gb` match official HWINFO units (cores, RAM GB, CPU
brand; GPU fields stay 0).

H3 RGB resize: portable bilinear + edge-extend (official vImage HQ path
without Accelerate). Identity geometry still copies.
