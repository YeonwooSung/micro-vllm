# micro-vllm

C/C++ inference server for consumer desktops and Macs. The original CUDA
Llama 3.2 1B demo is still here (`micro-vllm-cuda`). The `micro-vllm` binary
is a family-aware host engine that treats NVMe, RAM, and VRAM as one hierarchy
so Kimi K3, GLM-5.3, DeepSeek V4 Flash, and MiniMax-H3 can run without a datacenter GPU.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Build (host / Mac / no GPU)

```bash
cmake -B build -DMVLLM_HOST_ONLY=ON
cmake --build build -j
./build/mvllm_tests
```

## Commands

```bash
./build/micro-vllm info     --model /path/to/model
./build/micro-vllm smoke    --model /path/to/model
./build/micro-vllm generate --model /path/to/model --prompt "Ciao" --n 32 [--json] [--stop S]
./build/micro-vllm serve    --model /path/to/model --port 8000
./build/micro-vllm mux      --model /path/to/model
./build/micro-vllm video    --model /path/to/MiniMax-H3 --prompt "a red fox" -o out.mp4
#   [--width W] [--height H] [--frames N] [--seed S] [--audio WAV]
```

Official GLM-5.3-Flash is FP8. Convert experts to int4-g64 before load:

```bash
python3 python/convert_glm53.py --indir /path/GLM-5.3-Flash --outdir /path/glm53_i4
./build/micro-vllm smoke --model /path/glm53_i4
```

Kimi K3 official MXFP4 shards load as-is (`k3_repack.py` in colibri is optional).
Live dump smoke is header walk + one expert/`rope.inv_freq` pread; it does not
`Engine::load` the full dense overlay.

Optional live-dump paths (host tests skip when unset or the dir is missing):

```bash
export MVLLM_DUMP_K3=/path/to/kimi-k3          # or COLI_DUMP_K3
export MVLLM_DUMP_GLM53=/path/to/glm53_i4      # or COLI_DUMP_GLM53
export MVLLM_DUMP_H3=/path/to/minimax-h3       # or COLI_DUMP_H3
# MVLLM_DUMP_DSV4 / COLI_DUMP_DSV4 reserved for a later family
./build/micro-vllm smoke                       # first existing dump dir
./build/mvllm_tests                            # gated sniff + shard_probe
```

OpenAI-compatible:

- `GET  /health`
- `GET  /ready` / `/v1/ready`
- `GET  /version` / `/v1/version`
- `GET  /metrics`
- `POST /tokenize` / `/v1/tokenize`
- `POST /detokenize` / `/v1/detokenize`
- `POST /count` / `/v1/count`
- `GET  /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`         (echo)
- `POST /v1/messages`            (Anthropic; `x-api-key` or Bearer)
- `POST /v1/messages/count_tokens`
- `POST /v1/videos/generations`  (H3)

## Families

| Family | What runs today | Weights |
|--------|-----------------|---------|
| `kimi_k3` | CPU forward + disk-streamed native MXFP4 experts | HF shards / `k3_repack.py`; dense BF16 at load |
| `glm53` | CPU forward + disk-streamed int4-g64 experts (`convert_glm53.py` / HF names) | U8+`.qs` shards; dense BF16 at load |
| `dsv4` | CPU forward + disk-streamed MXFP4 experts (MLA + DSA + mHC) | official V4 shards / synth pack; dense FP8 at load |
| `h3` | 2-slot SSD DiT + TEXT/AUDIO/VIDEO pack + 3-axis RoPE; WAV / MP4 via ffmpeg | `FL2VA/{transformer,video_vae,audio_vae,text_encoder}` |
| `llama` | CPU stand-in; full CUDA GQA+PagedAttention in `micro-vllm-cuda` | `model.safetensors` |

## CUDA Llama demo

```bash
cmake -B build -DMVLLM_HOST_ONLY=OFF -DMVLLM_CUDA=ON
cmake --build build -j
# needs model.safetensors (Llama 3.2 1B) in the cwd
./build/micro-vllm-cuda
```

HIP: `-DUSE_HIP=ON -DMVLLM_HOST_ONLY=OFF`.
