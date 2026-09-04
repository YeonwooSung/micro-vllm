# micro-vllm

C/C++ inference server for consumer desktops and Macs. The original CUDA
Llama 3.2 1B demo is still here (`micro-vllm-cuda`). The `micro-vllm` binary
is a family-aware host engine that treats NVMe, RAM, and VRAM as one hierarchy
so Kimi K3, GLM-5.3, and MiniMax-H3 can run without a datacenter GPU.

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
./build/micro-vllm generate --model /path/to/model --prompt "Ciao" --n 32
./build/micro-vllm serve    --model /path/to/model --port 8000
./build/micro-vllm video    --model /path/to/MiniMax-H3 --prompt "a red fox" -o out.txt
```

OpenAI-compatible:

- `GET  /health`
- `GET  /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/videos/generations`  (H3)

## Families

| Family | What runs today | Weights |
|--------|-----------------|---------|
| `kimi_k3` | CPU forward + disk-streamed native MXFP4 experts | HF shards / `k3_repack.py`; dense BF16 at load |
| `glm53` | CPU forward + disk-streamed int4-g64 experts (`convert_glm53.py` / HF names) | U8+`.qs` shards; dense BF16 at load |
| `h3` | 2-slot SSD stream of 4 BF16 DiT matrices (`blocks.N.{qkv,out,fc1,fc2}`) + visual VAE decode | `FL2VA/transformer` shards; RGB via synth mix or `FL2VA/video_vae` |
| `llama` | CPU stand-in; full CUDA GQA+PagedAttention in `micro-vllm-cuda` | `model.safetensors` |

## CUDA Llama demo

```bash
cmake -B build -DMVLLM_HOST_ONLY=OFF -DMVLLM_CUDA=ON
cmake --build build -j
# needs model.safetensors (Llama 3.2 1B) in the cwd
./build/micro-vllm-cuda
```

HIP: `-DUSE_HIP=ON -DMVLLM_HOST_ONLY=OFF`.
