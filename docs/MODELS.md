# Prepare and run models

Two scripts cover every family the host engine sniffs:

```bash
./scripts/prepare FAMILY [options]     # download / convert
./scripts/run     FAMILY [cmd] [...]   # generate | video | serve | info | smoke
```

`FAMILY` is only used to pick a default Hugging Face repo and a default
command. The engine still sniffs `config.json` (then the directory name).
There is no `--family` flag on `micro-vllm`.

Default model root: `$MVLLM_MODELS_DIR` (falls back to `~/models/micro-vllm`).

```bash
export MVLLM_MODELS_DIR=/nvme/mvllm
./scripts/prepare --list
./scripts/prepare --doctor
```

## Quick start

```bash
# 1. Build the host (once)
./build.sh

# 2. Python bits used only by prepare / GLM convert
pip install -r scripts/requirements-prepare.txt

# 3. Fetch a small model and chat
./scripts/prepare olmoe --yes
./scripts/run olmoe "What is 2+2?" --n 32

# 4. Or the next-smallest Qwen
./scripts/prepare qwen36 --yes
./scripts/run qwen36 "hi" --n 32 --chat
```

Per-family aliases (same flags):

```bash
./scripts/prepare-glm.sh --out /nvme/glm53-i4
./scripts/run-glm.sh --chat --no-think --n 64
./scripts/run-h3.sh --prompt "a red fox" -o fox.mp4
```

## Family catalog

| Script id | Engine | Default repo | Prepare | Disk (approx.) |
|-----------|--------|--------------|---------|----------------|
| `k3` / `kimi` | kimi_k3 | [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) | download, **no convert** (native MXFP4) | ~1.5 TB |
| `glm` / `glm53` | glm53 | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) | download **+ int4-g64** (`python/convert_glm53.py`) | ~195 GB out |
| `glm52` | glm53 | [`mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp`](https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp) | download already-converted int4 | ~372 GB |
| `dsv4` / `deepseek` | dsv4 | [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | download, **no convert** (native MXFP4) | ~150 GB |
| `h3` / `minimax` | h3 | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) | `FL2VA/*` (add `--full` for `Ref2VA`) | tens of GB |
| `qwen36` / `qwen` | qwen36 | [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | official BF16; `--quant` → [`Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64`](https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64) | ~70 GB / ~20 GB |
| `qwen38` | qwen38 | [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | download, **no convert** (native FP8) | ~185 GB |
| `olmoe` | olmoe | [`allenai/OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) | download official BF16 | ~13 GB |
| `inkling` | inkling | [`thinkingmachines/Inkling`](https://huggingface.co/thinkingmachines/Inkling) | official; `--quant` → [`nbeerbower/Inkling-colibri-int4`](https://huggingface.co/nbeerbower/Inkling-colibri-int4) | huge / ~469 GB |
| `llama` | llama | [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | download (gated) | ~2.5 GB |

`--quant` is used only when a **public already-quantized** tree exists. GLM-5.3
has none; `prepare glm` always runs the in-tree converter so the host can load
routed int4-g64 (`name` U8 + `name.qs`). K3 / DSV4 / Qwen3.8 official dumps
already store the format the engine streams.

This host overlays Hugging Face tensor names for Qwen / OLMoE / Inkling.
A colibri `merged_weight` container (`--quant` on Qwen3.6 or Inkling) is
smaller on disk; dense HF names still overlay when present, and missing
routed experts fall back to the synthetic pack (same as a tiny fixture).

## `scripts/prepare`

```bash
./scripts/prepare FAMILY
./scripts/prepare glm --out /nvme/glm53-i4
./scripts/prepare glm --from-dir /path/GLM-5.3-Flash   # convert a local snapshot
./scripts/prepare h3 --full                             # FL2VA + Ref2VA
./scripts/prepare qwen36 --quant                        # public ~20 GB i4
./scripts/prepare inkling --quant
./scripts/prepare k3 --yes --smoke
```

| Flag | Meaning |
|------|---------|
| `--out DIR` | Destination (default `$MVLLM_MODELS_DIR/<name>`) |
| `--from-dir DIR` | Skip Hub; convert or copy a local tree |
| `--repo ID` | Override the Hugging Face repo |
| `--revision REV` | Pin a commit / tag |
| `--quant` | Prefer the public quantized container when the catalog lists one |
| `--official` | Force the official (usually larger) repo |
| `--full` | H3: also fetch `Ref2VA/*` |
| `--keep-source` | GLM: keep each FP8 shard after convert |
| `--limit-shards N` | GLM: trial convert of the first N shards |
| `--smoke` | Run `micro-vllm smoke --model DIR` afterwards |
| `--yes` | Skip the “this is huge” prompt |
| `--dry-run` | Print the plan only |

Auth: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or `~/.hf_token`. Llama is gated
(`huggingface-cli login`). Large public repos also throttle anonymous downloads.

After a successful prepare the script prints the dump-env export the engine
already understands:

```bash
export MVLLM_DUMP_GLM53=/nvme/glm53-i4     # or COLI_DUMP_GLM53
./build/micro-vllm smoke                   # picks the first existing dump dir
```

| Family | Env |
|--------|-----|
| k3 | `MVLLM_DUMP_K3` / `COLI_DUMP_K3` |
| glm53 / glm52 | `MVLLM_DUMP_GLM53` / `COLI_DUMP_GLM53` |
| dsv4 | `MVLLM_DUMP_DSV4` / `COLI_DUMP_DSV4` |
| h3 | `MVLLM_DUMP_H3` / `COLI_DUMP_H3` |
| qwen36 | `MVLLM_DUMP_QWEN36` / `COLI_DUMP_QWEN36` |
| qwen38 | `MVLLM_DUMP_QWEN38` / `COLI_DUMP_QWEN38` |
| olmoe | `MVLLM_DUMP_OLMOE` / `COLI_DUMP_OLMOE` |
| inkling | `MVLLM_DUMP_INKLING` / `COLI_DUMP_INKLING` |

## `scripts/run`

Model directory, first match:

1. `--model DIR` / `-m DIR`
2. `$MVLLM_MODEL`
3. The family dump env above
4. `$MVLLM_MODELS_DIR/<default-name>` (and the `--quant` sibling if present)
5. In-tree `fixtures/*_tiny` (synthetic; K3 / GLM / H3 only)

```bash
./scripts/run qwen36 "hello" --n 32
./scripts/run glm --chat --think --n 64 --temp 0.7
./scripts/run glm serve --port 8000
./scripts/run dsv4 info
./scripts/run h3 --prompt "a red fox" -o out.mp4 --width 864 --height 480 --frames 56
# H3 --frames aligns up to 5+17k and emits that many (8→22, 56 stays 56).
./scripts/run k3 smoke
```

`--device` defaults to `metal` on macOS, `cuda` when `nvidia-smi` is present,
otherwise `cpu`. Override with `--device` or `MVLLM_DEVICE`. Instruct families
get `--chat` (and GLM `--think`) unless you pass `--raw` / `--no-chat` /
`--no-think`.

Everything after the family name that is not a known command is forwarded to
`micro-vllm` (`--n`, `--temp`, `--top-p`, `--image`, `--tools`, `--seed`, …).

Llama: if `--model` points at a single `*.safetensors` and `micro-vllm-cuda`
is built (`-DMVLLM_HOST_ONLY=OFF -DMVLLM_CUDA=ON`), `scripts/run llama` execs
that demo instead of the host stand-in.

## When to convert

| Family | Official dump | This host |
|--------|---------------|-----------|
| Kimi K3 | MXFP4 routed experts | load as-is (`k3_repack.py` is optional, not in-tree) |
| GLM-5.3 | FP8 | **must** run `python/convert_glm53.py` (or `prepare glm`) |
| GLM-5.2 | use the public int4 container | load as glm53 |
| DeepSeek V4 | MXFP4 + FP8 dense | load as-is |
| MiniMax-H3 | BF16 FL2VA / Ref2VA | load as-is (`video`) |
| Qwen3.6 | BF16 HF names | load as-is; `--quant` is optional and smaller |
| Qwen3.8 | block-FP8 + pageable PLE | load as-is |
| OLMoE | BF16 HF names | load as-is |
| Inkling | BF16 HF names | load as-is; `--quant` is optional |
| Llama 3.2 1B | BF16 | host stand-in, or `micro-vllm-cuda` |

`convert_glm53.py` is disk-safe: one source shard at a time, then delete
(unless `--keep-source`). Peak extra disk is the growing output plus ~5 GB.

## Direct engine (no wrappers)

```bash
./build/micro-vllm info     --model DIR
./build/micro-vllm smoke    --model DIR
./build/micro-vllm generate --model DIR --prompt "..." --n 32 --chat
./build/micro-vllm serve    --model DIR --port 8000
./build/micro-vllm video    --model DIR --prompt "a red fox" -o out.mp4
```

`--device cpu|metal|cuda` and `--expert-gb N` apply to all of the above.
See the root [README](../README.md) for generate / video flags.
