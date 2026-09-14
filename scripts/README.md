# scripts/

User-facing wrappers around `./build/micro-vllm`.

| Script | Role |
|--------|------|
| [`prepare`](prepare) | Download official (or public quantized) weights; convert GLM FP8 → int4-g64 |
| [`run`](run) | `generate` / `video` / `serve` / `info` / `smoke` with a resolved model dir |
| `prepare-<family>.sh` / `run-<family>.sh` | Thin aliases (`qwen36`, `glm`, `h3`, …) |

Full table, disk sizes, and flags: [docs/MODELS.md](../docs/MODELS.md).

```bash
pip install -r scripts/requirements-prepare.txt   # once, for downloads + GLM convert
./scripts/prepare --doctor
./scripts/prepare qwen36                          # or glm / h3 / k3 / …
./scripts/run qwen36 "hello" --n 32
```
