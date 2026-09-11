# Official port plan (colibri + h3.c)

We **reimplement** host C++. Official C from `colibri-main/` and `h3.c-main/`
is not pasted. Licenses: colibri Apache-2.0, h3.c MIT — keep NOTICE if we
later vendor headers.

## Already in-tree (not this plan)

K3, GLM-5.3/5.2, DSV4, H3 host engines; Metal tails; CUDA CPU fallback;
Vulkan instance probe.

## Waves of 5 (exclusive files; commit after each wave)

### Wave 1 — new families + H3 tokenizer surface

| # | Deliverable | Files |
|---|-------------|-------|
| 1 | Qwen 3.6 host (GQA + GDN-as-dense stand-in + MoE) | `src/model/qwen36.cpp` |
| 2 | Qwen 3.8 host (text first; vision later) | `src/model/qwen38.cpp` |
| 3 | OLMoE host (GQA + routed experts) | `src/model/olmoe.cpp` |
| 4 | Inkling host (GQA, no RoPE, window, MoE) | `src/model/inkling.cpp` |
| 5 | H3 tokenizer host API | `src/model/h3_tok.hpp` + `h3_tok.cpp` |

Shared plumbing (parent): `Family` enum, sniff, `make_engine`, CMake,
`dump_env`, usage.

### Wave 2 — official GPU surfaces (no paste)

| # | Deliverable |
|---|-------------|
| 1 | Metal full-layer CB (MLA + router + MoE) | `metal_ops` |
| 2 | Vulkan compute dispatch for `vk_ops` GEMM | `vk_ops` + optional `.comp` |
| 3 | CUDA `.cu` device GEMM when nvcc exists | `coli_cuda` / `dsv4_cuda` |
| 4 | H3 `dit` Metal shaders (AdaLN + SDPA + SwiGLU) | `metal_h3` |
| 5 | H3 int8 GEMM + NAX kernels | `metal_h3` |

### Wave 3 — H3 full graphs + I/O (landed)

| # | Deliverable |
|---|-------------|
| 1 | `H3Vae::graph()` + `h3_read_ppm` | `h3_vae.cpp` |
| 2 | `H3AudioVae::graph()` | `h3_audio_vae.cpp` |
| 3 | `H3VisionEncoder::describe()` | `h3_vision.cpp` |
| 4 | `h3_read_mp4` ffmpeg pipe | `av_mux.cpp` |
| 5 | `H3GenParams::on_progress` + `ffmpeg=` | `h3.cpp` |

Stage-A engines may run **synthetic** weights (same as Llama/K3 fixtures)
until a dump tree exists.
