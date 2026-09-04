#!/usr/bin/env bash
set -euo pipefail
# Host engine (Mac / no GPU). For the CUDA Llama demo:
#   cmake -B build -DMVLLM_HOST_ONLY=OFF -DMVLLM_CUDA=ON && cmake --build build -j
cmake -B build -DMVLLM_HOST_ONLY=ON "$@"
cmake --build build -j