#!/usr/bin/env bash
set -euo pipefail
./build.sh
./build/mvllm_tests
