#!/usr/bin/env bash
# Shared helpers for scripts/prepare and scripts/run.
# shellcheck disable=SC2034

set -euo pipefail

mvllm_repo_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$here/.." && pwd
}

MVLLM_ROOT="$(mvllm_repo_root)"
MVLLM_MODELS_DIR="${MVLLM_MODELS_DIR:-${HOME}/models/micro-vllm}"
MVLLM_BIN="${MVLLM_BIN:-}"
MVLLM_PYTHON="${MVLLM_PYTHON:-python3}"

mvllm_die() { echo "error: $*" >&2; exit 1; }
mvllm_info() { echo "$*" >&2; }
mvllm_warn() { echo "warning: $*" >&2; }

mvllm_need_cmd() {
  command -v "$1" >/dev/null 2>&1 || mvllm_die "missing command: $1"
}

mvllm_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "$HF_TOKEN"
    return 0
  fi
  if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    printf '%s' "$HUGGING_FACE_HUB_TOKEN"
    return 0
  fi
  local p
  for p in "${HOME}/.hf_token" "${HOME}/.cache/huggingface/token"; do
    if [[ -f "$p" ]]; then
      tr -d '[:space:]' <"$p"
      return 0
    fi
  done
  return 1
}

mvllm_find_bin() {
  if [[ -n "$MVLLM_BIN" && -x "$MVLLM_BIN" ]]; then
    printf '%s' "$MVLLM_BIN"
    return 0
  fi
  local c
  for c in \
    "${MVLLM_ROOT}/build/micro-vllm" \
    "${MVLLM_ROOT}/build/Release/micro-vllm" \
    "${PWD}/build/micro-vllm" \
    "$(command -v micro-vllm 2>/dev/null || true)"; do
    if [[ -n "$c" && -x "$c" ]]; then
      printf '%s' "$c"
      return 0
    fi
  done
  return 1
}

mvllm_find_cuda_demo() {
  local c
  for c in \
    "${MVLLM_ROOT}/build/micro-vllm-cuda" \
    "${PWD}/build/micro-vllm-cuda"; do
    if [[ -x "$c" ]]; then
      printf '%s' "$c"
      return 0
    fi
  done
  return 1
}

mvllm_default_device() {
  if [[ -n "${MVLLM_DEVICE:-}" ]]; then
    printf '%s' "$MVLLM_DEVICE"
    return 0
  fi
  case "$(uname -s)" in
    Darwin) printf 'metal' ;;
    *)
      if command -v nvidia-smi >/dev/null 2>&1; then
        printf 'cuda'
      else
        printf 'cpu'
      fi
      ;;
  esac
}

mvllm_free_gb() {
  local path="${1:-.}"
  while [[ ! -e "$path" && "$path" != "/" && "$path" != "." ]]; do
    path="$(dirname "$path")"
  done
  [[ -e "$path" ]] || path="."
  if df -Pk "$path" >/dev/null 2>&1; then
    df -Pk "$path" | awk 'NR==2 { printf "%.0f", $4/1024/1024 }'
  else
    echo 0
  fi
}

mvllm_dir_ready() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  [[ -f "$dir/config.json" || -f "$dir/model_index.json" || -d "$dir/FL2VA" || -d "$dir/transformer" ]] || return 1
  # A fixture with only config.json is "ready" for smoke/synth, but warn later.
  return 0
}

mvllm_has_shards() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  local n
  n="$(find "$dir" -maxdepth 3 -name '*.safetensors' -print -quit 2>/dev/null || true)"
  [[ -n "$n" ]]
}

mvllm_write_marker() {
  local dir="$1" family="$2" repo="${3:-}"
  mkdir -p "$dir"
  cat >"${dir}/.mvllm_prepared" <<EOF
family=${family}
repo=${repo}
prepared_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
}

mvllm_dump_env_name() {
  case "$1" in
    k3|kimi_k3) echo MVLLM_DUMP_K3 ;;
    glm53|glm52) echo MVLLM_DUMP_GLM53 ;;
    dsv4) echo MVLLM_DUMP_DSV4 ;;
    h3) echo MVLLM_DUMP_H3 ;;
    qwen36) echo MVLLM_DUMP_QWEN36 ;;
    qwen38) echo MVLLM_DUMP_QWEN38 ;;
    olmoe) echo MVLLM_DUMP_OLMOE ;;
    inkling) echo MVLLM_DUMP_INKLING ;;
    *) echo "" ;;
  esac
}

mvllm_print_env_hint() {
  local family="$1" dir="$2"
  local ev
  ev="$(mvllm_dump_env_name "$family")"
  if [[ -n "$ev" ]]; then
    mvllm_info "export ${ev}=${dir}"
    mvllm_info "  (alias: COLI_DUMP_${ev#MVLLM_DUMP_})"
  fi
}
