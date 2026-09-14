#!/usr/bin/env bash
# Family catalog. Sourced by prepare/run. Do not execute directly.

# Canonical ids: k3 glm53 glm52 dsv4 h3 qwen36 qwen38 olmoe inkling llama

mvllm_normalize_family() {
  local a
  a="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  a="${a//_/-}"
  case "$a" in
    k3|kimi|kimi-k3|kimik3|moonshot)
      echo k3 ;;
    glm|glm53|glm-53|glm-5.3|glm5.3|glm-5-3|flash)
      echo glm53 ;;
    glm52|glm-52|glm-5.2|glm5.2|glm-5-2)
      echo glm52 ;;
    dsv4|ds-v4|deepseek|deepseek-v4|deepseekv4|v4)
      echo dsv4 ;;
    h3|minimax|minimax-h3|minimaxh3|video)
      echo h3 ;;
    qwen36|qwen3.6|qwen-36|qwen3-6|qwen3-36|q36|qwen3.6-35b)
      echo qwen36 ;;
    qwen38|qwen3.8|qwen-38|qwen3-8|qwen3-38|q38|qwen3.8-flash)
      echo qwen38 ;;
    qwen|qwen3)
      echo qwen36 ;;
    olmoe|olmo|olmo-e|allenai)
      echo olmoe ;;
    inkling|ink|tinker)
      echo inkling ;;
    llama|llama3|llama32|llama-3.2|llama-3-2-1b)
      echo llama ;;
    *)
      return 1 ;;
  esac
}

# Fill globals for a canonical family id:
#   CAT_ID CAT_ENGINE CAT_TITLE CAT_REPO CAT_QUANT_REPO CAT_DIR
#   CAT_KIND CAT_CONVERT CAT_SIZE CAT_CMD CAT_NOTES CAT_INCLUDE
mvllm_catalog_load() {
  local id="$1"
  CAT_ID="$id"
  CAT_ENGINE="$id"
  CAT_QUANT_REPO=""
  CAT_KIND="llm"
  CAT_CONVERT="none"
  CAT_INCLUDE=""
  CAT_REVISION=""
  CAT_GATED=0
  case "$id" in
    k3)
      CAT_ENGINE=kimi_k3
      CAT_TITLE="Kimi K3"
      CAT_REPO="moonshotai/Kimi-K3"
      CAT_DIR="kimi-k3"
      CAT_SIZE="~1.5 TB (native MXFP4 experts; no convert)"
      CAT_CMD="generate --chat"
      CAT_NOTES="Official shards load as-is. Vision shards are unused."
      ;;
    glm53)
      CAT_ENGINE=glm53
      CAT_TITLE="GLM-5.3-Flash"
      CAT_REPO="zai-org/GLM-5.3-Flash"
      CAT_DIR="glm53-i4"
      CAT_CONVERT="glm53"
      CAT_SIZE="~195 GB after int4-g64 (source ~328 GB, converted shard-by-shard)"
      CAT_CMD="generate --chat --think"
      CAT_NOTES="Official dump is FP8. convert_glm53.py is required."
      ;;
    glm52)
      CAT_ENGINE=glm53
      CAT_TITLE="GLM-5.2 (pre-converted int4)"
      CAT_REPO="mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp"
      CAT_DIR="glm52-i4"
      CAT_CONVERT="none"
      CAT_SIZE="~372 GB (already int4-g64)"
      CAT_CMD="generate --chat --think"
      CAT_NOTES="Pre-converted colibri container. Same glm53 engine."
      ;;
    dsv4)
      CAT_ENGINE=dsv4
      CAT_TITLE="DeepSeek V4 Flash"
      CAT_REPO="deepseek-ai/DeepSeek-V4-Flash-0731"
      CAT_DIR="deepseek-v4-flash"
      CAT_SIZE="~150 GB (native MXFP4 experts; no convert)"
      CAT_CMD="generate"
      CAT_NOTES="Official shards load as-is."
      ;;
    h3)
      CAT_ENGINE=h3
      CAT_TITLE="MiniMax-H3"
      CAT_REPO="MiniMaxAI/MiniMax-H3"
      CAT_DIR="minimax-h3"
      CAT_KIND="video"
      CAT_INCLUDE="model_index.json,FL2VA/*"
      CAT_SIZE="FL2VA only by default (add --full for Ref2VA)"
      CAT_CMD="video"
      CAT_NOTES="Not an LLM. Use video / POST /v1/videos/generations."
      ;;
    qwen36)
      CAT_ENGINE=qwen36
      CAT_TITLE="Qwen3.6-35B-A3B"
      CAT_REPO="Qwen/Qwen3.6-35B-A3B"
      CAT_QUANT_REPO="Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64"
      CAT_DIR="qwen36"
      CAT_SIZE="~70 GB official BF16; --quant ~20 GB colibri int4-gs64"
      CAT_CMD="generate --chat"
      CAT_NOTES="Default is the official HF tree (this host overlays HF names). --quant downloads the smaller colibri container."
      ;;
    qwen38)
      CAT_ENGINE=qwen38
      CAT_TITLE="Qwen3.8-Flash-Next"
      CAT_REPO="Qwen/Qwen3.8-Flash-Next-FP8"
      CAT_DIR="qwen38"
      CAT_REVISION=""
      CAT_SIZE="~185 GB (native block-FP8; no convert)"
      CAT_CMD="generate --chat"
      CAT_NOTES="Official FP8 checkpoint. PLE stays pageable when present."
      ;;
    olmoe)
      CAT_ENGINE=olmoe
      CAT_TITLE="OLMoE-1B-7B"
      CAT_REPO="allenai/OLMoE-1B-7B-0125-Instruct"
      CAT_DIR="olmoe"
      CAT_SIZE="~13 GB official BF16"
      CAT_CMD="generate --chat"
      CAT_NOTES="Official HF Instruct tree. Smallest LLM family here."
      ;;
    inkling)
      CAT_ENGINE=inkling
      CAT_TITLE="Inkling"
      CAT_REPO="thinkingmachines/Inkling"
      CAT_QUANT_REPO="nbeerbower/Inkling-colibri-int4"
      CAT_DIR="inkling"
      CAT_SIZE="official BF16 is huge; --quant ~469 GB colibri int4"
      CAT_CMD="generate"
      CAT_NOTES="Default official HF names. --quant downloads the public int4 container."
      ;;
    llama)
      CAT_ENGINE=llama
      CAT_TITLE="Llama 3.2 1B Instruct"
      CAT_REPO="meta-llama/Llama-3.2-1B-Instruct"
      CAT_DIR="llama-3.2-1b"
      CAT_GATED=1
      CAT_SIZE="~2.5 GB (gated; needs HF login)"
      CAT_CMD="generate"
      CAT_NOTES="Host is a small CPU stand-in. Full paged-attn loop is micro-vllm-cuda."
      ;;
    *)
      return 1 ;;
  esac
}

mvllm_catalog_ids() {
  echo k3 glm53 glm52 dsv4 h3 qwen36 qwen38 olmoe inkling llama
}

mvllm_catalog_print() {
  local id
  printf '%-8s %-28s %-44s %s\n' "ID" "ENGINE" "DEFAULT REPO" "PREP"
  printf '%-8s %-28s %-44s %s\n' "--" "------" "------------" "----"
  for id in $(mvllm_catalog_ids); do
    mvllm_catalog_load "$id"
    local prep="download"
    [[ "$CAT_CONVERT" == glm53 ]] && prep="download+int4"
    [[ -n "$CAT_QUANT_REPO" ]] && prep="${prep}|--quant"
    printf '%-8s %-28s %-44s %s\n' "$CAT_ID" "$CAT_ENGINE" "$CAT_REPO" "$prep"
  done
}
