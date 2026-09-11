#pragma once

#include "../core/types.hpp"

namespace mvllm {

// Non-empty getenv("MVLLM_DUMP_K3") or getenv("COLI_DUMP_K3"), else nullptr.
const char *dump_env_k3();
// Non-empty getenv("MVLLM_DUMP_GLM53") or getenv("COLI_DUMP_GLM53"), else nullptr.
const char *dump_env_glm53();
// Non-empty getenv("MVLLM_DUMP_H3") or getenv("COLI_DUMP_H3"), else nullptr.
const char *dump_env_h3();
// Non-empty getenv("MVLLM_DUMP_DSV4") or getenv("COLI_DUMP_DSV4"), else nullptr.
const char *dump_env_dsv4();
const char *dump_env_qwen36();
const char *dump_env_qwen38();
const char *dump_env_olmoe();
const char *dump_env_inkling();

// Family → matching dump env. Llama / Unknown → nullptr.
const char *dump_env_for(Family f);

// First dump env (K3, GLM53, H3, DSV4, Qwen, OLMoE, Inkling) whose path is a dir.
// *which receives the winning name ("MVLLM_DUMP_K3" or "COLI_DUMP_K3", ...).
const char *dump_env_first_dir(const char **which = nullptr);

} // namespace mvllm
