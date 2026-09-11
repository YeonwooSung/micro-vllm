#pragma once

// Optional wrap of vendored colibri backend_metal + h3_gpu hosts.
// Built only with -DMVLLM_OFFICIAL_METAL_HOST=ON (Apple + Metal).

namespace mvllm {
namespace official_metal {

bool init();
void shutdown();
bool coli_available();
bool h3_available();
const char *status(); // "coli+h3" | "coli" | "h3" | "off" | "err: ..."

} // namespace official_metal
} // namespace mvllm
