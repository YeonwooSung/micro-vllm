#pragma once

#include <string>

namespace mvllm {

struct HwInfo {
    int cores = 0;
    double ram_total_gb = 0;
    double ram_avail_gb = 0;
    int ngpu = 0;
    double vram_total_gb = 0;
    std::string cpu;
    std::string gpu; // empty / "none" on host
};

HwInfo hw_probe();
double rss_gb(); // peak RSS of this process

} // namespace mvllm
