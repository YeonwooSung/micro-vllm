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
// Probe + mux_format_hwinfo. rss is not part of HWINFO (STAT uses it).
std::string hwinfo_line();
// mux_format_hwinfo from a probed struct. No hw_probe().
std::string hwinfo_line(const HwInfo &info);
// Compact JSON. GBs are %.2f. cpu/gpu have " and \\ escaped; empty gpu → "none".
std::string hwinfo_json();
std::string hwinfo_json(const HwInfo &info);

} // namespace mvllm
