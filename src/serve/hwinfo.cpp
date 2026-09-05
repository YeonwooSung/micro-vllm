#include "hwinfo.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace mvllm {
namespace {

#if defined(__APPLE__)

void probe_cpu(HwInfo &info) {
    char brand[256] = {};
    std::size_t n = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &n, nullptr, 0) == 0)
        info.cpu = brand;
}

void probe_ram(HwInfo &info) {
    std::uint64_t bytes = 0;
    std::size_t n = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &n, nullptr, 0) == 0)
        info.ram_total_gb = static_cast<double>(bytes) / 1e9;

    std::int64_t page = 0;
    n = sizeof(page);
    if (sysctlbyname("hw.pagesize", &page, &n, nullptr, 0) != 0 || page <= 0)
        page = 16384;

    vm_statistics64_data_t vs{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vs),
                          &count) != KERN_SUCCESS)
        return;
    // macOS analogue of Linux MemAvailable: free + inactive + purgeable
    const double pages =
        static_cast<double>(vs.free_count + vs.inactive_count + vs.purgeable_count);
    info.ram_avail_gb = pages * static_cast<double>(page) / 1e9;
}

#else

void probe_cpu(HwInfo &info) {
    std::ifstream in("/proc/cpuinfo");
    if (!in)
        return;
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, 10, "model name") != 0)
            continue;
        const auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::size_t i = colon + 1;
            while (i < line.size() && line[i] == ' ')
                ++i;
            info.cpu = line.substr(i);
        }
        break;
    }
}

void probe_ram(HwInfo &info) {
    std::ifstream in("/proc/meminfo");
    if (!in)
        return;
    std::string line;
    while (std::getline(in, line)) {
        double kb = 0;
        if (std::sscanf(line.c_str(), "MemTotal: %lf", &kb) == 1)
            info.ram_total_gb = kb / 1e6;
        if (std::sscanf(line.c_str(), "MemAvailable: %lf", &kb) == 1)
            info.ram_avail_gb = kb / 1e6;
    }
}

#endif

void probe_cores(HwInfo &info) {
#ifdef _SC_NPROCESSORS_ONLN
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    info.cores = n > 0 ? static_cast<int>(n) : 0;
#else
    (void)info;
#endif
}

} // namespace

HwInfo hw_probe() {
    HwInfo info;
    probe_cpu(info);
    probe_cores(info);
    probe_ram(info);
    return info;
}

double rss_gb() {
    struct rusage ru {};
    if (getrusage(RUSAGE_SELF, &ru) != 0)
        return 0.0;
#if defined(__APPLE__)
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0 * 1024.0); // bytes
#else
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0); // kilobytes
#endif
}

} // namespace mvllm
