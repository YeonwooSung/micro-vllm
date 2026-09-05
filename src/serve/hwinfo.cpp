#include "hwinfo.hpp"
#include "mux_frames.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <dirent.h>
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

void probe_gpu(HwInfo &info) {
    char brand[256] = {};
    std::size_t n = sizeof(brand);
    if (sysctlbyname("machdep.gpu.brand_string", brand, &n, nullptr, 0) == 0 && brand[0])
        info.gpu = brand;
    else
        info.gpu = "apple";
    info.ngpu = 1;
    // unified memory: leave vram_total_gb at 0
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

void probe_gpu(HwInfo &info) {
    if (DIR *d = ::opendir("/proc/driver/nvidia/gpus")) {
        int n = 0;
        while (const dirent *e = ::readdir(d)) {
            if (e->d_name[0] == '.')
                continue;
            ++n;
            if (!info.gpu.empty())
                continue;
            std::ifstream in(std::string("/proc/driver/nvidia/gpus/") + e->d_name +
                             "/information");
            std::string line;
            while (in && std::getline(in, line)) {
                if (line.compare(0, 6, "Model:") != 0)
                    continue;
                std::size_t i = 6;
                while (i < line.size() && line[i] == ' ')
                    ++i;
                if (i < line.size())
                    info.gpu = line.substr(i);
                break;
            }
        }
        ::closedir(d);
        info.ngpu = n > 0 ? n : 1;
        if (info.gpu.empty())
            info.gpu = "nvidia";
        return;
    }
    if (access("/sys/class/drm/card0", F_OK) != 0)
        return;
    info.ngpu = 1;
    std::ifstream in("/sys/class/drm/card0/device/product_name");
    std::string name;
    if (in && std::getline(in, name)) {
        std::size_t i = 0;
        while (i < name.size() && name[i] == ' ')
            ++i;
        if (i < name.size())
            info.gpu = name.substr(i);
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
    probe_gpu(info);
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

std::string hwinfo_line() {
    HwInfo h = hw_probe();
    return mux_format_hwinfo(h.cores, h.ram_total_gb, h.ram_avail_gb, h.ngpu,
                             h.vram_total_gb, h.cpu, h.gpu);
}

} // namespace mvllm
