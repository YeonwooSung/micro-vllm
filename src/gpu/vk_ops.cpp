#include "vk_ops.hpp"

#include "../quant/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(MVLLM_WITH_VULKAN)
#include "vk_shaders.hpp"
#include <vulkan/vulkan.h>
#ifndef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#define VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR 0x00000001
#endif
#ifndef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#endif
#ifndef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#define VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME "VK_KHR_portability_subset"
#endif
#ifndef VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
#define VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME "VK_KHR_get_physical_device_properties2"
#endif
#endif

namespace mvllm {
namespace vk_ops {
namespace {

bool g_inited = false;
bool g_vk = false;

#if defined(MVLLM_WITH_VULKAN)

VkInstance g_inst = VK_NULL_HANDLE;
VkPhysicalDevice g_phys = VK_NULL_HANDLE;
VkDevice g_dev = VK_NULL_HANDLE;
VkQueue g_queue = VK_NULL_HANDLE;
uint32_t g_qfam = 0;
VkCommandPool g_cpool = VK_NULL_HANDLE;
VkCommandBuffer g_cmd = VK_NULL_HANDLE;
VkDescriptorPool g_dpool = VK_NULL_HANDLE;

struct Pipe {
    VkShaderModule mod = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    int nbind = 0;
    uint32_t pcsz = 0;
};

enum { P_ADD = 0, P_SILU, P_RMS, P_GEMM, P_COUNT };
Pipe g_pipes[P_COUNT];

struct GpuBuf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void *map = nullptr;
    VkDeviceSize size = 0;
};

uint32_t find_mem(uint32_t type_bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    }
    return UINT32_MAX;
}

void gpu_free(GpuBuf &b) {
    if (!g_dev)
        return;
    if (b.buf)
        vkDestroyBuffer(g_dev, b.buf, nullptr);
    if (b.mem)
        vkFreeMemory(g_dev, b.mem, nullptr);
    b = {};
}

bool gpu_make(GpuBuf &b, VkDeviceSize size) {
    if (!g_dev || size == 0)
        return false;
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g_dev, &ci, nullptr, &b.buf) != VK_SUCCESS)
        return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g_dev, b.buf, &req);
    const uint32_t mt = find_mem(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) {
        gpu_free(b);
        return false;
    }
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;
    if (vkAllocateMemory(g_dev, &ai, nullptr, &b.mem) != VK_SUCCESS) {
        gpu_free(b);
        return false;
    }
    if (vkBindBufferMemory(g_dev, b.buf, b.mem, 0) != VK_SUCCESS) {
        gpu_free(b);
        return false;
    }
    if (vkMapMemory(g_dev, b.mem, 0, req.size, 0, &b.map) != VK_SUCCESS) {
        gpu_free(b);
        return false;
    }
    b.size = size;
    return true;
}

bool gpu_dispatch(Pipe &p, const GpuBuf *bufs, int nbuf, const void *pc, uint32_t gx, uint32_t gy) {
    if (!g_dev || !p.pipe || !g_cmd || gx == 0 || gy == 0 || gx > 65535u || gy > 65535u ||
        nbuf != p.nbind)
        return false;
    VkDescriptorBufferInfo bi[4]{};
    VkWriteDescriptorSet wr[4]{};
    for (int i = 0; i < nbuf; ++i) {
        if (!bufs[i].buf)
            return false;
        bi[i].buffer = bufs[i].buf;
        bi[i].offset = 0;
        bi[i].range = VK_WHOLE_SIZE;
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = p.set;
        wr[i].dstBinding = static_cast<uint32_t>(i);
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(g_dev, static_cast<uint32_t>(nbuf), wr, 0, nullptr);
    if (vkResetCommandBuffer(g_cmd, 0) != VK_SUCCESS)
        return false;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(g_cmd, &begin) != VK_SUCCESS)
        return false;
    vkCmdBindPipeline(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.set, 0,
                            nullptr);
    if (pc && p.pcsz)
        vkCmdPushConstants(g_cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, p.pcsz, pc);
    vkCmdDispatch(g_cmd, gx, gy, 1);
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                         1, &mb, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(g_cmd) != VK_SUCCESS)
        return false;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd;
    if (vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
        return false;
    return vkQueueWaitIdle(g_queue) == VK_SUCCESS;
}

bool has_ext(const std::vector<VkExtensionProperties> &exts, const char *name) {
    for (const auto &e : exts) {
        if (std::strcmp(e.extensionName, name) == 0)
            return true;
    }
    return false;
}

bool create_instance() {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    if (n)
        vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data());
    std::vector<const char *> enable;
    if (has_ext(exts, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
        enable.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    const bool port = has_ext(exts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    if (port)
        enable.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "micro-vllm";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    if (port)
        ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    ci.enabledExtensionCount = static_cast<uint32_t>(enable.size());
    ci.ppEnabledExtensionNames = enable.empty() ? nullptr : enable.data();
    if (vkCreateInstance(&ci, nullptr, &g_inst) == VK_SUCCESS)
        return true;
    g_inst = VK_NULL_HANDLE;
    // Bare instance (original probe) if portability extras are rejected.
    ci.flags = 0;
    ci.enabledExtensionCount = 0;
    ci.ppEnabledExtensionNames = nullptr;
    return vkCreateInstance(&ci, nullptr, &g_inst) == VK_SUCCESS;
}

void destroy_device_objects() {
    if (g_dev)
        vkDeviceWaitIdle(g_dev);
    for (auto &p : g_pipes) {
        if (g_dev && p.pipe)
            vkDestroyPipeline(g_dev, p.pipe, nullptr);
        if (g_dev && p.layout)
            vkDestroyPipelineLayout(g_dev, p.layout, nullptr);
        if (g_dev && p.dsl)
            vkDestroyDescriptorSetLayout(g_dev, p.dsl, nullptr);
        if (g_dev && p.mod)
            vkDestroyShaderModule(g_dev, p.mod, nullptr);
        p = {};
    }
    if (g_dev && g_dpool)
        vkDestroyDescriptorPool(g_dev, g_dpool, nullptr);
    g_dpool = VK_NULL_HANDLE;
    g_cmd = VK_NULL_HANDLE;
    if (g_dev && g_cpool)
        vkDestroyCommandPool(g_dev, g_cpool, nullptr);
    g_cpool = VK_NULL_HANDLE;
    if (g_dev)
        vkDestroyDevice(g_dev, nullptr);
    g_dev = VK_NULL_HANDLE;
    g_queue = VK_NULL_HANDLE;
    g_phys = VK_NULL_HANDLE;
}

bool make_pipe(Pipe &p, const uint32_t *spv, size_t nbytes, int nbind, uint32_t pcsz) {
    p.nbind = nbind;
    p.pcsz = pcsz;
    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = nbytes;
    sm.pCode = spv;
    if (vkCreateShaderModule(g_dev, &sm, nullptr, &p.mod) != VK_SUCCESS)
        return false;

    VkDescriptorSetLayoutBinding binds[3]{};
    for (int i = 0; i < nbind; ++i) {
        binds[i].binding = static_cast<uint32_t>(i);
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{};
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = static_cast<uint32_t>(nbind);
    dl.pBindings = binds;
    if (vkCreateDescriptorSetLayout(g_dev, &dl, nullptr, &p.dsl) != VK_SUCCESS)
        return false;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = pcsz;
    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &p.dsl;
    if (pcsz) {
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pcr;
    }
    if (vkCreatePipelineLayout(g_dev, &pl, nullptr, &p.layout) != VK_SUCCESS)
        return false;

    VkComputePipelineCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = p.mod;
    cp.stage.pName = "main";
    cp.layout = p.layout;
    if (vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cp, nullptr, &p.pipe) != VK_SUCCESS)
        return false;

    VkDescriptorSetAllocateInfo da{};
    da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    da.descriptorPool = g_dpool;
    da.descriptorSetCount = 1;
    da.pSetLayouts = &p.dsl;
    return vkAllocateDescriptorSets(g_dev, &da, &p.set) == VK_SUCCESS;
}

bool create_device_and_pipelines() {
    uint32_t ndev = 0;
    if (vkEnumeratePhysicalDevices(g_inst, &ndev, nullptr) != VK_SUCCESS || ndev == 0)
        return false;
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(g_inst, &ndev, devs.data());

    g_phys = VK_NULL_HANDLE;
    for (VkPhysicalDevice pd : devs) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        if (nq)
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
        for (uint32_t i = 0; i < nq; ++i) {
            if (qf[i].queueCount > 0 && (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                g_phys = pd;
                g_qfam = i;
                break;
            }
        }
        if (g_phys)
            break;
    }
    if (!g_phys)
        return false;

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(g_phys, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> dexts(ne);
    if (ne)
        vkEnumerateDeviceExtensionProperties(g_phys, nullptr, &ne, dexts.data());
    std::vector<const char *> den;
    if (has_ext(dexts, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME))
        den.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);

    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = g_qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(den.size());
    dci.ppEnabledExtensionNames = den.empty() ? nullptr : den.data();
    if (vkCreateDevice(g_phys, &dci, nullptr, &g_dev) != VK_SUCCESS) {
        g_phys = VK_NULL_HANDLE;
        return false;
    }
    vkGetDeviceQueue(g_dev, g_qfam, 0, &g_queue);

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = g_qfam;
    if (vkCreateCommandPool(g_dev, &pci, nullptr, &g_cpool) != VK_SUCCESS)
        return false;
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = g_cpool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_dev, &cai, &g_cmd) != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize psz{};
    psz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psz.descriptorCount = 32;
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 8;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &psz;
    if (vkCreateDescriptorPool(g_dev, &dpi, nullptr, &g_dpool) != VK_SUCCESS)
        return false;

    if (!make_pipe(g_pipes[P_ADD], shaders::kAddSpv, sizeof(shaders::kAddSpv), 2, 4))
        return false;
    if (!make_pipe(g_pipes[P_SILU], shaders::kSiluSpv, sizeof(shaders::kSiluSpv), 2, 4))
        return false;
    if (!make_pipe(g_pipes[P_RMS], shaders::kRmsSpv, sizeof(shaders::kRmsSpv), 3, 12))
        return false;
    if (!make_pipe(g_pipes[P_GEMM], shaders::kGemmSpv, sizeof(shaders::kGemmSpv), 3, 12))
        return false;
    return true;
}

bool gpu_add(float *y, const float *a, size_t n) {
    if (n > 0xffffffffu)
        return false;
    const uint32_t nn = static_cast<uint32_t>(n);
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(n) * sizeof(float);
    GpuBuf by, ba;
    if (!gpu_make(by, bytes) || !gpu_make(ba, bytes)) {
        gpu_free(by);
        gpu_free(ba);
        return false;
    }
    std::memcpy(by.map, y, static_cast<size_t>(bytes));
    std::memcpy(ba.map, a, static_cast<size_t>(bytes));
    const uint32_t gx = (nn + 63u) / 64u;
    GpuBuf bufs[2] = {by, ba};
    const bool ok = gpu_dispatch(g_pipes[P_ADD], bufs, 2, &nn, gx, 1);
    if (ok)
        std::memcpy(y, by.map, static_cast<size_t>(bytes));
    gpu_free(by);
    gpu_free(ba);
    return ok;
}

bool gpu_silu(float *g, const float *u, size_t n) {
    if (n > 0xffffffffu)
        return false;
    const uint32_t nn = static_cast<uint32_t>(n);
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(n) * sizeof(float);
    GpuBuf bg, bu;
    if (!gpu_make(bg, bytes) || !gpu_make(bu, bytes)) {
        gpu_free(bg);
        gpu_free(bu);
        return false;
    }
    std::memcpy(bg.map, g, static_cast<size_t>(bytes));
    std::memcpy(bu.map, u, static_cast<size_t>(bytes));
    const uint32_t gx = (nn + 63u) / 64u;
    GpuBuf bufs[2] = {bg, bu};
    const bool ok = gpu_dispatch(g_pipes[P_SILU], bufs, 2, &nn, gx, 1);
    if (ok)
        std::memcpy(g, bg.map, static_cast<size_t>(bytes));
    gpu_free(bg);
    gpu_free(bu);
    return ok;
}

bool gpu_rmsnorm(float *y, const float *x, const float *w, int nrows, int D, float eps) {
    if (nrows < 1 || D < 1)
        return false;
    const uint32_t nr = static_cast<uint32_t>(nrows);
    const uint32_t d = static_cast<uint32_t>(D);
    const size_t ne = static_cast<size_t>(nrows) * static_cast<size_t>(D);
    if (ne > 0xffffffffu)
        return false;
    const VkDeviceSize xbytes = static_cast<VkDeviceSize>(ne) * sizeof(float);
    const VkDeviceSize wbytes = static_cast<VkDeviceSize>(D) * sizeof(float);
    GpuBuf by, bx, bw;
    if (!gpu_make(by, xbytes) || !gpu_make(bx, xbytes) || !gpu_make(bw, wbytes)) {
        gpu_free(by);
        gpu_free(bx);
        gpu_free(bw);
        return false;
    }
    std::memcpy(bx.map, x, static_cast<size_t>(xbytes));
    if (w) {
        std::memcpy(bw.map, w, static_cast<size_t>(wbytes));
    } else {
        float *wp = static_cast<float *>(bw.map);
        for (int i = 0; i < D; ++i)
            wp[i] = 1.f;
    }
    struct PC {
        uint32_t nrows, D;
        float eps;
    } pc{nr, d, eps};
    const uint32_t gx = (nr + 63u) / 64u;
    GpuBuf bufs[3] = {by, bx, bw};
    const bool ok = gpu_dispatch(g_pipes[P_RMS], bufs, 3, &pc, gx, 1);
    if (ok)
        std::memcpy(y, by.map, static_cast<size_t>(xbytes));
    gpu_free(by);
    gpu_free(bx);
    gpu_free(bw);
    return ok;
}

bool gpu_gemm(float *y, const float *x, const float *w, int S, int I, int O) {
    if (S < 1 || I < 1 || O < 1)
        return false;
    const size_t ny = static_cast<size_t>(S) * static_cast<size_t>(O);
    const size_t nx = static_cast<size_t>(S) * static_cast<size_t>(I);
    const size_t nw = static_cast<size_t>(O) * static_cast<size_t>(I);
    GpuBuf by, bx, bw;
    if (!gpu_make(by, ny * sizeof(float)) || !gpu_make(bx, nx * sizeof(float)) ||
        !gpu_make(bw, nw * sizeof(float))) {
        gpu_free(by);
        gpu_free(bx);
        gpu_free(bw);
        return false;
    }
    std::memcpy(bx.map, x, nx * sizeof(float));
    std::memcpy(bw.map, w, nw * sizeof(float));
    struct PC {
        uint32_t S, I, O;
    } pc{static_cast<uint32_t>(S), static_cast<uint32_t>(I), static_cast<uint32_t>(O)};
    const uint32_t gx = (static_cast<uint32_t>(O) + 15u) / 16u;
    const uint32_t gy = (static_cast<uint32_t>(S) + 15u) / 16u;
    GpuBuf bufs[3] = {by, bx, bw};
    const bool ok = gpu_dispatch(g_pipes[P_GEMM], bufs, 3, &pc, gx, gy);
    if (ok)
        std::memcpy(y, by.map, ny * sizeof(float));
    gpu_free(by);
    gpu_free(bx);
    gpu_free(bw);
    return ok;
}

#endif // MVLLM_WITH_VULKAN

} // namespace

bool init() {
#if defined(MVLLM_WITH_VULKAN)
    if (!g_inst) {
        if (create_instance()) {
            if (create_device_and_pipelines())
                g_vk = true;
            else
                destroy_device_objects();
        }
    }
#endif
    g_inited = true;
    return true;
}

void shutdown() {
#if defined(MVLLM_WITH_VULKAN)
    destroy_device_objects();
    if (g_inst) {
        vkDestroyInstance(g_inst, nullptr);
        g_inst = VK_NULL_HANDLE;
    }
    g_vk = false;
#endif
    g_inited = false;
}

bool available() { return g_inited; }

const char *backend_name() { return g_vk ? "vulkan" : "cpu"; }

bool rmsnorm(float *y, const float *x, const float *w, int nrows, int D, float eps) {
    if (!y || !x || nrows < 1 || D < 1)
        return false;
#if defined(MVLLM_WITH_VULKAN)
    if (g_vk && gpu_rmsnorm(y, x, w, nrows, D, eps))
        return true;
#endif
    for (int r = 0; r < nrows; ++r)
        quant::rmsnorm(x + static_cast<size_t>(r) * D, w, y + static_cast<size_t>(r) * D, D, eps);
    return true;
}

bool add(float *y, const float *a, size_t n) {
    if (!y || !a)
        return false;
#if defined(MVLLM_WITH_VULKAN)
    if (g_vk && n > 0 && gpu_add(y, a, n))
        return true;
#endif
    for (size_t i = 0; i < n; ++i)
        y[i] += a[i];
    return true;
}

bool silu_mul(float *g, const float *u, size_t n) {
    if (!g || !u)
        return false;
#if defined(MVLLM_WITH_VULKAN)
    if (g_vk && n > 0 && gpu_silu(g, u, n))
        return true;
#endif
    quant::silu_mul(g, u, static_cast<int>(n));
    return true;
}

bool gemm_f32(float *y, const float *x, const float *w, int S, int I, int O) {
    if (!y || !x || !w || S < 1 || I < 1 || O < 1)
        return false;
#if defined(MVLLM_WITH_VULKAN)
    if (g_vk && gpu_gemm(y, x, w, S, I, O))
        return true;
#endif
    quant::matmul_f32(y, x, w, S, I, O);
    return true;
}

bool layer_residual(float *x, const float *attn, const float *post_ln, float *nrm, int D,
                    float eps) {
    if (!x || !attn || !post_ln || !nrm || D < 1)
        return false;
    add(x, attn, static_cast<size_t>(D));
    rmsnorm(nrm, x, post_ln, 1, D, eps);
    return true;
}

bool moe_block_f32(int nb, int D, int Iinter, const float *const *g, const float *const *u,
                   const float *const *d, const float *xg, const int *xoff, const int *nr,
                   const int *rows, const float *rw, float *out, int S) {
    if (nb < 1 || D < 1 || !out || !xg || !xoff || !nr)
        return false;
    int base = 0;
    std::vector<float> gate, up, hh;
    for (int e = 0; e < nb; ++e) {
        const int nre = nr[e];
        if (nre <= 0)
            continue;
        if (Iinter < 1 || !g || !u || !d || !g[e] || !u[e] || !d[e]) {
            base += nre;
            continue;
        }
        const float *xe = xg + static_cast<size_t>(xoff[e]) * static_cast<size_t>(D);
        gate.assign(static_cast<size_t>(nre) * static_cast<size_t>(Iinter), 0.f);
        up.assign(static_cast<size_t>(nre) * static_cast<size_t>(Iinter), 0.f);
        hh.assign(static_cast<size_t>(nre) * static_cast<size_t>(D), 0.f);
        gemm_f32(gate.data(), xe, g[e], nre, D, Iinter);
        gemm_f32(up.data(), xe, u[e], nre, D, Iinter);
        silu_mul(gate.data(), up.data(), static_cast<size_t>(nre) * static_cast<size_t>(Iinter));
        gemm_f32(hh.data(), gate.data(), d[e], nre, Iinter, D);
        if (rows && rw) {
            for (int r = 0; r < nre; ++r) {
                const int dest = rows[base + r];
                if (dest < 0 || (S > 0 && dest >= S))
                    continue;
                const float w = rw[base + r];
                const float *hr = hh.data() + static_cast<size_t>(r) * static_cast<size_t>(D);
                float *orow = out + static_cast<size_t>(dest) * static_cast<size_t>(D);
                for (int i = 0; i < D; ++i)
                    orow[i] += w * hr[i];
            }
        }
        base += nre;
    }
    return true;
}

bool gdn_delta(float *S, float *ctx, const float *q, const float *k, const float *v, int nq,
               int nkv, int hd, int group) {
    if (!S || !ctx || !q || !k || !v || nq < 1 || nkv < 1 || hd < 1)
        return false;
    for (int hh = 0; hh < nq; ++hh) {
        const int kh = std::min(hh / std::max(group, 1), nkv - 1);
        const float *kk = k + kh * hd;
        const float *vv = v + kh * hd;
        const float *qq = q + hh * hd;
        float *Sh = S + static_cast<size_t>(hh) * hd * hd;
        const float beta = 1.f / (1.f + std::exp(-kk[0]));
        std::vector<float> attn(static_cast<size_t>(hd), 0.f);
        for (int j = 0; j < hd; ++j) {
            float a = 0.f;
            for (int i = 0; i < hd; ++i)
                a += Sh[static_cast<size_t>(i) * hd + j] * kk[i];
            attn[static_cast<size_t>(j)] = a;
        }
        for (int i = 0; i < hd; ++i)
            for (int j = 0; j < hd; ++j)
                Sh[static_cast<size_t>(i) * hd + j] +=
                    beta * (kk[i] * vv[j] - kk[i] * attn[static_cast<size_t>(j)]);
        for (int j = 0; j < hd; ++j) {
            float o = 0.f;
            for (int i = 0; i < hd; ++i)
                o += qq[i] * Sh[static_cast<size_t>(i) * hd + j];
            ctx[hh * hd + j] = o;
        }
    }
    return true;
}

} // namespace vk_ops
} // namespace mvllm
