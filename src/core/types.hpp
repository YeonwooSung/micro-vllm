#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mvllm {

enum class Status : int {
    Ok = 0,
    InvalidArgument = 1,
    NotFound = 2,
    IoError = 3,
    Oom = 4,
    Unsupported = 5,
    ParseError = 6,
};

enum class Device : int {
    Cpu = 0,
    Cuda = 1,
    Hip = 2,
    Metal = 3,
};

enum class DType : int {
    F32 = 0,
    BF16 = 1,
    F16 = 2,
    I8 = 3,
    U8 = 4,
    I4 = 5,
    MXFP4 = 6,
};

enum class Family : int {
    Unknown = 0,
    Llama = 1,
    KimiK3 = 2,
    Glm53 = 3,
    H3 = 4,
    Dsv4 = 5,
};

inline const char *device_name(Device d) {
    switch (d) {
    case Device::Cuda:
        return "cuda";
    case Device::Hip:
        return "hip";
    case Device::Metal:
        return "metal";
    default:
        return "cpu";
    }
}

inline Device parse_device(const std::string &s) {
    if (s == "cuda")
        return Device::Cuda;
    if (s == "hip")
        return Device::Hip;
    if (s == "metal")
        return Device::Metal;
    return Device::Cpu;
}

inline const char *family_name(Family f) {
    switch (f) {
    case Family::Llama:
        return "llama";
    case Family::KimiK3:
        return "kimi_k3";
    case Family::Glm53:
        return "glm53";
    case Family::H3:
        return "h3";
    case Family::Dsv4:
        return "dsv4";
    default:
        return "unknown";
    }
}

inline const char *status_name(Status s) {
    switch (s) {
    case Status::Ok:
        return "ok";
    case Status::InvalidArgument:
        return "invalid_argument";
    case Status::NotFound:
        return "not_found";
    case Status::IoError:
        return "io_error";
    case Status::Oom:
        return "oom";
    case Status::Unsupported:
        return "unsupported";
    case Status::ParseError:
        return "parse_error";
    }
    return "unknown";
}

struct TensorView {
    void *data = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    DType dtype = DType::F32;
    const float *scales = nullptr; // per-row or per-group, optional
    int group = 0;
};

} // namespace mvllm
