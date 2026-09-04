cmake_minimum_required(VERSION 3.24)

option(USE_HIP "Build with HIP for AMD GPUs" OFF)
option(MVLLM_CUDA "Build the legacy CUDA Llama demo (micro-vllm-cuda)" OFF)
option(MVLLM_HOST_ONLY "Build only the host engine and tests (no CUDA/HIP)" ON)

if(NOT USE_HIP AND MVLLM_CUDA AND NOT MVLLM_HOST_ONLY)
    if(EXISTS "/opt/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
        set(CUDACXX "/opt/cuda/bin/nvcc")
    endif()
endif()

if(USE_HIP AND NOT MVLLM_HOST_ONLY)
    project(micro-vllm LANGUAGES CXX HIP)
elseif(MVLLM_CUDA AND NOT MVLLM_HOST_ONLY)
    project(micro-vllm LANGUAGES CXX CUDA)
else()
    project(micro-vllm LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

set(MVLLM_HOST_SOURCES
    src/core/config.cpp
    src/io/file_io.cpp
    src/io/safetensors.cpp
    src/quant/quant.cpp
    src/quant/weight.cpp
    src/store/expert_store.cpp
    src/store/block_store.cpp
    src/tok/tokenizer.cpp
    src/model/ops.cpp
    src/model/family.cpp
    src/model/kimi_k3.cpp
    src/model/glm53.cpp
    src/model/h3.cpp
    src/model/h3_vae.cpp
    src/model/llama.cpp
    src/serve/http_server.cpp
    src/engine.cpp
)

add_library(mvllm_host STATIC ${MVLLM_HOST_SOURCES})
target_include_directories(mvllm_host PUBLIC src include)
if(UNIX)
    target_link_libraries(mvllm_host PUBLIC pthread)
endif()

add_executable(micro-vllm src/main.cpp)
target_link_libraries(micro-vllm PRIVATE mvllm_host)
target_include_directories(micro-vllm PRIVATE src include)

add_executable(mvllm_tests tests/test_host.cpp)
target_link_libraries(mvllm_tests PRIVATE mvllm_host)
target_include_directories(mvllm_tests PRIVATE src include)

if(USE_HIP AND NOT MVLLM_HOST_ONLY)
    find_package(hipblas REQUIRED)
    find_package(hip REQUIRED)
    add_executable(micro-vllm-cuda
        src/legacy/llama_main.cpp
        src/kernels.cu
    )
    set_source_files_properties(src/legacy/llama_main.cpp src/kernels.cu PROPERTIES LANGUAGE HIP)
    target_compile_definitions(micro-vllm-cuda PRIVATE USE_HIP)
    target_include_directories(micro-vllm-cuda PRIVATE src include)
    target_link_libraries(micro-vllm-cuda PRIVATE hip::host roc::hipblas)
    set_target_properties(micro-vllm-cuda PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
elseif(MVLLM_CUDA AND NOT MVLLM_HOST_ONLY)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 120)
    endif()
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -DDEBUG")
    find_package(CUDAToolkit REQUIRED)
    add_executable(micro-vllm-cuda
        src/legacy/llama_main.cpp
        src/kernels.cu
    )
    target_include_directories(micro-vllm-cuda PRIVATE src include)
    target_link_libraries(micro-vllm-cuda PRIVATE CUDA::cublas CUDA::cudart)
endif()
