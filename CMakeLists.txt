cmake_minimum_required(VERSION 3.24)

option(USE_HIP "Build with HIP for AMD GPUs" OFF)
option(MVLLM_CUDA "Build the legacy CUDA Llama demo (micro-vllm-cuda)" OFF)
option(MVLLM_HOST_ONLY "Build only the host engine and tests (no CUDA/HIP Llama demo)" ON)
if(APPLE)
    option(MVLLM_METAL "Metal expert GEMM and H3 DiT in the host engine" ON)
else()
    option(MVLLM_METAL "Metal expert GEMM and H3 DiT in the host engine" OFF)
endif()
option(MVLLM_GPU_CUDA "CUDA expert GEMM and H3 DiT in the host engine (not the Llama demo)" OFF)

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
    src/io/shard_probe.cpp
    src/io/image.cpp
    src/io/av_mux.cpp
    src/io/h3_resize.cpp
    src/quant/quant.cpp
    src/quant/weight.cpp
    src/quant/kv_fp8.cpp
    src/quant/kv_tq.cpp
    src/quant/int8_dyn.cpp
    src/quant/native_act.cpp
    src/quant/bf16.cpp
    src/store/expert_store.cpp
    src/store/block_store.cpp
    src/store/kv_persist.cpp
    src/store/kv_persist_v2.cpp
    src/store/kv_persist_v3.cpp
    src/store/route_usage.cpp
    src/store/route_trace.cpp
    src/store/kv_prefix.cpp
    src/store/kv_row.cpp
    src/tok/tokenizer.cpp
    src/tok/k3_tools.cpp
    src/tok/k3_chat1.cpp
    src/tok/gbnf.cpp
    src/tok/gbnf_forced.cpp
    src/tok/sample_nuc.cpp
    src/tok/stop_set.cpp
    src/tok/logprob.cpp
    src/tok/utf8.cpp
    src/tok/logit_dump.cpp
    src/tok/json_schema.cpp
    src/tok/decode_post.cpp
    src/tok/glm_tools.cpp
    src/serve/mux_stdio.cpp
    src/serve/mux_frames.cpp
    src/serve/mux_codec.cpp
    src/serve/mux_submit.cpp
    src/serve/hwinfo.cpp
    src/serve/cli_flags.cpp
    src/serve/anthropic.cpp
    src/model/ops.cpp
    src/model/family.cpp
    src/model/kimi_k3.cpp
    src/model/glm53.cpp
    src/model/h3.cpp
    src/model/h3_layout.cpp
    src/model/h3_seg.cpp
    src/model/h3_text.cpp
    src/model/h3_vision.cpp
    src/model/h3_mm.cpp
    src/model/h3_vae.cpp
    src/model/h3_audio_vae.cpp
    src/model/h3_dit_schedule.cpp
    src/model/h3_canvas.cpp
    src/model/h3_adaln.cpp
    src/model/h3_reuse.cpp
    src/model/h3_token_reduce.cpp
    src/model/moe_pick.cpp
    src/model/llama.cpp
    src/serve/http_server.cpp
    src/serve/session.cpp
    src/serve/scheduler.cpp
    src/engine.cpp
    src/gpu/backend.cpp
)

if(APPLE AND MVLLM_METAL)
    list(APPEND MVLLM_HOST_SOURCES src/gpu/metal.mm)
endif()

if(MVLLM_GPU_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        list(APPEND MVLLM_HOST_SOURCES src/gpu/cuda_gemm.cu)
    else()
        message(WARNING "MVLLM_GPU_CUDA=ON but no CUDA compiler was found")
        set(MVLLM_GPU_CUDA OFF)
    endif()
endif()

add_library(mvllm_host STATIC ${MVLLM_HOST_SOURCES})
target_include_directories(mvllm_host PUBLIC src include)
if(UNIX)
    target_link_libraries(mvllm_host PUBLIC pthread)
endif()
find_package(ZLIB)
if(ZLIB_FOUND)
    target_link_libraries(mvllm_host PUBLIC ZLIB::ZLIB)
else()
    target_link_libraries(mvllm_host PUBLIC z)
endif()
if(APPLE)
    target_compile_definitions(mvllm_host PUBLIC MVLLM_WITH_IMAGEIO=1)
    target_link_libraries(mvllm_host PUBLIC "-framework ImageIO" "-framework CoreGraphics"
                                            "-framework CoreFoundation")
endif()
if(APPLE AND MVLLM_METAL)
    set_source_files_properties(src/gpu/metal.mm PROPERTIES
        COMPILE_FLAGS "-x objective-c++ -fobjc-arc")
    target_compile_definitions(mvllm_host PUBLIC MVLLM_WITH_METAL=1)
    target_link_libraries(mvllm_host PUBLIC "-framework Metal" "-framework Foundation")
endif()
if(MVLLM_GPU_CUDA)
    set_target_properties(mvllm_host PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(mvllm_host PUBLIC MVLLM_WITH_CUDA_GEMM=1)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(mvllm_host PUBLIC CUDA::cudart)
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
