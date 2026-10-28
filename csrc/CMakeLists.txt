cmake_minimum_required(VERSION 3.20)
project(spark_eagle3_csrc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Lets editors (cpptools) resolve llama.h without knowing LLAMACPP_ROOT: the
# generated compile_commands.json carries the absolute -I paths.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# LLAMACPP_ROOT must point at a llama.cpp checkout that has been built with
# GGML_CUDA=ON and that carries the spark2_5 t_layer_inp patch.
if(NOT DEFINED LLAMACPP_ROOT)
    message(FATAL_ERROR "pass -DLLAMACPP_ROOT=/path/to/llama.cpp")
endif()
set(LLAMA_BUILD  "${LLAMACPP_ROOT}/build/bin")
set(LLAMA_LIB    "${LLAMA_BUILD}/libllama.so")

if(NOT EXISTS "${LLAMA_LIB}")
    message(FATAL_ERROR "libllama.so not found at ${LLAMA_LIB} (build llama.cpp first)")
endif()

add_library(spark_feat SHARED spark_feat.cpp)
target_include_directories(spark_feat PRIVATE
    "${LLAMACPP_ROOT}/include"
    "${LLAMACPP_ROOT}/ggml/include")
target_link_libraries(spark_feat PRIVATE "${LLAMA_LIB}")
target_link_options(spark_feat PRIVATE "-Wl,-rpath,${LLAMA_BUILD}")

# M1/G0 harness
add_executable(sp_feat_cli sp_feat_cli.cpp)
target_link_libraries(sp_feat_cli PRIVATE spark_feat)
target_link_options(sp_feat_cli PRIVATE "-Wl,-rpath,${LLAMA_BUILD}")

# coverage probe: generates with the target model and reports how much of its
# output the draft vocabulary can represent
add_executable(sp_gen_cli sp_gen_cli.cpp)
target_link_libraries(sp_gen_cli PRIVATE spark_feat)
target_link_options(sp_gen_cli PRIVATE "-Wl,-rpath,${LLAMA_BUILD}")
