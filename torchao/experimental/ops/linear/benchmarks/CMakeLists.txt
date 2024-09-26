# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(benchmarks)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
add_compile_options("-Wall" "-Werror")

set(TORCHAO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TORCHAO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

include(FetchContent)
FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main) # need main for benchmark::benchmark

set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(
  googlebenchmark)

include_directories(${TORCHAO_INCLUDE_DIRS})

set(TORCHAO_PARALLEL_BACKEND "OPENMP")
add_subdirectory(${TORCHAO_ROOT}/ops/linear ${CMAKE_CURRENT_BINARY_DIR}/torchao_ops_linear_${TORCHAO_PARALLEL_BACKEND})
add_subdirectory(${TORCHAO_ROOT}/kernels/cpu/aarch64 ${CMAKE_CURRENT_BINARY_DIR}/torchao_kernels_aarch64)

add_executable(benchmark_linear_operator benchmark_linear_operator.cpp)
target_link_libraries(
    benchmark_linear_operator
    PRIVATE
    benchmark::benchmark
    torchao_kernels_aarch64
    torchao_ops_linear_${TORCHAO_PARALLEL_BACKEND}
)
target_link_torchao_parallel_backend(benchmark_linear_operator "${TORCHAO_PARALLEL_BACKEND}")
