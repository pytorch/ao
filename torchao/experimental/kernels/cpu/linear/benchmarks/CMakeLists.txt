# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(benchmarks)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

include(FetchContent)
FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main) # need main for benchmark::benchmark

set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(
  googlebenchmark)

add_compile_options("-Wall" "-Werror")

include(CMakePrintHelpers)
message("TORCHAO_LIBRARIES: ${TORCHAO_LIBRARIES}")
include_directories(${TORCHAO_LIBRARIES})

add_library(
  dep
  ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/reduction/find_min_and_max.cpp
  ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/reduction/compute_sum.cpp
  ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/quantization/quantize.cpp
  ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/valpacking/interleave.cpp
)

add_executable(benchmark_linear_operator benchmark_linear_operator.cpp)
target_link_libraries(
    benchmark_linear_operator
    PRIVATE
    benchmark::benchmark
    dep
)

option(TORCHAO_PARALLEL_OMP "" OFF)
option(TORCHAO_PARALLEL_SINGLE_THREADED "" ON)

if (TORCHAO_PARALLEL_OMP)
  message("OpenMP_ROOT: ${OpenMP_ROOT}")
  add_definitions(-DTORCHAO_PARALLEL_OMP=1)
  find_package(OpenMP REQUIRED)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(benchmark_linear_operator PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if (TORCHAO_PARALLEL_SINGLE_THREADED)
  add_definitions(-DTORCHAO_PARALLEL_SINGLE_THREADED=1)
endif()
