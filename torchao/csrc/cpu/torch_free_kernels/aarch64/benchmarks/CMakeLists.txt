# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

project(torchao_benchmarks)
set(CMAKE_BUILD_TYPE Release)

set(TARGET_PREFIX "torchao_benchmarks_torch_free_kernels_aarch64_")

add_library(
  ${TARGET_PREFIX}dep
  ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/find_min_and_max.cpp
  ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/compute_sum.cpp
  ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/torch_free_kernels/aarch64/quantization/quantize.cpp
  ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/torch_free_kernels/aarch64/valpacking/interleave.cpp
)

add_executable(${TARGET_PREFIX}benchmark_quantization benchmark_quantization.cpp)
target_link_libraries(
    ${TARGET_PREFIX}benchmark_quantization
    PRIVATE
    benchmark::benchmark
    ${TARGET_PREFIX}dep
)

add_executable(${TARGET_PREFIX}benchmark_bitpacking benchmark_bitpacking.cpp)
target_link_libraries(
    ${TARGET_PREFIX}benchmark_bitpacking
    PRIVATE
    benchmark::benchmark
    ${TARGET_PREFIX}dep
)

# TODO: fix this, it's not working right now because of code refactors
# add_executable(${TARGET_PREFIX}benchmark_linear benchmark_linear.cpp)
# target_link_libraries(
#   ${TARGET_PREFIX}benchmark_linear
#     PRIVATE
#     benchmark::benchmark
#     ${TARGET_PREFIX}dep
# )
