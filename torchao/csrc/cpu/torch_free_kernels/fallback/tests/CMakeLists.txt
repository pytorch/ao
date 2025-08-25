# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

project(torchao_tests)

set(TEST_TARGET_PREFIX "torchao_tests_torch_free_kernels_fallback_")

add_library(
  ${TEST_TARGET_PREFIX}dep
  ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/find_min_and_max.cpp
  ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/compute_sum.cpp
  ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/torch_free_kernels/aarch64/quantization/quantize.cpp
)

enable_testing()

add_executable(${TEST_TARGET_PREFIX}test_bitpacking test_bitpacking.cpp)
target_link_libraries(
    ${TEST_TARGET_PREFIX}test_bitpacking
    PRIVATE
    GTest::gtest_main
   ${TEST_TARGET_PREFIX}dep
)

include(GoogleTest)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_bitpacking)
