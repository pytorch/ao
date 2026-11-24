# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

project(torchao_tests)

set(TEST_TARGET_PREFIX "torchao_tests_torch_free_kernels_fallback_")

enable_testing()

add_executable(${TEST_TARGET_PREFIX}test_bitpacking test_bitpacking.cpp)
target_link_libraries(
    ${TEST_TARGET_PREFIX}test_bitpacking
    PRIVATE
    GTest::gtest_main
)

add_executable(${TEST_TARGET_PREFIX}test_linear test_linear.cpp)
target_link_libraries(
    ${TEST_TARGET_PREFIX}test_linear
    PRIVATE
    GTest::gtest_main
)
if (TORCHAO_BUILD_CPU_AARCH64)
  # On aarch64, weight_packing.h uses the aarch64 compute_sum.
  target_link_libraries(
      ${TEST_TARGET_PREFIX}test_linear
      PRIVATE
      torchao_kernels_aarch64
  )
endif()

include(GoogleTest)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_bitpacking)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_linear)
