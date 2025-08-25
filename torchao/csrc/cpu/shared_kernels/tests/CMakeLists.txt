# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

project(torchao_tests)

set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

include_directories(${TORCHAO_INCLUDE_DIRS})

set(TEST_TARGET_PREFIX "torchao_tests_shared_kernels_")

add_executable(
  ${TEST_TARGET_PREFIX}test_linear_8bit_act_xbit_weight
  test_linear_8bit_act_xbit_weight.cpp
   ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.cpp
)
target_link_libraries(
  ${TEST_TARGET_PREFIX}test_linear_8bit_act_xbit_weight
  PRIVATE
  GTest::gtest_main
)
if (TORCHAO_BUILD_CPU_AARCH64)
  target_link_libraries(
    ${TEST_TARGET_PREFIX}test_linear_8bit_act_xbit_weight
    PRIVATE
    torchao_kernels_aarch64
  )
endif()
if (TORCHAO_BUILD_KLEIDIAI)
  target_link_libraries(
    ${TEST_TARGET_PREFIX}test_linear_8bit_act_xbit_weight
    PRIVATE
    kleidiai
  )
endif()
target_link_torchao_parallel_backend( ${TEST_TARGET_PREFIX}test_linear_8bit_act_xbit_weight test_dummy)

add_executable(
  ${TEST_TARGET_PREFIX}test_groupwise_lowbit_weight_lut
  test_groupwise_lowbit_weight_lut.cpp
  ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/shared_kernels/groupwise_lowbit_weight_lut/groupwise_lowbit_weight_lut.cpp
)
target_link_libraries(
  ${TEST_TARGET_PREFIX}test_groupwise_lowbit_weight_lut
  PRIVATE
  GTest::gtest_main
)
if (TORCHAO_BUILD_CPU_AARCH64)
  target_link_libraries(
    ${TEST_TARGET_PREFIX}test_groupwise_lowbit_weight_lut
    PRIVATE
    torchao_kernels_aarch64
  )
endif()
target_link_torchao_parallel_backend(${TEST_TARGET_PREFIX}test_groupwise_lowbit_weight_lut test_dummy)

include(GoogleTest)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_groupwise_lowbit_weight_lut)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_linear_8bit_act_xbit_weight)
