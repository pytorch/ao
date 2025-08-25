# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

project(torchao_tests)

 # Delay test discovery till runtime.  Useful for cross-compiling.
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

set(TEST_TARGET_PREFIX "torchao_tests_torch_free_kernels_aarch64_")

add_library(
  ${TEST_TARGET_PREFIX}dep
  ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/find_min_and_max.cpp
  ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/compute_sum.cpp
  ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/torch_free_kernels/aarch64/quantization/quantize.cpp
)

enable_testing()

add_executable(${TEST_TARGET_PREFIX}test_quantization test_quantization.cpp)
target_link_libraries(
  ${TEST_TARGET_PREFIX}test_quantization
    PRIVATE
    GTest::gtest_main
    ${TEST_TARGET_PREFIX}dep
)

add_executable(${TEST_TARGET_PREFIX}test_reduction test_reduction.cpp)
target_link_libraries(
  ${TEST_TARGET_PREFIX}test_reduction
    PRIVATE
    GTest::gtest_main
    ${TEST_TARGET_PREFIX}dep
)

add_executable(${TEST_TARGET_PREFIX}test_bitpacking test_bitpacking.cpp)
target_link_libraries(
  ${TEST_TARGET_PREFIX}test_bitpacking
    PRIVATE
    GTest::gtest_main
    ${TEST_TARGET_PREFIX}dep
)

add_executable(${TEST_TARGET_PREFIX}test_linear test_linear.cpp)
target_link_libraries(
  ${TEST_TARGET_PREFIX}test_linear
    PRIVATE
    GTest::gtest_main
    ${TEST_TARGET_PREFIX}dep
    torchao_kernels_aarch64
)

add_executable(${TEST_TARGET_PREFIX}test_embedding_lut test_embedding_lut.cpp)
target_link_libraries(
  ${TEST_TARGET_PREFIX}test_embedding_lut
    PRIVATE
    GTest::gtest_main
    ${TEST_TARGET_PREFIX}dep
)

add_executable(${TEST_TARGET_PREFIX}test_embedding test_embedding.cpp)
target_link_libraries(
  ${TEST_TARGET_PREFIX}test_embedding
    PRIVATE
    GTest::gtest_main
    ${TEST_TARGET_PREFIX}dep
)

add_executable(${TEST_TARGET_PREFIX}test_weight_packing test_weight_packing.cpp)
target_link_libraries(
  ${TEST_TARGET_PREFIX}test_weight_packing
    PRIVATE
    GTest::gtest_main
    ${TEST_TARGET_PREFIX}dep
)

add_executable(${TEST_TARGET_PREFIX}test_qmatmul test_qmatmul.cpp)
target_link_libraries(
  ${TEST_TARGET_PREFIX}test_qmatmul
    PRIVATE
    GTest::gtest_main
    ${TEST_TARGET_PREFIX}dep
)

add_executable(${TEST_TARGET_PREFIX}test_lut test_lut.cpp)
target_link_libraries(
  ${TEST_TARGET_PREFIX}test_lut
    PRIVATE
    GTest::gtest_main
    ${TEST_TARGET_PREFIX}dep
)

add_executable(${TEST_TARGET_PREFIX}test_bitpack_fallback_compatibility test_bitpack_fallback_compatibility.cpp)
target_link_libraries(
  ${TEST_TARGET_PREFIX}test_bitpack_fallback_compatibility
    PRIVATE
    GTest::gtest_main
    ${TEST_TARGET_PREFIX}dep
)

include(GoogleTest)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_quantization)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_reduction)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_bitpacking)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_linear)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_embedding)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_embedding_lut)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_weight_packing)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_qmatmul)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_lut)
gtest_discover_tests(${TEST_TARGET_PREFIX}test_bitpack_fallback_compatibility)
