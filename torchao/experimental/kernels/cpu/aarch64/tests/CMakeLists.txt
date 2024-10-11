# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(tests)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

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

if(NOT TORCHAO_INCLUDE_DIRS)
  set(TORCHAO_INCLUDE_DIRS ${TORCHAO_LIBRARIES})
endif()

add_subdirectory(${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64 ${CMAKE_CURRENT_BINARY_DIR}/torchao_kernels_aarch64)

# The TORCHAO_ENABLE_KLEIDI cmake variable should be set by `torchao_kernels_aarch64"
# This is a temporary work around.
if(TORCHAO_ENABLE_KLEIDI)
  add_compile_definitions(TORCHAO_ENABLE_KLEIDI)
endif()

enable_testing()

add_executable(test_quantization test_quantization.cpp)
target_link_libraries(
  test_quantization
    PRIVATE
    GTest::gtest_main
    dep
)

add_executable(test_reduction test_reduction.cpp)
target_link_libraries(
  test_reduction
    PRIVATE
    GTest::gtest_main
    dep
)

add_executable(test_bitpacking test_bitpacking.cpp)
target_link_libraries(
  test_bitpacking
    PRIVATE
    GTest::gtest_main
    dep
)

add_executable(test_linear test_linear.cpp)
target_link_libraries(
  test_linear
    PRIVATE
    GTest::gtest_main
    dep
    torchao_kernels_aarch64
)

add_executable(test_valpacking test_valpacking.cpp)
target_link_libraries(
  test_valpacking
    PRIVATE
    GTest::gtest_main
    dep
)

include(GoogleTest)
gtest_discover_tests(test_quantization)
gtest_discover_tests(test_reduction)
gtest_discover_tests(test_bitpacking)
gtest_discover_tests(test_linear)
gtest_discover_tests(test_valpacking)
