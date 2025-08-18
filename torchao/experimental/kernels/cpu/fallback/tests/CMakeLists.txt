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
)
if(NOT TORCHAO_INCLUDE_DIRS)
  set(TORCHAO_INCLUDE_DIRS ${TORCHAO_LIBRARIES})
endif()

add_subdirectory(
${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/fallback
${CMAKE_CURRENT_BINARY_DIR}/torchao_kernels_cpu_fallback
)

enable_testing()

add_executable(test_bitpacking test_bitpacking.cpp)
target_link_libraries(
    test_bitpacking
    PRIVATE
    GTest::gtest_main
    dep
)

include(GoogleTest)
gtest_discover_tests(test_bitpacking)
