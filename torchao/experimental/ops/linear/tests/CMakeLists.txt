# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
add_compile_options("-Wall" "-Werror")

set(TORCHAO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TORCHAO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

include_directories(${TORCHAO_INCLUDE_DIRS})

set(TORCHAO_PARALLEL_BACKEND "TEST_DUMMY")
add_subdirectory(${TORCHAO_ROOT}/ops/linear ${CMAKE_CURRENT_BINARY_DIR}/torchao_ops_linear_${TORCHAO_PARALLEL_BACKEND})
add_subdirectory(${TORCHAO_ROOT}/kernels/cpu/aarch64 ${CMAKE_CURRENT_BINARY_DIR}/torchao_kernels_aarch64)

include(${TORCHAO_ROOT}/Utils.cmake)
add_executable(test_linear_operator test_linear_operator.cpp)
target_link_libraries(
    test_linear_operator
    PRIVATE
    GTest::gtest_main
    torchao_kernels_aarch64
    torchao_ops_linear_${TORCHAO_PARALLEL_BACKEND}
)
target_link_torchao_parallel_backend(test_linear_operator "${TORCHAO_PARALLEL_BACKEND}")

include(GoogleTest)
gtest_discover_tests(test_linear_operator)
