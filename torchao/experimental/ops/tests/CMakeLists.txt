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

set(TORCHAO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(TORCHAO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()
if(TORCHAO_BUILD_KLEIDIAI)
    add_compile_definitions(TORCHAO_ENABLE_KLEIDI=1)
endif()

include_directories(${TORCHAO_INCLUDE_DIRS})

set(TORCHAO_PARALLEL_BACKEND "test_dummy")
add_subdirectory(${TORCHAO_ROOT}/kernels/cpu/aarch64 ${CMAKE_CURRENT_BINARY_DIR}/torchao_kernels_aarch64)

include(${TORCHAO_ROOT}/Utils.cmake)
add_executable(
  test_linear_8bit_act_xbit_weight
  test_linear_8bit_act_xbit_weight.cpp
  ${TORCHAO_ROOT}/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.cpp
)
target_link_libraries(
  test_linear_8bit_act_xbit_weight
  PRIVATE
  GTest::gtest_main
  torchao_kernels_aarch64
)
target_link_torchao_parallel_backend(test_linear_8bit_act_xbit_weight "${TORCHAO_PARALLEL_BACKEND}")

include(GoogleTest)
gtest_discover_tests(test_linear_8bit_act_xbit_weight)
