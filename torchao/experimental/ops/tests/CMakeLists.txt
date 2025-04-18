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

if(TORCHAO_BUILD_CPU_AARCH64)
  add_compile_definitions(TORCHAO_BUILD_CPU_AARCH64=1)
  add_compile_definitions(TORCHAO_ENABLE_ARM_NEON_DOT)
endif()

if(TORCHAO_BUILD_KLEIDIAI)
    add_compile_definitions(TORCHAO_ENABLE_KLEIDI=1)
endif()

if(TORCHAO_BUILD_ARM_I8MM)
  add_compile_definitions(TORCHAO_ENABLE_ARM_I8MM)
endif()

if (ANDROID_ABI)
  # We are cross compiling, delay test discovery till runtime
  set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
endif()

include_directories(${TORCHAO_INCLUDE_DIRS})

set(TORCHAO_PARALLEL_BACKEND "test_dummy")

if (TORCHAO_BUILD_CPU_AARCH64)
  add_subdirectory(${TORCHAO_ROOT}/kernels/cpu/aarch64 ${CMAKE_CURRENT_BINARY_DIR}/torchao_kernels_aarch64)
  add_compile_definitions(TORCHAO_BUILD_CPU_AARCH64)
endif()

include(${TORCHAO_ROOT}/Utils.cmake)

if (ANDROID_ABI)
  # Given where we are today this is sufficent. But needs to be revisited.
  # This is also needed for native builds, but keeping it only for cross builds
  # for now given the hacky nature.
  file(GLOB DOTPROD_SRC_FILES test*.cpp)
  message(SRC_FILES: ${DOTPROD_SRC_FILES})
  set_property(SOURCE
    ${DOTPROD_SRC_FILES}
    APPEND_STRING PROPERTY
    COMPILE_FLAGS " -march=armv8.2-a+dotprod ")
endif()

add_executable(
  test_linear_8bit_act_xbit_weight
  test_linear_8bit_act_xbit_weight.cpp
  ${TORCHAO_ROOT}/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.cpp
)
target_link_libraries(
  test_linear_8bit_act_xbit_weight
  PRIVATE
  GTest::gtest_main
)
if (TORCHAO_BUILD_CPU_AARCH64)
  target_link_libraries(
    test_linear_8bit_act_xbit_weight
    PRIVATE
    torchao_kernels_aarch64
  )
endif()
target_link_torchao_parallel_backend(test_linear_8bit_act_xbit_weight "${TORCHAO_PARALLEL_BACKEND}")

include(GoogleTest)
gtest_discover_tests(test_linear_8bit_act_xbit_weight)
