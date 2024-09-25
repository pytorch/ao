# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

project(examples)

cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

include(CMakePrintHelpers)

set(TORCHAO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TORCHAO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

include_directories(${TORCHAO_INCLUDE_DIRS})

set(TORCHAO_PARALLEL_BACKEND "OPENMP")
add_subdirectory(${TORCHAO_ROOT}/ops/linear ${CMAKE_CURRENT_BINARY_DIR}/torchao_ops_linear_${TORCHAO_PARALLEL_BACKEND})
add_subdirectory(${TORCHAO_ROOT}/kernels/cpu/aarch64 ${CMAKE_CURRENT_BINARY_DIR}/torchao_kernels_aarch64)

include(${TORCHAO_ROOT}/Utils.cmake)

add_executable(separate_function_wrappers separate_function_wrappers.cpp)
target_link_libraries(
  separate_function_wrappers
    PRIVATE
    torchao_kernels_aarch64
    torchao_ops_linear_${TORCHAO_PARALLEL_BACKEND}
)
target_link_torchao_parallel_backend(separate_function_wrappers "${TORCHAO_PARALLEL_BACKEND}")

add_executable(stateful_class_wrapper stateful_class_wrapper.cpp)
target_link_libraries(
  stateful_class_wrapper
    PRIVATE
    torchao_kernels_aarch64
    torchao_ops_linear_${TORCHAO_PARALLEL_BACKEND}
)
target_link_torchao_parallel_backend(stateful_class_wrapper "${TORCHAO_PARALLEL_BACKEND}")
