# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

project(examples)

cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

add_compile_options("-Wall" "-Werror")

include(CMakePrintHelpers)
message("TORCHAO_LIBRARIES: ${TORCHAO_LIBRARIES}")
include_directories(${TORCHAO_LIBRARIES})

add_subdirectory(${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64 ${CMAKE_CURRENT_BINARY_DIR}/kernel_aarch64)

add_executable(separate_function_wrappers separate_function_wrappers.cpp)
target_link_libraries(
  separate_function_wrappers
    PRIVATE
    kernel_aarch64
)

add_executable(stateful_class_wrapper stateful_class_wrapper.cpp)
target_link_libraries(
  stateful_class_wrapper
    PRIVATE
    kernel_aarch64
)

include(${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/Utils.cmake)

target_link_torchao_parallel_backend(stateful_class_wrapper "openmp")
target_link_torchao_parallel_backend(separate_function_wrappers "openmp")
