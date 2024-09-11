# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

project(torch_custom_op)

cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

add_compile_options("-Wall" "-Werror")

include(CMakePrintHelpers)
message("TORCHAO_LIBRARIES: ${TORCHAO_LIBRARIES}")
include_directories(${TORCHAO_LIBRARIES})

add_subdirectory(${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64 ${CMAKE_CURRENT_BINARY_DIR}/kernel_aarch64)

find_package(Torch REQUIRED)
include_directories("${TORCH_INCLUDE_DIRS}")

add_library(torch_custom_op SHARED torch_custom_op.cpp)
target_link_libraries(torch_custom_op PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(torch_custom_op PRIVATE kernel_aarch64)

include(${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/Utils.cmake)
set(TORCHAO_PARALLEL_BACKEND "ATEN_OPENMP" CACHE STRING "Choose parallel backend to use for torchao parallelism (aten_openmp, openmp, pthreadpool, single_threaded)")
target_link_torchao_parallel_backend(torch_custom_op "${TORCHAO_PARALLEL_BACKEND}")
