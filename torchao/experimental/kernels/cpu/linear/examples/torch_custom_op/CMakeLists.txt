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
message("TORCHAO_INCLUDE_DIRS: ${TORCHAO_INCLUDE_DIRS}")
include_directories(${TORCHAO_INCLUDE_DIRS})

add_subdirectory(${TORCHAO_INCLUDE_DIRS}/torchao/experimental/kernels/cpu/aarch64 ${CMAKE_CURRENT_BINARY_DIR}/kernel_aarch64)

include(${TORCHAO_INCLUDE_DIRS}/torchao/experimental/kernels/cpu/Utils.cmake)

set(PLATFORM "ATEN" CACHE STRING "Choose platform surface: ATEN, EXECUTORCH")
string(TOUPPER ${PLATFORM} PLATFORM_TO_UPPER)

if(PLATFORM_TO_UPPER STREQUAL "ATEN")
message(STATUS "Building with PLATFORM=ATEN")

find_package(Torch REQUIRED)
add_library(lowbit_op_aten SHARED lowbit_op_aten.cpp)
target_link_libraries(lowbit_op_aten PRIVATE kernel_aarch64)
target_include_directories(lowbit_op_aten PRIVATE "${TORCH_INCLUDE_DIRS}")
target_link_libraries(lowbit_op_aten PRIVATE "${TORCH_LIBRARIES}")
target_compile_definitions(lowbit_op_aten PRIVATE USE_ATEN=1)
target_link_torchao_parallel_backend(lowbit_op_aten "ATEN_OPENMP")

elseif(PLATFORM_TO_UPPER STREQUAL "EXECUTORCH")
message(STATUS "Building with PLATFORM=EXECUTORCH")

add_library(lowbit_op_executorch SHARED
    lowbit_op_executorch/w2s.cpp
    lowbit_op_executorch/w2sz.cpp
    lowbit_op_executorch/w3s.cpp
    lowbit_op_executorch/w3sz.cpp
    lowbit_op_executorch/w4s.cpp
    lowbit_op_executorch/w4sz.cpp
    lowbit_op_executorch/w5s.cpp
    lowbit_op_executorch/w5sz.cpp
)
target_include_directories(lowbit_op_executorch PRIVATE ${EXECUTORCH_INCLUDE_DIRS})
target_compile_definitions(lowbit_op_executorch PRIVATE USE_EXECUTORCH=1)
target_link_torchao_parallel_backend(lowbit_op_executorch "SINGLE_THREADED")
target_link_libraries(lowbit_op_executorch PRIVATE ${EXECUTORCH_LIBRARIES})
target_link_libraries(lowbit_op_executorch PRIVATE kernel_aarch64)

else()
message(FATAL_ERROR "Unknown PLATFORM: ${PLATFORM}. Please choose one of: ATEN, EXECUTORCH.")
endif()
