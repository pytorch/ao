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

add_library(
  torchao_dep
  ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/reduction/find_min_and_max.cpp
  ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/reduction/compute_sum.cpp
  ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/quantization/quantize.cpp
  ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/valpacking/interleave.cpp
)

include(FetchContent)
FetchContent_Declare(pthreadpool
        GIT_REPOSITORY https://github.com/Maratyszcza/pthreadpool.git
        GIT_TAG master)
FetchContent_MakeAvailable(
  pthreadpool)

find_package(Torch REQUIRED)
message("TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
include_directories("${TORCH_INCLUDE_DIRS}")

add_library(torch_custom_op SHARED torch_custom_op.cpp)
target_link_libraries(torch_custom_op PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(torch_custom_op PRIVATE torchao_dep)
target_compile_definitions(torch_custom_op PRIVATE TORCHAO_PARALLEL_PTHREADPOOL=1)
target_link_libraries(torch_custom_op PRIVATE pthreadpool)
