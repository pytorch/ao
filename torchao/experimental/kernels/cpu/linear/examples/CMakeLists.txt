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


add_executable(separate_function_wrappers separate_function_wrappers.cpp)
target_link_libraries(
  separate_function_wrappers
    PRIVATE
    torchao_dep
)

add_executable(stateful_class_wrapper stateful_class_wrapper.cpp)
target_link_libraries(
  stateful_class_wrapper
    PRIVATE
    torchao_dep
)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(separate_function_wrappers PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(separate_function_wrappers PRIVATE TORCHAO_PARALLEL_OMP=1)

  target_link_libraries(stateful_class_wrapper PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(stateful_class_wrapper PRIVATE TORCHAO_PARALLEL_OMP=1)
else()
  target_compile_definitions(separate_function_wrappers PRIVATE TORCHAO_PARALLEL_SINGLE_THREADED=1)
  target_compile_definitions(stateful_class_wrapper PRIVATE TORCHAO_PARALLEL_SINGLE_THREADED=1)
endif()
