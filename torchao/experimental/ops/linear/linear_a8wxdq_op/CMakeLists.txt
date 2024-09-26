# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

include(${TORCHAO_ROOT}/Utils.cmake)

if(TORCHAO_OP_TARGET STREQUAL "ATEN")
    message(STATUS "Building with TORCHAO_OP_TARGET=ATEN")
    find_package(Torch REQUIRED)
    add_library(linear_a8wxdq_${TORCHAO_OP_TARGET} SHARED linear_a8wxdq_aten.cpp)
    target_link_libraries(linear_a8wxdq_${TORCHAO_OP_TARGET} PRIVATE torchao_kernels_aarch64)
    target_link_libraries(linear_a8wxdq_${TORCHAO_OP_TARGET} PRIVATE torchao_ops_linear_${TORCHAO_PARALLEL_BACKEND})
    target_include_directories(linear_a8wxdq_${TORCHAO_OP_TARGET} PRIVATE "${TORCH_INCLUDE_DIRS}")
    target_link_libraries(linear_a8wxdq_${TORCHAO_OP_TARGET} PRIVATE "${TORCH_LIBRARIES}")
    target_compile_definitions(linear_a8wxdq_${TORCHAO_OP_TARGET} PRIVATE USE_ATEN=1)
elseif(TORCHAO_OP_TARGET STREQUAL "EXECUTORCH")
    message(STATUS "Building with TORCHAO_OP_TARGET=EXECUTORCH")
    add_library(linear_a8wxdq_${TORCHAO_OP_TARGET} STATIC
        linear_a8wxdq_executorch/w2s.cpp
        linear_a8wxdq_executorch/w2sz.cpp
        linear_a8wxdq_executorch/w3s.cpp
        linear_a8wxdq_executorch/w3sz.cpp
        linear_a8wxdq_executorch/w4s.cpp
        linear_a8wxdq_executorch/w4sz.cpp
        linear_a8wxdq_executorch/w5s.cpp
        linear_a8wxdq_executorch/w5sz.cpp
    )
    target_include_directories(linear_a8wxdq_${TORCHAO_OP_TARGET} PRIVATE "${EXECUTORCH_INCLUDE_DIRS}")
    target_compile_definitions(linear_a8wxdq_${TORCHAO_OP_TARGET} PRIVATE USE_EXECUTORCH=1)
    target_link_libraries(linear_a8wxdq_${TORCHAO_OP_TARGET} PRIVATE "${EXECUTORCH_LIBRARIES}")
    target_link_libraries(linear_a8wxdq_${TORCHAO_OP_TARGET} PRIVATE torchao_kernels_aarch64)
    target_link_libraries(linear_a8wxdq_${TORCHAO_OP_TARGET} PRIVATE torchao_ops_linear_${TORCHAO_PARALLEL_BACKEND})
else()
    message(FATAL_ERROR "Unknown TORCHAO_OP_TARGET: ${TORCHAO_OP_TARGET}. Please choose one of: ATEN, EXECUTORCH.")
endif()


install(
  TARGETS linear_a8wxdq_${TORCHAO_OP_TARGET}
  DESTINATION lib
)
