# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

include(${TORCHAO_ROOT}/Utils.cmake)


if(TORCHAO_OP_TARGET STREQUAL "aten")
    message(STATUS "Building with TORCHAO_OP_TARGET=aten")
    find_package(Torch REQUIRED)
    add_library(torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET} OBJECT
        linear_8bit_act_xbit_weight.cpp
        op_linear_8bit_act_xbit_weight_aten.cpp
    )
    target_link_torchao_parallel_backend(torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET} "${TORCHAO_PARALLEL_BACKEND}")
    target_link_libraries(torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET} PRIVATE torchao_kernels_aarch64)
    target_include_directories(torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET} PRIVATE "${TORCH_INCLUDE_DIRS}")
    target_link_libraries(torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET} PRIVATE "${TORCH_LIBRARIES}")
    target_compile_definitions(torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET} PRIVATE USE_ATEN=1)
elseif(TORCHAO_OP_TARGET STREQUAL "executorch")
    message(STATUS "Building with TORCHAO_OP_TARGET=executorch")
    add_library(torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET} OBJECT
        linear_8bit_act_xbit_weight.cpp
        op_linear_8bit_act_xbit_weight_executorch/w2s.cpp
        op_linear_8bit_act_xbit_weight_executorch/w2sz.cpp
        op_linear_8bit_act_xbit_weight_executorch/w3s.cpp
        op_linear_8bit_act_xbit_weight_executorch/w3sz.cpp
        op_linear_8bit_act_xbit_weight_executorch/w4s.cpp
        op_linear_8bit_act_xbit_weight_executorch/w4sz.cpp
        op_linear_8bit_act_xbit_weight_executorch/w5s.cpp
        op_linear_8bit_act_xbit_weight_executorch/w5sz.cpp
    )
    target_link_torchao_parallel_backend(torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET} "${TORCHAO_PARALLEL_BACKEND}")
    target_include_directories(torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET} PRIVATE "${EXECUTORCH_INCLUDE_DIRS}")
    target_compile_definitions(torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET} PRIVATE USE_EXECUTORCH=1)
    target_link_libraries(torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET} PRIVATE "${EXECUTORCH_LIBRARIES}")
    target_link_libraries(torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET} PRIVATE torchao_kernels_aarch64)
else()
    message(FATAL_ERROR "Unknown TORCHAO_OP_TARGET: ${TORCHAO_OP_TARGET}. Please choose one of: aten, executorch.")
endif()
