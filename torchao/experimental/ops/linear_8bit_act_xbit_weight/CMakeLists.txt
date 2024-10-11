# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../Utils.cmake)

find_package(Torch REQUIRED)
add_library(torchao_ops_linear_8bit_act_xbit_weight_aten OBJECT
    linear_8bit_act_xbit_weight.cpp
    op_linear_8bit_act_xbit_weight_aten.cpp
)
target_link_torchao_parallel_backend(torchao_ops_linear_8bit_act_xbit_weight_aten aten_openmp)
target_link_libraries(torchao_ops_linear_8bit_act_xbit_weight_aten PRIVATE torchao_kernels_aarch64)
target_include_directories(torchao_ops_linear_8bit_act_xbit_weight_aten PRIVATE "${TORCH_INCLUDE_DIRS}")
target_link_libraries(torchao_ops_linear_8bit_act_xbit_weight_aten PRIVATE "${TORCH_LIBRARIES}")
target_compile_definitions(torchao_ops_linear_8bit_act_xbit_weight_aten PRIVATE USE_ATEN=1)

if(TORCHAO_OP_EXECUTORCH_BUILD)
    find_package(ExecuTorch REQUIRED HINTS ${CMAKE_PREFIX_PATH}/executorch/share/cmake)
    add_library(torchao_ops_linear_8bit_act_xbit_weight_executorch OBJECT
        linear_8bit_act_xbit_weight.cpp
        op_linear_8bit_act_xbit_weight_executorch/w2s.cpp
        op_linear_8bit_act_xbit_weight_executorch/w2sz.cpp
        op_linear_8bit_act_xbit_weight_executorch/w3s.cpp
        op_linear_8bit_act_xbit_weight_executorch/w3sz.cpp
        op_linear_8bit_act_xbit_weight_executorch/w4s.cpp
        op_linear_8bit_act_xbit_weight_executorch/w4sz.cpp
        op_linear_8bit_act_xbit_weight_executorch/w5s.cpp
        op_linear_8bit_act_xbit_weight_executorch/w5sz.cpp
        op_linear_8bit_act_xbit_weight_executorch/w6s.cpp
        op_linear_8bit_act_xbit_weight_executorch/w6sz.cpp
    )
    target_link_torchao_parallel_backend(torchao_ops_linear_8bit_act_xbit_weight_executorch executorch)
    target_include_directories(torchao_ops_linear_8bit_act_xbit_weight_executorch PRIVATE "${EXECUTORCH_INCLUDE_DIRS}")
    target_compile_definitions(torchao_ops_linear_8bit_act_xbit_weight_executorch PRIVATE USE_EXECUTORCH=1)
    target_link_libraries(torchao_ops_linear_8bit_act_xbit_weight_executorch PRIVATE "${EXECUTORCH_LIBRARIES}")
    target_link_libraries(torchao_ops_linear_8bit_act_xbit_weight_executorch PRIVATE torchao_kernels_aarch64)
endif()
