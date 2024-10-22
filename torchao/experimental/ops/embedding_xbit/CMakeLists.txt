# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../Utils.cmake)

find_package(Torch REQUIRED)
add_library(torchao_ops_embedding_xbit_aten OBJECT
    op_embedding_xbit_aten.cpp
)
target_link_torchao_parallel_backend(torchao_ops_embedding_xbit_aten "aten_openmp")
target_link_libraries(torchao_ops_embedding_xbit_aten PRIVATE torchao_kernels_aarch64)
target_include_directories(torchao_ops_embedding_xbit_aten PRIVATE "${TORCH_INCLUDE_DIRS}")
target_link_libraries(torchao_ops_embedding_xbit_aten PRIVATE "${TORCH_LIBRARIES}")
target_compile_definitions(torchao_ops_embedding_xbit_aten PRIVATE USE_ATEN=1)

if(TORCHAO_BUILD_EXECUTORCH_OPS)
    # ExecuTorch package is not required, but EXECUTORCH_INCLUDE_DIRS and EXECUTORCH_LIBRARIES must
    # be defined and EXECUTORCH_LIBRARIES must include the following libraries installed by ExecuTorch:
    # libexecutorch.a
    # libextension_threadpool.a
    # libcpuinfo.a
    # libpthreadpool.a
    add_library(torchao_ops_embedding_xbit_executorch OBJECT
        op_embedding_xbit_executorch.cpp
    )
    target_link_torchao_parallel_backend(torchao_ops_embedding_xbit_executorch executorch)
    target_include_directories(torchao_ops_embedding_xbit_executorch PRIVATE "${EXECUTORCH_INCLUDE_DIRS}")
    target_compile_definitions(torchao_ops_embedding_xbit_executorch PRIVATE USE_EXECUTORCH=1)
    target_link_libraries(torchao_ops_embedding_xbit_executorch PRIVATE "${EXECUTORCH_LIBRARIES}")
    target_link_libraries(torchao_ops_embedding_xbit_executorch PRIVATE torchao_kernels_aarch64)
endif()
