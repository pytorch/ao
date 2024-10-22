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
