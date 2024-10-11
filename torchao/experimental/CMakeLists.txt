# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
cmake_minimum_required(VERSION 3.19)

project(torchao)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(TORCHAO_OP_EXECUTORCH_BUILD "Building torchao ops for ExecuTorch." OFF)


if(NOT TORCHAO_INCLUDE_DIRS)
  set(TORCHAO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()


include(CMakePrintHelpers)

add_compile_options("-Wall" "-Werror" "-Wno-deprecated")

include(CMakePrintHelpers)
message("TORCHAO_INCLUDE_DIRS: ${TORCHAO_INCLUDE_DIRS}")
include_directories(${TORCHAO_INCLUDE_DIRS})

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    # Defines target torchao_kernels_aarch64
    add_subdirectory(kernels/cpu/aarch64)
    add_subdirectory(ops/linear_8bit_act_xbit_weight)

    add_library(torchao_ops_aten SHARED)
    target_link_libraries(
        torchao_ops_aten PRIVATE
        torchao_ops_linear_8bit_act_xbit_weight_aten
    )
    install(
        TARGETS torchao_ops_aten
        DESTINATION lib
    )
    if(TORCHAO_OP_EXECUTORCH_BUILD)
        add_library(torchao_ops_executorch STATIC)
        target_link_libraries(torchao_ops_executorch PRIVATE torchao_ops_linear_8bit_act_xbit_weight_executorch)

        install(
            TARGETS torchao_ops_executorch
            DESTINATION lib
        )
    endif()
else()
    message(FATAL_ERROR "Torchao experimental ops can only be built on arm64 CPUs.")
endif()
