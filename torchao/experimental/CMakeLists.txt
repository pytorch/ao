# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

project(torchao)

cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


# Source root directory for torchao/experimental
if(NOT TORCHAO_ROOT)
  set(TORCHAO_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(NOT TORCHAO_INCLUDE_DIRS)
  set(TORCHAO_INCLUDE_DIRS ${TORCHAO_ROOT}/../..)
endif()

if (NOT TORCHAO_PARALLEL_BACKEND)
    if (TORCHAO_OP_TARGET STREQUAL "ATEN")
        set(TORCHAO_PARALLEL_BACKEND "ATEN_OPENMP")
    elseif(TORCHAO_OP_TARGET STREQUAL "EXECUTORCH")
        set(TORCHAO_PARALLEL_BACKEND "PTHREADPOOL")
    else()
        message(TORCHAO_PARALLEL_BACKEND "TORCHAO_PARALLEL_BACKEND is not set.  Please set it directly or set TORCHAO_OP_TARGET to get a default.")
    endif()
endif()

include(CMakePrintHelpers)

add_compile_options("-Wall" "-Werror")

include(CMakePrintHelpers)
message("TORCHAO_INCLUDE_DIRS: ${TORCHAO_INCLUDE_DIRS}")
include_directories(${TORCHAO_INCLUDE_DIRS})

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    # Defines target torchao_kernels_aarch64
    add_subdirectory(${TORCHAO_ROOT}/kernels/cpu/aarch64)
    add_subdirectory(${TORCHAO_ROOT}/ops/linear)
    add_subdirectory(${TORCHAO_ROOT}/ops/linear/linear_a8wxdq_op)
endif()
