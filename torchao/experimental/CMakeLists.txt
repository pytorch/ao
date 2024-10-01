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

if (NOT TORCHAO_OP_TARGET)
    message(FATAL_ERROR "TORCHAO_OP_TARGET is not set.  Set it to aten or executorch.")
endif()

if (NOT TORCHAO_PARALLEL_BACKEND)
    if (TORCHAO_OP_TARGET STREQUAL "aten")
        set(TORCHAO_PARALLEL_BACKEND "aten_openmp")
    elseif(TORCHAO_OP_TARGET STREQUAL "executorch")
        set(TORCHAO_PARALLEL_BACKEND "executorch")
    else()
        message(TORCHAO_PARALLEL_BACKEND "TORCHAO_PARALLEL_BACKEND is not set.  Please set it directly or set TORCHAO_OP_TARGET to get a default.")
    endif()
endif()

include(CMakePrintHelpers)

add_compile_options("-Wall" "-Werror")

include(CMakePrintHelpers)
message("TORCHAO_INCLUDE_DIRS: ${TORCHAO_INCLUDE_DIRS}")
include_directories(${TORCHAO_INCLUDE_DIRS})

if(TORCHAO_OP_TARGET STREQUAL "aten")
    add_library(torchao_ops_${TORCHAO_OP_TARGET} SHARED)
elseif(TORCHAO_OP_TARGET STREQUAL "executorch")
    add_library(torchao_ops_${TORCHAO_OP_TARGET} STATIC)
else()
    message(FATAL_ERROR "Unknown TORCHAO_OP_TARGET: ${TORCHAO_OP_TARGET}. Please choose one of: aten, executorch.")
endif()

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    # Defines target torchao_kernels_aarch64
    add_subdirectory(${TORCHAO_ROOT}/kernels/cpu/aarch64)
    add_subdirectory(${TORCHAO_ROOT}/ops/linear_8bit_act_xbit_weight)

    target_link_libraries(
        torchao_ops_${TORCHAO_OP_TARGET} PRIVATE
        torchao_ops_linear_8bit_act_xbit_weight_${TORCHAO_OP_TARGET}
    )
endif()

install(
  TARGETS  torchao_ops_${TORCHAO_OP_TARGET}
  DESTINATION lib
)
