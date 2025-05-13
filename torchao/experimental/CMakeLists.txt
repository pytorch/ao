# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
include(CMakeDependentOption)

project(torchao)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform options
option(TORCHAO_BUILD_EXECUTORCH_OPS "Building torchao ops for ExecuTorch." OFF)
option(TORCHAO_BUILD_MPS_OPS "Building torchao MPS ops" OFF)
option(TORCHAO_BUILD_CPU_AARCH64 "Build torchao's CPU aarch64 kernels" OFF)
option(TORCHAO_BUILD_KLEIDIAI "Download, build, and link against Arm KleidiAI library (arm64 only)" OFF)
option(TORCHAO_ENABLE_ARM_NEON_DOT "Enable ARM Neon Dot Product extension" OFF)
option(TORCHAO_ENABLE_ARM_I8MM "Enable ARM 8-bit Integer Matrix Multiply instructions" OFF)

if(NOT TORCHAO_INCLUDE_DIRS)
  set(TORCHAO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

if(NOT DEFINED TORCHAO_PARALLEL_BACKEND)
    set(TORCHAO_PARALLEL_BACKEND aten_openmp)
endif()

# Set compiler flags
add_compile_options("-Wall" "-Werror" "-Wno-deprecated")

include(CMakePrintHelpers)
message("TORCHAO_INCLUDE_DIRS: ${TORCHAO_INCLUDE_DIRS}")
include_directories(${TORCHAO_INCLUDE_DIRS})

if(TORCHAO_BUILD_CPU_AARCH64)
    message(STATUS "Building with cpu/aarch64")
    add_compile_definitions(TORCHAO_BUILD_CPU_AARCH64)

    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(STATUS "Add aarch64 linux compiler options")
        add_compile_options(
            "-fPIC"
            "-Wno-error=unknown-pragmas"
            "-Wno-array-parameter"
            "-Wno-maybe-uninitialized"
            "-Wno-sign-compare"
        )

        # Since versions are hierarchical (each includes features from prior versions):
        # - dotprod is included by default in armv8.4-a and later
        # - i8mm is included by default in armv8.6-a and later
        if(TORCHAO_ENABLE_ARM_I8MM)
            message(STATUS "Using armv8.6-a (includes I8MM and NEON dot product support)")
            add_compile_options("-march=armv8.6-a")
        elseif(TORCHAO_ENABLE_ARM_NEON_DOT)
            message(STATUS "Using armv8.4-a (includes NEON dot product support)")
            add_compile_options("-march=armv8.4-a")
        endif()
    endif()

    if(TORCHAO_ENABLE_ARM_NEON_DOT)
        message(STATUS "Building with ARM NEON dot product support")
        add_compile_definitions(TORCHAO_ENABLE_ARM_NEON_DOT)
    endif()

    if(TORCHAO_ENABLE_ARM_I8MM)
        message(STATUS "Building with ARM I8MM support")
        add_compile_definitions(TORCHAO_ENABLE_ARM_I8MM)
    endif()

    # Defines torchao_kernels_aarch64
    add_subdirectory(kernels/cpu/aarch64)

    if(TORCHAO_BUILD_KLEIDIAI)
        message(STATUS "Building with Arm KleidiAI library")
        add_compile_definitions(TORCHAO_ENABLE_KLEIDI)
    endif()
endif()

# Add quantized operation dir
add_subdirectory(ops/linear_8bit_act_xbit_weight)
add_subdirectory(ops/embedding_xbit)

# ATen ops lib
add_library(torchao_ops_aten SHARED)
target_link_libraries(
    torchao_ops_aten PRIVATE
    torchao_ops_linear_8bit_act_xbit_weight_aten
    torchao_ops_embedding_xbit_aten
)

# Add MPS support if enabled
if (TORCHAO_BUILD_MPS_OPS)
    message(STATUS "Building with MPS support")
    add_subdirectory(ops/mps)
    target_link_libraries(torchao_ops_aten PRIVATE torchao_ops_mps_aten)
endif()

# Install ATen targets
install(
    TARGETS torchao_ops_aten
    EXPORT _targets
    DESTINATION lib
)

# Build executorch lib if enabled
if(TORCHAO_BUILD_EXECUTORCH_OPS)
    add_library(torchao_ops_executorch STATIC)
    target_link_libraries(torchao_ops_executorch PRIVATE
        torchao_ops_linear_8bit_act_xbit_weight_executorch
        torchao_ops_embedding_xbit_executorch
    )
    install(
        TARGETS
            torchao_ops_executorch
            torchao_ops_linear_8bit_act_xbit_weight_executorch
            torchao_ops_embedding_xbit_executorch
        EXPORT _targets
        DESTINATION lib
    )
endif()
