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

# Hardware-specific backends
option(TORCHAO_BUILD_MPS_OPS "Building torchao MPS ops" OFF)
option(TORCHAO_BUILD_CPU_AARCH64 "Build torchao's CPU aarch64 kernels" OFF)
option(TORCHAO_BUILD_KLEIDIAI "Download, build, and link against Arm KleidiAI library (arm64 only)" OFF)

# Include directories
if(NOT TORCHAO_INCLUDE_DIRS)
    set(TORCHAO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include_directories(${TORCHAO_INCLUDE_DIRS})

# Set compiler flags
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64") 
    add_compile_options("-Wall" "-Werror" "-Wno-deprecated" "-march=armv8.4-a+dotprod" "-fPIC" "-Wno-error=unknown-pragmas" "-Wno-array-parameter" "-Wno-maybe-uninitialized" "-Wno-sign-compare") 
else()
    add_compile_options("-Wall" "-Werror" "-Wno-deprecated")   
endif()

# Find PyTorch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# Parallel backend options
if(NOT DEFINED TORCHAO_PARALLEL_BACKEND)
    set(TORCHAO_PARALLEL_BACKEND aten_openmp)
endif()

if(TORCHAO_BUILD_CPU_AARCH64)
    message(STATUS "Building with cpu/aarch64")
    add_compile_definitions(TORCHAO_BUILD_CPU_AARCH64)
    add_compile_definitions(TORCHAO_ENABLE_ARM_NEON_DOT)

    # Defines torchao_kernels_aarch64
    add_subdirectory(kernels/cpu/aarch64)

    # Add KleidiAI support if enabled
    if(TORCHAO_BUILD_KLEIDIAI)
        message(STATUS "Building with Arm KleidiAI library")
        add_compile_definitions(TORCHAO_ENABLE_KLEIDI)
    endif()
endif()

# Add quantized operation subdirectories
add_subdirectory(ops/linear_8bit_act_xbit_weight)
add_subdirectory(ops/embedding_xbit)

# ATen ops library
add_library(torchao_ops_aten SHARED)
target_link_libraries(
    torchao_ops_aten PRIVATE
    torchao_ops_linear_8bit_act_xbit_weight_aten
    torchao_ops_embedding_xbit_aten
    ${TORCH_LIBRARIES}
)

# Add MPS support if enabled
if(TORCHAO_BUILD_MPS_OPS)
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

# Build ExecuTorch library if enabled
if(TORCHAO_BUILD_EXECUTORCH_OPS)
    add_library(torchao_ops_executorch STATIC)
    target_link_libraries(torchao_ops_executorch PRIVATE
        torchao_ops_linear_8bit_act_xbit_weight_executorch
        torchao_ops_embedding_xbit_executorch
        ${TORCH_LIBRARIES}
    )
    
    # Install ExecuTorch targets
    install(
        TARGETS
            torchao_ops_executorch
            torchao_ops_linear_8bit_act_xbit_weight_executorch
            torchao_ops_embedding_xbit_executorch
        EXPORT _targets
        DESTINATION lib
    )
endif()
