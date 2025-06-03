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
option(TORCHAO_BUILD_ATEN_OPS "Building torchao ops for ATen." ON)
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

# Set default compiler options
add_compile_options("-Wall" "-Werror" "-Wno-deprecated" "-Wno-shorten-64-to-32")

include(CMakePrintHelpers)
include(${CMAKE_CURRENT_SOURCE_DIR}/Utils.cmake)

message("TORCHAO_INCLUDE_DIRS: ${TORCHAO_INCLUDE_DIRS}")
include_directories(${TORCHAO_INCLUDE_DIRS})


# Build cpu/aarch64 kernels
if(TORCHAO_BUILD_CPU_AARCH64)
    message(STATUS "Building with cpu/aarch64")
    add_compile_definitions(TORCHAO_BUILD_CPU_AARCH64)

    # Set aarch64 compiler options
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
            message(STATUS "Using armv8.6-a (includes 'i8mm' and 'dotprod' flags)")
            add_compile_options("-march=armv8.6-a")
        elseif(TORCHAO_ENABLE_ARM_NEON_DOT)
            message(STATUS "Using armv8.4-a (includes '+dotprod' flag)")
            add_compile_options("-march=armv8.4-a")
        endif()
    endif()

    if(TORCHAO_ENABLE_ARM_NEON_DOT)
        message(STATUS "Building with ARM NEON dot product support")
        add_compile_definitions(TORCHAO_ENABLE_ARM_NEON_DOT)
        add_compile_options("-march=armv8.4-a+dotprod")
    endif()

    if(TORCHAO_ENABLE_ARM_I8MM)
        message(STATUS "Building with ARM I8MM support")
        add_compile_definitions(TORCHAO_ENABLE_ARM_I8MM)
    endif()

    if(TORCHAO_BUILD_KLEIDIAI)
        message(STATUS "Building with Arm KleidiAI library")
        add_compile_definitions(TORCHAO_ENABLE_KLEIDI)
    endif()

    # Defines torchao_kernels_aarch64
    add_subdirectory(kernels/cpu/aarch64)
endif()



if (NOT TARGET cpuinfo)
    # For some reason cpuinfo package has unused functions/variables
    # TODO (T215533422): fix upstream
    add_compile_options(-Wno-unused-function -Wno-unused-variable)
    include(FetchContent)
    FetchContent_Declare(cpuinfo
        GIT_REPOSITORY https://github.com/pytorch/cpuinfo.git
        GIT_TAG c61fe919607bbc534d7a5a5707bdd7041e72c5ff)
    FetchContent_MakeAvailable(
        cpuinfo)
endif()

# Build ATen ops
if(TORCHAO_BUILD_ATEN_OPS)
    find_package(Torch REQUIRED)
    set(_torchao_op_srcs_aten)
    list(APPEND _torchao_op_srcs_aten
        ops/embedding_xbit/op_embedding_xbit_aten.cpp
        ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.cpp
        ops/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight_aten.cpp
    )
    list(TRANSFORM _torchao_op_srcs_aten PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
    add_library(torchao_ops_aten SHARED ${_torchao_op_srcs_aten})
    target_link_torchao_parallel_backend(torchao_ops_aten "${TORCHAO_PARALLEL_BACKEND}")
    if (TORCHAO_BUILD_CPU_AARCH64)
        target_link_libraries(torchao_ops_aten PRIVATE torchao_kernels_aarch64)
    endif()
    target_link_libraries(torchao_ops_aten PRIVATE cpuinfo)
    target_include_directories(torchao_ops_aten PRIVATE "${TORCH_INCLUDE_DIRS}")
    target_link_libraries(torchao_ops_aten PRIVATE "${TORCH_LIBRARIES}")
    target_compile_definitions(torchao_ops_aten PRIVATE USE_ATEN=1)

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
endif()


# Build ExecuTorch ops
if(TORCHAO_BUILD_EXECUTORCH_OPS)
    # ExecuTorch package is not required, but EXECUTORCH_INCLUDE_DIRS and EXECUTORCH_LIBRARIES must
    # be defined and EXECUTORCH_LIBRARIES must include the following libraries installed by ExecuTorch:
    # libexecutorch.a
    # libextension_threadpool.a
    # libcpuinfo.a
    # libpthreadpool.a
    if(NOT DEFINED EXECUTORCH_INCLUDE_DIRS AND NOT DEFINED EXECUTORCH_LIBRARIES)
        message(WARNING "EXECUTORCH_INCLUDE_DIRS and EXECUTORCH_LIBRARIES are not defined. Looking for ExecuTorch.")
        find_package(ExecuTorch HINTS ${CMAKE_PREFIX_PATH}/executorch/share/cmake)
    endif()
    set(_torchao_op_srcs_executorch)
    list(APPEND _torchao_op_srcs_executorch
        ops/embedding_xbit/op_embedding_xbit_executorch.cpp
        ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.cpp
        ops/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight_executorch.cpp
    )
    list(TRANSFORM _torchao_op_srcs_executorch PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
    add_library(torchao_ops_executorch STATIC ${_torchao_op_srcs_executorch})
    target_link_torchao_parallel_backend(torchao_ops_executorch executorch)
    target_include_directories(torchao_ops_executorch PRIVATE "${EXECUTORCH_INCLUDE_DIRS}")
    target_compile_definitions(torchao_ops_executorch PRIVATE USE_EXECUTORCH=1)
    target_link_libraries(torchao_ops_executorch PRIVATE "${EXECUTORCH_LIBRARIES}")
    if (TORCHAO_BUILD_CPU_AARCH64)
        target_link_libraries(torchao_ops_executorch PRIVATE torchao_kernels_aarch64)
    endif()
    target_link_libraries(torchao_ops_executorch PRIVATE cpuinfo)
    install(
        TARGETS
            torchao_ops_executorch
        EXPORT _targets
        DESTINATION lib
    )
endif()
