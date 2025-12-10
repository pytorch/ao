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
option(TORCHAO_BUILD_CPU_AARCH64 "Build torchao's CPU aarch64 kernels" OFF)
option(TORCHAO_BUILD_KLEIDIAI "Download, build, and link against Arm KleidiAI library (arm64 only)" OFF)
option(TORCHAO_ENABLE_ARM_NEON_DOT "Enable ARM Neon Dot Product extension" OFF)
option(TORCHAO_ENABLE_ARM_I8MM "Enable ARM 8-bit Integer Matrix Multiply instructions" OFF)
option(TORCHAO_BUILD_TESTS "Build tests" OFF)
option(TORCHAO_BUILD_BENCHMARKS "Build tests" OFF)

# Set default compiler options
add_compile_options("-fPIC" "-Wall" "-Werror" "-Wno-deprecated")
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_compile_options(
        "-Wno-error=unknown-pragmas"
        "-Wno-array-parameter"
        "-Wno-maybe-uninitialized"
        "-Wno-sign-compare"
    )
elseif (APPLE)
    add_compile_options("-Wno-shorten-64-to-32")
endif()



if (NOT TARGET cpuinfo)
    cmake_policy(PUSH)
    cmake_policy(VERSION 3.5)  # cpuinfo requires CMake 3.5

    # For some reason cpuinfo package has unused functions/variables
    # TODO (T215533422): fix upstream
    add_compile_options(-Wno-unused-function -Wno-unused-variable)

    # set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    include(FetchContent)
    set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "" FORCE)
    set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(cpuinfo
        GIT_REPOSITORY https://github.com/pytorch/cpuinfo.git
        GIT_TAG c61fe919607bbc534d7a5a5707bdd7041e72c5ff
    )
    FetchContent_MakeAvailable(
        cpuinfo)

    cmake_policy(POP)
endif()

if (TORCHAO_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

if (TORCHAO_BUILD_BENCHMARKS)
    include(FetchContent)
    FetchContent_Declare(googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG main) # need main for benchmark::benchmark

    set(BENCHMARK_ENABLE_TESTING OFF)
    FetchContent_MakeAvailable(
        googlebenchmark)
endif()

if(NOT TORCHAO_INCLUDE_DIRS)
  set(TORCHAO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

if(NOT DEFINED TORCHAO_PARALLEL_BACKEND)
    set(TORCHAO_PARALLEL_BACKEND aten_openmp)
endif()

# Set default compiler options

include(CMakePrintHelpers)
include(${CMAKE_CURRENT_SOURCE_DIR}/shared_kernels/Utils.cmake)

message("TORCHAO_INCLUDE_DIRS: ${TORCHAO_INCLUDE_DIRS}")
include_directories(${TORCHAO_INCLUDE_DIRS})


# Build fallback kernels
add_subdirectory(torch_free_kernels/fallback)

# Build cpu/aarch64 kernels
if(TORCHAO_BUILD_CPU_AARCH64)
    message(STATUS "Building with cpu/aarch64")
    add_compile_definitions(TORCHAO_BUILD_CPU_AARCH64)

    if(TORCHAO_ENABLE_ARM_NEON_DOT)
        message(STATUS "Building with ARM NEON dot product support")
        add_compile_definitions(TORCHAO_ENABLE_ARM_NEON_DOT)
        add_compile_options("-march=armv8.4-a+dotprod")
    endif()

    if(TORCHAO_ENABLE_ARM_I8MM)
        message(STATUS "Building with ARM I8MM support")
        add_compile_definitions(TORCHAO_ENABLE_ARM_I8MM)
        add_compile_options("-march=armv8.6-a")
    endif()

    if(TORCHAO_BUILD_KLEIDIAI)
        message(STATUS "Building with Arm KleidiAI library")
        add_compile_definitions(TORCHAO_ENABLE_KLEIDI)
        if (NOT TARGET kleidiai)
            include(FetchContent)
            # KleidiAI is an open-source library that provides optimized
            # performance-critical routines, also known as micro-kernels, for artificial
            # intelligence (AI) workloads tailored for Arm® CPUs.
            set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
            set(KLEIDIAI_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
            FetchContent_Declare(kleidiai
                GIT_REPOSITORY https://git.gitlab.arm.com/kleidi/kleidiai.git
                GIT_TAG v1.12.0
            )
            FetchContent_MakeAvailable(kleidiai)
        endif()
    endif()

    # Defines torchao_kernels_aarch64
    add_subdirectory(torch_free_kernels/aarch64)
endif()

# Build ATen ops
if(TORCHAO_BUILD_ATEN_OPS)
    find_package(Torch REQUIRED)
    set(_torchao_op_srcs_aten)
    list(APPEND _torchao_op_srcs_aten
        shared_kernels/embedding_xbit/op_embedding_xbit_aten.cpp
        shared_kernels/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.cpp
        shared_kernels/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight_aten.cpp
        shared_kernels/groupwise_lowbit_weight_lut/groupwise_lowbit_weight_lut.cpp
        shared_kernels/groupwise_lowbit_weight_lut/op_groupwise_lowbit_weight_lut_aten.cpp
    )
    list(TRANSFORM _torchao_op_srcs_aten PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

    # Use the Python extension name if provided
    add_library(torchao_ops_aten SHARED ${_torchao_op_srcs_aten})
    if(DEFINED TORCHAO_CMAKE_EXT_SO_NAME)
        message(STATUS "Setting output name to: ${TORCHAO_CMAKE_EXT_SO_NAME}.so")
        set_target_properties(torchao_ops_aten PROPERTIES
            OUTPUT_NAME ${TORCHAO_CMAKE_EXT_SO_NAME}
            PREFIX ""  # Remove "lib" prefix for Python extensions
            SUFFIX ".so"  # Add ".so" suffix for Python extensions
        )
    endif()

    target_link_torchao_parallel_backend(torchao_ops_aten "${TORCHAO_PARALLEL_BACKEND}")
    if (TORCHAO_BUILD_CPU_AARCH64)
        target_link_libraries(torchao_ops_aten PRIVATE torchao_kernels_aarch64)
        if (TORCHAO_BUILD_KLEIDIAI)
            target_link_libraries(torchao_ops_aten PRIVATE kleidiai)
        endif()
    endif()
    target_link_libraries(torchao_ops_aten PRIVATE cpuinfo)
    target_include_directories(torchao_ops_aten PRIVATE "${TORCH_INCLUDE_DIRS}")
    target_link_libraries(torchao_ops_aten PRIVATE "${TORCH_LIBRARIES}")
    target_compile_definitions(torchao_ops_aten PRIVATE TORCHAO_SHARED_KERNELS_BUILD_ATEN=1)

    if (TORCHAO_BUILD_TESTS)
        add_subdirectory(shared_kernels/tests)
    endif()

    if (TORCHAO_BUILD_BENCHMARKS)
        add_subdirectory(shared_kernels/benchmarks)
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
        shared_kernels/embedding_xbit/op_embedding_xbit_executorch.cpp
        shared_kernels/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.cpp
        shared_kernels/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight_executorch.cpp
        shared_kernels/groupwise_lowbit_weight_lut/groupwise_lowbit_weight_lut.cpp
        shared_kernels/groupwise_lowbit_weight_lut/op_groupwise_lowbit_weight_lut_executorch.cpp)

    list(TRANSFORM _torchao_op_srcs_executorch PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
    add_library(torchao_ops_executorch STATIC ${_torchao_op_srcs_executorch})

    target_compile_definitions(torchao_ops_executorch PRIVATE TORCHAO_SHARED_KERNELS_BUILD_EXECUTORCH=1)

    # This links to ExecuTorch
    target_link_torchao_parallel_backend(torchao_ops_executorch executorch)
    if (TORCHAO_BUILD_CPU_AARCH64)
        target_link_libraries(torchao_ops_executorch PRIVATE torchao_kernels_aarch64)
        if (TORCHAO_BUILD_KLEIDIAI)
            target_link_libraries(torchao_ops_executorch PRIVATE kleidiai)
        endif()
    endif()
    target_link_libraries(torchao_ops_executorch PRIVATE cpuinfo)
endif()
