# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(torchao_core CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find PyTorch package
find_package(Torch REQUIRED)

# Global compile definitions
add_compile_definitions(Py_LIMITED_API=0x03090000)

# Set compiler flags based on platform and build type
if(MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /ZI /DEBUG)
    else()
        add_compile_options(/O2 /permissive-)
    endif()
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
        if(CMAKE_CUDA_COMPILER)
            add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-g>)
        endif()
    else()
        add_compile_options(-O3)
        if(CMAKE_CUDA_COMPILER)
            add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
        endif()
    endif()

    # Add color diagnostics for non-Windows builds
    add_compile_options(-fdiagnostics-color=always)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# CUDA Setup
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DTORCH_CUDA_AVAILABLE)

    # Set CUDA architectures if not already set
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    endif()

    # Set CUDA flags
    # Set CUDA architectures and TORCH_CUDA_ARCH_LIST
    if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
        set(TORCH_CUDA_ARCH_LIST "7.0;7.5;8.0;8.6;9.0")
    endif()

    # CUTLASS support for non-Windows CUDA builds
    if(NOT WIN32)
        set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass)
        if(EXISTS ${CUTLASS_DIR})
            add_definitions(-DTORCHAO_USE_CUTLASS)
            include_directories(${CUTLASS_DIR}/include)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DTORCHAO_USE_CUTLASS -I${CUTLASS_DIR}/include")
        endif()
    endif()
endif()

# Find source files
file(GLOB_RECURSE CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/**/*.cpp")
if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/torchao/csrc/cuda/**/*.cu")
endif()

# Create the core library
add_library(torchao_core SHARED
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(torchao_core PRIVATE
    ${TORCH_LIBRARIES}
)

# Set Python limited API version
target_compile_definitions(torchao_core PRIVATE Py_LIMITED_API=0x03090000)

# Installation
install(TARGETS torchao_core
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
)
