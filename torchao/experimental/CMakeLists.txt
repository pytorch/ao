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
option(TORCHAO_BUILD_MPS_OPS "Building torchao MPS ops" OFF)

if(NOT TORCHAO_INCLUDE_DIRS)
  set(TORCHAO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

if(NOT DEFINED TORCHAO_PARALLEL_BACKEND)
    set(TORCHAO_PARALLEL_BACKEND aten_openmp)
endif()

# Set default compiler options
add_compile_options("-Wall" "-Werror" "-Wno-deprecated" "-Wno-shorten-64-to-32")

include(CMakePrintHelpers)

message("TORCHAO_INCLUDE_DIRS: ${TORCHAO_INCLUDE_DIRS}")
include_directories(${TORCHAO_INCLUDE_DIRS})


# Build ATen ops
if(TORCHAO_BUILD_ATEN_OPS)
    find_package(Torch REQUIRED)

    # Use the Python extension name if provided
    add_library(torchao_ops_aten SHARED)

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
