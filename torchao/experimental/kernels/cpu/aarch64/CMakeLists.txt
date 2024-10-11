# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

include(FetchContent)

# KleidiAI is an open-source library that provides optimized
# performance-critical routines, also known as micro-kernels, for artificial
# intelligence (AI) workloads tailored for Arm® CPUs.
FetchContent_Declare(kleidiai
        GIT_REPOSITORY https://git.gitlab.arm.com/kleidi/kleidiai.git
        GIT_TAG 35e156d62d1d7e4d27a39f56ed7770a665628b31) # same as xnnpack for now, TODO - revisit this

FetchContent_MakeAvailable(kleidiai)

# Disabled by default. Force enable if we are on a suitable system.
# TODO: Introduce ISA specific flags for i8mm.
CMAKE_DEPENDENT_OPTION(BUILD_KLEIDI "Download, build, and link against Arm KleidiAI library"
  OFF "CMAKE_SYSTEM_PROCESSOR STREQUAL \"arm64\"" ON)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  add_library(
    torchao_kernels_aarch64
    ${CMAKE_CURRENT_SOURCE_DIR}/reduction/find_min_and_max.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduction/compute_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/valpacking/interleave.cpp
  )
  if (BUILD_KLEIDI)
    # Temporarily exposing this to the parent scope until we wire
    # this up properly from the top level
    set(TORCHAO_ENABLE_KLEIDI ON PARENT_SCOPE)
    message(STATUS "Building with Kleidi")
    target_link_libraries(torchao_kernels_aarch64 PUBLIC kleidiai)
  endif()
endif()

install(
  TARGETS torchao_kernels_aarch64
  DESTINATION lib
)
