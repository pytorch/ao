# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

if (TORCHAO_BUILD_CPU_AARCH64)
  add_library(
    torchao_kernels_aarch64
    ${CMAKE_CURRENT_SOURCE_DIR}/reduction/find_min_and_max.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduction/compute_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/valpacking/interleave.cpp
  )
  if (TORCHAO_BUILD_KLEIDIAI)
    include(FetchContent)
    # KleidiAI is an open-source library that provides optimized
    # performance-critical routines, also known as micro-kernels, for artificial
    # intelligence (AI) workloads tailored for Arm® CPUs.
    FetchContent_Declare(kleidiai
          GIT_REPOSITORY https://git.gitlab.arm.com/kleidi/kleidiai.git
          GIT_TAG v1.2.0)
    FetchContent_MakeAvailable(kleidiai)

    target_link_libraries(torchao_kernels_aarch64 PUBLIC kleidiai)
  endif()

install(
  TARGETS torchao_kernels_aarch64
  DESTINATION lib
)
endif()
