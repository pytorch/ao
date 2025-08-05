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
endif()
