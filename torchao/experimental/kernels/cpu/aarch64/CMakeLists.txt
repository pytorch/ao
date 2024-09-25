# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

add_library(
  kernel_aarch64
  ${TORCHAO_INCLUDE_DIRS}/torchao/experimental/kernels/cpu/aarch64/reduction/find_min_and_max.cpp
  ${TORCHAO_INCLUDE_DIRS}/torchao/experimental/kernels/cpu/aarch64/reduction/compute_sum.cpp
  ${TORCHAO_INCLUDE_DIRS}/torchao/experimental/kernels/cpu/aarch64/quantization/quantize.cpp
  ${TORCHAO_INCLUDE_DIRS}/torchao/experimental/kernels/cpu/aarch64/valpacking/interleave.cpp
)
