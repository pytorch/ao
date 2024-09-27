# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

include(${TORCHAO_ROOT}/Utils.cmake)

add_library(torchao_ops_linear_${TORCHAO_PARALLEL_BACKEND} STATIC channelwise_8bit_activation_groupwise_lowbit_weight.cpp)
target_link_torchao_parallel_backend(torchao_ops_linear_${TORCHAO_PARALLEL_BACKEND} "${TORCHAO_PARALLEL_BACKEND}")

install(
  TARGETS torchao_ops_linear_${TORCHAO_PARALLEL_BACKEND}
  DESTINATION lib
)
