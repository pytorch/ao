# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

project(torchao_benchmarks)
set(CMAKE_BUILD_TYPE Release)

set(TARGET_PREFIX "torchao_benchmarks_shared_kernels_")


# TODO: fix benchmark.  Got broken from refactor

# add_executable(${TARGET_PREFIX}benchmark_linear_8bit_act_xbit_weight
#   benchmark_linear_8bit_act_xbit_weight.cpp
#   ${TORCHAO_INCLUDE_DIRS}/torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.cpp
# )

# target_link_torchao_parallel_backend(${TARGET_PREFIX}benchmark_linear_8bit_act_xbit_weight openmp)
# target_link_libraries(
#     ${TARGET_PREFIX}benchmark_linear_8bit_act_xbit_weight
#     PRIVATE
#     benchmark::benchmark
#     torchao_kernels_aarch64
# )
