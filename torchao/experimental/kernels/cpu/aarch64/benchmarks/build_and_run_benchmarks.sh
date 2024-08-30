#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Call script with sh build_and_run_benchmarks.sh {BENCHAMRK}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
export TORCHAO_LIBRARIES=${SCRIPT_DIR}/../../../../../..
export CMAKE_OUT=/tmp/cmake-out/torch_ao/benchmarks
cmake -DTORCHAO_LIBRARIES=${TORCHAO_LIBRARIES} \
    -S ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/benchmarks \
    -B ${CMAKE_OUT}

cmake --build ${CMAKE_OUT}

# Run
case "$1" in
    quantization) ${CMAKE_OUT}/benchmark_quantization; ;;
    bitpacking) ${CMAKE_OUT}/benchmark_bitpacking; ;;
    linear) ${CMAKE_OUT}/benchmark_linear; ;;
    *) echo "Unknown benchmark: $1. Please specify quantization, bitpacking, or linear."; exit 1; ;;
esac
