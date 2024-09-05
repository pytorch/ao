#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
export TORCHAO_LIBRARIES=${SCRIPT_DIR}/../../../../../..
export CMAKE_OUT=/tmp/cmake-out/torch_ao/tests
cmake -DTORCHAO_LIBRARIES=${TORCHAO_LIBRARIES} -S ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/linear/tests -B ${CMAKE_OUT}

cmake --build  ${CMAKE_OUT}

# Run
${CMAKE_OUT}/test_linear_operator
