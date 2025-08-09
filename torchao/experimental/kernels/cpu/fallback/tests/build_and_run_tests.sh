#!/bin/bash -eu
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -eu
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
export TORCHAO_LIBRARIES=${SCRIPT_DIR}/../../../../../..
export CMAKE_OUT=/tmp/cmake-out/torch_ao/kernel_fallback_tests

target=${1:-"native"}

EXTRA_ARGS=""

cmake \
    ${EXTRA_ARGS} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTORCHAO_LIBRARIES=${TORCHAO_LIBRARIES} \
    -DTORCHAO_BUILD_CPU_AARCH64=ON \
    -S ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/fallback/tests \
    -B ${CMAKE_OUT}

cmake --build ${CMAKE_OUT}

echo "Successfully built tests."

if [[ "${target}" != "native" ]]; then
    echo "Skip running tests when cross compiling.";
    exit 0;
fi

# Run
${CMAKE_OUT}/test_bitpacking
