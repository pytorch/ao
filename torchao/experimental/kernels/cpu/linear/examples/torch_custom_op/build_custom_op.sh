#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
export TORCHAO_INCLUDE_DIRS=${SCRIPT_DIR}/../../../../../../..

export CMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"
echo "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}"
export CMAKE_OUT=/tmp/cmake-out/torchao
cmake -DTORCHAO_INCLUDE_DIRS=${TORCHAO_INCLUDE_DIRS} \
    -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} \
    -DPLATFORM="ATEN" \
    -S ${TORCHAO_INCLUDE_DIRS}/torchao/experimental/kernels/cpu/linear/examples/torch_custom_op \
    -B ${CMAKE_OUT}
cmake --build  ${CMAKE_OUT}
