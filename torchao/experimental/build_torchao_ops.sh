#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

export CMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"
echo "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}"
export CMAKE_OUT=/tmp/cmake-out/torchao
cmake -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_OUT} \
    -DTORCHAO_OP_TARGET="$1" \
    -DEXECUTORCH_LIBRARIES=${EXECUTORCH_LIBRARIES} \
    -DEXECUTORCH_INCLUDE_DIRS=${EXECUTORCH_INCLUDE_DIRS} \
    -S . \
    -B ${CMAKE_OUT}
cmake --build  ${CMAKE_OUT} --target install --config Release
