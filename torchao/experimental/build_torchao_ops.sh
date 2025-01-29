#!/bin/bash -eu
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <aten|executorch>";
    exit 1;
fi
TARGET="${1}"
export CMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')
echo "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}"
if [[ $TARGET == "executorch" ]]; then
    TORCHAO_BUILD_EXECUTORCH_OPS=ON
else
    TORCHAO_BUILD_EXECUTORCH_OPS=OFF
fi
export CMAKE_OUT=cmake-out
cmake -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_OUT} \
    -DTORCHAO_BUILD_EXECUTORCH_OPS="${TORCHAO_BUILD_EXECUTORCH_OPS}" \
    -S . \
    -B ${CMAKE_OUT}
cmake --build  ${CMAKE_OUT} -j 16 --target install --config Release
