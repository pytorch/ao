#!/bin/bash -eu
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cd "$(dirname "$BASH_SOURCE")"

# export CMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')
export CMAKE_PREFIX_PATH="~/Projects/pytorch/torch"
echo "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}"
export CMAKE_OUT=${PWD}/cmake-out
echo "CMAKE_OUT: ${CMAKE_OUT}"

cmake -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_OUT} \
    -S . \
    -B ${CMAKE_OUT}
cmake --build  ${CMAKE_OUT} -j 16 --target install --config Release
