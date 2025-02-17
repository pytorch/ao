#!/bin/bash -eu
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

export CMAKE_OUT=cmake-out
cmake -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_OUT} \
    -DTORCHAO_BUILD_EXECUTORCH_OPS="${TORCHAO_BUILD_EXECUTORCH_OPS}" \
    -S . \
    -B ${CMAKE_OUT}
cmake --build  ${CMAKE_OUT} -j 16 --target install --config Release

./cmake-out/main
