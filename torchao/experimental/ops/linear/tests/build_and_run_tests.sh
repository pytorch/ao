#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

export CMAKE_OUT=/tmp/cmake-out/torchao/tests
cmake -DTORCHAO_LIBRARIES=${TORCHAO_LIBRARIES} -S . -B ${CMAKE_OUT}

cmake --build  ${CMAKE_OUT}

# Run
${CMAKE_OUT}/test_linear_operator
