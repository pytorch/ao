#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

IS_ARM64=0
hash arch; retval=$?
if [[ ${retval} -eq 0 && $(arch) == "arm64" ]]; then
    IS_ARM64=1
fi

export CMAKE_OUT=/tmp/cmake-out/torchao/tests
cmake \
    -DTORCHAO_LIBRARIES=${TORCHAO_LIBRARIES} \
    -DTORCHAO_BUILD_KLEIDIAI=${IS_ARM64} \
    -S . \
    -B ${CMAKE_OUT}

cmake --build ${CMAKE_OUT}

# Run
${CMAKE_OUT}/test_linear_8bit_act_xbit_weight
