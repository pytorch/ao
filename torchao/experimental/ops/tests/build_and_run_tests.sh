#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

target=${1:-"native"}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
export CMAKE_OUT=/tmp/cmake-out/torch_ao/tests

export TORCH_DIR=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib() + '/torch/share/cmake/Torch')")

IS_ARM64=0
BUILD_ARM_I8MM=0
EXTRA_ARGS=""
if [[ "${target}" == "android" ]]; then
    if [[ -z ${ANDROID_NDK} ]]; then
        echo "Need to set ANDROID_NDK env variable to build for Android";
        exit 1;
    fi
    android_abi=arm64-v8a
    android_platform=28 # must be >=28 for aligned_alloc
    IS_ARM64=1
    BUILD_ARM_I8MM=1 # Hardcoded for now
    CMAKE_OUT=${CMAKE_OUT/cmake-out/cmake-out-android}
    toolchain_file="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
    if [[ -z ${toolchain_file} ]]; then
        echo "Unable to find toolchain file at ANDROID_NDK location, looking for ${toolchain_file}"
        exit 1;
    fi
    EXTRA_ARGS="\
        -DCMAKE_TOOLCHAIN_FILE=${toolchain_file} \
        -DANDROID_ABI=${android_abi} \
        -DANDROID_PLATFORM=${android_platform}
    "
    echo "Building tests for Android (${android_abi}) @ ${CMAKE_OUT}"
fi

hash arch; retval=$?
if [[ ${retval} -eq 0 && $(arch) == "arm64" ]]; then
    IS_ARM64=1
fi

cmake \
    ${EXTRA_ARGS} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTORCHAO_BUILD_CPU_AARCH64=${IS_ARM64} \
    -DTORCHAO_BUILD_KLEIDIAI=${IS_ARM64} \
    -DTORCHAO_BUILD_ARM_I8MM=${BUILD_ARM_I8MM} \
    -DTorch_DIR=${TORCH_DIR} \
    -S . \
    -B ${CMAKE_OUT}

cmake --build ${CMAKE_OUT}

echo "Successfully built tests."

if [[ "${target}" != "native" ]]; then
    echo "Skip running tests when cross compiling.";
    exit 0;
fi

# Run
${CMAKE_OUT}/test_linear_8bit_act_xbit_weight
${CMAKE_OUT}/test_groupwise_lowbit_weight_lut
