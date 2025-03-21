#!/bin/bash -eu
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -eu
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
export TORCHAO_LIBRARIES=${SCRIPT_DIR}/../../../../../..
export CMAKE_OUT=/tmp/cmake-out/torch_ao/kernel_tests

target=${1:-"native"}

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

cmake \
    ${EXTRA_ARGS} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTORCHAO_LIBRARIES=${TORCHAO_LIBRARIES} \
    -DTORCHAO_BUILD_CPU_AARCH64=ON \
    -S ${TORCHAO_LIBRARIES}/torchao/experimental/kernels/cpu/aarch64/tests \
    -B ${CMAKE_OUT}

cmake --build ${CMAKE_OUT}

echo "Successfully built tests."

if [[ "${target}" != "native" ]]; then
    echo "Skip running tests when cross compiling.";
    exit 0;
fi

# Run
${CMAKE_OUT}/test_quantization
${CMAKE_OUT}/test_reduction
${CMAKE_OUT}/test_bitpacking
${CMAKE_OUT}/test_linear
${CMAKE_OUT}/test_valpacking
${CMAKE_OUT}/test_embedding
${CMAKE_OUT}/test_weight_packing
