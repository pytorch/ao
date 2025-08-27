#!/bin/bash -eu
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -eu


target=${1:-"native"}
export CMAKE_OUT=cmake-out

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




export CMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')
echo "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}"


cmake -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_OUT} \
    -DTORCHAO_BUILD_EXECUTORCH_OPS=OFF \
    -DTORCHAO_BUILD_CPU_AARCH64=ON \
    -DTORCHAO_ENABLE_ARM_NEON_DOT=ON \
    -DTORCHAO_BUILD_KLEIDIAI=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTORCHAO_BUILD_TESTS=ON \
    -S . \
    -B ${CMAKE_OUT}
cmake --build  ${CMAKE_OUT} -j 16 --config Debug



echo "Successfully built tests."

if [[ "${target}" != "native" ]]; then
    echo "Skip running tests when cross compiling.";
    exit 0;
fi

# Torch-free aarch64
TEST_TARGET_PREFIX="${CMAKE_OUT}/torch_free_kernels/aarch64/tests/torchao_tests_torch_free_kernels_aarch64_"
${TEST_TARGET_PREFIX}test_quantization
${TEST_TARGET_PREFIX}test_reduction
${TEST_TARGET_PREFIX}test_reduction
${TEST_TARGET_PREFIX}test_bitpacking
${TEST_TARGET_PREFIX}test_linear
${TEST_TARGET_PREFIX}test_embedding
${TEST_TARGET_PREFIX}test_weight_packing
${TEST_TARGET_PREFIX}test_qmatmul
${TEST_TARGET_PREFIX}test_lut
${TEST_TARGET_PREFIX}test_bitpack_fallback_compatibility
${TEST_TARGET_PREFIX}test_embedding_lut

# Torch-free fallback
TEST_TARGET_PREFIX="${CMAKE_OUT}/torch_free_kernels/fallback/tests/torchao_tests_torch_free_kernels_fallback_"
${TEST_TARGET_PREFIX}test_bitpacking

# Shared kernels
TEST_TARGET_PREFIX="${CMAKE_OUT}/shared_kernels/tests/torchao_tests_shared_kernels_"
${TEST_TARGET_PREFIX}test_linear_8bit_act_xbit_weight
${TEST_TARGET_PREFIX}test_groupwise_lowbit_weight_lut
