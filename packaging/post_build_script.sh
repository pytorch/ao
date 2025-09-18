#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

# Prepare manywheel, only for CUDA.
# The wheel is a pure python wheel for other platforms.
if [[ "$CU_VERSION" == cu* ]]; then
    WHEEL_NAME=$(ls dist/)

    pushd dist
    manylinux_plat=manylinux_2_28_x86_64
    # Only run auditwheel if the wheel contains at least one shared object (.so)
    if unzip -l "$WHEEL_NAME" | awk '{print $4}' | grep -E '\\.so($|\.)' >/dev/null 2>&1; then
        auditwheel repair --plat "$manylinux_plat" -w . \
    --exclude libtorch.so \
    --exclude libtorch_python.so \
    --exclude libtorch_cuda.so \
    --exclude libtorch_cpu.so \
    --exclude libc10.so \
    --exclude libc10_cuda.so \
    --exclude libcuda.so.* \
    --exclude libcudart.so.* \
    "${WHEEL_NAME}"
    else
        echo "No shared libraries detected in wheel ${WHEEL_NAME}; skipping auditwheel."
    fi

    ls -lah .
    # Clean up the linux_x86_64 wheel
    rm -f "${WHEEL_NAME}"
    popd
fi

MANYWHEEL_NAME=$(ls dist/)
# Try to install the new wheel
pip install "dist/${MANYWHEEL_NAME}"
python -c "import torchao"
