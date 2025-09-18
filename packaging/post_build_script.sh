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
    pushd dist
    # Determine the original wheel produced by build (there should be exactly one)
    ORIG_WHEEL=$(ls -1 *.whl | head -n 1)
    manylinux_plat=manylinux_2_28_x86_64
    # Only run auditwheel if the wheel contains at least one shared object (.so)
    if unzip -l "$ORIG_WHEEL" | awk '{print $4}' | grep -E '\\.so($|\.)' >/dev/null 2>&1; then
        auditwheel repair --plat "$manylinux_plat" -w . \
    --exclude libtorch.so \
    --exclude libtorch_python.so \
    --exclude libtorch_cuda.so \
    --exclude libtorch_cpu.so \
    --exclude libc10.so \
    --exclude libc10_cuda.so \
    --exclude libcuda.so.* \
    --exclude libcudart.so.* \
        "${ORIG_WHEEL}"
    else
        echo "No shared libraries detected in wheel ${ORIG_WHEEL}; skipping auditwheel."
    fi

    ls -lah .
    popd
fi

INSTALL_WHEEL=$(ls -1t dist/*.whl | head -n 1)
# Try to install the new wheel (pick most recent wheel file)
pip install "${INSTALL_WHEEL}"
python -c "import torchao"
