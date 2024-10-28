#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

WHEEL_NAME=$(ls dist/)

pushd dist
# Prepare manywheel
manylinux_plat=manylinux2014_x86_64
if [[ "$CU_VERSION" == "xpu" ]]; then
    manylinux_plat=manylinux_2_28_x86_64
fi
auditwheel repair --plat "$manylinux_plat" -w . \
    --exclude libtorch.so \
    --exclude libtorch_python.so \
    --exclude libtorch_cuda.so \
    --exclude libtorch_cpu.so \
    --exclude libc10.so \
    --exclude libc10_cuda.so \
    --exclude libcudart.so.12 \
    --exclude libcudart.so.11.0 \
    "${WHEEL_NAME}"

ls -lah .
# Clean up the linux_x86_64 wheel
rm "${WHEEL_NAME}"
popd

MANYWHEEL_NAME=$(ls dist/)
# Try to install the new wheel
pip install "dist/${MANYWHEEL_NAME}"
python -c "import torchao"
