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
auditwheel repair --plat manylinux2014_x86_64 -w . \
    --exclude libtorch.so \
    --exclude libtorch_python.so \
    --exclude libtorch_cuda.so \
    --exclude libtorch_cpu.so \
    --exclude libc10.so \
    --exclude libc10_cuda.so \
    "${WHEEL_NAME}"

ls -lah .
# Clean up the linux_x86_64 wheel
rm "${WHEEL_NAME}"
popd
