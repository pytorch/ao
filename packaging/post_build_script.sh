#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

# Prepare manywheel for any non-pure wheel.
WHEELS=(dist/*.whl)
if [[ ${#WHEELS[@]} -gt 0 ]]; then
    case "$(uname -m)" in
        x86_64) manylinux_plat=manylinux_2_28_x86_64 ;;
        aarch64) manylinux_plat=manylinux_2_28_aarch64 ;;
        *) echo "Unsupported arch for auditwheel: $(uname -m)"; exit 1 ;;
    esac

    for WHEEL_PATH in "${WHEELS[@]}"; do
        WHEEL_NAME=$(basename "${WHEEL_PATH}")
        if [[ "${WHEEL_NAME}" == *"none-any.whl" ]]; then
            echo "Skipping pure Python wheel: ${WHEEL_NAME}"
            continue
        fi

        pushd dist
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

        ls -lah .
        # Clean up the original linux_* wheel after repair.
        rm "${WHEEL_NAME}"
        popd
    done
fi

MANYWHEEL_NAME=$(ls dist/)
# Try to install the new wheel
pip install "dist/${MANYWHEEL_NAME}"
python -c "import torchao"
