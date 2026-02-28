# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is sourced into the environment before building a pip wheel. It
# should typically only contain shell variable assignments. Be sure to export
# any variables so that subprocesses will see them.
if [[ ${CHANNEL:-nightly} == "nightly" ]]; then
  export TORCHAO_NIGHTLY=1
fi

# Set ARCH list so that we can build fp16 with SM75+, the logic is copied from
# pytorch/builder
TORCH_CUDA_ARCH_LIST="8.0;8.6"
if [[ ${CU_VERSION:-} == "cu124" ]]; then
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};9.0"
fi

# Enable C++ kernels + kleidiai in aarch64 build
if [[ $(uname -m) == "aarch64" ]]; then
    echo "Enabling aarch64-specific build"
    export USE_CPP=1
    export USE_CPU_KERNELS=1
    export TORCHAO_BUILD_KLEIDIAI=1
    export TORCHAO_BUILD_CPU_AARCH64=1
    export BUILD_TORCHAO_EXPERIMENTAL=1
    export TORCHAO_ENABLE_ARM_NEON_DOT=1
    echo " - USE_CPP: $USE_CPP"
    echo " - USE_CPU_KERNELS: $USE_CPU_KERNELS"
    echo " - TORCHAO_BUILD_KLEIDIAI: $TORCHAO_BUILD_KLEIDIAI"
    echo " - TORCHAO_BUILD_CPU_AARCH64: $TORCHAO_BUILD_CPU_AARCH64"
    echo " - TORCHAO_ENABLE_ARM_NEON_DOT: $TORCHAO_ENABLE_ARM_NEON_DOT"
    echo " - BUILD_TORCHAO_EXPERIMENTAL: $BUILD_TORCHAO_EXPERIMENTAL"
fi
