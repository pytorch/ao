# Copyright (c) Meta Platforms, Inc. and affiliates.
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

# CI-ONLY: Disable PEP 517 build isolation to reuse pre-installed build tools
# from pre_build_script.sh (setuptools, wheel), significantly reducing build time.
# This is safe in CI because the build environment is controlled and reproducible.
# WARNING: Do not use this in local development unless you understand the implications
# for build reproducibility and dependency management.
export PIP_NO_BUILD_ISOLATION=1
