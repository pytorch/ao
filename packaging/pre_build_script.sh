#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

echo "This script is run before building torchao binaries"

pip install setuptools wheel twine auditwheel

# NB: This will make the nightly wheel compatible with the latest stable torch,
# maybe we could remove this line to make it compatible with PyTorch nightly
# instead
pip uninstall torch
if [[ "${CU_VERSION:-}" == "cu118" ]]; then
  pip install torch --index-url "https://download.pytorch.org/whl/${CU_VERSION}"
else
  pip install torch
fi

pip install -r requirements.txt
pip install -r dev-requirements.txt
