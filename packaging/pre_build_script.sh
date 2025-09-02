#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

echo "This script is run before building torchao binaries"

python -m pip install --upgrade pip
if [ -z ${PYTORCH_VERSION:-} ]; then
    PYTORCH_DEP="torch"
else
    PYTORCH_DEP="torch==$PYTORCH_VERSION"
fi
pip install $PYTORCH_DEP

pip install setuptools wheel twine auditwheel scikit-build-core cmake ninja
