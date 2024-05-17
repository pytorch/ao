#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

echo "This script is run before building torchao binaries"

pip install setuptools wheel twine auditwheel
# NB: This will make the nightly wheel compatible with torch 2.3.0, maybe we could
# remove this to make it compatible with PyTorch nighly instead
pip install torch==2.3.0
pip install -r requirements.txt
pip install -r dev-requirements.txt
