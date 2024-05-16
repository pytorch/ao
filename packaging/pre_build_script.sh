#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

echo "This script is run before building torchao binaries"

pip install torch setuptools wheel twine
pip install -r requirements.txt
pip install -r dev-requirements.txt
