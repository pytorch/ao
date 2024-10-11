#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install requirements for experimental torchao ops.
if [[ -z $PIP ]];
then
    PIP=pip
fi

NIGHTLY_VERSION="dev20241011"
$PIP install "executorch==0.5.0.$NIGHTLY_VERSION" --extra-index-url https://download.pytorch.org/whl/nightly/cpu
