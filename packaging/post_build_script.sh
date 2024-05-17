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
auditwheel repair "${WHEEL_NAME}"
ls -lah
popd
