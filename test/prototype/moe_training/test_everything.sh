# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash

# terminate script on first error
set -e
IS_ROCM=$(rocm-smi --version || true)

# These tests do not work on ROCm yet
if [ -z "$IS_ROCM" ]
then
./test_fsdp.sh
./test_tp.sh
./test_fsdp_tp.sh
fi

echo "all tests successful"
