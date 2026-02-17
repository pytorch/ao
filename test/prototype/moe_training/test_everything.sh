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
pytest test/prototype/moe_training/test_kernels.py -s -v
pytest test/prototype/moe_training/test_scaled_grouped_mm.py -s -v
pytest test/prototype/moe_training/test_training.py -s -v
pytest test/prototype/moe_training/test_fqn_to_config.py -s -v
./test/prototype/moe_training/test_distributed.sh -s -v
fi

echo "all tests successful"
