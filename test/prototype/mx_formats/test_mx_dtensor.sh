#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# terminate script on first error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# integration tests for TP/SP
if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "CUDA available, proceeding with test."
    exec env NCCL_DEBUG=WARN torchrun --nproc_per_node 2 "${SCRIPT_DIR}/test_mx_dtensor.py"

elif python -c 'import torch; assert torch.xpu.is_available()' 2>/dev/null; then
    echo "XPU available, proceeding with test."
    exec torchrun --nproc_per_node 2 "${SCRIPT_DIR}/test_mx_dtensor.py"

else
    echo "Skipping test_mx_dtensor.sh because no CUDA or XPU devices are available."
    exit 0
fi
