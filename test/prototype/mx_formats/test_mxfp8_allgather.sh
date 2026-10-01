#!/bin/bash

# terminate script on first error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# integration tests for TP/SP
if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "CUDA available, proceeding with test."
    exec env NCCL_DEBUG=WARN torchrun --nproc_per_node 2 "${SCRIPT_DIR}/test_mxfp8_allgather.py"

elif python -c 'import torch; assert torch.xpu.is_available()' 2>/dev/null; then
    echo "XPU available, proceeding with test."
    exec torchrun --nproc_per_node 2 "${SCRIPT_DIR}/test_mxfp8_allgather.py"

else
    echo "Skipping test_mxfp8_allgather.sh because no CUDA or XPU devices are available."
    exit 0
fi
