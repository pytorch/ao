# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash

# terminate script on first error
set -e

if ! python - <<'PY'
import sys
import torch
has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
has_cuda = torch.cuda.is_available()
sys.exit(0 if (has_xpu or has_cuda) else 1)
PY
then
    echo "Skipping test_dtensor.sh because no XPU/CUDA devices are available."
    exit
fi

# integration tests for TP/SP
NCCL_DEBUG=WARN torchrun --nproc_per_node 2 test/float8/test_dtensor.py

# integration smoke tests for FSDP2 + TP
NCCL_DEBUG=WARN torchrun --nproc_per_node 4 test/float8/test_fsdp2_tp.py
