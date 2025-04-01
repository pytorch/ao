# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash

# terminate script on first error
set -e

if python -c 'import torch;print(torch.cuda.is_available())' | grep -q "False"; then
    echo "Skipping test_dtensor.sh because no CUDA devices are available."
    exit
fi

# integration tests for TP/SP
NCCL_DEBUG=WARN torchrun --nproc_per_node 2 test/float8/test_dtensor.py

# integration smoke tests for FSDP2 + TP
NCCL_DEBUG=WARN torchrun --nproc_per_node 4 test/float8/test_fsdp2_tp.py
