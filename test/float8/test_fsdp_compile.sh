# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash

# terminate script on first error
set -e
if python -c 'import torch;print(torch.cuda.is_available())' | grep -q "False"; then
    echo "Skipping test_fsdp_compile.sh because no CUDA devices are available."
    exit
fi

# Code to be executed if CUDA devices are available
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 python test/float8/test_fsdp_compile.py
