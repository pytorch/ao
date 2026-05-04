# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchao.prototype.quantization import Int8DynamicActivationUIntxWeightConfig
from torchao.quantization import quantize_

model = nn.Sequential(nn.Linear(512, 256, device="cuda", dtype=torch.float16))

# int8 dynamic activation + 4-bit grouped weight quantization
config = Int8DynamicActivationUIntxWeightConfig(
    group_size=128,
    bit_width=4,
    packing_bitwidth=32,
)
quantize_(model, config)

# int8 dynamic activation + 8-bit per-channel weight quantization
model_8bit = nn.Sequential(nn.Linear(512, 256, device="cuda", dtype=torch.float16))
config_8bit = Int8DynamicActivationUIntxWeightConfig(
    group_size=None,  # per-channel (required for 8-bit)
    bit_width=8,
)
quantize_(model_8bit, config_8bit)
