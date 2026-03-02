# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchao.quantization import Int8DynamicActivationIntxWeightConfig, quantize_
from torchao.quantization.granularity import PerGroup

model = nn.Sequential(nn.Linear(2048, 2048, device="cuda"))
quantize_(
    model,
    Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4, weight_granularity=PerGroup(32)
    ),
)
