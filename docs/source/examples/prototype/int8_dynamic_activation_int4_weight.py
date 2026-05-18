# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.prototype.quantization.int4 import Int8DynamicActivationInt4WeightConfig
from torchao.quantization import quantize_
from torchao.quantization.quant_primitives import MappingType

config = Int8DynamicActivationInt4WeightConfig(
    group_size=32,
    act_mapping_type=MappingType.SYMMETRIC,
)
m = torch.nn.Linear(64, 64).to(torch.float)
quantize_(m, config)
