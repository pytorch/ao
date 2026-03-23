# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.prototype.quantization.int4 import Int8DynamicActivationInt4WeightConfig
from torchao.quantization import quantize_
from torchao.quantization.quant_primitives import MappingType
from torchao.utils import torch_version_at_least

if not torch_version_at_least("2.8.0"):
    print("This example requires PyTorch 2.8 or later")
    exit()


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64, bias=False):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=bias).to(torch.float)
        self.linear2 = torch.nn.Linear(n, k, bias=bias).to(torch.float)

    def example_inputs(self, batch_size=1, dtype=torch.float, device="cpu"):
        return (
            torch.randn(
                batch_size, self.linear1.in_features, dtype=dtype, device=device
            ),
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


device = "cpu"
bias = True
dtype = torch.bfloat16
bs = 32
act_mapping_type = MappingType.SYMMETRIC

m = ToyLinearModel(bias=bias).eval().to(dtype).to(device)
example_inputs = m.example_inputs(batch_size=bs, dtype=dtype, device=device)


with torch.no_grad():
    quantize_(
        m,
        Int8DynamicActivationInt4WeightConfig(
            group_size=32,
            act_mapping_type=act_mapping_type,
        ),
    )
    opt_model = torch.compile(m, fullgraph=True, dynamic=True)
    opt_model(*example_inputs)
