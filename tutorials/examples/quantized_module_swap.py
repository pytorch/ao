# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import torch


def int8_symmetric_quantize(
    fp32_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetrically quantize the torch.float32 tensor into torch.int8.
    Return a 2-tuple of (quantized value, scale).

    input: dimensions=[M, N], dtype=torch.float32
    output: dimensions=[M, N], dtype=torch.int8
    scale: dimensions=[M, 1], dtype=torch.float32
    """
    quant_min = -128
    quant_max = 127
    min_val = torch.amin(fp32_tensor, dim=[1], keepdim=False)
    max_val = torch.amax(fp32_tensor, dim=[1], keepdim=False)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    scale = scale.view(fp32_tensor.shape[0], -1)
    out = torch.round(fp32_tensor * (1.0 / scale))
    out = torch.clamp(out, quant_min, quant_max).to(torch.int8)
    return out, scale


class QuantizedLinear(torch.nn.Linear):
    """
    Linear module that performs dynamic and symmetric weight-only
    int8 quantization.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_int8, scale = int8_symmetric_quantize(self.weight)
        return torch.matmul(x, w_int8.t().to(x.dtype)) * scale.t()

    @classmethod
    def from_float(cls, mod: torch.nn.Linear):
        new_linear = cls(mod.in_features, mod.out_features, mod.bias)
        new_linear.weight = mod.weight
        return new_linear


class ToyModel(torch.nn.Module):
    def __init__(self, m: int, n: int, k: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    # Set up toy model
    model = ToyModel(64, 128, 32).cuda()
    example_inputs = torch.randn((1, 64), dtype=torch.float32, device="cuda")

    # Swap torch.nn.Linear with QuantizedLinear
    for name, child in model.named_children():
        if type(child) == torch.nn.Linear:
            new_linear = QuantizedLinear.from_float(child)
            setattr(model, name, new_linear)

    print("quantized model: ", model)
    print("output: ", model(example_inputs))
