# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchao.prototype.paretoq.models.utils_quant import (
    LsqBinaryTernaryExtension,
    QuantizeLinear,
    StretchedElasticQuant,
)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 256, bias=False).to(torch.float32)

    def forward(self, x):
        return self.linear(x)


class TestParetoQ(unittest.TestCase):
    def test_quantized_linear(self):
        m = M()
        example_inputs = torch.randn(1, 256).to(torch.float32)
        for w_bits in [0, 1, 2, 3, 4, 16]:
            m.linear = QuantizeLinear(
                m.linear.in_features,
                m.linear.out_features,
                bias=False,
                w_bits=w_bits,
            )
            m(example_inputs)

    def test_quantize_functions(self):
        x = torch.randn(256, 256).to(torch.float32)
        alpha = torch.Tensor(256, 1)
        for layerwise in [True, False]:
            LsqBinaryTernaryExtension.apply(x, alpha, 1, layerwise)
            LsqBinaryTernaryExtension.apply(x, alpha, 3, layerwise)
            LsqBinaryTernaryExtension.apply(x, alpha, 4, layerwise)
            StretchedElasticQuant.apply(x, alpha, 0, layerwise)
            StretchedElasticQuant.apply(x, alpha, 2, layerwise)


if __name__ == "__main__":
    unittest.main()
