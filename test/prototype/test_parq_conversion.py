# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn

from torchao.prototype.parq.optim import ProxPARQ, QuantOptimizer
from torchao.prototype.parq.quant import UnifTorchaoQuantizer
from torchao.quantization.quantize_.workflows import IntxUnpackedToInt8Tensor


class TestPARQConversion(unittest.TestCase):
    def test_int8_weight_only_cpu(self) -> None:
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(32, 16, bias=False, device="cpu"))
        groups = [{"params": [model[0].weight], "quant_bits": 8}]
        # foreach=True enables multi-tensor optimizer updates.
        base_optimizer = torch.optim.AdamW(groups, lr=1e-3, foreach=True)
        optimizer = QuantOptimizer(
            base_optimizer,
            UnifTorchaoQuantizer(),
            ProxPARQ(anneal_start=0, anneal_end=2),
            quant_period=1,
            quant_per_channel=True,
        )
        inputs = torch.randn(4, 32, device="cpu")
        targets = torch.randn(4, 16, device="cpu")
        for _ in range(3):
            optimizer.zero_grad()
            nn.functional.mse_loss(model(inputs), targets).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            expected_output = model(inputs).clone()
            expected_weight = model[0].weight.detach().clone()
            optimizer.torchao_convert(model, weight_only=True)
            actual_output = model(inputs)

        self.assertFalse(model.training)
        self.assertIsInstance(model[0].weight, IntxUnpackedToInt8Tensor)
        self.assertIsNone(model[0].weight.activation_quantization)
        torch.testing.assert_close(actual_output, expected_output)
        torch.testing.assert_close(
            model[0].weight.dequantize(), expected_weight, atol=0, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
