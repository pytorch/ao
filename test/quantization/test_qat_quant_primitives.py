# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run

import copy
import unittest

import torch
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib
from torchao.quantization.quant_primitives import (
    get_group_qparams_symmetric,
)
from torchao.quantization._qat_quant_primitives import (
    group_fake_quantize_tensor_symmetric,
)

class TestQATQuantPrimitives(unittest.TestCase):
    SEED = 123

    def _get_qmin_qmax(self, n_bit: int):
        qmin = -(2 ** (n_bit - 1))
        qmax = 2 ** (n_bit - 1) - 1
        return (qmin, qmax)

    def test_fake_quantize_per_channel_group(self):
        n_bit = 4
        (qmin, qmax) = self._get_qmin_qmax(n_bit)
        group_size = 128

        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256).requires_grad_()
        (s, zp) = get_group_qparams_symmetric(x, n_bit, group_size)
        out = torch.ops.quantized_decomposed.fake_quantize_per_channel_group(
            x, s, zp, qmin, qmax, group_size,
        )
        out.sum().backward()

    def test_group_fake_quantize_tensor_symmetric(self):
        n_bit = 4
        (qmin, qmax) = self._get_qmin_qmax(n_bit)
        group_size = 128

        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256).requires_grad_()
        (out, _, _) = group_fake_quantize_tensor_symmetric(x, n_bit, group_size)
        out.sum().backward()


if __name__ == "__main__":
    unittest.main()
