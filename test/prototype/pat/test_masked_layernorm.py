# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn

from torchao.prototype.pat.layers import MaskedLayerNorm


class TestMaskedLayerNorm(unittest.TestCase):
    def test_no_bias_matches_layernorm_without_masked_values(self):
        for dtype in (torch.float32, torch.float64):
            for shape in (4, (2, 4)):
                with self.subTest(dtype=dtype, normalized_shape=shape):
                    torch.manual_seed(0)
                    layer = MaskedLayerNorm(shape, bias=False, dtype=dtype)
                    reference = nn.LayerNorm(shape, bias=False, dtype=dtype)
                    with torch.no_grad():
                        layer.weight.uniform_(0.5, 1.5)
                    reference.load_state_dict(layer.state_dict())
                    x = (torch.rand(3, 2, 4, dtype=dtype) + 0.1).requires_grad_()
                    x_ref = x.detach().clone().requires_grad_()
                    output = layer(x)
                    expected = reference(x_ref)
                    torch.testing.assert_close(output, expected)
                    gradient = torch.rand_like(output)
                    output.backward(gradient)
                    expected.backward(gradient)
                    torch.testing.assert_close(x.grad, x_ref.grad)
                    torch.testing.assert_close(layer.weight.grad, reference.weight.grad)
                    self.assertIsNone(layer.bias)

    def test_no_bias_ignores_zero_values_in_statistics(self):
        layer = MaskedLayerNorm(4, bias=False, dtype=torch.float64)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        x = torch.tensor([[0.0, 1.0, 3.0, 0.0]], dtype=torch.float64)
        expected = torch.tensor([[0.0, -1.0, 1.5, 0.0]], dtype=torch.float64)
        expected /= (1.0 + layer.eps) ** 0.5
        torch.testing.assert_close(layer(x), expected)

    def test_no_bias_all_zero_input(self):
        layer = MaskedLayerNorm(4, bias=False)
        x = torch.zeros(2, 4, requires_grad=True)
        output = layer(x)
        torch.testing.assert_close(output, torch.zeros_like(x))
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.zeros_like(x))
        torch.testing.assert_close(layer.weight.grad, torch.zeros_like(layer.weight))

    def test_existing_affine_options(self):
        x = torch.tensor([[0.0, 1.0, 3.0, 0.0]])
        for affine, bias in ((True, True), (False, False), (False, True)):
            with self.subTest(elementwise_affine=affine, bias=bias):
                layer = MaskedLayerNorm(4, elementwise_affine=affine, bias=bias)
                expected = torch.tensor([[0.0, -1.0, 1.0, 0.0]])
                expected /= (1.0 + layer.eps) ** 0.5
                if affine:
                    with torch.no_grad():
                        layer.weight.fill_(2.0)
                        layer.bias.fill_(0.5)
                    expected = expected * 2.0 + 0.5
                torch.testing.assert_close(layer(x), expected)


if __name__ == "__main__":
    unittest.main()
