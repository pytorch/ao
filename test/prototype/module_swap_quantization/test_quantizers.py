# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchao.prototype.quantization.module_swap import (
    CodeBookQuantizer,
    IntQuantizer,
    QuantizedLinear,
)
from torchao.prototype.quantization.module_swap.quantizers import (
    VectorQuantizerFunction,
)


class TestIntQuantizer(unittest.TestCase):
    def test_get_scale_param_size(self) -> None:
        x = torch.FloatTensor([0, 5, 10, 15])
        group_size = 4
        scale_param_size = IntQuantizer.get_scale_param_size(x, group_size)
        assert scale_param_size == torch.Size([1])

        x = torch.FloatTensor([0, 5, 10, 15])
        group_size = 2
        scale_param_size = IntQuantizer.get_scale_param_size(x, group_size)
        assert scale_param_size == torch.Size([2])

        x = torch.FloatTensor([0, 5, 10, 15, 20, 25, 30, 35]).reshape(2, 4)
        group_size = 4
        scale_param_size = IntQuantizer.get_scale_param_size(x, group_size)
        assert scale_param_size == torch.Size([2, 1])

        x = torch.FloatTensor([0, 5, 10, 15, 20, 25, 30, 35]).reshape(2, 4)
        group_size = 2
        scale_param_size = IntQuantizer.get_scale_param_size(x, group_size)
        assert scale_param_size == torch.Size([2, 2])

    def test_get_qmin_qmax(self) -> None:
        qmin, qmax = IntQuantizer.get_qmin_qmax(4, signed=False)
        assert qmin == 0
        assert qmax == 15

        qmin, qmax = IntQuantizer.get_qmin_qmax(4, signed=True)
        assert qmin == -8
        assert qmax == 7

    def test_get_scale_offset_asymmetric(self) -> None:
        x = torch.FloatTensor([0, 5, 10, 15])
        group_size = 4
        quantization_mode = "asymmetric"
        q_min = 0
        q_max = 15
        scale, offset = IntQuantizer.get_scale_offset(
            x, group_size, quantization_mode, q_min, q_max
        )
        assert scale == 1
        assert offset == 0

    def test_get_scale_offset_symmetric(self) -> None:
        x = torch.FloatTensor([-8, -5, -3, -1, 0, 1, 3, 5, 7])
        group_size = 9
        quantization_mode = "symmetric"
        q_min = -8
        q_max = 7
        scale, offset = IntQuantizer.get_scale_offset(
            x, group_size, quantization_mode, q_min, q_max
        )
        assert scale == 1
        assert offset is None

    def test_quantize_forward(self) -> None:
        x = torch.FloatTensor([0, 5, 10, 15])
        scale = torch.FloatTensor([1])
        offset = torch.FloatTensor([0])
        group_size = 4
        q_min = 0
        q_max = 15
        output = IntQuantizer.quantize_forward(
            x, scale, offset, q_min, q_max, group_size
        )

        torch.testing.assert_close(output, torch.FloatTensor([0, 5, 10, 15]))

    def test_quantize_forward_asymmetric_clipping(self) -> None:
        x = torch.FloatTensor([0, 5, 10, 100])
        scale = torch.FloatTensor([1])
        offset = torch.FloatTensor([0])
        group_size = 4
        q_min = 0
        q_max = 15
        output = IntQuantizer.quantize_forward(
            x, scale, offset, q_min, q_max, group_size
        )

        torch.testing.assert_close(output, torch.FloatTensor([0, 5, 10, 15]))

    def test_quantize_forward_symmetric(self) -> None:
        x = torch.FloatTensor([0, 1, 2, 3])
        scale = torch.FloatTensor([1.0])
        group_size = 4
        q_min = -8
        q_max = 7
        output = IntQuantizer.quantize_forward(
            x, scale, offset=None, group_size=group_size, q_min=q_min, q_max=q_max
        )

        torch.testing.assert_close(output, torch.FloatTensor([0, 1, 2, 3]))

    def test_quantize_forward_symmetric_clipping(self) -> None:
        x = torch.FloatTensor([0, 1, 2, 10])
        scale = torch.FloatTensor([1.0])
        group_size = 4
        q_min = -8
        q_max = 7
        output = IntQuantizer.quantize_forward(
            x, scale, offset=None, group_size=group_size, q_min=q_min, q_max=q_max
        )

        torch.testing.assert_close(output, torch.FloatTensor([0, 1, 2, 7]))

    def test_get_scale_offset_from_min_max(self) -> None:
        x_min = torch.FloatTensor([-8])
        x_max = torch.FloatTensor([7])
        q_min = 0
        q_max = 15
        scale, offset = IntQuantizer.get_scale_offset_from_min_max(
            x_min, x_max, q_min, q_max
        )
        assert scale == 1
        assert offset == 8

    def test_get_scale_offset_from_min_max_tensorized(self) -> None:
        x_min = torch.FloatTensor([-8, 0])
        x_max = torch.FloatTensor([7, 15])
        q_min = 0
        q_max = 15
        scale, offset = IntQuantizer.get_scale_offset_from_min_max(
            x_min, x_max, q_min, q_max
        )
        assert torch.allclose(scale, torch.FloatTensor([1, 1]))
        assert torch.allclose(offset, torch.FloatTensor([8, 0]))

    def test_get_scale_from_min_max(self) -> None:
        x_min = torch.FloatTensor([-8])
        x_max = torch.FloatTensor([7])
        q_min = -8
        q_max = 7
        scale = IntQuantizer.get_scale_from_min_max(x_min, x_max, q_min, q_max)

        assert scale == 1

    def test_get_scale_from_min_max_vectorized(self) -> None:
        x_min = torch.FloatTensor([-8, -16])
        x_max = torch.FloatTensor([7, 14])
        q_min = -8
        q_max = 7
        scale = IntQuantizer.get_scale_from_min_max(x_min, x_max, q_min, q_max)

        assert torch.allclose(scale, torch.FloatTensor([1, 2]))


class TestCodebookQuantizer(unittest.TestCase):
    def devices(self):
        return ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    def test_codebook_quantizer(self) -> None:
        for device in self.devices():
            with self.subTest(device=device):
                quantizer = CodeBookQuantizer(n_bits=1, features=4, codebook_dim=2).to(
                    device
                )
                with torch.no_grad():
                    quantizer.codebook.copy_(
                        torch.tensor(
                            [[-1, -1], [1, 1], [9, 9], [-9, -9]], device=device
                        )
                    )
                inputs = torch.tensor(
                    [[-2, -2, -1, -3], [2, 2, 1, 3]],
                    dtype=torch.float32,
                    device=device,
                    requires_grad=True,
                )
                output = quantizer(inputs)
                expected = torch.tensor(
                    [[-1.5, -2.5, -1.5, -2.5], [1.5, 2.5, 1.5, 2.5]], device=device
                )
                self.assertEqual(output.device, inputs.device)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                output.sum().backward()
                torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))

    def test_vector_quantizer(self) -> None:
        for device in self.devices():
            with self.subTest(device=device):
                inputs = torch.tensor(
                    [[-3, -1], [-1, -3], [1, 3], [3, 1]],
                    dtype=torch.float32,
                    device=device,
                    requires_grad=True,
                )
                codebook = torch.tensor(
                    [[-2.0, -2.0], [2.0, 2.0], [99.0, 99.0]], device=device
                )
                output = VectorQuantizerFunction.apply(inputs, codebook)
                expected = torch.tensor(
                    [[-2, -2], [-2, -2], [2, 2], [2, 2]], device=device
                )
                self.assertEqual(output.device, inputs.device)
                torch.testing.assert_close(output, expected.float(), rtol=0, atol=0)
                torch.testing.assert_close(
                    codebook,
                    torch.tensor([[-2.0, -2.0], [2.0, 2.0], [0.0, 0.0]], device=device),
                    rtol=0,
                    atol=0,
                )
                gradient = torch.arange(8, device=device, dtype=torch.float32).reshape(
                    4, 2
                )
                output.backward(gradient)
                torch.testing.assert_close(inputs.grad, gradient, rtol=0, atol=0)

    def test_quantized_linear_with_codebook(self) -> None:
        quantizer = CodeBookQuantizer(n_bits=1, features=4, codebook_dim=2)
        layer = QuantizedLinear(
            in_features=4,
            out_features=2,
            bias=False,
            activation_bits=8,
            weight_quantizer=quantizer,
            input_quantization=False,
            activation_quantization=False,
        )
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor([[-2.0, -2.0, -1.0, -3.0], [2.0, 2.0, 1.0, 3.0]])
            )
            quantizer.codebook.copy_(
                torch.tensor([[-1.0, -1.0], [1.0, 1.0], [9.0, 9.0], [-9.0, -9.0]])
            )
        actual = layer(torch.eye(4))
        expected = torch.tensor([[-1.5, 1.5], [-2.5, 2.5], [-1.5, 1.5], [-2.5, 2.5]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
