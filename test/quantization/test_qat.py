# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run

import copy
import unittest

import torch
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torchao.quantization.prototype.qat import (
    _choose_qparams_per_token_asymmetric,
    fake_quantize_per_channel_group,
    fake_quantize_per_token,
)
from torchao.quantization.quant_primitives import get_group_qparams_symmetric
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_3


# TODO: put this in a common test utils file
class Sub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32, bias=False).to(torch.float)

    def example_inputs(self):
        return (torch.randn(1, 32).to(torch.float),)

    def forward(self, x):
        return self.linear(x)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 32, bias=False).to(torch.float)
        self.sub = Sub()
        self.linear2 = torch.nn.Linear(32, 64, bias=False).to(torch.float)

    def example_inputs(self):
        return (torch.randn(1, 64).to(torch.float),)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sub(x)
        x = self.linear2(x)
        return x


class TestQAT(unittest.TestCase):
    SEED = 123

    def _get_qmin_qmax(self, n_bit: int):
        qmin = -(2 ** (n_bit - 1))
        qmax = 2 ** (n_bit - 1) - 1
        return (qmin, qmax)

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "skipping when torch verion is 2.3 or lower")
    def test_fake_quantize_per_channel_group(self):
        n_bit = 4
        (qmin, qmax) = self._get_qmin_qmax(n_bit)
        group_size = 128

        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256).requires_grad_()
        (s, zp) = get_group_qparams_symmetric(x, n_bit, group_size)
        x2 = copy.deepcopy(x)

        # fake quant op
        out = fake_quantize_per_channel_group(
            x, s, zp, qmin, qmax, group_size,
        )
        out.sum().backward()

        # compare against PTQ ops
        out_ptq = torch.ops.quantized_decomposed.quantize_per_channel_group(
            x2, s, zp, qmin, qmax, torch.int8, group_size,
        )
        out_ptq = torch.ops.quantized_decomposed.dequantize_per_channel_group(
            out_ptq, s, zp, qmin, qmax, torch.int8, group_size, torch.float32,
        )
        torch.testing.assert_close(out, out_ptq, atol=0, rtol=0)

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "skipping when torch verion is 2.3 or lower")
    def test_fake_quantize_per_token(self):
        (qmin, qmax) = self._get_qmin_qmax(8)

        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256).requires_grad_()
        x2 = copy.deepcopy(x)
        # TODO: use torch.ops.aten.quantized_decomposed version instead
        (s, zp) = _choose_qparams_per_token_asymmetric(
            x,
            torch.int8,  # not used
        )

        # fake quant op
        out = fake_quantize_per_token(x, s, zp, qmin, qmax)
        out.sum().backward()

        # compare against PTQ ops
        out_ptq = torch.ops.quantized_decomposed.quantize_per_token(
            x2, s, zp, qmin, qmax, torch.int8,
        )
        out_ptq = torch.ops.quantized_decomposed.dequantize_per_token(
            out_ptq, s, zp, qmin, qmax, torch.int8, torch.float32,
        )
        torch.testing.assert_close(out, out_ptq, atol=0, rtol=0)

    def _set_ptq_weight(
        self,
        ptq_linear: "Int8DynActInt4WeightLinear",
        fp32_weight: torch.Tensor,
        group_size: int,
    ):
        """
        Set the weight to the quantized version of the given fp32 weights,
        for making linear outputs comparable with QAT.
        """
        n_bit = 4
        (qmin, qmax) = self._get_qmin_qmax(n_bit)
        (s, zp) = get_group_qparams_symmetric(fp32_weight, n_bit, group_size)
        q_weight = torch.ops.quantized_decomposed.quantize_per_channel_group(
            fp32_weight, s, zp, qmin, qmax, torch.int8, group_size,
        )
        ptq_linear.weight = q_weight
        ptq_linear.scales = s
        ptq_linear.zeros = zp

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "skipping when torch verion is 2.3 or lower")
    def test_qat_8da4w_linear(self):
        from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATLinear
        from torchao.quantization.GPTQ import Int8DynActInt4WeightLinear

        group_size = 128
        torch.manual_seed(self.SEED)
        qat_linear = Int8DynActInt4WeightQATLinear(
            256, 688, bias=False, groupsize=group_size,
        )
        ptq_linear = Int8DynActInt4WeightLinear(
            256, 688, bias=False, groupsize=group_size,
        )

        # Force the weights to be the same
        self._set_ptq_weight(ptq_linear, qat_linear.weight, group_size)

        # Compare linear values
        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256)
        x2 = copy.deepcopy(x)
        qat_out = qat_linear(x)
        ptq_out = ptq_linear(x2)
        torch.testing.assert_close(ptq_out, qat_out, atol=0, rtol=0)

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "skipping when torch verion is 2.3 or lower")
    def test_qat_8da4w_quantizer(self):
        from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer
        from torchao.quantization.GPTQ import Int8DynActInt4WeightQuantizer

        group_size = 16
        torch.manual_seed(self.SEED)
        m = M()
        m2 = copy.deepcopy(m)
        qat_quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=group_size)
        ptq_quantizer = Int8DynActInt4WeightQuantizer(groupsize=group_size)
        qat_model = qat_quantizer.prepare(m)
        ptq_model = ptq_quantizer.quantize(m2)

        # Force the weights to be the same
        self._set_ptq_weight(
            ptq_model.linear1, qat_model.linear1.weight, group_size,
        )
        self._set_ptq_weight(
            ptq_model.sub.linear, qat_model.sub.linear.weight, group_size,
        )
        self._set_ptq_weight(
            ptq_model.linear2, qat_model.linear2.weight, group_size,
        )

        # Compare model values
        torch.manual_seed(self.SEED)
        x = m.example_inputs()
        x2 = copy.deepcopy(x)
        qat_out = qat_model(*x)
        ptq_out = ptq_model(*x2)
        torch.testing.assert_close(ptq_out, qat_out, atol=0, rtol=0)

        # Convert QAT model and compare model values
        converted_model = qat_quantizer.convert(qat_model)
        converted_out = converted_model(*x)
        torch.testing.assert_close(ptq_out, converted_out, atol=0, rtol=0)

        # Compare converted state dict
        ptq_state_dict = ptq_model.state_dict()
        converted_state_dict = converted_model.state_dict()
        self.assertEqual(ptq_state_dict.keys(), converted_state_dict.keys())
        for k in ptq_state_dict.keys():
            torch.testing.assert_close(ptq_state_dict[k], converted_state_dict[k], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
