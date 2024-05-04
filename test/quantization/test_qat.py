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

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "skipping when torch verion is 2.3 or lower")
    def test_qat_8da4w_quantizer_disable_fake_quant(self):
        """
        Test that 8da4w QAT with disabled fake quant matches nn.Linear in forward.
        """
        from torchao.quantization.prototype.qat import (
            Int8DynActInt4WeightQATQuantizer,
            disable_8da4w_fake_quant,
            enable_8da4w_fake_quant,
        )

        group_size = 16
        torch.manual_seed(self.SEED)
        m = M()
        m2 = copy.deepcopy(m)
        m3 = copy.deepcopy(m)
        quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=group_size)
        qat_model = quantizer.prepare(m)
        qat_model.apply(disable_8da4w_fake_quant)
        self.assertFalse(qat_model.linear1._fake_quant_enabled)
        self.assertFalse(qat_model.linear2._fake_quant_enabled)
        self.assertFalse(qat_model.sub.linear._fake_quant_enabled)

        # Disabled fake quant is just a normal linear
        m2.linear1.weight = qat_model.linear1.weight
        m2.linear2.weight = qat_model.linear2.weight
        m2.sub.linear.weight = qat_model.sub.linear.weight
        torch.manual_seed(self.SEED)
        x = m.example_inputs()
        x2 = copy.deepcopy(x)
        qat_out = qat_model(*x)
        nn_out = m2(*x2)
        torch.testing.assert_close(nn_out, qat_out, atol=0, rtol=0)

        # Renable fake quant
        qat_model.apply(enable_8da4w_fake_quant)
        self.assertTrue(qat_model.linear1._fake_quant_enabled)
        self.assertTrue(qat_model.linear2._fake_quant_enabled)
        self.assertTrue(qat_model.sub.linear._fake_quant_enabled)

        # Fake quant should be applied as normal
        quantizer2 = Int8DynActInt4WeightQATQuantizer(groupsize=group_size)
        qat_model2 = quantizer2.prepare(m3)
        qat_model2.linear1.weight = qat_model.linear1.weight
        qat_model2.linear2.weight = qat_model.linear2.weight
        qat_model2.sub.linear.weight = qat_model.sub.linear.weight
        torch.manual_seed(self.SEED)
        x = m.example_inputs()
        x2 = copy.deepcopy(x)
        qat_out = qat_model(*x)
        qat_out2 = qat_model2(*x2)
        torch.testing.assert_close(qat_out, qat_out2, atol=0, rtol=0)

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "skipping when torch verion is 2.3 or lower")
    def test_qat_8da4w_quantizer_disable_fake_quant_backward(self):
        """
        Test that 8da4w QAT with disabled fake quant matches nn.Linear in backward.
        """
        from torchao.quantization.prototype.qat import (
            Int8DynActInt4WeightQATQuantizer,
            disable_8da4w_fake_quant,
        )

        group_size = 16
        torch.manual_seed(self.SEED)
        m = M()
        nn_model = copy.deepcopy(m)
        quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=group_size)
        qat_model = quantizer.prepare(m)
        qat_model.apply(disable_8da4w_fake_quant)
        nn_model.linear1.weight = qat_model.linear1.weight
        nn_model.linear2.weight = qat_model.linear2.weight
        nn_model.sub.linear.weight = qat_model.sub.linear.weight

        # Simulate training for both models
        optimizer1 = torch.optim.SGD(nn_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
        optimizer2 = torch.optim.SGD(qat_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
        loss_fn1 = torch.nn.CrossEntropyLoss()
        loss_fn2 = torch.nn.CrossEntropyLoss()
        example_inputs = nn_model.example_inputs()
        target = torch.randn(1, 64).float()
        output1 = nn_model(*example_inputs)
        output2 = qat_model(*example_inputs)
        torch.testing.assert_close(output1, output2, atol=0, rtol=0)
        loss1 = loss_fn1(output1, target)
        loss2 = loss_fn2(output2, target)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss1.backward()
        loss2.backward()
        optimizer1.step()
        optimizer2.step()

        # After 1 training step, weights should match exactly
        torch.testing.assert_close(nn_model.linear1.weight, qat_model.linear1.weight, atol=0, rtol=0)
        torch.testing.assert_close(nn_model.linear2.weight, qat_model.linear2.weight, atol=0, rtol=0)
        torch.testing.assert_close(nn_model.sub.linear.weight, qat_model.sub.linear.weight, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
