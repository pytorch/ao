# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run

import copy
import unittest

import torch
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401

from torchao.quantization.GPTQ import _replace_linear_8da4w, _replace_linear_int4
from torchao.quantization.granularity import (
    PerAxis,
    PerGroup,
    PerRow,
    PerToken,
)
from torchao.quantization.qat.api import (
    ComposableQATQuantizer,
    FakeQuantizeConfig,
)
from torchao.quantization.qat.embedding import (
    FakeQuantizedEmbedding,
)
from torchao.quantization.qat.linear import (
    FakeQuantizedLinear,
    Int4WeightOnlyQATLinear,
    Int8DynActInt4WeightQATLinear,
)
from torchao.quantization.qat.utils import (
    _choose_qparams_per_token_asymmetric,
    _fake_quantize_per_channel_group,
    _fake_quantize_per_token,
    _GenericFakeQuantize,
    _get_qmin_qmax,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    TorchAODType,
    ZeroPointDomain,
    fake_quantize_affine,
)
from torchao.quantization.unified import (
    TwoStepQuantizer,
)
from torchao.quantization.utils import (
    get_group_qparams_symmetric,
    get_groupwise_affine_qparams,
    groupwise_affine_quantize_tensor,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_3,
    TORCH_VERSION_AT_LEAST_2_4,
)

# TODO: put this in a common test utils file
_CUDA_IS_AVAILABLE = torch.cuda.is_available()


class Sub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 256, bias=False).to(torch.float)

    def example_inputs(self):
        return (torch.randn(1, 256).to(torch.float),)

    def forward(self, x):
        return self.linear(x)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(512, 256, bias=False).to(torch.float)
        self.sub = Sub()
        self.linear2 = torch.nn.Linear(256, 512, bias=False).to(torch.float)

    def example_inputs(self):
        return (torch.randn(1, 512).to(torch.float),)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sub(x)
        x = self.linear2(x)
        return x


class M2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 512)

    def example_inputs(self):
        return (torch.randint(1, 10, (1, 512)),)

    def forward(self, x):
        return self.embedding(x)


class TestQAT(unittest.TestCase):
    SEED = 123

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_fake_quantize_per_channel_group(self):
        n_bit = 4
        (qmin, qmax) = _get_qmin_qmax(n_bit)
        group_size = 128

        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256).requires_grad_()
        (s, zp) = get_group_qparams_symmetric(x, n_bit, group_size)
        zp = zp.to(torch.int32)
        x2 = copy.deepcopy(x)

        # fake quant op
        out = _fake_quantize_per_channel_group(
            x,
            s,
            zp,
            qmin,
            qmax,
            group_size,
        )
        out.sum().backward()

        # compare against PTQ ops
        out_ptq = torch.ops.quantized_decomposed.quantize_per_channel_group(
            x2,
            s,
            zp,
            qmin,
            qmax,
            torch.int8,
            group_size,
        )
        out_ptq = torch.ops.quantized_decomposed.dequantize_per_channel_group(
            out_ptq,
            s,
            zp,
            qmin,
            qmax,
            torch.int8,
            group_size,
            torch.float32,
        )
        torch.testing.assert_close(out, out_ptq, atol=0, rtol=0)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_fake_quantize_per_token(self):
        (qmin, qmax) = _get_qmin_qmax(8)

        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256).requires_grad_()
        x2 = copy.deepcopy(x)
        # TODO: use torch.ops.aten.quantized_decomposed version instead
        (s, zp) = _choose_qparams_per_token_asymmetric(x, torch.float32, torch.int32)

        # fake quant op
        out = _fake_quantize_per_token(x, s, zp, qmin, qmax)
        out.sum().backward()

        # compare against PTQ ops
        out_ptq = torch.ops.quantized_decomposed.quantize_per_token(
            x2,
            s,
            zp,
            qmin,
            qmax,
            torch.int8,
        )
        out_ptq = torch.ops.quantized_decomposed.dequantize_per_token(
            out_ptq,
            s,
            zp,
            qmin,
            qmax,
            torch.int8,
            torch.float32,
        )
        torch.testing.assert_close(out, out_ptq, atol=0, rtol=0)

    def _set_ptq_weight(
        self,
        ptq_linear: torch.nn.Module,
        qat_linear: torch.nn.Module,
    ):
        """
        Set the weight to the quantized version of the given fp32 weights,
        for making linear outputs comparable with QAT.
        """
        from torchao.quantization.GPTQ import (
            Int8DynActInt4WeightLinear,
            WeightOnlyInt4Linear,
        )
        from torchao.quantization.qat.linear import (
            Int4WeightOnlyQATLinear,
            Int8DynActInt4WeightQATLinear,
        )

        n_bit = 4
        (qmin, qmax) = _get_qmin_qmax(n_bit)
        group_size = qat_linear.weight_fake_quantizer.config.group_size
        if isinstance(ptq_linear, Int8DynActInt4WeightLinear):
            assert isinstance(qat_linear, Int8DynActInt4WeightQATLinear)
            fp32_weight = qat_linear.weight
            (s, zp) = get_group_qparams_symmetric(fp32_weight, n_bit, group_size)
            q_weight = torch.ops.quantized_decomposed.quantize_per_channel_group(
                fp32_weight,
                s,
                zp,
                qmin,
                qmax,
                torch.int8,
                group_size,
            )
            ptq_linear.weight = q_weight
            ptq_linear.scales = s
            ptq_linear.zeros = zp
        elif isinstance(ptq_linear, WeightOnlyInt4Linear):
            assert isinstance(qat_linear, Int4WeightOnlyQATLinear)
            (q_weight, scales_and_zeros) = groupwise_affine_quantize_tensor(
                qat_linear.weight,
                n_bit,
                group_size,
            )
            q_weight = torch.ops.aten._convert_weight_to_int4pack(
                q_weight.to("cuda"),
                qat_linear.inner_k_tiles,
            )
            ptq_linear.weight = q_weight
            ptq_linear.scales_and_zeros = scales_and_zeros
        else:
            raise ValueError("Unknown ptq_linear type: %s" % type(ptq_linear))

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_qat_8da4w_linear(self):
        from torchao.quantization.GPTQ import Int8DynActInt4WeightLinear
        from torchao.quantization.qat.linear import Int8DynActInt4WeightQATLinear

        group_size = 128
        torch.manual_seed(self.SEED)
        qat_linear = Int8DynActInt4WeightQATLinear(
            256,
            688,
            bias=False,
            groupsize=group_size,
        )
        ptq_linear = Int8DynActInt4WeightLinear(
            256,
            688,
            bias=False,
            groupsize=group_size,
        )

        # Force the weights to be the same
        self._set_ptq_weight(ptq_linear, qat_linear)

        # Compare linear values
        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256)
        x2 = copy.deepcopy(x)
        qat_out = qat_linear(x)
        ptq_out = ptq_linear(x2)
        torch.testing.assert_close(ptq_out, qat_out, atol=0, rtol=0)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_qat_8da4w_quantizer(self):
        from torchao.quantization.GPTQ import Int8DynActInt4WeightQuantizer
        from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer

        group_size = 16
        torch.manual_seed(self.SEED)
        m = M()
        m2 = copy.deepcopy(m)
        qat_quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=group_size)
        ptq_quantizer = Int8DynActInt4WeightQuantizer(groupsize=group_size)
        qat_model = qat_quantizer.prepare(m)
        ptq_model = ptq_quantizer.quantize(m2)

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
            torch.testing.assert_close(
                ptq_state_dict[k], converted_state_dict[k], atol=0, rtol=0
            )

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_qat_8da4w_quantizer_meta_weights(self):
        from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer

        with torch.device("meta"):
            m = M()
        self.assertTrue(all(v.is_meta for v in m.state_dict().values()))
        group_size = 16
        qat_quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=group_size)
        qat_model = qat_quantizer.prepare(m)
        self.assertTrue(all(v.is_meta for v in qat_model.state_dict().values()))

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_qat_8da4w_quantizer_disable_fake_quant(self):
        """
        Test that 8da4w QAT with disabled fake quant matches nn.Linear in forward.
        """
        from torchao.quantization.qat.linear import (
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
        self.assertFalse(qat_model.linear1.activation_fake_quantizer.enabled)
        self.assertFalse(qat_model.linear1.weight_fake_quantizer.enabled)
        self.assertFalse(qat_model.linear2.activation_fake_quantizer.enabled)
        self.assertFalse(qat_model.linear2.weight_fake_quantizer.enabled)
        self.assertFalse(qat_model.sub.linear.activation_fake_quantizer.enabled)
        self.assertFalse(qat_model.sub.linear.weight_fake_quantizer.enabled)

        # Disabled fake quant is just a normal linear
        m2.linear1.weight = torch.nn.Parameter(qat_model.linear1.weight)
        m2.linear2.weight = torch.nn.Parameter(qat_model.linear2.weight)
        m2.sub.linear.weight = torch.nn.Parameter(qat_model.sub.linear.weight)
        torch.manual_seed(self.SEED)
        x = m.example_inputs()
        x2 = copy.deepcopy(x)
        qat_out = qat_model(*x)
        nn_out = m2(*x2)
        torch.testing.assert_close(nn_out, qat_out, atol=0, rtol=0)

        # Renable fake quant
        qat_model.apply(enable_8da4w_fake_quant)
        self.assertTrue(qat_model.linear1.activation_fake_quantizer.enabled)
        self.assertTrue(qat_model.linear1.weight_fake_quantizer.enabled)
        self.assertTrue(qat_model.linear2.activation_fake_quantizer.enabled)
        self.assertTrue(qat_model.linear2.weight_fake_quantizer.enabled)
        self.assertTrue(qat_model.sub.linear.activation_fake_quantizer.enabled)
        self.assertTrue(qat_model.sub.linear.weight_fake_quantizer.enabled)

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

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_qat_8da4w_quantizer_disable_fake_quant_backward(self):
        """
        Test that 8da4w QAT with disabled fake quant matches nn.Linear in backward.
        """
        from torchao.quantization.qat.linear import (
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
        nn_model.linear1.weight = torch.nn.Parameter(qat_model.linear1.weight)
        nn_model.linear2.weight = torch.nn.Parameter(qat_model.linear2.weight)
        nn_model.sub.linear.weight = torch.nn.Parameter(qat_model.sub.linear.weight)

        # Simulate training for both models
        optimizer1 = torch.optim.SGD(
            nn_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5
        )
        optimizer2 = torch.optim.SGD(
            qat_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5
        )
        loss_fn1 = torch.nn.CrossEntropyLoss()
        loss_fn2 = torch.nn.CrossEntropyLoss()
        example_inputs = nn_model.example_inputs()
        target = torch.randn(1, 512).float()
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
        torch.testing.assert_close(
            nn_model.linear1.weight, qat_model.linear1.weight, atol=0, rtol=0
        )
        torch.testing.assert_close(
            nn_model.linear2.weight, qat_model.linear2.weight, atol=0, rtol=0
        )
        torch.testing.assert_close(
            nn_model.sub.linear.weight, qat_model.sub.linear.weight, atol=0, rtol=0
        )

    def _test_qat_quantized_gradients(self, quantizer):
        """
        Test that QAT produces gradients in the backward pass.
        """
        num_steps = 10
        torch.manual_seed(self.SEED)
        m = M()
        model = quantizer.prepare(m)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        # Simulate training
        current_step = 0
        last_linear1_grad = None
        last_linear2_grad = None
        last_sub_linear_grad = None
        while current_step < num_steps:
            example_inputs = model.example_inputs()
            target = torch.randn(1, 512).float()
            output = model(*example_inputs)
            loss = loss_fn(output, target)
            loss.backward()
            # assert each linear grad is updated
            new_linear1_grad = model.linear1.weight.grad
            new_linear2_grad = model.linear2.weight.grad
            new_sub_linear_grad = model.sub.linear.weight.grad
            self.assertIsNotNone(new_linear1_grad)
            self.assertIsNotNone(new_linear2_grad)
            self.assertIsNotNone(new_sub_linear_grad)
            if current_step > 0:
                self.assertFalse(torch.equal(last_linear1_grad, new_linear1_grad))
                self.assertFalse(torch.equal(last_linear2_grad, new_linear2_grad))
                self.assertFalse(torch.equal(last_sub_linear_grad, new_sub_linear_grad))
            last_linear1_grad = new_linear1_grad
            last_linear2_grad = new_linear2_grad
            last_sub_linear_grad = new_sub_linear_grad
            optimizer.zero_grad()
            optimizer.step()
            current_step += 1

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_qat_8da4w_quantizer_gradients(self):
        from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer

        quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=16)
        self._test_qat_quantized_gradients(quantizer)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_qat_generic_fake_quantize(self):
        """
        Test that the generic fake quantize used in 8da4w QAT matches
        the numerics of existing fake quantize ops in Pytorch in both
        the forward and the backward passes.
        """
        (qmin, qmax) = _get_qmin_qmax(4)
        py_input = torch.randn(16, 64).float().requires_grad_()
        py_s = torch.randn(16).float()
        py_zp = torch.randint(qmax, size=(16,), dtype=torch.int32)
        py_out = torch.fake_quantize_per_channel_affine(
            py_input, py_s, py_zp, 0, qmin, qmax
        )
        py_out.sum().backward()

        ao_input = copy.deepcopy(py_input)
        ao_input.grad.data.zero_()
        block_size = (1, ao_input.shape[-1])
        ao_s = copy.deepcopy(py_s)
        ao_zp = copy.deepcopy(py_zp)
        ao_out = _GenericFakeQuantize.apply(
            ao_input, block_size, ao_s, ao_zp, qmin, qmax
        )
        ao_out.sum().backward()

        torch.testing.assert_close(py_out, ao_out, atol=0, rtol=0)

        # Test that gradients are close enough
        num_grads = py_input.grad.numel()
        num_equal_grads = torch.eq(py_input.grad, ao_input.grad).flatten().sum().item()
        num_equal_grad_threshold = 0.8
        self.assertGreaterEqual(num_equal_grads / num_grads, num_equal_grad_threshold)

    def _assert_close_4w(self, val, ref):
        # Note: for int4 weight-only quantization, we do not expect exact match
        # because torch._weight_int4pack_mm and torch.mm do not match exactly.
        # Here we use the same error bar as PyTorch core to determine closeness:
        # https://github.com/pytorch/pytorch/blob/6079c5091091d872b8dafbaa4e31a5b6194647ad/test/test_linalg.py#L6079
        mean_err = ((val - ref) / ref).mean().abs()
        print(mean_err)
        self.assertTrue(mean_err < 0.05)

    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    def test_qat_4w_primitives(self):
        n_bit = 4
        group_size = 32
        inner_k_tiles = 8
        scales_precision = torch.bfloat16
        device = torch.device("cuda")
        dtype = torch.bfloat16
        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256, dtype=dtype, device=device)
        weight = torch.randn(512, 256, dtype=dtype, device=device)

        # PTQ
        (q_weight, scales_and_zeros) = groupwise_affine_quantize_tensor(
            weight,
            n_bit,
            group_size,
            scales_precision,
        )
        q_weight = torch.ops.aten._convert_weight_to_int4pack(
            q_weight.to(device),
            inner_k_tiles,
        )
        ptq_out = torch.ops.aten._weight_int4pack_mm(
            x, q_weight, group_size, scales_and_zeros
        )

        # QAT
        block_size = (1, group_size)
        quant_min = 0
        quant_max = 2**n_bit - 1
        scales, zero_points = get_groupwise_affine_qparams(
            weight,
            n_bit,
            group_size,
            scales_precision,
        )
        w_fq = fake_quantize_affine(
            weight,
            block_size,
            scales,
            zero_points,
            torch.int32,
            quant_min,
            quant_max,
            zero_point_domain=ZeroPointDomain.FLOAT,
        )
        qat_out = torch.nn.functional.linear(x, w_fq)

        self._assert_close_4w(qat_out, ptq_out)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    def test_qat_4w_linear(self):
        from torchao.quantization.GPTQ import WeightOnlyInt4Linear
        from torchao.quantization.qat.linear import Int4WeightOnlyQATLinear

        group_size = 128
        device = torch.device("cuda")
        dtype = torch.bfloat16
        torch.manual_seed(self.SEED)
        qat_linear = Int4WeightOnlyQATLinear(
            256,
            688,
            bias=False,
            groupsize=group_size,
            device=device,
        )
        ptq_linear = WeightOnlyInt4Linear(
            256,
            688,
            bias=False,
            groupsize=group_size,
            device=device,
        )

        # Force the weights to be the same
        self._set_ptq_weight(ptq_linear, qat_linear)

        # Compare linear values
        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256, dtype=dtype, device=device)
        x2 = copy.deepcopy(x)
        qat_out = qat_linear(x)
        ptq_out = ptq_linear(x2)
        self._assert_close_4w(qat_out, ptq_out)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_qat_4w_quantizer_gradients(self):
        from torchao.quantization.qat import Int4WeightOnlyQATQuantizer

        quantizer = Int4WeightOnlyQATQuantizer(groupsize=32, inner_k_tiles=8)
        self._test_qat_quantized_gradients(quantizer)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    def test_qat_4w_quantizer(self):
        from torchao.quantization.GPTQ import Int4WeightOnlyQuantizer
        from torchao.quantization.qat import Int4WeightOnlyQATQuantizer

        group_size = 32
        inner_k_tiles = 8
        device = torch.device("cuda")
        dtype = torch.bfloat16
        torch.manual_seed(self.SEED)
        m = M().to(device).to(dtype)
        m2 = copy.deepcopy(m)
        qat_quantizer = Int4WeightOnlyQATQuantizer(
            groupsize=group_size,
            inner_k_tiles=inner_k_tiles,
        )
        ptq_quantizer = Int4WeightOnlyQuantizer(
            groupsize=group_size,
            inner_k_tiles=inner_k_tiles,
        )
        qat_model = qat_quantizer.prepare(m)
        ptq_model = ptq_quantizer.quantize(m2)

        # Compare model values
        torch.manual_seed(self.SEED)
        x = [i.to(device).to(dtype) for i in m.example_inputs()]
        x2 = copy.deepcopy(x)
        qat_out = qat_model(*x)
        ptq_out = ptq_model(*x2)
        self._assert_close_4w(qat_out, ptq_out)

        # Convert QAT model and compare model values
        converted_model = qat_quantizer.convert(qat_model)
        converted_out = converted_model(*x)
        torch.testing.assert_close(converted_out, ptq_out, atol=0, rtol=0)

        # Compare converted state dict
        ptq_state_dict = ptq_model.state_dict()
        converted_state_dict = converted_model.state_dict()
        self.assertEqual(ptq_state_dict.keys(), converted_state_dict.keys())
        for k in ptq_state_dict.keys():
            torch.testing.assert_close(
                ptq_state_dict[k], converted_state_dict[k], atol=0, rtol=0
            )

    class _MyQATQuantizer(TwoStepQuantizer):
        """
        Dummy quantizer that attaches a certain value to each nn.Linear's
        `_temp_quantizer_values` attribute.
        """

        ATTR_NAME = "_temp_quantizer_values"

        def __init__(self, value: str):
            self.value = value

        def _attach_value(self, module: torch.nn.Module):
            if isinstance(module, torch.nn.Linear):
                if not hasattr(module, self.ATTR_NAME):
                    setattr(module, self.ATTR_NAME, [])
                getattr(module, self.ATTR_NAME).append(self.value)

        def prepare(self, model: torch.nn.Module):
            model.apply(self._attach_value)
            return model

        def convert(self, model: torch.nn.Module):
            model.apply(self._attach_value)
            return model

    def test_composable_qat_quantizer(self):
        quantizer1 = self._MyQATQuantizer("quantizer1")
        quantizer2 = self._MyQATQuantizer("quantizer2")
        composable_quantizer = ComposableQATQuantizer([quantizer1, quantizer2])
        model = M()
        model = composable_quantizer.prepare(model)
        self.assertTrue(hasattr(model.linear1, self._MyQATQuantizer.ATTR_NAME))
        values_list = getattr(model.linear1, self._MyQATQuantizer.ATTR_NAME)
        self.assertEqual(values_list, ["quantizer1", "quantizer2"])
        composable_quantizer.convert(model)
        values_list = getattr(model.linear1, self._MyQATQuantizer.ATTR_NAME)
        self.assertEqual(
            values_list, ["quantizer1", "quantizer2", "quantizer1", "quantizer2"]
        )

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_qat_4w_embedding(self):
        from torchao.quantization.qat import Int4WeightOnlyEmbeddingQATQuantizer

        model = M2()
        x = model.example_inputs()
        model(*x)
        quantizer = Int4WeightOnlyEmbeddingQATQuantizer()
        prepared = quantizer.prepare(model)
        prepared(*x)
        converted = quantizer.convert(model)
        converted(*x)

    def test_fake_quantize_config_granularity(self):
        """
        Test initialization and property setting of `FakeQuantizeConfig`'s granularity.
        """
        # per token
        per_token_config1 = FakeQuantizeConfig(torch.int8, PerToken())
        per_token_config2 = FakeQuantizeConfig(torch.int8, "per_token")
        self.assertIsInstance(per_token_config1.granularity, PerToken)
        self.assertIsInstance(per_token_config2.granularity, PerToken)

        # per channel
        per_channel_config1 = FakeQuantizeConfig(torch.int8, PerAxis(0))
        per_channel_config2 = FakeQuantizeConfig(torch.int8, "per_channel")
        self.assertIsInstance(per_channel_config1.granularity, PerAxis)
        self.assertIsInstance(per_channel_config2.granularity, PerAxis)
        self.assertEqual(per_channel_config1.granularity.axis, 0)
        self.assertEqual(per_channel_config2.granularity.axis, 0)

        # per group
        per_group_config1 = FakeQuantizeConfig(torch.int8, PerGroup(32))
        per_group_config2 = FakeQuantizeConfig(torch.int8, "per_group", group_size=32)
        per_group_config3 = FakeQuantizeConfig(torch.int8, group_size=32)
        self.assertIsInstance(per_group_config1.granularity, PerGroup)
        self.assertIsInstance(per_group_config2.granularity, PerGroup)
        self.assertIsInstance(per_group_config3.granularity, PerGroup)
        self.assertEqual(per_group_config1.group_size, 32)
        self.assertEqual(per_group_config2.group_size, 32)
        self.assertEqual(per_group_config3.group_size, 32)

        # set `group_size` after initialization
        per_token_config1.group_size = 64
        per_channel_config1.group_size = 64
        per_group_config1.group_size = 64
        self.assertIsInstance(per_token_config1.granularity, PerGroup)
        self.assertIsInstance(per_channel_config1.granularity, PerGroup)
        self.assertIsInstance(per_group_config1.granularity, PerGroup)
        self.assertEqual(per_token_config1.granularity.group_size, 64)
        self.assertEqual(per_channel_config1.granularity.group_size, 64)
        self.assertEqual(per_group_config1.granularity.group_size, 64)

    def test_fake_quantize_config_granularity_error_cases(self):
        """
        Test incorrect settings of `FakeQuantizeConfig`'s granularity.
        """
        # no granularity provided
        with self.assertRaisesRegex(
            ValueError, "`granularity` or `group_size` must be set"
        ):
            FakeQuantizeConfig(torch.int8)

        # group_size with conflicting granularity
        msg = "`group_size` conflicts with granularity"
        with self.assertRaisesRegex(ValueError, msg):
            FakeQuantizeConfig(torch.int8, PerToken(), group_size=32)
        with self.assertRaisesRegex(ValueError, msg):
            FakeQuantizeConfig(torch.int8, PerGroup(64), group_size=32)
        with self.assertRaisesRegex(ValueError, msg):
            FakeQuantizeConfig(torch.int8, "per_token", group_size=32)

        # 'per_group' but no group_size
        msg = "Granularity was 'per_group' but no `group_size` was set"
        with self.assertRaisesRegex(ValueError, msg):
            FakeQuantizeConfig(torch.int8, "per_group")

        # not supported
        with self.assertRaisesRegex(ValueError, "not supported"):
            FakeQuantizeConfig(torch.int8, PerRow())
        with self.assertRaisesRegex(ValueError, "Only axis=0 is supported"):
            FakeQuantizeConfig(torch.int8, PerAxis(1))
        with self.assertRaisesRegex(ValueError, "Unexpected granularity"):
            FakeQuantizeConfig(torch.int8, "blah")
        with self.assertRaisesRegex(ValueError, "unexpected type"):
            FakeQuantizeConfig(torch.int8, 1234)

    def test_fake_quantize_config_mapping_type(self):
        """
        Test initialization and property setting of `FakeQuantizeConfig`'s mapping type.
        """
        # symmetric
        symmetric_config1 = FakeQuantizeConfig(torch.int8, "per_token")
        symmetric_config2 = FakeQuantizeConfig(
            torch.int8, "per_token", is_symmetric=True
        )
        symmetric_config3 = FakeQuantizeConfig(
            torch.int8, "per_token", MappingType.SYMMETRIC
        )
        self.assertEqual(symmetric_config1.mapping_type, MappingType.SYMMETRIC)
        self.assertEqual(symmetric_config2.mapping_type, MappingType.SYMMETRIC)
        self.assertEqual(symmetric_config3.mapping_type, MappingType.SYMMETRIC)
        self.assertTrue(symmetric_config1.is_symmetric)
        self.assertTrue(symmetric_config2.is_symmetric)
        self.assertTrue(symmetric_config3.is_symmetric)

        # asymmetric
        asymmetric_config1 = FakeQuantizeConfig(
            torch.int8, "per_token", is_symmetric=False
        )
        asymmetric_config2 = FakeQuantizeConfig(
            torch.int8, "per_token", MappingType.ASYMMETRIC
        )
        self.assertEqual(asymmetric_config1.mapping_type, MappingType.ASYMMETRIC)
        self.assertEqual(asymmetric_config2.mapping_type, MappingType.ASYMMETRIC)
        self.assertFalse(asymmetric_config1.is_symmetric)
        self.assertFalse(asymmetric_config2.is_symmetric)

        # set `is_symmetric` after initialization
        asymmetric_config1.is_symmetric = True
        self.assertEqual(asymmetric_config1.mapping_type, MappingType.SYMMETRIC)
        self.assertTrue(asymmetric_config1.is_symmetric)

        # bad config1: both mapping_type and is_symmetric are set
        msg = "Cannot set both `mapping_type` and `is_symmetric`"
        with self.assertRaisesRegex(ValueError, msg):
            FakeQuantizeConfig(
                torch.int8, "per_token", MappingType.SYMMETRIC, is_symmetric=False
            )

        # bad config2: not supported
        with self.assertRaisesRegex(ValueError, "not supported"):
            FakeQuantizeConfig(
                torch.int8, "per_token", MappingType.SYMMETRIC_NO_CLIPPING_ERR
            )

    def test_fake_quantize_config_dtype(self):
        """
        Test that unsupported dtypes are caught in `FakeQuantizeConfig`.
        """
        msg = "Unsupported dtype"
        with self.assertRaisesRegex(ValueError, msg):
            FakeQuantizeConfig(torch.int16, "per_token")
        with self.assertRaisesRegex(ValueError, msg):
            FakeQuantizeConfig(torch.int32, "per_token")
        with self.assertRaisesRegex(ValueError, msg):
            FakeQuantizeConfig(torch.bfloat16, "per_token")
        with self.assertRaisesRegex(ValueError, msg):
            FakeQuantizeConfig(torch.float32, "per_token")
        # OK
        if TORCH_VERSION_AT_LEAST_2_3:
            FakeQuantizeConfig(torch.uint1, "per_token")
            FakeQuantizeConfig(torch.uint2, "per_token")
            FakeQuantizeConfig(torch.uint3, "per_token")
            FakeQuantizeConfig(torch.uint4, "per_token")
            FakeQuantizeConfig(torch.uint5, "per_token")
            FakeQuantizeConfig(torch.uint6, "per_token")
            FakeQuantizeConfig(torch.uint7, "per_token")
            FakeQuantizeConfig(torch.uint8, "per_token")
        FakeQuantizeConfig(TorchAODType.INT1, "per_token")
        FakeQuantizeConfig(TorchAODType.INT2, "per_token")
        FakeQuantizeConfig(TorchAODType.INT3, "per_token")
        FakeQuantizeConfig(TorchAODType.INT4, "per_token")
        FakeQuantizeConfig(TorchAODType.INT5, "per_token")
        FakeQuantizeConfig(TorchAODType.INT6, "per_token")
        FakeQuantizeConfig(TorchAODType.INT7, "per_token")
        FakeQuantizeConfig(torch.int8, "per_token")

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_fake_quantized_linear_8da4w(self):
        """
        Test that we can express int8 dynamic activations + int4 weights with `FakeQuantizedLinear`.
        """
        group_size = 128
        torch.manual_seed(self.SEED)
        fq_linear = FakeQuantizedLinear(
            256,
            688,
            bias=False,
            activation_config=FakeQuantizeConfig(
                torch.int8, "per_token", is_symmetric=False
            ),
            weight_config=FakeQuantizeConfig(TorchAODType.INT4, group_size=group_size),
        )

        def linear_forward_8da4w(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            """
            Baseline for int8 dynamic per token asymmetric + int4 per group symmetric quant.
            """
            # activations
            (s, zp) = _choose_qparams_per_token_asymmetric(
                x, torch.float32, torch.int32
            )
            (qmin, qmax) = _get_qmin_qmax(8)
            x_fq = _fake_quantize_per_token(x, s, zp, qmin, qmax)

            # weights
            (s, zp) = get_group_qparams_symmetric(weight, 4, group_size, torch.float32)
            zp = zp.to(torch.int32)
            (qmin, qmax) = _get_qmin_qmax(4)
            w_fq = _fake_quantize_per_channel_group(
                weight, s, zp, qmin, qmax, group_size
            )
            return F.linear(x_fq, w_fq)

        # Compare linear values
        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256)
        x2 = copy.deepcopy(x)
        fq_out = fq_linear(x)
        baseline_out = linear_forward_8da4w(x2, fq_linear.weight)
        torch.testing.assert_close(baseline_out, fq_out, atol=0, rtol=0)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_fake_quantized_linear_4w(self):
        """
        Test that we can express int4 weight only (tinygemm) with `FakeQuantizedLinear`.
        """
        group_size = 128
        weight_config = FakeQuantizeConfig(
            dtype=torch.uint4,
            group_size=group_size,
            is_symmetric=False,
            zero_point_domain=ZeroPointDomain.FLOAT,
        )
        torch.manual_seed(self.SEED)
        fq_linear = FakeQuantizedLinear(
            256,
            688,
            bias=False,
            activation_config=None,
            weight_config=weight_config,
        )

        def linear_forward_4w(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            """
            Baseline for int4 weight only fake quantization that simulates the tinygemm kernel.
            """
            (qmin, qmax) = _get_qmin_qmax(4, symmetric=False)
            (s, zp) = get_groupwise_affine_qparams(weight, 4, group_size, torch.float32)
            zp = zp.to(torch.int32)
            w_fq = _fake_quantize_per_channel_group(
                weight,
                s,
                zp,
                qmin,
                qmax,
                group_size,
                zero_point_domain=ZeroPointDomain.FLOAT,
            )
            return F.linear(x, w_fq)

        # Compare linear values
        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256)
        x2 = copy.deepcopy(x)
        fq_out = fq_linear(x)
        baseline_out = linear_forward_4w(x2, fq_linear.weight)
        torch.testing.assert_close(baseline_out, fq_out, atol=0, rtol=0)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_replace_linear_8da4w(self):
        module = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=256, out_features=50, bias=True)]
        )
        _replace_linear_8da4w(
            module,
            256,
            False,
            torch.float32,
            torch.float32,
            Int8DynActInt4WeightQATLinear,
            copy_weights=True,
        )
        assert not isinstance(module[0], Int8DynActInt4WeightQATLinear) and isinstance(
            module[0], torch.nn.Linear
        )
        module = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=256, out_features=50, bias=False)]
        )
        _replace_linear_8da4w(
            module,
            256,
            False,
            torch.float32,
            torch.float32,
            Int8DynActInt4WeightQATLinear,
            copy_weights=True,
        )
        assert isinstance(module[0], Int8DynActInt4WeightQATLinear)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_replace_linear_int4(self):
        module = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=256, out_features=50, bias=True)]
        )
        _replace_linear_int4(
            module,
            256,
            8,
            padding_allowed=True,
            precision=torch.bfloat16,
            scales_precision=torch.bfloat16,
            linear_class=Int4WeightOnlyQATLinear,
            copy_weights=True,
        )
        assert not isinstance(module[0], Int4WeightOnlyQATLinear) and isinstance(
            module[0], torch.nn.Linear
        )
        module = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=256, out_features=50, bias=False)]
        )
        _replace_linear_int4(
            module,
            256,
            8,
            padding_allowed=True,
            precision=torch.bfloat16,
            scales_precision=torch.bfloat16,
            linear_class=Int4WeightOnlyQATLinear,
            copy_weights=True,
        )
        assert isinstance(module[0], Int4WeightOnlyQATLinear)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_fake_quantized_embedding_4w(self):
        """
        Test that we can express int4 per group symmetric weight only fake quantization
        with `FakeQuantizedEmbedding`.
        """
        num_embeddings = 64
        embedding_dim = 128
        group_size = 32
        torch.manual_seed(self.SEED)
        fq_embedding = FakeQuantizedEmbedding(
            num_embeddings,
            embedding_dim,
            weight_config=FakeQuantizeConfig(TorchAODType.INT4, group_size=group_size),
        )

        def embedding_forward_4w(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            """
            Baseline for int4 per group symmetric weight only fake quantization.
            """
            (s, zp) = get_group_qparams_symmetric(weight, 4, group_size, torch.float32)
            zp = zp.to(torch.int32)
            (qmin, qmax) = _get_qmin_qmax(4)
            w_fq = _fake_quantize_per_channel_group(
                weight, s, zp, qmin, qmax, group_size
            )
            return F.embedding(x, w_fq)

        # Compare embedding values
        torch.manual_seed(self.SEED)
        x = torch.randint(num_embeddings, (5, 10))
        x2 = copy.deepcopy(x)
        fq_out = fq_embedding(x)
        baseline_out = embedding_forward_4w(x2, fq_embedding.weight)
        torch.testing.assert_close(baseline_out, fq_out, atol=0, rtol=0)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_qat_prototype_bc(self):
        """
        Just to make sure we can import all the old prototype paths.
        We will remove this test in the near future when we actually break BC.
        """


if __name__ == "__main__":
    unittest.main()
