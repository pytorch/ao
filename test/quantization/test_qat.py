# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run

import copy
import unittest
import warnings
from typing import List, Type

import torch
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
)

from torchao import quantize_
from torchao.core.config import AOBaseConfig
from torchao.float8.config import e4m3_dtype
from torchao.quantization import Float8Tensor
from torchao.quantization.granularity import (
    Granularity,
    PerAxis,
    PerGroup,
    PerRow,
    PerTensor,
    PerToken,
)
from torchao.quantization.linear_quant_modules import (
    _replace_linear_8da4w,
    _replace_linear_int4,
)
from torchao.quantization.qat.api import (
    ComposableQATQuantizer,
    FromIntXQuantizationAwareTrainingConfig,
    IntXQuantizationAwareTrainingConfig,
    QATConfig,
    QATStep,
    initialize_fake_quantizers,
)
from torchao.quantization.qat.embedding import (
    FakeQuantizedEmbedding,
)
from torchao.quantization.qat.fake_quantize_config import (
    Float8FakeQuantizeConfig,
    Int4WeightFakeQuantizeConfig,
    IntxFakeQuantizeConfig,
)
from torchao.quantization.qat.fake_quantizer import (
    Float8FakeQuantizer,
    IntxFakeQuantizer,
)
from torchao.quantization.qat.linear import (
    FakeQuantizedLinear,
    Float8ActInt4WeightQATQuantizer,
    Int4WeightOnlyQATLinear,
    Int8DynActInt4WeightQATLinear,
)
from torchao.quantization.qat.utils import (
    _fake_quantize_per_channel_group,
    _fake_quantize_per_token,
    _get_qmin_qmax,
)
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    TorchAODType,
    ZeroPointDomain,
    _fake_quantize_affine,
    choose_qparams_affine,
    dequantize_affine,
    quantize_affine,
)
from torchao.quantization.quantize_.workflows import Int4PackingFormat
from torchao.quantization.unified import (
    TwoStepQuantizer,
)
from torchao.quantization.utils import (
    _get_per_token_block_size,
    compute_error,
    get_group_qparams_symmetric,
    get_groupwise_affine_qparams,
    groupwise_affine_quantize_tensor,
)
from torchao.utils import (
    _is_fbgemm_genai_gpu_available,
    is_fbcode,
    is_sm_at_least_89,
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

    def example_inputs(self, device: torch.device = None):
        return (torch.randn((1, 512), device=device).to(torch.float),)

    def _get_all_weight_scales(self) -> List[torch.Tensor]:
        return [
            self.linear1.weight_fake_quantizer.scale,
            self.sub.linear.weight_fake_quantizer.scale,
            self.linear2.weight_fake_quantizer.scale,
        ]

    def _get_all_weight_zero_points(self) -> List[torch.Tensor]:
        return [
            self.linear1.weight_fake_quantizer.zero_point,
            self.sub.linear.weight_fake_quantizer.zero_point,
            self.linear2.weight_fake_quantizer.zero_point,
        ]

    def _get_all_weight_qparams(self) -> List[torch.Tensor]:
        return self._get_all_weight_scales() + self._get_all_weight_zero_points()

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


class M3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 512)
        self.linear1 = torch.nn.Linear(512, 256, bias=False).to(torch.float)
        self.linear2 = torch.nn.Linear(256, 512, bias=False).to(torch.float)
        self.relu = torch.nn.ReLU()

    def example_inputs(self):
        return (torch.randint(1, 10, (1, 512)),)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x


class M4(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dtype = dtype
        self.linear = torch.nn.Linear(512, 256, bias=False).to(dtype)

    def example_inputs(self):
        return (torch.randn(1, 512).to(self.dtype),)

    def forward(self, x):
        return self.linear(x)


class ModelWithLinearBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(512, 256, bias=True)
        self.linear2 = torch.nn.Linear(256, 512, bias=True)

    def example_inputs(self):
        return (torch.randn(1, 512),)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TestQAT(TestCase):
    SEED = 123

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

    def test_fake_quantize_per_token(self):
        (qmin, qmax) = _get_qmin_qmax(8)

        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256).requires_grad_()
        x2 = copy.deepcopy(x)
        block_size = _get_per_token_block_size(x)
        (s, zp) = choose_qparams_affine(
            x,
            mapping_type=MappingType.ASYMMETRIC,
            block_size=block_size,
            target_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.int32,
        )

        # fake quant op
        out = _fake_quantize_per_token(x, s, zp, qmin, qmax)
        out.sum().backward()

        # compare against PTQ ops
        out_ptq = quantize_affine(
            x2,
            block_size,
            s,
            zp,
            torch.int8,
            qmin,
            qmax,
        )
        out_ptq = dequantize_affine(
            out_ptq,
            block_size,
            s,
            zp,
            torch.int8,
            qmin,
            qmax,
            output_dtype=torch.float32,
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

    def test_qat_8da4w_quantizer_meta_weights(self):
        from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer

        with torch.device("meta"):
            m = M()
        self.assertTrue(all(v.is_meta for v in m.state_dict().values()))
        group_size = 16
        qat_quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=group_size)
        qat_model = qat_quantizer.prepare(m)
        self.assertTrue(all(v.is_meta for v in qat_model.state_dict().values()))

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

    def test_qat_8da4w_quantizer_gradients(self):
        from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer

        quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=16)
        self._test_qat_quantized_gradients(quantizer)

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
        w_fq = _fake_quantize_affine(
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

    def test_qat_4w_quantizer_gradients(self):
        from torchao.quantization.qat import Int4WeightOnlyQATQuantizer

        quantizer = Int4WeightOnlyQATQuantizer(groupsize=32, inner_k_tiles=8)
        self._test_qat_quantized_gradients(quantizer)

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

    def test_qat_4w_embedding(self):
        from torchao._executorch_ops import (
            _quantized_decomposed_quantize_per_channel_group_wrapper,
        )
        from torchao.quantization.qat import Int4WeightOnlyEmbeddingQATQuantizer

        group_size = 256
        model = M2()
        x = model.example_inputs()
        model(*x)
        quantizer = Int4WeightOnlyEmbeddingQATQuantizer(group_size)
        prepared = quantizer.prepare(model)
        prepared_embedding_weight = copy.deepcopy(prepared.embedding.weight)
        prepared(*x)
        converted = quantizer.convert(model)
        converted(*x)

        # Assert the scales, zero points, and weights are correct after convert
        qmin, qmax = -8, 7
        (s, zp) = get_group_qparams_symmetric(
            prepared_embedding_weight,
            4,
            group_size,
        )
        zp = zp.to(torch.int32)
        q_weight = _quantized_decomposed_quantize_per_channel_group_wrapper(
            prepared_embedding_weight,
            s,
            zp,
            qmin,
            qmax,
            torch.int8,
            group_size,
        )
        torch.testing.assert_close(converted.embedding.weight, q_weight)
        torch.testing.assert_close(converted.embedding.scale, s)
        torch.testing.assert_close(converted.embedding.zero_point, zp)

    def test_fake_quantize_config_granularity(self):
        """
        Test initialization and property setting of `IntxFakeQuantizeConfig`'s granularity.
        """
        # per token
        per_token_config1 = IntxFakeQuantizeConfig(torch.int8, PerToken())
        per_token_config2 = IntxFakeQuantizeConfig(torch.int8, "per_token")
        self.assertIsInstance(per_token_config1.granularity, PerToken)
        self.assertIsInstance(per_token_config2.granularity, PerToken)

        # per channel
        per_channel_config1 = IntxFakeQuantizeConfig(torch.int8, PerAxis(0))
        per_channel_config2 = IntxFakeQuantizeConfig(torch.int8, "per_channel")
        self.assertIsInstance(per_channel_config1.granularity, PerAxis)
        self.assertIsInstance(per_channel_config2.granularity, PerAxis)
        self.assertEqual(per_channel_config1.granularity.axis, 0)
        self.assertEqual(per_channel_config2.granularity.axis, 0)

        # per group
        per_group_config1 = IntxFakeQuantizeConfig(torch.int8, PerGroup(32))
        per_group_config2 = IntxFakeQuantizeConfig(
            torch.int8, "per_group", group_size=32
        )
        per_group_config3 = IntxFakeQuantizeConfig(torch.int8, group_size=32)
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
        Test incorrect settings of `IntxFakeQuantizeConfig`'s granularity.
        """
        # no granularity provided
        with self.assertRaisesRegex(
            ValueError, "`granularity` or `group_size` must be set"
        ):
            IntxFakeQuantizeConfig(torch.int8)

        # group_size with conflicting granularity
        msg = "`group_size` conflicts with granularity"
        with self.assertRaisesRegex(ValueError, msg):
            IntxFakeQuantizeConfig(torch.int8, PerToken(), group_size=32)
        with self.assertRaisesRegex(ValueError, msg):
            IntxFakeQuantizeConfig(torch.int8, PerGroup(64), group_size=32)
        with self.assertRaisesRegex(ValueError, msg):
            IntxFakeQuantizeConfig(torch.int8, "per_token", group_size=32)

        # 'per_group' but no group_size
        msg = "Granularity was 'per_group' but no `group_size` was set"
        with self.assertRaisesRegex(ValueError, msg):
            IntxFakeQuantizeConfig(torch.int8, "per_group")

        # not supported
        with self.assertRaisesRegex(ValueError, "not supported"):
            IntxFakeQuantizeConfig(torch.int8, PerRow())
        with self.assertRaisesRegex(ValueError, "Only axis=0 is supported"):
            IntxFakeQuantizeConfig(torch.int8, PerAxis(1))
        with self.assertRaisesRegex(ValueError, "Unexpected granularity"):
            IntxFakeQuantizeConfig(torch.int8, "blah")
        with self.assertRaisesRegex(ValueError, "unexpected type"):
            IntxFakeQuantizeConfig(torch.int8, 1234)

    def test_fake_quantize_config_mapping_type(self):
        """
        Test initialization and property setting of `IntxFakeQuantizeConfig`'s mapping type.
        """
        # symmetric
        symmetric_config1 = IntxFakeQuantizeConfig(torch.int8, "per_token")
        symmetric_config2 = IntxFakeQuantizeConfig(
            torch.int8, "per_token", is_symmetric=True
        )
        symmetric_config3 = IntxFakeQuantizeConfig(
            torch.int8, "per_token", MappingType.SYMMETRIC
        )
        self.assertEqual(symmetric_config1.mapping_type, MappingType.SYMMETRIC)
        self.assertEqual(symmetric_config2.mapping_type, MappingType.SYMMETRIC)
        self.assertEqual(symmetric_config3.mapping_type, MappingType.SYMMETRIC)
        self.assertTrue(symmetric_config1.is_symmetric)
        self.assertTrue(symmetric_config2.is_symmetric)
        self.assertTrue(symmetric_config3.is_symmetric)

        # asymmetric
        asymmetric_config1 = IntxFakeQuantizeConfig(
            torch.int8, "per_token", is_symmetric=False
        )
        asymmetric_config2 = IntxFakeQuantizeConfig(
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
            IntxFakeQuantizeConfig(
                torch.int8, "per_token", MappingType.SYMMETRIC, is_symmetric=False
            )

        # bad config2: not supported
        with self.assertRaisesRegex(ValueError, "not supported"):
            IntxFakeQuantizeConfig(
                torch.int8, "per_token", MappingType.SYMMETRIC_NO_CLIPPING_ERR
            )

    def test_fake_quantize_config_dtype(self):
        """
        Test that unsupported dtypes are caught in `IntxFakeQuantizeConfig`.
        """
        msg = "Unsupported dtype"
        with self.assertRaisesRegex(ValueError, msg):
            IntxFakeQuantizeConfig(torch.int16, "per_token")
        with self.assertRaisesRegex(ValueError, msg):
            IntxFakeQuantizeConfig(torch.int32, "per_token")
        with self.assertRaisesRegex(ValueError, msg):
            IntxFakeQuantizeConfig(torch.bfloat16, "per_token")
        with self.assertRaisesRegex(ValueError, msg):
            IntxFakeQuantizeConfig(torch.float32, "per_token")
        # OK
        IntxFakeQuantizeConfig(torch.uint1, "per_token")
        IntxFakeQuantizeConfig(torch.uint2, "per_token")
        IntxFakeQuantizeConfig(torch.uint3, "per_token")
        IntxFakeQuantizeConfig(torch.uint4, "per_token")
        IntxFakeQuantizeConfig(torch.uint5, "per_token")
        IntxFakeQuantizeConfig(torch.uint6, "per_token")
        IntxFakeQuantizeConfig(torch.uint7, "per_token")
        IntxFakeQuantizeConfig(torch.uint8, "per_token")
        IntxFakeQuantizeConfig(TorchAODType.INT1, "per_token")
        IntxFakeQuantizeConfig(TorchAODType.INT2, "per_token")
        IntxFakeQuantizeConfig(TorchAODType.INT3, "per_token")
        IntxFakeQuantizeConfig(TorchAODType.INT4, "per_token")
        IntxFakeQuantizeConfig(TorchAODType.INT5, "per_token")
        IntxFakeQuantizeConfig(TorchAODType.INT6, "per_token")
        IntxFakeQuantizeConfig(TorchAODType.INT7, "per_token")
        IntxFakeQuantizeConfig(torch.int8, "per_token")

    def test_fake_quantize_config_dynamic_and_range_learning(self):
        """
        Test that `is_dynamic` and `range_learning` cannot both be set.
        """
        IntxFakeQuantizeConfig(
            torch.int8, "per_channel", is_dynamic=True, range_learning=False
        )
        IntxFakeQuantizeConfig(
            torch.int8, "per_channel", is_dynamic=False, range_learning=True
        )
        with self.assertRaisesRegex(ValueError, "not compatible"):
            IntxFakeQuantizeConfig(
                torch.int8, "per_channel", is_dynamic=True, range_learning=True
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
            activation_config=IntxFakeQuantizeConfig(
                torch.int8, "per_token", is_symmetric=False
            ),
            weight_config=IntxFakeQuantizeConfig(
                TorchAODType.INT4, group_size=group_size
            ),
        )

        def linear_forward_8da4w(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            """
            Baseline for int8 dynamic per token asymmetric + int4 per group symmetric quant.
            """
            # activations
            (s, zp) = choose_qparams_affine(
                x,
                mapping_type=MappingType.ASYMMETRIC,
                block_size=_get_per_token_block_size(x),
                target_dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
                scale_dtype=torch.float32,
                zero_point_dtype=torch.int32,
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

    def test_fake_quantized_linear_4w(self):
        """
        Test that we can express int4 weight only (tinygemm) with `FakeQuantizedLinear`.
        """
        group_size = 128
        weight_config = IntxFakeQuantizeConfig(
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

    def test_replace_linear_8da4w(self):
        module = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=256, out_features=50, bias=True),
                torch.nn.Linear(in_features=256, out_features=50, bias=False),
            ]
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
        assert isinstance(module[1], Int8DynActInt4WeightQATLinear)

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
            weight_config=IntxFakeQuantizeConfig(
                TorchAODType.INT4, group_size=group_size
            ),
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

    def test_qat_prototype_bc(self):
        """
        Just to make sure we can import all the old prototype paths.
        We will remove this test in the near future when we actually break BC.
        """
        from torchao.quantization.prototype.qat import (  # noqa: F401, F811, I001
            disable_4w_fake_quant,
            disable_8da4w_fake_quant,
            enable_4w_fake_quant,
            enable_8da4w_fake_quant,
            ComposableQATQuantizer,
            Int8DynActInt4WeightQATLinear,
            Int4WeightOnlyEmbeddingQATQuantizer,
            Int4WeightOnlyQATQuantizer,
            Int8DynActInt4WeightQATQuantizer,
        )
        from torchao.quantization.prototype.qat._module_swap_api import (  # noqa: F401, F811
            disable_4w_fake_quant_module_swap,
            enable_4w_fake_quant_module_swap,
            disable_8da4w_fake_quant_module_swap,
            enable_8da4w_fake_quant_module_swap,
            Int4WeightOnlyQATQuantizerModuleSwap,
            Int8DynActInt4WeightQATQuantizerModuleSwap,
        )
        from torchao.quantization.prototype.qat.affine_fake_quantized_tensor import (  # noqa: F401, F811
            _AffineFakeQuantizedTensor,
            _to_affine_fake_quantized,
        )
        from torchao.quantization.prototype.qat.api import (  # noqa: F401, F811
            ComposableQATQuantizer,
            FakeQuantizeConfig,
        )
        from torchao.quantization.prototype.qat.embedding import (  # noqa: F401, F811
            FakeQuantizedEmbedding,
            Int4WeightOnlyEmbeddingQATQuantizer,
            Int4WeightOnlyEmbedding,
            Int4WeightOnlyQATEmbedding,
        )
        from torchao.quantization.prototype.qat.fake_quantizer import (  # noqa: F401, F811
            FakeQuantizer,
        )
        from torchao.quantization.prototype.qat.linear import (  # noqa: F401, F811
            disable_4w_fake_quant,
            disable_8da4w_fake_quant,
            enable_4w_fake_quant,
            enable_8da4w_fake_quant,
            FakeQuantizedLinear,
            Int4WeightOnlyQATLinear,
            Int4WeightOnlyQATQuantizer,
            Int8DynActInt4WeightQATLinear,
            Int8DynActInt4WeightQATQuantizer,
        )

    def test_qat_config_init(self):
        """
        Test that the correct errors are thrown if `QATConfig` is not instantiated properly.
        """
        base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
        fq_config = IntxFakeQuantizeConfig(torch.int8, "per_channel")

        # OK
        QATConfig(base_config, step="prepare")
        QATConfig(base_config, step="convert")
        QATConfig(base_config, step=QATStep.PREPARE)
        QATConfig(base_config, step=QATStep.CONVERT)
        QATConfig(activation_config=fq_config, weight_config=fq_config, step="prepare")
        QATConfig(weight_config=fq_config, step="prepare")
        QATConfig(step="convert")

        # OK: good step values
        self.assertEqual(QATConfig(base_config).step, "prepare")
        self.assertEqual(QATConfig(base_config, step="Prepare").step, "prepare")
        self.assertEqual(QATConfig(base_config, step="CONVERT").step, "convert")

        # Bad step
        with self.assertRaisesRegex(ValueError, "`step` must be one of"):
            QATConfig(base_config, step="blah")

        # Step was not a keyword arg
        with self.assertRaisesRegex(
            TypeError, "4 positional arguments but 5 were given"
        ):
            QATConfig(base_config, None, None, "prepare")

        # No configs were provided in prepare step
        with self.assertRaisesRegex(
            ValueError,
            "Must specify `base_config`, `activation_config`, or `weight_config` in the prepare step",
        ):
            QATConfig(step="prepare")

        # Clashing configs are provided
        with self.assertRaisesRegex(ValueError, "Cannot specify both"):
            QATConfig(base_config, weight_config=fq_config, step="prepare")
        with self.assertRaisesRegex(ValueError, "Cannot specify both"):
            QATConfig(base_config, activation_config=fq_config, step="prepare")
        with self.assertRaisesRegex(
            ValueError, "Cannot specify .* in the convert step"
        ):
            QATConfig(weight_config=fq_config, step="convert")

        # FakeQuantizeConfigBase was specified as base_config
        with self.assertRaisesRegex(
            ValueError,
            "was passed as `base_config`. Did you mean to do the following instead?",
        ):
            QATConfig(fq_config, step="prepare")

    def test_quantize_api_prepare(self):
        """
        Test that the following:

            quantize_(model, QATConfig(...))

        can produce the same results as `ComposableQATQuantizer`.
        """
        from torchao.quantization.qat import (
            ComposableQATQuantizer,
            Int4WeightOnlyEmbeddingQATQuantizer,
            Int8DynActInt4WeightQATQuantizer,
        )

        group_size = 16
        torch.manual_seed(self.SEED)
        m = M3()
        baseline_model = copy.deepcopy(m)

        # Baseline quantizer
        baseline_quantizer = ComposableQATQuantizer(
            [
                Int8DynActInt4WeightQATQuantizer(groupsize=group_size),
                Int4WeightOnlyEmbeddingQATQuantizer(group_size=group_size),
            ]
        )
        baseline_model = baseline_quantizer.prepare(baseline_model)

        # quantize_ API
        act_config = IntxFakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
        weight_config = IntxFakeQuantizeConfig(TorchAODType.INT4, group_size=group_size)
        qat_config1 = QATConfig(
            activation_config=act_config, weight_config=weight_config
        )
        qat_config2 = QATConfig(weight_config=weight_config)
        quantize_(m, qat_config1)
        quantize_(
            m, qat_config2, filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding)
        )

        # Compare model values
        torch.manual_seed(self.SEED)
        x = m.example_inputs()
        x2 = copy.deepcopy(x)
        out = m(*x)
        baseline_out = baseline_model(*x2)
        torch.testing.assert_close(out, baseline_out, atol=0, rtol=0)

    def test_quantize_api_errors(self):
        """
        Test that we throw exceptions with helpful error messages if `quantize_`
        runs into unexpected configurations.
        """
        fq_config = IntxFakeQuantizeConfig(torch.int8, group_size=32)
        qat_config = QATConfig(activation_config=fq_config, weight_config=fq_config)
        m = M3()

        # Embedding currently only supports weight-only quantization
        with self.assertRaisesRegex(
            ValueError, "Activation fake quantization is not supported for embedding"
        ):
            quantize_(m, qat_config, lambda m, _: isinstance(m, torch.nn.Embedding))

        # Only linear and embedding are supported currently
        with self.assertRaisesRegex(ValueError, "does not have QAT support"):
            quantize_(m, qat_config, lambda m, _: isinstance(m, torch.nn.ReLU))

    def test_quantize_api_e2e(self):
        """
        Test that the following:

            quantize_(model, QATConfig(Int8DynamicActivationInt4WeightConfig(), step="prepare"))
            quantize_(model, QATConfig(Int8DynamicActivationInt4WeightConfig(), step="convert"))

        can produce the same results as `Int8DynActInt4WeightQATQuantizer` prepare + convert.
        """
        from torchao.quantization.qat import (
            Int8DynActInt4WeightQATQuantizer,
        )

        group_size = 16
        torch.manual_seed(self.SEED)
        m = M()
        baseline_model = copy.deepcopy(m)

        # Baseline prepare
        baseline_quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=group_size)
        baseline_model = baseline_quantizer.prepare(baseline_model)

        # quantize_ prepare
        base_config = Int8DynamicActivationInt4WeightConfig(group_size=group_size)
        quantize_(m, QATConfig(base_config, step="prepare"))

        # Compare prepared values
        torch.manual_seed(self.SEED)
        x = m.example_inputs()
        x2 = copy.deepcopy(x)
        out = m(*x)
        baseline_out = baseline_model(*x2)
        torch.testing.assert_close(out, baseline_out, atol=0, rtol=0)

        # Baseline convert
        baseline_model = baseline_quantizer.convert(baseline_model)

        # quantize_ convert
        quantize_(m, QATConfig(base_config, step="convert"))

        # Compare converted values
        torch.manual_seed(self.SEED)
        x = m.example_inputs()
        x2 = copy.deepcopy(x)
        out = m(*x)
        baseline_out = baseline_model(*x2)
        torch.testing.assert_close(out, baseline_out, atol=0, rtol=0)

    def test_fake_quantize_config_torch_intx(self):
        """
        Test that `IntxFakeQuantizeConfig` works with torch.intx.
        """
        group_size = 16
        config1 = IntxFakeQuantizeConfig(TorchAODType.INT4, group_size=group_size)
        config2 = IntxFakeQuantizeConfig(torch.int4, group_size=group_size)
        linear1 = FakeQuantizedLinear(32, 64, weight_config=config1)
        linear2 = FakeQuantizedLinear(32, 64, weight_config=config2)
        linear2.weight = linear1.weight
        torch.manual_seed(self.SEED)
        x = torch.randn((1, 32)).to(torch.float)
        x2 = copy.deepcopy(x)
        out1 = linear1(*x)
        out2 = linear2(*x2)
        torch.testing.assert_close(out1, out2, atol=0, rtol=0)

    def test_fake_quantizer_repr(self):
        """
        Test that `repr(IntxFakeQuantizer(config))` exposes useful config details.
        """
        config = IntxFakeQuantizeConfig(torch.int4, group_size=128)
        fake_quantizer = IntxFakeQuantizer(config)
        fake_quantizer_repr = repr(fake_quantizer)
        self.assertTrue("dtype=torch.int4" in fake_quantizer_repr)
        self.assertTrue("group_size=128" in fake_quantizer_repr)
        self.assertTrue("PerGroup" in fake_quantizer_repr)
        self.assertTrue("MappingType.SYMMETRIC" in fake_quantizer_repr)

    def test_qat_linear_bias(self):
        """
        Test that QAT supports linear bias.
        """
        m = ModelWithLinearBias()
        act_config = IntxFakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
        weight_config = IntxFakeQuantizeConfig(TorchAODType.INT4, group_size=32)
        qat_config = QATConfig(
            activation_config=act_config, weight_config=weight_config
        )
        quantize_(m, qat_config)
        example_inputs = m.example_inputs()
        m(*example_inputs)

    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_fake_quantize_per_token_vs_convert(self, dtype: torch.dtype):
        """
        Test that the following produce the exact same numerics:
          1. IntxFakeQuantizer with asymmetric per_token config
          2. torchao.quantization.utils.per_token_dynamic_quant
        """
        from torchao.quantization.utils import per_token_dynamic_quant

        torch.manual_seed(self.SEED)
        x = torch.randn(1, 235, 2048).to(dtype)
        config = IntxFakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
        fake_quantizer = IntxFakeQuantizer(config)
        fake_quantizer_out = fake_quantizer(x)
        baseline_out = per_token_dynamic_quant(x)
        torch.testing.assert_close(fake_quantizer_out, baseline_out, atol=0, rtol=0)

    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_qat_8da4w_prepare_vs_convert(self, dtype: torch.dtype):
        """
        Test that the prepare and convert steps of Int8DynActInt4QATQuantizer produces
        numerics that match exactly over N trials.
        """
        from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer

        num_trials = 1000
        group_size = 16
        non_inf_sqnr = []

        for seed in range(self.SEED, self.SEED + num_trials):
            torch.manual_seed(seed)
            m = M4(dtype)
            torch.manual_seed(seed)
            x = m.example_inputs()

            quantizer = Int8DynActInt4WeightQATQuantizer(
                groupsize=group_size, precision=dtype, scales_precision=dtype
            )
            prepared = quantizer.prepare(m)
            prepared_out = prepared(*x)
            converted = quantizer.convert(prepared)
            converted_out = converted(*x)
            sqnr = compute_error(prepared_out, converted_out).item()
            if sqnr != float("inf"):
                non_inf_sqnr.append(sqnr)

        avg_sqnr = (
            sum(non_inf_sqnr) / len(non_inf_sqnr) if len(non_inf_sqnr) > 0 else -1
        )
        fail_message = "%s/%s trials did not match exactly, average sqnr = %s" % (
            len(non_inf_sqnr),
            num_trials,
            avg_sqnr,
        )
        self.assertEqual(len(non_inf_sqnr), 0, fail_message)

    def test_fake_quantize_config_eps(self):
        """
        Test that users can set arbitrary eps value in `IntxFakeQuantizeConfig`.
        """
        eps = 0.00123
        x = torch.randn(2, 3).to(torch.float32)
        scale, zp = choose_qparams_affine(
            x,
            mapping_type=MappingType.ASYMMETRIC,
            block_size=(1, 3),
            target_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            eps=eps,
        )
        expected_out = _fake_quantize_per_token(x, scale, zp, -128, 127)
        config = IntxFakeQuantizeConfig(
            torch.int8,
            "per_token",
            is_symmetric=False,
            eps=eps,
        )
        fake_quantizer = IntxFakeQuantizer(config)
        actual_out = fake_quantizer(x)
        torch.testing.assert_close(expected_out, actual_out, atol=0, rtol=0)

    def test_qat_8da4w_eps(self):
        """
        Test that the 8da4w QAT flow uses the expected eps.
        """
        from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer
        from torchao.quantization.utils import per_token_dynamic_quant

        group_size = 16
        torch.manual_seed(self.SEED)
        m = M()
        quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=group_size)

        # prepare
        prepared_model = quantizer.prepare(m)
        self.assertEqual(
            prepared_model.linear1.activation_fake_quantizer.config.eps,
            torch.finfo(torch.float32).eps,
        )

        # convert
        converted_model = quantizer.convert(m)
        x = m.example_inputs()[0]
        _input = per_token_dynamic_quant(
            x,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.float32,
            eps=torch.finfo(torch.float32).eps,
        )
        _weight_dq = dequantize_affine(
            converted_model.linear1.weight,
            (1, group_size),
            converted_model.linear1.scales,
            converted_model.linear1.zeros,
            torch.int8,
            quant_min=-8,
            quant_max=7,
            output_dtype=torch.float32,
        )
        expected_out = torch.nn.functional.linear(
            _input,
            _weight_dq,
            converted_model.linear1.bias,
        )
        actual_out = converted_model.linear1(x)
        torch.testing.assert_close(expected_out, actual_out, atol=0, rtol=0)

    @parametrize("is_symmetric", [True, False])
    def test_fake_quantizer_range_learning(self, is_symmetric):
        """
        Test that range learning requires `IntxFakeQuantizer`s to be initialized correctly.
        """
        config = IntxFakeQuantizeConfig(
            torch.int8,
            "per_channel",
            is_dynamic=False,
            range_learning=True,
            scale_precision=torch.float32,
            zero_point_precision=torch.float32,
            is_symmetric=is_symmetric,
        )
        fake_quantizer = IntxFakeQuantizer(config)
        example_inputs = (torch.randn(2, 3),)

        # Not initialized, should fail
        self.assertFalse(fake_quantizer._initialized)
        self.assertIsNone(fake_quantizer.scale)
        self.assertIsNone(fake_quantizer.zero_point)
        with self.assertRaisesRegex(
            ValueError,
            "Please call `torchao.quantization.qat.initialize_fake_quantizers` "
            "before initializing the optimizer and beginning training.",
        ):
            fake_quantizer(*example_inputs)

        # Should pass after initializing
        initialize_fake_quantizers(fake_quantizer, example_inputs)
        self.assertTrue(fake_quantizer._initialized)
        self.assertIsInstance(fake_quantizer.scale, torch.nn.Parameter)
        self.assertTrue(fake_quantizer.scale.requires_grad)
        if config.is_symmetric:
            self.assertFalse(isinstance(fake_quantizer.zero_point, torch.nn.Parameter))
            self.assertTrue(torch.all(fake_quantizer.zero_point == 0))
        else:
            self.assertIsInstance(fake_quantizer.zero_point, torch.nn.Parameter)
            self.assertTrue(fake_quantizer.zero_point.requires_grad)
        fake_quantizer(*example_inputs)

    @parametrize("is_symmetric", [True, False])
    def test_qat_range_learning(self, is_symmetric):
        """
        Test end-to-end QAT flow with range learning.
        """
        config = IntxFakeQuantizeConfig(
            torch.int8,
            "per_channel",
            is_dynamic=False,
            range_learning=True,
            scale_precision=torch.float32,
            zero_point_precision=torch.float32,
            is_symmetric=is_symmetric,
        )
        m = M()
        example_inputs = m.example_inputs()
        quantize_(m, QATConfig(weight_config=config))

        # Not initialized, should fail
        for t in m._get_all_weight_qparams():
            self.assertIsNone(t)
        with self.assertRaisesRegex(
            ValueError,
            "Please call `torchao.quantization.qat.initialize_fake_quantizers` "
            "before initializing the optimizer and beginning training.",
        ):
            m(*example_inputs)

        # Should pass after initializing
        # All scales and zero points should be in `m.parameters()`
        initialize_fake_quantizers(m, example_inputs)
        params = set(m.parameters())

        for scale in m._get_all_weight_scales():
            self.assertIsInstance(scale, torch.nn.Parameter)
            self.assertTrue(scale.requires_grad)
            self.assertTrue(scale in params)

        for zero_point in m._get_all_weight_zero_points():
            if config.is_symmetric:
                self.assertFalse(isinstance(zero_point, torch.nn.Parameter))
                self.assertTrue(torch.all(zero_point == 0))
            else:
                self.assertIsInstance(zero_point, torch.nn.Parameter)
                self.assertTrue(zero_point.requires_grad)
                self.assertTrue(zero_point in params)

        m(*example_inputs)

        # Simulate training
        num_steps = 10
        optimizer = torch.optim.SGD(
            m.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        for i in range(num_steps):
            prev_scale = copy.deepcopy(m.linear1.weight_fake_quantizer.scale)
            prev_weight = copy.deepcopy(m.linear1.weight)
            optimizer.zero_grad()
            target = torch.randn(1, 512).float()
            out = m(*example_inputs)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            # Assert that scales have valid gradients and are being updated
            new_scale = m.linear1.weight_fake_quantizer.scale
            self.assertIsNotNone(new_scale.grad)
            self.assertNotEqual(torch.count_nonzero(new_scale.grad), 0)
            self.assertFalse(torch.equal(new_scale, prev_scale))
            # Assert that weights have valid gradients and are being updated
            new_weight = m.linear1.weight
            self.assertIsNotNone(new_weight.grad)
            self.assertNotEqual(torch.count_nonzero(new_weight.grad), 0)
            self.assertFalse(torch.equal(new_weight, prev_weight))

    def test_qat_fp8a4w_quantizer(self):
        """
        Test basic model training with `Float8ActInt4WeightQATQuantizer`.
        """
        torch.manual_seed(self.SEED)
        m = M()
        qat_quantizer = Float8ActInt4WeightQATQuantizer()
        qat_model = qat_quantizer.prepare(m)
        for linear in [m.linear1, m.sub.linear, m.linear2]:
            self.assertIsInstance(linear, FakeQuantizedLinear)
            self.assertIsInstance(
                linear.activation_fake_quantizer,
                Float8FakeQuantizer,
            )
            self.assertIsInstance(linear.weight_fake_quantizer, IntxFakeQuantizer)
        prev_weight = copy.deepcopy(m.linear1.weight)

        # Simulate training
        optimizer = torch.optim.SGD(
            m.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        target = torch.randn(1, 512).float()
        example_inputs = m.example_inputs()
        out = qat_model(*example_inputs)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        # Assert that weights have valid gradients and are being updated
        new_weight = m.linear1.weight
        self.assertIsNotNone(new_weight.grad)
        self.assertNotEqual(torch.count_nonzero(new_weight.grad), 0)
        self.assertFalse(torch.equal(new_weight, prev_weight))

    def test_legacy_quantize_api_e2e(self):
        """
        Test that the following two APIs are numerically equivalent:

        New API:
            quantize_(model, QATConfig(Int8DynamicActivationInt4WeightConfig(), step="prepare"))
            quantize_(model, QATConfig(Int8DynamicActivationInt4WeightConfig(), step="convert"))

        Old API:
            quantize_(model, IntXQuantizationAwareTrainingConfig(...))
            quantize_(model, FromIntXQuantizationAwareTrainingConfig())
            quantize_(model, Int8DynamicActivationInt4WeightConfig())
        """
        group_size = 16
        torch.manual_seed(self.SEED)
        m = M()
        baseline_model = copy.deepcopy(m)

        # Baseline prepare
        act_config = IntxFakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
        weight_config = IntxFakeQuantizeConfig(TorchAODType.INT4, group_size=group_size)
        old_qat_config = IntXQuantizationAwareTrainingConfig(act_config, weight_config)
        quantize_(baseline_model, old_qat_config)

        # QATConfig prepare
        base_config = Int8DynamicActivationInt4WeightConfig(group_size=group_size)
        quantize_(m, QATConfig(base_config, step="prepare"))

        # Compare prepared values
        torch.manual_seed(self.SEED)
        x = m.example_inputs()
        x2 = copy.deepcopy(x)
        out = m(*x)
        baseline_out = baseline_model(*x2)
        torch.testing.assert_close(out, baseline_out, atol=0, rtol=0)

        # Baseline convert
        quantize_(baseline_model, FromIntXQuantizationAwareTrainingConfig())
        quantize_(baseline_model, base_config)

        # quantize_ convert
        quantize_(m, QATConfig(base_config, step="convert"))

        # Compare converted values
        torch.manual_seed(self.SEED)
        x = m.example_inputs()
        x2 = copy.deepcopy(x)
        out = m(*x)
        baseline_out = baseline_model(*x2)
        torch.testing.assert_close(out, baseline_out, atol=0, rtol=0)

    def test_qat_api_deprecation(self):
        """
        Test that the appropriate deprecation warning is logged exactly once per class.
        """
        from torchao.quantization.qat import (
            FakeQuantizeConfig,
            FakeQuantizer,
            from_intx_quantization_aware_training,
            intx_quantization_aware_training,
        )

        # Reset deprecation warning state, otherwise we won't log warnings here
        warnings.resetwarnings()

        # Map from deprecated API to the args needed to instantiate it
        deprecated_apis_to_args = {
            IntXQuantizationAwareTrainingConfig: (),
            FromIntXQuantizationAwareTrainingConfig: (),
            intx_quantization_aware_training: (),
            from_intx_quantization_aware_training: (),
            FakeQuantizeConfig: (torch.int8, "per_channel"),
            FakeQuantizer: (IntxFakeQuantizeConfig(torch.int8, "per_channel"),),
        }

        with warnings.catch_warnings(record=True) as _warnings:
            # Call each deprecated API twice
            for cls, args in deprecated_apis_to_args.items():
                cls(*args)
                cls(*args)

            # Each call should trigger the warning only once
            self.assertEqual(len(_warnings), len(deprecated_apis_to_args))
            for w in _warnings:
                self.assertIn(
                    "is deprecated and will be removed in a future release",
                    str(w.message),
                )

    def test_qat_api_convert_no_quantization(self):
        """
        Test that `QATConfig(step="convert")` swaps back to nn modules without quantization.
        """
        torch.manual_seed(self.SEED)
        m = M()
        baseline_model = copy.deepcopy(m)

        # Prepare swaps to FakeQuantizedLinear
        quantize_(m, QATConfig(Int8DynamicActivationInt4WeightConfig(), step="prepare"))
        self.assertEqual(type(m.linear1), FakeQuantizedLinear)
        self.assertEqual(type(m.sub.linear), FakeQuantizedLinear)
        self.assertEqual(type(m.linear2), FakeQuantizedLinear)

        # Convert without a `base_config` swaps back to nn.Linear
        quantize_(m, QATConfig(step="convert"))
        self.assertEqual(type(m.linear1), torch.nn.Linear)
        self.assertEqual(type(m.sub.linear), torch.nn.Linear)
        self.assertEqual(type(m.linear2), torch.nn.Linear)

        # Model weights should be identical to before
        torch.manual_seed(self.SEED)
        x = m.example_inputs()
        x2 = copy.deepcopy(x)
        out = m(*x)
        baseline_out = baseline_model(*x2)
        torch.testing.assert_close(out, baseline_out, atol=0, rtol=0)

    def test_float8_fake_quantize_config(self):
        """
        Test that the correct errors are thrown if `Float8FakeQuantizeConfig` is not instantiated properly.
        """
        # OK
        Float8FakeQuantizeConfig(torch.float8_e4m3fn)
        Float8FakeQuantizeConfig(torch.float8_e4m3fn, PerRow())
        Float8FakeQuantizeConfig(torch.float8_e4m3fn, PerTensor())

        with self.assertRaisesRegex(ValueError, "not a float8 dtype"):
            Float8FakeQuantizeConfig(torch.int8)
        with self.assertRaisesRegex(
            ValueError, "Please specify the granularity object instead of the class"
        ):
            Float8FakeQuantizeConfig(granularity=PerRow)
        with self.assertRaisesRegex(
            ValueError, "Expected PerRow or PerTensor granularity"
        ):
            Float8FakeQuantizeConfig(granularity=PerToken())

    @parametrize("granularity", [PerTensor(), PerRow()])
    def test_float8_fake_quantize(self, granularity: Granularity):
        """
        Test that `Float8FakeQuantizer` is numerically close to `Float8Tensor`.
        """
        dtype = torch.float8_e4m3fn
        fq_config = Float8FakeQuantizeConfig(dtype, granularity)
        fake_quantizer = Float8FakeQuantizer(fq_config)
        torch.manual_seed(self.SEED)
        x = torch.randn(32, 64)
        out = fake_quantizer(x)
        out_expected = Float8Tensor.from_hp(x, dtype, granularity).dequantize()
        sqnr = compute_error(out, out_expected)
        self.assertGreater(sqnr, 16)

    def _test_quantize_api_against_ptq(
        self,
        base_config: AOBaseConfig,
        target_prepare_sqnr: float,
        target_convert_sqnr: float,
        dtype: torch.dtype = torch.bfloat16,
        module_type: str = "linear",
    ):
        """
        Test the following:

            quantize_(model, QATConfig(base_config, step="prepare"))
            quantize_(model, QATConfig(base_config, step="convert"))

        and compare model outputs of each step against:

            quantize_(model, base_config)
        """
        torch.manual_seed(self.SEED)

        if module_type == "linear":
            m = M().to(dtype).cuda()
            example_inputs = (m.example_inputs()[0].to(dtype).cuda(),)
            filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear)
        elif module_type == "embedding":
            m = M3().to(dtype).cuda()
            example_inputs = (m.example_inputs()[0].cuda(),)
            filter_fn = lambda m, fqn: isinstance(m, torch.nn.Embedding)
        else:
            raise ValueError(f"Unknown module type {module_type}")

        # baseline
        m_baseline = copy.deepcopy(m)
        quantize_(m_baseline, base_config, filter_fn)
        out_baseline = m_baseline(*example_inputs)

        # compare prepare
        quantize_(m, QATConfig(base_config, step="prepare"), filter_fn)
        out_prepared = m(*example_inputs)
        prepare_sqnr = compute_error(out_prepared, out_baseline)
        self.assertGreaterEqual(prepare_sqnr, target_prepare_sqnr)

        # compare convert
        quantize_(m, QATConfig(base_config, step="convert"), filter_fn)
        out_converted = m(*example_inputs)
        convert_sqnr = compute_error(out_converted, out_baseline)
        self.assertGreaterEqual(convert_sqnr, target_convert_sqnr)

    @parametrize("granularity", [PerTensor(), PerRow()])
    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    @unittest.skipIf(not is_sm_at_least_89(), "Need sm89+")
    def test_quantize_api_fp8_fp8(self, granularity: Granularity):
        """
        Test the following:
            quantize_(model, QATConfig(Float8DynamicActivationFloat8Weight(), step="prepare"))
            quantize_(model, QATConfig(Float8DynamicActivationFloat8Weight(), step="convert"))
        """
        self._test_quantize_api_against_ptq(
            Float8DynamicActivationFloat8WeightConfig(granularity=granularity),
            target_prepare_sqnr=15,
            target_convert_sqnr=float("inf"),
        )

    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    @unittest.skipIf(not is_sm_at_least_89(), "Need sm89+")
    @unittest.skipIf(
        not _is_fbgemm_genai_gpu_available(), "Requires fbgemm-gpu-genai >= 1.2.0"
    )
    def test_quantize_api_fp8_int4(self):
        """
        Test the following:
            quantize_(model, QATConfig(Float8DynamicActivationInt4WeightConfig(), step="prepare"))
            quantize_(model, QATConfig(Float8DynamicActivationInt4WeightConfig(), step="convert"))
        """
        self._test_quantize_api_against_ptq(
            Float8DynamicActivationInt4WeightConfig(),
            target_prepare_sqnr=22,
            target_convert_sqnr=float("inf"),
        )

    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    @unittest.skipIf(
        not _is_fbgemm_genai_gpu_available(), "Requires fbgemm-gpu-genai >= 1.2.0"
    )
    @unittest.skipIf(is_fbcode(), "cutlass cannot initialize")
    @parametrize("version", [1, 2])
    @parametrize(
        "packing_format", [Int4PackingFormat.PLAIN, Int4PackingFormat.PRESHUFFLED]
    )
    def test_quantize_api_int4(self, version: int, packing_format: Int4PackingFormat):
        """
        Test the following:
            quantize_(model, QATConfig(Int4WeightOnlyConfig(), step="prepare"))
            quantize_(model, QATConfig(Int4WeightOnlyConfig(), step="convert"))
        """
        self._test_quantize_api_against_ptq(
            Int4WeightOnlyConfig(version=version, int4_packing_format=packing_format),
            target_prepare_sqnr=45 if version == 2 else 12,
            target_convert_sqnr=float("inf"),
        )

    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    def test_quantize_api_int8_int4(self):
        """
        Test the following:
            quantize_(model, QATConfig(Int8DynamicActivationInt4WeightConfig(), step="prepare"))
            quantize_(model, QATConfig(Int8DynamicActivationInt4WeightConfig(), step="convert"))
        """
        self._test_quantize_api_against_ptq(
            Int8DynamicActivationInt4WeightConfig(group_size=32),
            target_prepare_sqnr=30,
            target_convert_sqnr=float("inf"),
        )

    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    @parametrize(
        "weight_dtype, weight_granularity, dtype",
        [
            (weight_dtype, weight_granularity, dtype)
            for weight_dtype in [getattr(torch, f"int{i}") for i in range(2, 9)]
            for weight_granularity in [PerGroup(32), PerAxis(0)]
            for dtype in [torch.bfloat16, torch.float32]
        ],
    )
    def test_quantize_api_int8_intx(self, weight_dtype, weight_granularity, dtype):
        """
        Test the following:
            quantize_(model, QATConfig(Int8DynamicActivationIntxWeightConfig(), step="prepare"))
            quantize_(model, QATConfig(Int8DynamicActivationIntxWeightConfig(), step="convert"))
        """
        self._test_quantize_api_against_ptq(
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype, weight_granularity=weight_granularity
            ),
            target_prepare_sqnr=float("inf"),
            target_convert_sqnr=float("inf"),
            dtype=dtype,
        )

    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    @parametrize(
        "weight_dtype, granularity, dtype, module_type",
        [
            (weight_dtype, granularity, dtype, module_type)
            for weight_dtype in [getattr(torch, f"int{i}") for i in range(2, 9)]
            for granularity in [PerGroup(32), PerAxis(0)]
            for dtype in [torch.bfloat16, torch.float32]
            for module_type in ["linear", "embedding"]
        ],
    )
    def test_quantize_api_intx(self, weight_dtype, granularity, dtype, module_type):
        """
        Test the following:
            quantize_(model, QATConfig(IntxWeightOnlyConfig(), step="prepare"))
            quantize_(model, QATConfig(IntxWeightOnlyConfig(), step="convert"))
        """
        self._test_quantize_api_against_ptq(
            IntxWeightOnlyConfig(weight_dtype=weight_dtype, granularity=granularity),
            target_prepare_sqnr=float("inf"),
            target_convert_sqnr=float("inf"),
            dtype=dtype,
            module_type=module_type,
        )

    def test_infer_fp8_int4_config(self):
        """
        Test that fake quantize configs are correctly inferred from
        `Float8DynamicActivationInt4WeightConfig`.
        """
        from torchao.quantization.qat.fake_quantize_config import (
            _infer_fake_quantize_configs,
        )

        base_config = Float8DynamicActivationInt4WeightConfig()
        (act_config, weight_config) = _infer_fake_quantize_configs(base_config)
        self.assertIsInstance(act_config, Float8FakeQuantizeConfig)
        self.assertEqual(act_config.dtype, e4m3_dtype)
        self.assertIsInstance(act_config.granularity, PerRow)
        self.assertIsInstance(weight_config, Int4WeightFakeQuantizeConfig)
        self.assertEqual(weight_config.group_size, 128)
        self.assertEqual(weight_config.activation_dtype, e4m3_dtype)

    def test_infer_int4_weight_only_config(self):
        """
        Test that fake quantize configs are correctly inferred from `Int4WeightOnlyConfig`.
        """
        from torchao.quantization.qat.fake_quantize_config import (
            _infer_fake_quantize_configs,
        )

        base_config = Int4WeightOnlyConfig(version=1)
        (act_config, weight_config) = _infer_fake_quantize_configs(base_config)
        self.assertIsNone(act_config)
        self.assertIsInstance(weight_config, IntxFakeQuantizeConfig)
        self.assertEqual(weight_config.dtype, torch.uint4)
        self.assertEqual(weight_config.group_size, 128)
        self.assertFalse(weight_config.is_symmetric)

        base_config = Int4WeightOnlyConfig(version=2)
        (act_config, weight_config) = _infer_fake_quantize_configs(base_config)
        self.assertIsNone(act_config)
        self.assertIsInstance(weight_config, Int4WeightFakeQuantizeConfig)
        self.assertEqual(weight_config.group_size, 128)
        self.assertEqual(weight_config.activation_dtype, torch.bfloat16)

    @unittest.skipIf(not is_sm_at_least_89(), "Need sm89+")
    @parametrize("use_per_tensor_scale", [True, False])
    def test_quantize_api_nvfp4(self, use_per_tensor_scale: bool):
        """
        Test the following:
            quantize_(model, QATConfig(NVFP4InferenceConfig(), step="prepare"))
            quantize_(model, QATConfig(NVFP4InferenceConfig(), step="convert"))
        """
        from torchao.prototype.mx_formats import NVFP4InferenceConfig

        if use_per_tensor_scale:
            target_prepare_sqnr = 21
        else:
            target_prepare_sqnr = float("inf")

        self._test_quantize_api_against_ptq(
            NVFP4InferenceConfig(use_dynamic_per_tensor_scale=use_per_tensor_scale),
            target_prepare_sqnr=target_prepare_sqnr,
            target_convert_sqnr=float("inf"),
        )

    @unittest.skipIf(not is_sm_at_least_89(), "Need sm89+")
    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    @parametrize("use_per_tensor_scale", [True, False])
    def test_qat_nvfp4(self, use_per_tensor_scale: bool):
        """
        Test QAT with `NVFP4FakeQuantizeConfig`.
        """
        from torchao.prototype.mx_formats import NVFP4InferenceConfig
        from torchao.prototype.qat import NVFP4FakeQuantizeConfig

        torch.manual_seed(self.SEED)
        m = M().cuda()
        baseline_model = copy.deepcopy(m)
        quantize_(
            baseline_model,
            NVFP4InferenceConfig(use_dynamic_per_tensor_scale=use_per_tensor_scale),
        )
        qat_config = QATConfig(
            activation_config=NVFP4FakeQuantizeConfig(use_per_tensor_scale),
            weight_config=NVFP4FakeQuantizeConfig(use_per_tensor_scale),
            step="prepare",
        )
        quantize_(m, qat_config)

        # Compare prepared values
        torch.manual_seed(self.SEED)
        x = m.example_inputs("cuda")
        out = m(*x)
        baseline_out = baseline_model(*x)
        sqnr = compute_error(out, baseline_out).item()
        if use_per_tensor_scale:
            target_sqnr = 130
        else:
            target_sqnr = float("inf")
        self.assertGreaterEqual(sqnr, target_sqnr)

    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    @unittest.skipIf(
        not _is_fbgemm_genai_gpu_available(), "Requires fbgemm-gpu-genai >= 1.2.0"
    )
    @unittest.skipIf(is_fbcode(), "triton compilation error")
    def test_fbgemm_fp8_primitives(self):
        """
        Compare numerics between:
            (1) fbgemm_gpu.experimental.gen_ai.quantize.quantize_fp8_row
            (2) Our reference QAT version in `Float8FakeQuantizer`
        """
        from fbgemm_gpu.experimental.gen_ai.quantize import quantize_fp8_row

        from torchao.quantization.quant_primitives import (
            _choose_scale_float8,
            _quantize_affine_float8,
        )

        x1 = torch.randn([128, 256], dtype=torch.bfloat16).cuda()
        x2 = copy.deepcopy(x1)

        # (1) Just call `quantize_fp8_row`
        (q1, scale1) = quantize_fp8_row(x1)

        # (2) Our reference implementation for QAT without the dequantize
        scale2 = _choose_scale_float8(
            x2,
            (1, x2.shape[-1]),
            torch.float8_e4m3fn,
            hp_value_lb=1e-12,
        )
        q2 = _quantize_affine_float8(x2, scale2, torch.float8_e4m3fn)
        sqnr = compute_error(q1.to(torch.float32), q2.to(torch.float32))
        scale_sqnr = compute_error(
            scale1.to(torch.float32).flatten(),
            scale2.to(torch.float32).flatten(),
        )
        self.assertGreater(sqnr, 40)
        self.assertGreater(scale_sqnr, 50)

    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    @unittest.skipIf(
        not _is_fbgemm_genai_gpu_available(), "Requires fbgemm-gpu-genai >= 1.2.0"
    )
    @unittest.skipIf(is_fbcode(), "triton compilation error")
    def test_fbgemm_fp8_int4_preshuffled_primitives(self):
        """
        Compare numerics between:
            (1) fbgemm_gpu.experimental.gen_ai.quantize.quantize_int4_preshuffle
            (2) Our reference QAT version in `Int4WeightFakeQuantizer`
        """
        from fbgemm_gpu.experimental.gen_ai.quantize import (
            int4_row_quantize,
            pack_int4,
            quantize_fp8_row,
            quantize_int4_preshuffle,
        )

        from torchao.quantization.quant_primitives import (
            _choose_scale_float8,
            _quantize_affine_float8,
            _quantize_affine_no_dtype_cast,
        )

        group_size = 128
        x1 = torch.randn([128, 256], dtype=torch.bfloat16).cuda()
        x2 = copy.deepcopy(x1)
        x3 = copy.deepcopy(x1)

        # (1) Just call `quantize_int4_preshuffle`
        (q1, (scale1, _)) = quantize_int4_preshuffle(x1, group_size, dtype="fp8")

        # (2) Call `quantize_int4_preshuffle` but skip packing and shuffling
        (q2, _) = quantize_fp8_row(x2)
        (q2, scale2) = int4_row_quantize(q2, group_size)

        # (3) Reference implementation for QAT without the dequantize
        fp8_scale = _choose_scale_float8(
            x3,
            (1, x3.shape[-1]),
            torch.float8_e4m3fn,
            hp_value_lb=1e-12,
        )
        x3_fp8 = _quantize_affine_float8(x3, fp8_scale, torch.float8_e4m3fn)
        x3_fp8 = x3_fp8.to(torch.float32)
        x3_fp8_grouped = x3_fp8.view(x3_fp8.shape[0], -1, group_size)
        max_abs = torch.amax(torch.abs(x3_fp8_grouped), dim=-1, keepdim=False)
        scale = torch.clamp(max_abs / 8, min=1e-6)
        zero_point = torch.zeros_like(scale)
        q3 = _quantize_affine_no_dtype_cast(
            x3_fp8,
            (1, group_size),
            scale,
            zero_point,
            quant_min=-8,
            quant_max=7,
        )
        scale3 = scale

        def shuffle_and_pack(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            t = pack_int4(t.to(torch.int8))
            return torch.ops.fbgemm.preshuffle_i4(t, scale.to(torch.float8_e4m3fn))[0]

        # First, sanity check that shuffle_and_pack(q2) == q1
        torch.testing.assert_close(q1, shuffle_and_pack(q2, scale2), atol=0, rtol=0)

        # Now check q2 vs q3 with and without shuffle
        sqnr_q2_q3 = compute_error(q2.to(torch.float32), q3.to(torch.float32))
        sqnr_q2_q3_preshuffle = compute_error(
            shuffle_and_pack(q2, scale2).to(torch.float32),
            shuffle_and_pack(q3, scale3).to(torch.float32),
        )
        self.assertGreater(sqnr_q2_q3, 32)
        self.assertGreater(sqnr_q2_q3_preshuffle, 32)

        # Now check shuffle_and_pack(q3) vs q1
        sqnr_q1_q3_preshuffle = compute_error(
            q1.to(torch.float32),
            shuffle_and_pack(q3, scale3).to(torch.float32),
        )
        self.assertGreater(sqnr_q1_q3_preshuffle, 32)

    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping when cuda is not available")
    @unittest.skipIf(
        not _is_fbgemm_genai_gpu_available(), "Requires fbgemm-gpu-genai >= 1.2.0"
    )
    @unittest.skipIf(is_fbcode(), "triton compilation error")
    def test_fbgemm_int4_weight_only_primitives(self):
        """
        Compare numerics between:
            (1) fbgemm_gpu.experimental.gen_ai.quantize.int4_row_quantize_zp
            (2) Our reference QAT version in `Int4WeightFakeQuantizer`
        """
        from fbgemm_gpu.experimental.gen_ai.quantize import (
            int4_row_quantize_zp,
            pack_int4,
            quantize_int4_preshuffle,
        )

        group_size = 128
        x1 = torch.randn([128, 256], dtype=torch.bfloat16).cuda()
        x2 = copy.deepcopy(x1)
        x3 = copy.deepcopy(x1)

        # (1) Just call `quantize_int4_preshuffle` with dtype="bf16"
        (q1, (scale1, _)) = quantize_int4_preshuffle(x1, group_size, dtype="bf16")

        # (2) Call `int4_row_quantize_zp`, which should be the same as (1)
        # but without the packing and shuffling
        (q2, scale2, _) = int4_row_quantize_zp(x2, group_size)

        # (3) Reference implementation for QAT without the dequantize
        eps = 1e-6
        qmin, qmax = 0, 15
        fbgemm_symmetric_qmax = 8
        w_grouped = x3.to(torch.float32).view(x3.shape[0], -1, group_size)
        max_val = torch.amax(w_grouped, dim=-1, keepdim=True)
        min_val = torch.amin(w_grouped, dim=-1, keepdim=True)
        scale3 = torch.clamp(max_val - min_val, min=eps) / qmax
        q3 = (w_grouped.sub(min_val).div(scale3)).round().clamp_(qmin, qmax)
        q3 = q3 - fbgemm_symmetric_qmax
        q3 = q3.view(x3.shape)

        def shuffle_and_pack(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            t = pack_int4(t.to(torch.int8))
            return torch.ops.fbgemm.preshuffle_i4(t, scale.to(torch.bfloat16))[0]

        # First, sanity check that shuffle_and_pack(q2) == q1
        torch.testing.assert_close(q1, shuffle_and_pack(q2, scale2), atol=0, rtol=0)

        # Now check q2 vs q3 with and without shuffle
        torch.testing.assert_close(q2.to(torch.float32), q3, atol=0, rtol=0)
        torch.testing.assert_close(
            shuffle_and_pack(q2, scale2).to(torch.float32),
            shuffle_and_pack(q3, scale3).to(torch.float32),
            atol=0,
            rtol=0,
        )

        # Now check shuffle_and_pack(q3) vs q1
        torch.testing.assert_close(
            q1.to(torch.float32),
            shuffle_and_pack(q3, scale3).to(torch.float32),
            atol=0,
            rtol=0,
        )

    @parametrize(
        "base_config_cls",
        [
            IntxWeightOnlyConfig,
            Int8DynamicActivationInt4WeightConfig,
            Int8DynamicActivationIntxWeightConfig,
        ],
    )
    def test_range_learning_convert_pass_qparams(
        self, base_config_cls: Type[AOBaseConfig]
    ):
        """
        Verify that range learning QAT can pass qparams from the prepared
        model to the convert model.
        """
        group_size = 32
        config = IntxFakeQuantizeConfig(
            torch.int4,
            group_size=group_size,
            is_symmetric=True,
            is_dynamic=False,
            range_learning=True,
        )
        m = M()
        example_inputs = m.example_inputs()
        quantize_(m, QATConfig(weight_config=config, step="prepare"))
        initialize_fake_quantizers(m, example_inputs)

        # convert and verify scales are what we expect
        scale1 = m.linear1.weight_fake_quantizer.scale
        scale2 = m.linear2.weight_fake_quantizer.scale
        sub_scale = m.sub.linear.weight_fake_quantizer.scale
        if base_config_cls == Int8DynamicActivationInt4WeightConfig:
            base_config = base_config_cls()
            quantize_(m, QATConfig(base_config, step="convert"))
            torch.testing.assert_close(
                m.linear1.weight.original_weight_tensor.tensor_impl.scale, scale1
            )
            torch.testing.assert_close(
                m.linear2.weight.original_weight_tensor.tensor_impl.scale, scale2
            )
            torch.testing.assert_close(
                m.sub.linear.weight.original_weight_tensor.tensor_impl.scale, sub_scale
            )
        else:
            base_config = base_config_cls(torch.int4, PerGroup(group_size))
            quantize_(m, QATConfig(base_config, step="convert"))
            torch.testing.assert_close(m.linear1.weight.scale, scale1)
            torch.testing.assert_close(m.linear2.weight.scale, scale2)
            torch.testing.assert_close(m.sub.linear.weight.scale, sub_scale)


instantiate_parametrized_tests(TestQAT)


if __name__ == "__main__":
    unittest.main()
