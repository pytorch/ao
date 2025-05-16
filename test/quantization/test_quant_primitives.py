# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run
import unittest

import torch

from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
    choose_qparams_affine,
    choose_qparams_affine_tinygemm,
    dequantize_affine,
    fake_quantize_affine,
    fake_quantize_affine_cachemask,
    quantize_affine,
)

# TODO: remove test for utils?
from torchao.quantization.utils import (
    get_group_qparams_symmetric,
    groupwise_affine_dequantize_tensor_from_qparams,
    groupwise_affine_quantize_tensor_from_qparams,
    quantize_activation_per_token_absmax,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_3,
    TORCH_VERSION_AT_LEAST_2_4,
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
    check_cpu_version,
    check_xpu_version,
    is_fbcode,
)

_SEED = 1234
torch.manual_seed(_SEED)


# Helper function to run a function twice
# and verify that the result is the same.
# Adds some verification to avoid side effects.
# NOTE:
# - Does not verify the args and kwargs are unchanged.
# - Assumes the output is a single Tensor
def check_idempotent(self, fn, *args, **kwargs):
    output0 = fn(*args, **kwargs)
    assert torch.is_tensor(output0)
    output1 = fn(*args, **kwargs)
    self.assertTrue(
        torch.equal(output0, output1), f"Expected given function {fn} to be idempotent."
    )
    return output1


# Legacy tinygemm ops
def _get_groupwise_affine_qparams(
    w,
    n_bit=4,
    groupsize=128,
    dtype=torch.bfloat16,
    zero_point_domain=ZeroPointDomain.FLOAT,
    zero_point_dtype=torch.bfloat16,
):
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    # assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    quant_min = 0
    quant_max = max_int
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    if zero_point_domain == ZeroPointDomain.FLOAT:
        zeros = min_val + scales * (2 ** (n_bit - 1))
        zeros = zeros.to(dtype=zero_point_dtype).reshape(w.shape[0], -1)
    else:
        zeros = quant_min - torch.round(min_val / scales)
        zeros = torch.clamp(zeros, quant_min, quant_max)
        zeros = zeros.to(dtype=zero_point_dtype).reshape(w.shape[0], -1)
    scales = scales.to(dtype=dtype).reshape(w.shape[0], -1)
    return scales, zeros


def _groupwise_affine_quantize_tensor_from_qparams(
    w, scales, zeros, n_bit=4, groupsize=128, zero_point_domain=ZeroPointDomain.FLOAT
):
    assert groupsize > 1
    assert n_bit == 4
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    # assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    max_int = 2**n_bit - 1
    min_int = 0
    if zero_point_domain == ZeroPointDomain.FLOAT:
        min_val = zeros - scales * (2 ** (n_bit - 1))
        w_int4x8 = (
            to_quant.sub(min_val)
            .div(scales)
            .round()
            .clamp_(min_int, max_int)
            .to(torch.int32)
            .reshape_as(w)
        )
    else:
        w_int4x8 = (
            to_quant.div(scales)
            .round()
            .add(zeros)
            .clamp_(min_int, max_int)
            .to(torch.int32)
            .reshape_as(w)
        )

    if TORCH_VERSION_AT_LEAST_2_5:
        if (not (check_cpu_version(w.device))) and (not (check_xpu_version(w.device))):
            w_int4x8 = (w_int4x8[::, ::2] << 4 | w_int4x8[::, 1::2]).to(torch.uint8)

    return w_int4x8


def _groupwise_affine_dequantize_tensor_from_qparams(
    w_int4x8,
    scales,
    zeros,
    n_bit=4,
    groupsize=128,
    zero_point_domain=ZeroPointDomain.FLOAT,
):
    assert groupsize > 1
    # needed for GPTQ single column dequantize
    if groupsize > w_int4x8.shape[-1] and scales.shape[-1] == 1:
        groupsize = w_int4x8.shape[-1]
    assert w_int4x8.shape[-1] % groupsize == 0
    assert w_int4x8.dim() == 2

    w_int4x8_grouped = w_int4x8.reshape(-1, groupsize)
    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)

    if zero_point_domain == ZeroPointDomain.FLOAT:
        w_dq = (
            w_int4x8_grouped.sub(2 ** (n_bit - 1))
            .mul(scales)
            .add(zeros)
            .reshape_as(w_int4x8)
        )
    else:
        w_dq = w_int4x8_grouped.sub(zeros).mul(scales).reshape_as(w_int4x8)
    return w_dq


class TestQuantPrimitives(unittest.TestCase):
    SEED = 123

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_3, "skipping when torch version is 2.3 or lower"
    )
    def test_get_group_qparams_symmetric(self):
        """
        Test that `get_group_qparams_symmetric` produces the exact same scales as
        `PerChannelMinMaxObserver._calculate_qparams`.
        """
        n_bit = 4
        qmin = -(2 ** (n_bit - 1))
        qmax = 2 ** (n_bit - 1) - 1
        groupsize = 256
        torch.manual_seed(self.SEED)
        weight = torch.randn(100, 256).to(torch.float16)

        # calculate observer scales
        obs = torch.ao.quantization.PerChannelMinMaxObserver(
            ch_axis=0,
            qscheme=torch.per_channel_symmetric,
            quant_min=qmin,
            quant_max=qmax,
            # This is needed to ensure `min_val` and `max_val` are fp16,
            # otherwise they default to fp32 and the qparams will be slightly off
            factory_kwargs={"dtype": torch.float16},
        )
        obs(weight)
        (scale_obs, _) = obs.calculate_qparams()
        scale_obs = scale_obs.reshape(weight.shape[0], -1)

        # assert that scales are identical
        (scale_ao, _) = get_group_qparams_symmetric(
            weight, n_bit, groupsize, precision=torch.float16
        )
        torch.testing.assert_close(scale_obs, scale_ao, rtol=0, atol=0)

    def test_choose_qparams_group_sym(self):
        """Note: groupwise asymmetric quant is using a different way of computing zero_points, so
        we don't include it here. We may just replace it with per block quant
        """
        input = torch.randn(10, 10)
        mapping_type = MappingType.SYMMETRIC
        dtype = torch.int8
        block_size = (1, 2)
        eps = torch.finfo(torch.float32).eps
        precision = torch.float32
        scale, zero_point = choose_qparams_affine(
            input,
            mapping_type,
            block_size,
            dtype,
            eps=eps,
            scale_dtype=precision,
            zero_point_dtype=precision,
        )

        scale_ref, zp_ref = get_group_qparams_symmetric(
            input, n_bit=8, groupsize=2, precision=precision, mapping_type=mapping_type
        )

        self.assertTrue(torch.equal(scale, scale_ref))
        self.assertTrue(torch.equal(zero_point, zp_ref))

    def test_choose_qparams_group_sym_no_clipping_err(self):
        """
        Test the added MappingType.SYMMETRIC_NO_CLIPPING_ERR
        """
        input = torch.randn(10, 10)
        mapping_type = MappingType.SYMMETRIC_NO_CLIPPING_ERR
        dtype = torch.int8
        block_size = (1, 2)
        eps = torch.finfo(torch.float32).eps
        precision = torch.float32
        scale, zero_point = choose_qparams_affine(
            input,
            mapping_type,
            block_size,
            dtype,
            eps=eps,
            scale_dtype=precision,
            zero_point_dtype=precision,
        )

        scale_ref, zp_ref = get_group_qparams_symmetric(
            input, n_bit=8, groupsize=2, precision=precision, mapping_type=mapping_type
        )

        self.assertTrue(torch.equal(scale, scale_ref))
        self.assertTrue(torch.equal(zero_point, zp_ref))

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_3, "skipping when torch version is 2.3 or lower"
    )
    @unittest.skipIf(is_fbcode(), "broken in fbcode")
    def test_choose_qparams_token_asym(self):
        input = torch.randn(10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (1, 10)
        if TORCH_VERSION_AT_LEAST_2_6:
            scale, zero_point = choose_qparams_affine(
                input,
                mapping_type,
                block_size,
                dtype,
                eps=torch.finfo(torch.float32).eps,
                scale_dtype=torch.float64,
                zero_point_dtype=torch.int64,
            )
        else:
            scale, zero_point = choose_qparams_affine(
                input,
                mapping_type,
                block_size,
                dtype,
                eps=torch.finfo(torch.float32).eps,
            )

        scale_ref, zp_ref = (
            torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric(
                input, dtype
            )
        )
        scale_ref = scale_ref.squeeze()
        zp_ref = zp_ref.squeeze()

        torch.testing.assert_close(scale, scale_ref, atol=10e-3, rtol=10e-3)
        self.assertTrue(torch.equal(zero_point, zp_ref))

    @unittest.skipIf(is_fbcode(), "broken in fbcode")
    def test_choose_qparams_tensor_asym(self):
        input = torch.randn(10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (10, 10)
        eps = torch.finfo(torch.float32).eps
        scale, zero_point = choose_qparams_affine(
            input, mapping_type, block_size, dtype, eps=eps
        )

        quant_min = -128
        quant_max = 127
        scale_ref, zp_ref = torch.ops.quantized_decomposed.choose_qparams(
            input, quant_min, quant_max, eps, dtype
        )
        scale_ref = scale_ref.squeeze()
        zp_ref = zp_ref.squeeze()

        self.assertTrue(torch.equal(scale, scale_ref))
        self.assertTrue(torch.equal(zero_point, zp_ref))

    @unittest.skipIf(is_fbcode(), "broken in fbcode")
    def test_choose_qparams_tensor_sym(self):
        input = torch.randn(10, 10)
        mapping_type = MappingType.SYMMETRIC
        dtype = torch.int8
        block_size = (10, 10)
        eps = torch.finfo(torch.float32).eps
        scale, zero_point = choose_qparams_affine(
            input, mapping_type, block_size, dtype, eps=eps
        )

        quant_min = -128
        quant_max = 127
        scale_ref, zp_ref = torch.ops.quantized_decomposed.choose_qparams_symmetric(
            input, quant_min, quant_max, eps, dtype
        )
        scale_ref = scale_ref.squeeze()
        zp_ref = zp_ref.squeeze()

        self.assertTrue(torch.equal(scale, scale_ref))
        self.assertTrue(torch.equal(zero_point, zp_ref))

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_quantize_activation_per_token_abs_max(self):
        input = torch.randn(10, 10)
        quantized_ref, scale_ref = quantize_activation_per_token_absmax(input)

        mapping_type = MappingType.SYMMETRIC
        block_size = list(input.shape)
        for i in range(len(block_size) - 1):
            block_size[i] = 1
        dtype = torch.int8
        eps = 1e-5
        quant_min = -127
        quant_max = 127
        scale, zero_point = choose_qparams_affine(
            input,
            mapping_type,
            block_size,
            dtype,
            quant_min,
            quant_max,
            eps=eps,
            scale_dtype=torch.float,
        )

        quantized = quantize_affine(
            input, block_size, scale, zero_point, dtype, quant_min, quant_max
        )

        self.assertTrue(torch.equal(quantized, quantized_ref))
        self.assertTrue(torch.equal(scale, scale_ref))

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_quantize_activation_per_token_abs_max_zero_input(self):
        input = torch.zeros(10, 10)
        # make sure it still works
        quantized_ref, scale_ref = quantize_activation_per_token_absmax(input)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_quantize_activation_per_token_abs_max_dtype(self):
        input = torch.zeros(10, 10, dtype=torch.bfloat16)
        quantized_ref, scale_ref = quantize_activation_per_token_absmax(input)
        self.assertTrue(scale_ref.dtype, torch.bfloat16)

        input = torch.zeros(10, 10, dtype=torch.float32)
        quantized_ref, scale_ref = quantize_activation_per_token_absmax(input)
        self.assertTrue(scale_ref.dtype, torch.float32)

        input = torch.zeros(10, 10, dtype=torch.float16)
        quantized_ref, scale_ref = quantize_activation_per_token_absmax(input)
        self.assertTrue(scale_ref.dtype, torch.float32)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    @unittest.skipIf(is_fbcode(), "broken in fbcode")
    def test_quantize_dequantize_group_sym(self):
        input = torch.randn(10, 10)
        mapping_type = MappingType.SYMMETRIC
        dtype = torch.int8
        block_size = (1, 2)
        scale, zero_point = choose_qparams_affine(
            input, mapping_type, block_size, dtype, eps=torch.finfo(torch.float32).eps
        )

        quantized = quantize_affine(input, block_size, scale, zero_point, dtype)
        dequantized = check_idempotent(
            self,
            dequantize_affine,
            quantized,
            block_size,
            scale,
            zero_point,
            dtype,
            output_dtype=torch.float32,
        )

        group_size = 2
        quant_min = -128
        quant_max = 127
        quantized_ref = torch.ops.quantized_decomposed.quantize_per_channel_group(
            input, scale, zero_point, quant_min, quant_max, torch.int8, group_size
        )
        dequantized_ref = torch.ops.quantized_decomposed.dequantize_per_channel_group(
            quantized_ref,
            scale,
            zero_point,
            quant_min,
            quant_max,
            torch.int8,
            group_size,
            output_dtype=torch.float32,
        )

        self.assertTrue(torch.equal(quantized, quantized_ref))
        self.assertTrue(torch.equal(dequantized, dequantized_ref))

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    @unittest.skipIf(is_fbcode(), "broken in fbcode")
    def test_quantize_dequantize_channel_asym(self):
        input = torch.randn(10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (10, 1)
        scale, zero_point = choose_qparams_affine(
            input, mapping_type, block_size, dtype, eps=torch.finfo(torch.float32).eps
        )
        output_dtype = torch.float32
        quantized = quantize_affine(input, block_size, scale, zero_point, dtype)
        dequantized = check_idempotent(
            self,
            dequantize_affine,
            quantized,
            block_size,
            scale,
            zero_point,
            dtype,
            output_dtype=output_dtype,
        )

        axis = 1
        quant_min = -128
        quant_max = 127
        quantized_ref = torch.ops.quantized_decomposed.quantize_per_channel(
            input, scale, zero_point, axis, quant_min, quant_max, torch.int8
        )
        dequantized_ref = torch.ops.quantized_decomposed.dequantize_per_channel(
            quantized_ref,
            scale,
            zero_point,
            axis,
            quant_min,
            quant_max,
            torch.int8,
            out_dtype=output_dtype,
        )
        self.assertTrue(torch.equal(quantized, quantized_ref))
        self.assertTrue(torch.equal(dequantized, dequantized_ref))

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    @unittest.skipIf(is_fbcode(), "broken in fbcode")
    def test_quantize_dequantize_tensor_asym(self):
        input = torch.randn(10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (10, 10)
        output_dtype = torch.float32
        scale, zero_point = choose_qparams_affine(
            input, mapping_type, block_size, dtype, eps=torch.finfo(torch.float32).eps
        )
        quantized = quantize_affine(input, block_size, scale, zero_point, dtype)
        dequantized = check_idempotent(
            self,
            dequantize_affine,
            quantized,
            block_size,
            scale,
            zero_point,
            dtype,
            output_dtype=output_dtype,
        )

        quant_min = -128
        quant_max = 127
        quantized_ref = torch.ops.quantized_decomposed.quantize_per_tensor(
            input, scale, zero_point, quant_min, quant_max, torch.int8
        )
        dequantized_ref = torch.ops.quantized_decomposed.dequantize_per_tensor(
            quantized_ref,
            scale,
            zero_point,
            quant_min,
            quant_max,
            torch.int8,
            out_dtype=output_dtype,
        )
        self.assertTrue(torch.equal(quantized, quantized_ref))
        self.assertTrue(torch.equal(dequantized, dequantized_ref))

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    @unittest.skipIf(is_fbcode(), "broken in fbcode")
    def test_quantize_dequantize_channel_asym_4d(self):
        input = torch.randn(3, 3, 10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (3, 3, 1, 10)
        scale, zero_point = choose_qparams_affine(
            input, mapping_type, block_size, dtype, eps=torch.finfo(torch.float32).eps
        )
        quantized = quantize_affine(input, block_size, scale, zero_point, dtype)
        dequantized = check_idempotent(
            self,
            dequantize_affine,
            quantized,
            block_size,
            scale,
            zero_point,
            dtype,
            output_dtype=torch.float32,
        )

        axis = 2
        quant_min = -128
        quant_max = 127
        quantized_ref = torch.ops.quantized_decomposed.quantize_per_channel(
            input, scale, zero_point, axis, quant_min, quant_max, torch.int8
        )
        dequantized_ref = torch.ops.quantized_decomposed.dequantize_per_channel(
            quantized_ref,
            scale,
            zero_point,
            axis,
            quant_min,
            quant_max,
            torch.int8,
            out_dtype=torch.float32,
        )
        self.assertTrue(torch.equal(quantized, quantized_ref))
        self.assertTrue(torch.equal(dequantized, dequantized_ref))

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_3, "skipping when torch version is 2.3 or lower"
    )
    def test_quantize_dequantize_channel_asym_4d_multi_dim_reduction(self):
        input = torch.randn(3, 3, 10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (3, 3, 2, 2)
        scale, zero_point = choose_qparams_affine(
            input, mapping_type, block_size, dtype, eps=torch.finfo(torch.float32).eps
        )
        quantized = quantize_affine(input, block_size, scale, zero_point, dtype)
        dequantized = check_idempotent(
            self,
            dequantize_affine,
            quantized,
            block_size,
            scale,
            zero_point,
            dtype,
            output_dtype=torch.float32,
        )
        # we don't have corresponding ops in existing primitives, so just make sure it runs and it's close to float
        torch.testing.assert_close(dequantized, input, rtol=2, atol=0.02)

    def test_choose_qparams_tensor_asym_eps(self):
        input = torch.zeros(10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (10, 10)
        scale, zero_point = choose_qparams_affine(
            input, mapping_type, block_size, dtype
        )
        eps = torch.finfo(torch.float32).eps
        self.assertEqual(scale, eps)

    @unittest.skipIf(
        not torch.cuda.is_available(), "skipping when cuda is not available"
    )
    def test_get_group_qparams_symmetric_memory(self):
        """Check the memory usage of the op"""
        weight = torch.randn(1024, 1024).to(device="cuda")
        original_mem_use = torch.cuda.memory_allocated()
        n_bit = 4
        groupsize = 128
        (scale_ao, _) = get_group_qparams_symmetric(weight, n_bit, groupsize)
        after_choose_qparams_mem_use = torch.cuda.memory_allocated()
        self.assertTrue(after_choose_qparams_mem_use < 1.2 * original_mem_use)

    def test_raises(self):
        """Make sure some errors are raised when user requested an unsupported type of quantization"""
        input = torch.randn(10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (10, 10)
        scale, zero_point = choose_qparams_affine(
            input, mapping_type, block_size, dtype
        )

        # make sure we can't quantize int32 tensors:
        with self.assertRaisesRegex(AssertionError, "Unsupported input dtype:"):
            _ = quantize_affine(
                input.to(torch.int32), block_size, scale, zero_point, dtype
            )

        # block_size and scale/zero_point shape mismatch
        block_size = (1, 1)
        with self.assertRaisesRegex(RuntimeError, "is invalid for input of size 1"):
            _ = quantize_affine(input, block_size, scale, zero_point, dtype)

    def test_get_groupwise_affine_qparams(self):
        input = torch.randn(10, 256)
        n_bit = 4

        zero_point_domains = [ZeroPointDomain.FLOAT, ZeroPointDomain.INT]
        zero_point_dtypes = [torch.bfloat16, torch.int32]
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (1, 128)
        quant_min = 0
        quant_max = 2**n_bit - 1
        eps = 1e-6
        scale_dtype = torch.bfloat16
        for zero_point_domain, zero_point_dtype in zip(
            zero_point_domains, zero_point_dtypes
        ):
            scale_ref, zero_point_ref = _get_groupwise_affine_qparams(
                input,
                n_bit=n_bit,
                groupsize=128,
                dtype=torch.bfloat16,
                zero_point_domain=zero_point_domain,
            )
            if zero_point_domain == ZeroPointDomain.FLOAT:
                scale, zero_point = choose_qparams_affine_tinygemm(
                    input,
                    mapping_type,
                    block_size,
                    dtype,
                    quant_min,
                    quant_max,
                    eps,
                    scale_dtype=scale_dtype,
                    zero_point_dtype=zero_point_dtype,
                )
            else:
                scale, zero_point = choose_qparams_affine(
                    input,
                    mapping_type,
                    block_size,
                    dtype,
                    quant_min,
                    quant_max,
                    eps,
                    scale_dtype=scale_dtype,
                    zero_point_dtype=zero_point_dtype,
                )

            self.assertTrue(torch.equal(scale, scale_ref))
            self.assertTrue(torch.equal(zero_point, zero_point_ref))

    def test_groupwise_affine_quantize_tensor_from_qparams(self):
        input = torch.randn(10, 256)
        scales = torch.randn(10, 2)
        zeros = torch.randn(10, 2)
        n_bit = 4
        groupsize = 128

        for zero_point_domain in [ZeroPointDomain.FLOAT, ZeroPointDomain.INT]:
            w_int4x8 = groupwise_affine_quantize_tensor_from_qparams(
                input, scales, zeros, n_bit, groupsize, zero_point_domain
            )
            w_int4x8_ref = _groupwise_affine_quantize_tensor_from_qparams(
                input, scales, zeros, n_bit, groupsize, zero_point_domain
            )

            self.assertTrue(torch.equal(w_int4x8, w_int4x8_ref))

    def test_groupwise_affine_dequantize_tensor_from_qparams(self):
        input = torch.randint(0, 15, (10, 256), dtype=torch.int32)
        scales = torch.randn(10, 2).bfloat16()
        zeros = torch.randn(10, 2).bfloat16()
        n_bit = 4
        groupsize = 128

        for zero_point_domain in [ZeroPointDomain.FLOAT, ZeroPointDomain.INT]:
            if zero_point_domain == ZeroPointDomain.INT:
                zeros = torch.randint(0, 15, (10, 2), dtype=torch.int32)
            if TORCH_VERSION_AT_LEAST_2_5:
                input_tmp = input
                if (not (check_cpu_version(input.device))) and (
                    not (check_xpu_version(input.device))
                ):
                    input_tmp = (input[::, ::2] << 4 | input[::, 1::2]).to(torch.uint8)
                w_bf16 = groupwise_affine_dequantize_tensor_from_qparams(
                    input_tmp, scales, zeros, n_bit, groupsize, zero_point_domain
                )
            else:
                if zero_point_domain == ZeroPointDomain.INT:
                    continue
                w_bf16 = groupwise_affine_dequantize_tensor_from_qparams(
                    input, scales, zeros, n_bit, groupsize
                )
            w_bf16_ref = _groupwise_affine_dequantize_tensor_from_qparams(
                input, scales, zeros, n_bit, groupsize, zero_point_domain
            )

        self.assertTrue(torch.equal(w_bf16, w_bf16_ref))

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_fake_quantize_affine(self):
        input = torch.randn(10, 10)

        mapping_type = MappingType.SYMMETRIC
        block_size = list(input.shape)
        for i in range(len(block_size) - 1):
            block_size[i] = 1
        dtype = torch.int8
        eps = 1e-5
        quant_min = -127
        quant_max = 127
        scale, zero_point = choose_qparams_affine(
            input,
            mapping_type,
            block_size,
            dtype,
            quant_min,
            quant_max,
            eps=eps,
            scale_dtype=torch.float,
        )

        quantized = quantize_affine(
            input, block_size, scale, zero_point, dtype, quant_min, quant_max
        )
        dequantized = dequantize_affine(
            quantized, block_size, scale, zero_point, dtype, quant_min, quant_max
        )
        fake_quantized = fake_quantize_affine(
            input, block_size, scale, zero_point, dtype, quant_min, quant_max
        )
        torch.testing.assert_close(dequantized, fake_quantized)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4, "skipping when torch version is 2.4 or lower"
    )
    def test_fake_quantize_affine_cachemask(self):
        input = torch.randn(10, 10)

        mapping_type = MappingType.SYMMETRIC
        block_size = list(input.shape)
        for i in range(len(block_size) - 1):
            block_size[i] = 1
        dtype = torch.int8
        eps = 1e-5
        quant_min = -127
        quant_max = 127
        scale, zero_point = choose_qparams_affine(
            input,
            mapping_type,
            block_size,
            dtype,
            quant_min,
            quant_max,
            eps=eps,
            scale_dtype=torch.float,
        )

        quantized = quantize_affine(
            input, block_size, scale, zero_point, dtype, quant_min, quant_max
        )
        dequantized = dequantize_affine(
            quantized, block_size, scale, zero_point, dtype, quant_min, quant_max
        )
        (fake_quantized, mask) = fake_quantize_affine_cachemask(
            input,
            block_size,
            scale,
            zero_point,
            dtype,
            quant_min,
            quant_max,
        )
        expected_mask = torch.full(input.shape, True)
        torch.testing.assert_close(dequantized, fake_quantized)
        torch.testing.assert_close(expected_mask, mask)


if __name__ == "__main__":
    unittest.main()
