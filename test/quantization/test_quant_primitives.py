# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run
import unittest
import torch
from torchao.quantization.quant_primitives import (
    get_group_qparams_symmetric,
    get_groupwise_affine_qparams,
    quantize_affine,
    dequantize_affine,
    choose_qparams_affine,
    MappingType,
)

from torchao.quantization.utils import (
    TORCH_VERSION_AFTER_2_3,
    TORCH_VERSION_AFTER_2_4,
)

_SEED = 1234
torch.manual_seed(_SEED)

class TestQuantPrimitives(unittest.TestCase):
    SEED = 123

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "skipping when torch verion is 2.3 or lower")
    def test_get_group_qparams_symmetric(self):
        """
        Test that `get_group_qparams_symmetric` produces the exact same scales as
        `PerChannelMinMaxObserver._calculate_qparams`.
        """
        n_bit = 4
        qmin = -(2 ** (n_bit - 1))
        qmax = 2 ** (n_bit - 1) - 1
        eps = torch.finfo(torch.float32).eps
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
            factory_kwargs={"dtype": torch.float16}
        )
        obs(weight)
        (scale_obs, _) = obs.calculate_qparams()
        scale_obs = scale_obs.reshape(weight.shape[0], -1)

        # assert that scales are identical
        (scale_ao, _) = get_group_qparams_symmetric(weight, n_bit, groupsize, precision=torch.float16)
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
        scale, zero_point = choose_qparams_affine(input, mapping_type, block_size, dtype, eps=eps, scale_dtype=precision, zero_point_dtype=precision)

        scale_ref, zp_ref = get_group_qparams_symmetric(input, n_bit=8, groupsize=2, precision=precision)

        self.assertTrue(torch.equal(scale, scale_ref))
        self.assertTrue(torch.equal(zero_point, zp_ref))

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "skipping when torch verion is 2.3 or lower")
    def test_choose_qparams_token_asym(self):
        input = torch.randn(10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (1, 10)
        scale, zero_point = choose_qparams_affine(input, mapping_type, block_size, dtype, eps=torch.finfo(torch.float32).eps)

        scale_ref, zp_ref = torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric(input, dtype)
        scale_ref = scale_ref.squeeze()
        zp_ref = zp_ref.squeeze()

        torch.testing.assert_close(scale, scale_ref, atol=10e-3, rtol=10e-3)
        self.assertTrue(torch.equal(zero_point, zp_ref))

    def test_choose_qparams_tensor_asym(self):
        input = torch.randn(10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (10, 10)
        eps = torch.finfo(torch.float32).eps
        scale, zero_point = choose_qparams_affine(input, mapping_type, block_size, dtype, eps=eps)


        quant_min = -128
        quant_max = 127
        scale_ref, zp_ref = torch.ops.quantized_decomposed.choose_qparams(input, quant_min, quant_max, eps, dtype)
        scale_ref = scale_ref.squeeze()
        zp_ref = zp_ref.squeeze()

        self.assertTrue(torch.equal(scale, scale_ref))
        self.assertTrue(torch.equal(zero_point, zp_ref))

    def test_choose_qparams_tensor_sym(self):
        input = torch.randn(10, 10)
        mapping_type = MappingType.SYMMETRIC
        dtype = torch.int8
        block_size = (10, 10)
        eps = torch.finfo(torch.float32).eps
        scale, zero_point = choose_qparams_affine(input, mapping_type, block_size, dtype, eps=eps)

        quant_min = -128
        quant_max = 127
        scale_ref, zp_ref = torch.ops.quantized_decomposed.choose_qparams_symmetric(input, quant_min, quant_max, eps, dtype)
        scale_ref = scale_ref.squeeze()
        zp_ref = zp_ref.squeeze()

        self.assertTrue(torch.equal(scale, scale_ref))
        self.assertTrue(torch.equal(zero_point, zp_ref))

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "skipping when torch verion is 2.4 or lower")
    def test_quantize_activation_per_token_abs_max(self):
        from torchao.quantization.quant_primitives import quantize_activation_per_token_absmax
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
        scale, zero_point = choose_qparams_affine(input, mapping_type, block_size, dtype, quant_min, quant_max, eps=eps, scale_dtype=torch.float)

        quantized = quantize_affine(input, block_size, scale, zero_point, dtype, quant_min, quant_max)

        self.assertTrue(torch.equal(quantized, quantized_ref))
        self.assertTrue(torch.equal(scale, scale_ref))

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "skipping when torch verion is 2.4 or lower")
    def test_quantize_activation_per_token_abs_max_zero_input(self):
        from torchao.quantization.quant_primitives import quantize_activation_per_token_absmax
        input = torch.zeros(10, 10)
        # make sure it still works
        quantized_ref, scale_ref = quantize_activation_per_token_absmax(input)


    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "skipping when torch verion is 2.4 or lower")
    def test_quantize_dequantize_group_sym(self):
        input = torch.randn(10, 10)
        mapping_type = MappingType.SYMMETRIC
        dtype = torch.int8
        block_size = (1, 2)
        scale, zero_point = choose_qparams_affine(input, mapping_type, block_size, dtype, eps=torch.finfo(torch.float32).eps)

        quantized = quantize_affine(input, block_size, scale, zero_point, dtype)
        dequantized = dequantize_affine(quantized, block_size, scale, zero_point, dtype, output_dtype=torch.float32)

        group_size = 2
        quant_min = -128
        quant_max = 127
        quantized_ref = torch.ops.quantized_decomposed.quantize_per_channel_group(
            input, scale, zero_point, quant_min, quant_max, torch.int8, group_size
        )
        dequantized_ref = torch.ops.quantized_decomposed.dequantize_per_channel_group(
            quantized_ref, scale, zero_point, quant_min, quant_max, torch.int8, group_size, output_dtype=torch.float32
        )

        self.assertTrue(torch.equal(quantized, quantized_ref))
        self.assertTrue(torch.equal(dequantized, dequantized_ref))

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "skipping when torch verion is 2.4 or lower")
    def test_quantize_dequantize_channel_asym(self):
        input = torch.randn(10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (10, 1)
        scale, zero_point = choose_qparams_affine(input, mapping_type, block_size, dtype, eps=torch.finfo(torch.float32).eps)
        output_dtype = torch.float32
        quantized = quantize_affine(input, block_size, scale, zero_point, dtype)
        dequantized = dequantize_affine(quantized, block_size, scale, zero_point, dtype, output_dtype=output_dtype)

        axis = 1
        quant_min = -128
        quant_max = 127
        quantized_ref = torch.ops.quantized_decomposed.quantize_per_channel(
            input, scale, zero_point, axis, quant_min, quant_max, torch.int8
        )
        dequantized_ref = torch.ops.quantized_decomposed.dequantize_per_channel(
            quantized_ref, scale, zero_point, axis, quant_min, quant_max, torch.int8, out_dtype=output_dtype
        )
        self.assertTrue(torch.equal(quantized, quantized_ref))
        self.assertTrue(torch.equal(dequantized, dequantized_ref))

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "skipping when torch verion is 2.4 or lower")
    def test_quantize_dequantize_tensor_asym(self):
        input = torch.randn(10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (10, 10)
        output_dtype = torch.float32
        scale, zero_point = choose_qparams_affine(input, mapping_type, block_size, dtype, eps=torch.finfo(torch.float32).eps)
        quantized = quantize_affine(input, block_size, scale, zero_point, dtype)
        dequantized = dequantize_affine(quantized, block_size, scale, zero_point, dtype, output_dtype=output_dtype)

        axis = 1
        quant_min = -128
        quant_max = 127
        quantized_ref = torch.ops.quantized_decomposed.quantize_per_tensor(
            input, scale, zero_point, quant_min, quant_max, torch.int8
        )
        dequantized_ref = torch.ops.quantized_decomposed.dequantize_per_tensor(
            quantized_ref, scale, zero_point, quant_min, quant_max, torch.int8, out_dtype=output_dtype
        )
        self.assertTrue(torch.equal(quantized, quantized_ref))
        self.assertTrue(torch.equal(dequantized, dequantized_ref))

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "skipping when torch verion is 2.4 or lower")
    def test_quantize_dequantize_channel_asym_4d(self):
        input = torch.randn(3, 3, 10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (3, 3, 1, 10)
        scale, zero_point = choose_qparams_affine(input, mapping_type, block_size, dtype, eps=torch.finfo(torch.float32).eps)
        quantized = quantize_affine(input, block_size, scale, zero_point, dtype)
        dequantized = dequantize_affine(quantized, block_size, scale, zero_point, dtype, output_dtype=torch.float32)

        axis = 2
        quant_min = -128
        quant_max = 127
        quantized_ref = torch.ops.quantized_decomposed.quantize_per_channel(
            input, scale, zero_point, axis, quant_min, quant_max, torch.int8
        )
        dequantized_ref = torch.ops.quantized_decomposed.dequantize_per_channel(
            quantized_ref, scale, zero_point, axis, quant_min, quant_max, torch.int8, out_dtype=torch.float32
        )
        self.assertTrue(torch.equal(quantized, quantized_ref))
        self.assertTrue(torch.equal(dequantized, dequantized_ref))

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "skipping when torch verion is 2.3 or lower")
    def test_quantize_dequantize_channel_asym_4d_multi_dim_reduction(self):
        input = torch.randn(3, 3, 10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (3, 3, 2, 2)
        scale, zero_point = choose_qparams_affine(input, mapping_type, block_size, dtype, eps=torch.finfo(torch.float32).eps)
        quantized = quantize_affine(input, block_size, scale, zero_point, dtype)
        dequantized = dequantize_affine(quantized, block_size, scale, zero_point, dtype, output_dtype=torch.float32)
        # we don't have corresponding ops in existing primitives, so just make sure it runs and it's close to float
        torch.testing.assert_close(dequantized, input, rtol=2, atol=0.02)

    def test_choose_qparams_tensor_asym_eps(self):
        input = torch.zeros(10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (10, 10)
        scale, zero_point = choose_qparams_affine(input, mapping_type, block_size, dtype)
        eps = torch.finfo(torch.float32).eps
        self.assertEqual(scale, eps)

    @unittest.skipIf(not torch.cuda.is_available(), "skipping when cuda is not available")
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
        """Make sure some errors are raised when user requested an unsupported type of quantization
        """
        input = torch.randn(10, 10)
        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (10, 10)
        scale, zero_point = choose_qparams_affine(input, mapping_type, block_size, dtype)


        # make sure we can't quantize int32 tensors:
        with self.assertRaisesRegex(AssertionError, "Unsupported input dtype:"):
            _ = quantize_affine(input.to(torch.int32), block_size, scale, zero_point, dtype)

        # block_size and scale/zero_point shape mismatch
        block_size = (1, 1)
        with self.assertRaisesRegex(RuntimeError, "is invalid for input of size 1"):
            _ = quantize_affine(input, block_size, scale, zero_point, dtype)

    def test_not_preserve_zero_not_supported(self):
        """Making sure preserve_zero == False is not supported for symmetric quant"""
        input = torch.randn(10, 256)
        n_bit = 4
        mapping_type = MappingType.SYMMETRIC
        dtype = torch.int8
        block_size = (1, 128)
        quant_min = 0
        quant_max = 2**n_bit - 1
        eps = 1e-6
        scale_dtype = torch.bfloat16
        zero_point_dtype = torch.bfloat16
        with self.assertRaisesRegex(ValueError, "preserve_zero == False is not supported for symmetric quantization"):
            choose_qparams_affine(
                input,
                mapping_type,
                block_size,
                dtype,
                quant_min,
                quant_max,
                eps,
                scale_dtype=scale_dtype,
                zero_point_dtype=zero_point_dtype,
                preserve_zero=False,
            )


    def test_tinygemm_get_groupwise_affine_qparams(self):
        from torchao.quantization.quant_primitives import ZeroPointDomain

        input = torch.randn(10, 256)
        n_bit = 4
        scale_ref, zero_point_ref = get_groupwise_affine_qparams(input, n_bit=n_bit, groupsize=128, dtype=torch.bfloat16)

        mapping_type = MappingType.ASYMMETRIC
        dtype = torch.int8
        block_size = (1, 128)
        quant_min = 0
        quant_max = 2**n_bit - 1
        eps = 1e-6
        scale_dtype = torch.bfloat16
        zero_point_dtype = torch.bfloat16
        scale, zero_point = \
            choose_qparams_affine(
                input,
                mapping_type,
                block_size,
                dtype,
                quant_min,
                quant_max,
                eps,
                scale_dtype=scale_dtype,
                zero_point_dtype=zero_point_dtype,
                preserve_zero=False,
                zero_point_domain=ZeroPointDomain.FLOAT,
            )

        self.assertTrue(torch.equal(scale, scale_ref))
        self.assertTrue(torch.equal(zero_point, zero_point_ref))


if __name__ == "__main__":
    unittest.main()
