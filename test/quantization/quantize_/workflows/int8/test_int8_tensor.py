# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal import common_utils

from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8StaticActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quantize_.common import ObservedLinear
from torchao.quantization.utils import compute_error, get_block_size
from torchao.testing.model_architectures import ToyTwoLinearModel
from torchao.testing.utils import TorchAOIntegrationTestCase
from torchao.utils import torch_version_at_least

INT8_TEST_CONFIGS = [
    Int8WeightOnlyConfig(version=2, granularity=PerTensor()),
    Int8WeightOnlyConfig(version=2, granularity=PerRow()),
    Int8DynamicActivationInt8WeightConfig(
        version=2, granularity=PerTensor(), act_mapping_type=MappingType.SYMMETRIC
    ),
    Int8DynamicActivationInt8WeightConfig(
        version=2, granularity=PerRow(), act_mapping_type=MappingType.SYMMETRIC
    ),
]


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@common_utils.instantiate_parametrized_tests
class TestInt8Tensor(TorchAOIntegrationTestCase):
    def setUp(self):
        super().setUp()

        self.test_shape = (32, 20)
        self.dtype = torch.bfloat16
        self.batch_size = 32

        torch.manual_seed(42)

    @common_utils.parametrize("config", INT8_TEST_CONFIGS)
    def test_creation_and_attributes(self, config):
        """Test tensor creation, dtypes, and ranges"""
        linear = torch.nn.Linear(
            self.test_shape[1],
            self.test_shape[0],
            bias=False,
            dtype=self.dtype,
            device="cuda",
        )
        quantize_(linear, config)

        w = linear.weight

        self.assertEqual(w.shape, self.test_shape)
        self.assertEqual(w.qdata.dtype, torch.int8)
        self.assertTrue(torch.all(w.qdata >= -128) and torch.all(w.qdata <= 127))

        if isinstance(config.granularity, PerRow):
            self.assertEqual(w.scale.shape, (w.shape[0], 1))
        elif isinstance(config.granularity, PerTensor):
            self.assertEqual(w.scale.shape, (1, 1))

        if hasattr(config, "act_mapping_type"):
            self.assertEqual(w.act_quant_kwargs.mapping_type, config.act_mapping_type)

    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize("config", INT8_TEST_CONFIGS)
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),  # 2D
            ((32, 128), 64, 256),  # 3D
        ],
    )
    def test_int8_linear_variants(
        self,
        dtype: torch.dtype,
        config,
        compile: bool,
        sizes: tuple,
    ):
        """Test linear operation supports including shape and compile"""
        torch.compiler.reset()

        M, N, K = sizes
        input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")
        model = ToyTwoLinearModel(K, N, K, dtype=dtype, device="cuda").eval()
        model_q = copy.deepcopy(model)

        quantize_(model_q, config)

        if isinstance(config.granularity, PerRow):
            self.assertEqual(model_q.linear2.weight.scale.shape, (K, 1))
        elif isinstance(config.granularity, PerTensor):
            self.assertEqual(model_q.linear2.weight.scale.shape, (1, 1))

        self.assertEqual(model_q.linear2.weight.scale.ndim, 2)

        if compile:
            if isinstance(config, Int8WeightOnlyConfig) and isinstance(
                config.granularity, PerTensor
            ):
                # currently the inductor lowering for weight only quant in core does not support per-tensor gpu, so this errors. Skipping for now, but will address this in core
                return
            model_q = torch.compile(model_q, fullgraph=True)

        output_fp = model(input_tensor)
        output_quantized = model_q(input_tensor)

        assert compute_error(output_fp, output_quantized) > 20, (
            f"Quantization error is too high got a SQNR of {compute_error(output_fp, output_quantized)}"
        )

    @common_utils.parametrize("config", INT8_TEST_CONFIGS)
    @common_utils.parametrize("device", ["cpu", "cuda"])
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_slice(self, config, device, dtype):
        """Test tensor slicing with per-row quantization"""
        tensor_size = 256
        slice_sizes = (64, 128)

        dummy = torch.nn.Linear(
            tensor_size, tensor_size, bias=False, dtype=dtype, device=device
        )
        quantize_(dummy, config)

        weight1 = dummy.weight.clone().narrow(0, 0, slice_sizes[0])
        weight2 = dummy.weight.clone().narrow(1, 0, slice_sizes[1])

        self.assertEqual(weight1.qdata, dummy.weight.qdata.narrow(0, 0, slice_sizes[0]))
        self.assertEqual(weight2.qdata, dummy.weight.qdata.narrow(1, 0, slice_sizes[1]))

        if isinstance(config.granularity, PerRow):
            self.assertEqual(
                weight1.scale, dummy.weight.scale.narrow(0, 0, slice_sizes[0])
            )

        self.assertEqual(weight2.scale, dummy.weight.scale)
        with self.assertRaises(NotImplementedError):
            _ = dummy.weight[::2]

    @common_utils.parametrize("config", INT8_TEST_CONFIGS)
    def test_index_select(self, config):
        """test that `x_0 = x[0]` works when `x` is a 2D quantized tensor."""
        N, K = 256, 512
        x = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        linear = torch.nn.Linear(K, N, bias=False, dtype=torch.bfloat16, device="cuda")
        linear.weight.data = x

        quantize_(linear, config)

        x_int8 = linear.weight
        x_int8_0 = x_int8[0]

        # Test dequantization consistency
        torch.testing.assert_close(
            x_int8.dequantize()[0], x_int8_0.dequantize(), atol=0, rtol=0
        )

        # Test block_size granularity
        if isinstance(config.granularity, PerRow):
            self.assertEqual(
                list(get_block_size(x_int8.shape, config.granularity)), [1, K]
            )
        elif isinstance(config.granularity, PerTensor):
            self.assertEqual(
                list(get_block_size(x_int8.shape, config.granularity)), [N, K]
            )

    @common_utils.parametrize("config", INT8_TEST_CONFIGS)
    def test_dequantization_accuracy(self, config):
        """Test dequantization accuracy separately"""
        linear = torch.nn.Linear(
            256, 512, bias=False, dtype=torch.bfloat16, device="cuda"
        )
        weight_fp = copy.deepcopy(linear.weight)
        quantize_(linear, config)

        tensor = linear.weight
        dequantized = tensor.dequantize()
        self.assertEqual(dequantized.shape, weight_fp.shape)
        assert compute_error(dequantized, weight_fp) > 20, (
            f"Dequantization error is too high to get a SQNR of {compute_error(dequantized, weight_fp)}"
        )

    @unittest.skipIf(
        not torch_version_at_least("2.7.0"), "torch 2.6.0 and below has custom fx pass"
    )
    def test_available_gpu_kernels(self):
        """Check which GPU kernels are used"""
        torch.compiler.reset()

        M, K, N = 128, 256, 512
        m = torch.nn.Sequential(
            torch.nn.Linear(K, N, device="cuda", dtype=torch.bfloat16)
        )

        config = Int8DynamicActivationInt8WeightConfig(version=2)
        quantize_(m, config)

        m = torch.compile(m)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        out, code = run_and_get_code(m, x)

        # Check expected kernels are present
        FileCheck().check_count("triton_per_fused", 1).check_count(
            "extern_kernels._int_mm", 1
        ).check_count("triton_poi_fused", 1).run(code[0])

    @common_utils.parametrize("config", INT8_TEST_CONFIGS)
    def test_pin_memory(self, config):
        linear = torch.nn.Linear(
            256, 512, bias=False, dtype=torch.bfloat16, device="cuda"
        )
        quantize_(linear, config)
        weight_cpu = linear.weight.cpu()
        self.assertFalse(weight_cpu.is_pinned())

        weight_pinned = weight_cpu.pin_memory()

        self.assertTrue(weight_pinned.is_pinned())
        self.assertFalse(weight_cpu.is_pinned())

        self.assertTrue(weight_pinned.qdata.is_pinned())
        self.assertTrue(weight_pinned.scale.is_pinned())
        if weight_pinned.act_quant_scale is not None:
            self.assertTrue(weight_pinned.act_quant_scale.is_pinned())

        self.assertEqual(
            weight_cpu.dequantize(), weight_pinned.dequantize(), atol=0, rtol=0
        )

    @common_utils.parametrize("granularity", [PerRow()])
    def test_static_quantization(self, granularity):
        """Test observer-based static quantization: prepare -> calibrate -> convert"""
        M, K, N = 32, 64, 32
        model = ToyTwoLinearModel(K, N, K, dtype=torch.bfloat16, device="cuda").eval()

        # PREPARE
        quantize_(
            model,
            Int8StaticActivationInt8WeightConfig(
                step="prepare", granularity=granularity
            ),
        )

        # CALIBRATE
        for _ in range(5):
            with torch.no_grad():
                model(*model.example_inputs(batch_size=M))

        # CONVERT
        quantize_(model, Int8StaticActivationInt8WeightConfig(step="convert"))

        # Verify quantized
        self.assertNotIsInstance(model.linear1, ObservedLinear)
        self.assertNotIsInstance(model.linear2, ObservedLinear)

        # Test inference
        output = model(*model.example_inputs(batch_size=M))
        self.assertEqual(output.shape, (M, K))

    def test_static_activation_per_row_dim_0_not_supported(self):
        """Test that PerRow(dim=0) activation quantization raises an error.

        Per-token activation quantization (PerRow(dim=0)) would require slicing
        act_quant_scale when weight is sliced, which is not currently supported.
        """
        with self.assertRaises(ValueError) as cm:
            Int8StaticActivationInt8WeightConfig(
                step="prepare",
                granularity=PerRow(dim=0),  # This should fail
            )

        self.assertIn("PerRow(dim=-1)", str(cm.exception))

    @common_utils.parametrize("granularity", [PerRow()])
    def test_static_act_quant_slice_and_select(self, granularity):
        """Test static activation quantization with slice and select operations.

        This test validates that PerRow(dim=-1) works correctly with weight slicing.
        Per-token activation quantization (PerRow(dim=-1)) should work with weight
        slicing since act_quant_scale doesn't need to be sliced.
        """
        N, K = 256, 512
        M = 32  # batch size
        dtype = torch.bfloat16
        device = "cuda"

        model = torch.nn.Sequential(
            torch.nn.Linear(K, N, bias=False, dtype=dtype, device=device)
        )
        input_tensor = torch.randn(M, K, dtype=dtype, device=device)

        # PREPARE: Insert observer
        quantize_(
            model,
            Int8StaticActivationInt8WeightConfig(
                step="prepare",
                granularity=granularity,
                act_mapping_type=MappingType.SYMMETRIC,
            ),
        )

        # CALIBRATE: Run forward passes to collect statistics
        for _ in range(5):
            with torch.no_grad():
                model(input_tensor)

        # CONVERT: Convert to quantized model
        quantize_(
            model,
            Int8StaticActivationInt8WeightConfig(
                step="convert",
            ),
        )

        # Verify the weight has the correct act_quant_scale
        linear = model[0]
        original_weight = linear.weight
        original_act_quant_scale = original_weight.act_quant_scale
        assert original_act_quant_scale is not None

        # Slice the weight on dim=1 (input features)
        K_sliced = 256
        sliced_weight = linear.weight.narrow(1, 0, K_sliced)

        # Verify that unsupported indexing operations raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            _ = linear.weight[::2]  # Strided indexing not supported

        # Test functional correctness: sliced weight should work with appropriate input
        input_full = torch.randn(M, K, dtype=dtype, device=device)
        input_sliced = input_full[:, :K_sliced]

        # Create a reference linear with the sliced weight
        linear_ref = torch.nn.Linear(
            K_sliced, N, bias=False, dtype=dtype, device=device
        )
        linear_ref.weight.data = original_weight.dequantize()[:, :K_sliced]

        # Both should produce similar outputs
        with torch.no_grad():
            output_ref = linear_ref(input_sliced)
            output_quantized = torch.nn.functional.linear(input_sliced, sliced_weight)

        # Verify reasonable quantization error
        error = compute_error(output_ref, output_quantized)
        self.assertGreater(error, 15, f"Quantization SQNR too low: {error}")

    @common_utils.parametrize("dtype", [torch.bfloat16])
    def test_int8_weight_only_v2_correct_eps(self, dtype):
        """
        Ensure that v2 of Int8WeightOnlyConfig uses the correct eps value.
        This test will fail if we use bfloat16 eps
        """
        torch.manual_seed(42)

        # Create test model
        model = ToyTwoLinearModel(256, 128, 256, dtype=dtype, device="cuda").eval()
        model_baseline = copy.deepcopy(model)

        # Create input
        input_tensor = torch.randn(32, 256, dtype=dtype, device="cuda")

        # Get baseline output
        output_baseline = model_baseline(input_tensor)

        # Apply Int8WeightOnlyConfig quantization (uses Int8Tensor)
        config = Int8WeightOnlyConfig(version=2, granularity=PerRow())
        quantize_(model, config)
        output = model(input_tensor)

        # Compute SQNR and make sure it's above 40
        sqnr = compute_error(output_baseline, output)
        self.assertGreater(sqnr, 40, f"SQNR too low: {sqnr}")


if __name__ == "__main__":
    common_utils.run_tests()
