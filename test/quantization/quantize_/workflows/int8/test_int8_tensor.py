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
from torchao.quantization.quantize_.common import (
    _choose_quant_func_and_quantize_tensor,
)
from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
    QuantizeTensorToInt8Kwargs,
)
from torchao.quantization.utils import compute_error, get_block_size
from torchao.testing.model_architectures import ToyTwoLinearModel
from torchao.testing.utils import TorchAOIntegrationTestCase, skip_if_xpu
from torchao.utils import (
    get_available_devices,
    get_current_accelerator_device,
    torch_version_at_least,
)

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


@unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
@common_utils.instantiate_parametrized_tests
class TestInt8Tensor(TorchAOIntegrationTestCase):
    def setUp(self):
        super().setUp()

        self.test_shape = (32, 20)
        self.dtype = torch.bfloat16
        self.batch_size = 32

        torch.manual_seed(42)

    @common_utils.parametrize("config", INT8_TEST_CONFIGS)
    @common_utils.parametrize("device", get_available_devices())
    def test_creation_and_attributes(self, config, device):
        """Test tensor creation, dtypes, and ranges"""
        linear = torch.nn.Linear(
            self.test_shape[1],
            self.test_shape[0],
            bias=False,
            dtype=self.dtype,
            device=device,
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
    @common_utils.parametrize("device", get_available_devices())
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
        device,
        compile: bool,
        sizes: tuple,
    ):
        """Test linear operation supports including shape and compile"""
        torch.compiler.reset()

        M, N, K = sizes
        input_tensor = torch.randn(*M, K, dtype=dtype, device=device)
        model = ToyTwoLinearModel(K, N, K, dtype=dtype, device=device).eval()
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
    @common_utils.parametrize("device", get_available_devices())
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
    @common_utils.parametrize("device", get_available_devices())
    def test_index_select(self, config, device):
        """test that `x_0 = x[0]` works when `x` is a 2D quantized tensor."""
        N, K = 256, 512
        x = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        linear = torch.nn.Linear(K, N, bias=False, dtype=torch.bfloat16, device=device)
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
    @common_utils.parametrize("device", get_available_devices())
    def test_dequantization_accuracy(self, config, device):
        """Test dequantization accuracy separately"""
        linear = torch.nn.Linear(
            256, 512, bias=False, dtype=torch.bfloat16, device=device
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
        device = get_current_accelerator_device()
        m = torch.nn.Sequential(
            torch.nn.Linear(K, N, device=device, dtype=torch.bfloat16)
        )

        config = Int8DynamicActivationInt8WeightConfig(version=2)
        quantize_(m, config)

        m = torch.compile(m)
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

        out, code = run_and_get_code(m, x)

        # Check expected kernels are present
        FileCheck().check_count("triton_per_fused", 1).check_count(
            "extern_kernels._int_mm", 1
        ).check_count("triton_poi_fused", 1).run(code[0])

    @skip_if_xpu("pin memory is not supported on XPU")
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


@unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
@common_utils.instantiate_parametrized_tests
class TestInt8StaticQuant(TorchAOIntegrationTestCase):
    @common_utils.parametrize("granularity", [PerRow(), PerTensor()])
    @common_utils.parametrize("dtype", [torch.bfloat16])
    @common_utils.parametrize("device", get_available_devices())
    def test_static_activation_per_row_int8_weight(self, granularity, dtype, device):
        torch.compiler.reset()

        M, N, K = 128, 128, 128

        input_tensor = torch.randn(M, K, dtype=dtype, device=device)

        model = torch.nn.Linear(K, N, bias=False).eval().to(device=device, dtype=dtype)
        model_static_quant = copy.deepcopy(model)
        model_dynamic_quant = copy.deepcopy(model)

        model_out_baseline = model(input_tensor)

        dynamic_config = Int8DynamicActivationInt8WeightConfig(
            version=2, granularity=granularity
        )
        quantize_(model_dynamic_quant, dynamic_config)

        dynamic_out_eager = model_dynamic_quant(input_tensor)
        sqnr_dynamic_eager = compute_error(model_out_baseline, dynamic_out_eager)

        model_dynamic_quant = torch.compile(model_dynamic_quant, fullgraph=True)

        dynamic_out_compile = model_dynamic_quant(input_tensor)
        sqnr_dynamic_compile = compute_error(model_out_baseline, dynamic_out_compile)

        # we use eager scales to calculate
        int8_input = _choose_quant_func_and_quantize_tensor(
            input_tensor, model_dynamic_quant.weight.act_quant_kwargs
        )

        static_config = Int8StaticActivationInt8WeightConfig(
            act_quant_scale=int8_input.scale.detach().clone(),
            granularity=granularity,
        )
        quantize_(model_static_quant, static_config)

        static_out_eager = model_static_quant(input_tensor)
        sqnr_static_eager = compute_error(model_out_baseline, static_out_eager)

        model_static_quant = torch.compile(model_static_quant, fullgraph=True)

        static_out_compile = model_static_quant(input_tensor)
        sqnr_static_compile = compute_error(model_out_baseline, static_out_compile)

        assert abs(sqnr_static_compile - sqnr_dynamic_compile) < 1, (
            f"Static compile vs dynamic compile SQNR difference too large: "
            f"static_compile={sqnr_static_compile} vs dynamic_compile={sqnr_dynamic_compile}, "
            f"diff={abs(sqnr_static_compile - sqnr_dynamic_compile)}"
        )
        assert abs(sqnr_static_eager - sqnr_dynamic_eager) < 1, (
            f"Static eager vs dynamic eager SQNR difference too large: "
            f"static_eager={sqnr_static_eager} vs dynamic_eager={sqnr_dynamic_eager}, "
            f"diff={abs(sqnr_static_eager - sqnr_dynamic_eager)}"
        )
        # eager numerics should match exactly
        # for compile, we can't compare dynamic vs static because we may get slightly different qparams when fused
        torch.testing.assert_close(dynamic_out_eager, static_out_eager)

    def test_static_activation_per_row_dim_0_not_supported(self):
        """Test that PerRow(dim=0) activation quantization raises an error.

        Per-token activation quantization (PerRow(dim=0)) would require slicing
        act_quant_scale when weight is sliced, which is not currently supported.
        """
        with self.assertRaises(ValueError) as cm:
            Int8StaticActivationInt8WeightConfig(
                granularity=PerRow(dim=0),  # This should fail
            )

        self.assertIn("PerRow(dim=-1)", str(cm.exception))

    @common_utils.parametrize("granularity", [PerRow(), PerTensor()])
    @common_utils.parametrize("device", get_available_devices())
    def test_static_act_quant_slice_and_select(self, granularity, device):
        """Test static activation quantization with slice and select operations.

        This test validates that PerRow(dim=-1) and PerTensor() work correctly
        with weight slicing. Per-token activation quantization (PerRow(dim=-1))
        should work with weight slicing since act_quant_scale doesn't need to be sliced.
        """
        N, K = 256, 512
        M = 32  # batch size
        dtype = torch.bfloat16

        linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device=device)
        input_tensor = torch.randn(M, K, dtype=dtype, device=device)
        int8_input = _choose_quant_func_and_quantize_tensor(
            input_tensor,
            QuantizeTensorToInt8Kwargs(
                granularity=granularity,
                mapping_type=MappingType.SYMMETRIC,
            ),
        )

        act_quant_scale = int8_input.scale.detach().clone()
        static_config = Int8StaticActivationInt8WeightConfig(
            act_quant_scale=act_quant_scale,
            granularity=granularity,
            act_mapping_type=MappingType.SYMMETRIC,
        )
        quantize_(linear, static_config)

        # Verify the weight has the correct act_quant_scale
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
    @common_utils.parametrize("device", get_available_devices())
    def test_int8_weight_only_v2_correct_eps(self, dtype, device):
        """
        Ensure that v2 of Int8WeightOnlyConfig uses the correct eps value.
        This test will fail if we use bfloat16 eps
        """
        torch.manual_seed(42)

        # Create test model
        model = ToyTwoLinearModel(256, 128, 256, dtype=dtype, device=device).eval()
        model_baseline = copy.deepcopy(model)

        # Create input
        input_tensor = torch.randn(32, 256, dtype=dtype, device=device)

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
