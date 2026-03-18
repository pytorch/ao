# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.quantization import quantize_
from torchao.quantization.utils import compute_error

try:
    import gemlite  # noqa: F401

    has_gemlite = True
except ModuleNotFoundError:
    has_gemlite = False


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not has_gemlite, "gemlite not available")
class TestUIntxBitPackedTensor(TestCase):
    def _test_quantize_and_linear(
        self, bit_width, group_size, packing_bitwidth, sqnr_threshold=24
    ):
        """Helper: quantize a linear layer and verify forward pass produces valid output."""
        from torchao.prototype.quantization.quant_api import UIntxWeightOnlyConfig

        in_features = 512
        out_features = 256
        model = torch.nn.Linear(in_features, out_features, bias=False).to(
            device="cuda", dtype=torch.float16
        )

        # Compute reference output before quantization
        x = torch.randn(2, in_features, device="cuda", dtype=torch.float16)
        ref_out = model(x)

        config = UIntxWeightOnlyConfig(
            group_size=group_size,
            bit_width=bit_width,
            packing_bitwidth=packing_bitwidth,
        )
        quantize_(model, config)

        # Verify weight is now UIntxBitPackedTensor
        from torchao.prototype.quantization.uintx.uintx_bit_packed_tensor import (
            UIntxBitPackedTensor,
        )

        self.assertIsInstance(model.weight, UIntxBitPackedTensor)

        # Verify forward pass works and is close to non-quantized output
        out = model(x)
        self.assertEqual(out.shape, (2, out_features))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

        sqnr = compute_error(ref_out, out)
        self.assertGreater(
            sqnr,
            sqnr_threshold,
            f"SQNR {sqnr:.1f} dB is below threshold {sqnr_threshold} dB",
        )

    def test_4bit_group64_pack32(self):
        self._test_quantize_and_linear(bit_width=4, group_size=64, packing_bitwidth=32)

    def test_4bit_group128_pack32(self):
        self._test_quantize_and_linear(bit_width=4, group_size=128, packing_bitwidth=32)

    def test_4bit_group64_pack8(self):
        self._test_quantize_and_linear(bit_width=4, group_size=64, packing_bitwidth=8)

    def test_8bit_perchannel_pack32(self):
        self._test_quantize_and_linear(
            bit_width=8, group_size=None, packing_bitwidth=32
        )

    def test_8bit_perchannel_pack8(self):
        self._test_quantize_and_linear(bit_width=8, group_size=None, packing_bitwidth=8)

    def _test_dynamic_quantize_and_linear(
        self, bit_width, group_size, packing_bitwidth, sqnr_threshold=22
    ):
        """Helper: quantize with dynamic activation and verify forward pass."""
        from torchao.prototype.quantization.quant_api import (
            Int8DynamicActivationUIntxWeightConfig,
        )

        in_features = 512
        out_features = 256
        model = torch.nn.Linear(in_features, out_features, bias=False).to(
            device="cuda", dtype=torch.float16
        )

        # Compute reference output before quantization
        x = torch.randn(2, in_features, device="cuda", dtype=torch.float16)
        ref_out = model(x)

        config = Int8DynamicActivationUIntxWeightConfig(
            group_size=group_size,
            bit_width=bit_width,
            packing_bitwidth=packing_bitwidth,
        )
        quantize_(model, config)

        from torchao.prototype.quantization.uintx.uintx_bit_packed_tensor import (
            UIntxBitPackedTensor,
        )

        self.assertIsInstance(model.weight, UIntxBitPackedTensor)

        # Verify forward pass works and is close to non-quantized output
        out = model(x)
        self.assertEqual(out.shape, (2, out_features))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

        sqnr = compute_error(ref_out, out)
        self.assertGreater(
            sqnr,
            sqnr_threshold,
            f"SQNR {sqnr:.1f} dB is below threshold {sqnr_threshold} dB",
        )

    def test_dynamic_4bit_group64_pack32(self):
        self._test_dynamic_quantize_and_linear(
            bit_width=4, group_size=64, packing_bitwidth=32
        )

    def test_dynamic_4bit_group128_pack32(self):
        self._test_dynamic_quantize_and_linear(
            bit_width=4, group_size=128, packing_bitwidth=32
        )

    def test_dynamic_4bit_group64_pack8(self):
        self._test_dynamic_quantize_and_linear(
            bit_width=4, group_size=64, packing_bitwidth=8
        )

    def test_dynamic_8bit_perchannel_pack32(self):
        self._test_dynamic_quantize_and_linear(
            bit_width=8, group_size=None, packing_bitwidth=32
        )

    def test_dynamic_8bit_perchannel_pack8(self):
        self._test_dynamic_quantize_and_linear(
            bit_width=8, group_size=None, packing_bitwidth=8
        )

    def test_slice_dim0(self):
        """Test narrow/slice on dim 0 (out_features) for tensor parallelism."""
        from torchao.prototype.quantization.quant_api import UIntxWeightOnlyConfig

        model = torch.nn.Linear(512, 256, bias=False).to(
            device="cuda", dtype=torch.float16
        )
        x = torch.randn(2, 512, device="cuda", dtype=torch.float16)
        quantize_(
            model,
            UIntxWeightOnlyConfig(group_size=64, bit_width=4, packing_bitwidth=32),
        )
        full_out = model(x)

        weight = model.weight
        sliced = weight.narrow(0, 0, 64)
        self.assertEqual(sliced.shape[0], 64)

        # Verify internal tensors match direct slicing
        # Data is stored transposed (K x N), so logical dim 0 -> data dim 1
        self.assertEqual(
            sliced.packed_weight,
            weight.packed_weight.narrow(1, 0, 64),
        )
        self.assertEqual(
            sliced.scale,
            weight.scale.narrow(1, 0, 64),
        )

        # Verify forward pass with sliced weight matches full output
        model_sliced = torch.nn.Linear(512, 64, bias=False).to(
            device="cuda", dtype=torch.float16
        )
        model_sliced.weight = torch.nn.Parameter(sliced, requires_grad=False)
        sliced_out = model_sliced(x)
        self.assertEqual(sliced_out.shape, (2, 64))
        self.assertTrue(torch.equal(sliced_out, full_out[:, :64]))

    def test_slice_dim1(self):
        """Test narrow/slice on dim 1 (in_features) for tensor parallelism."""
        from torchao.prototype.quantization.quant_api import UIntxWeightOnlyConfig

        model = torch.nn.Linear(512, 256, bias=False).to(
            device="cuda", dtype=torch.float16
        )
        quantize_(
            model,
            UIntxWeightOnlyConfig(group_size=64, bit_width=4, packing_bitwidth=32),
        )

        weight = model.weight
        sliced = weight.narrow(1, 0, 128)
        self.assertEqual(sliced.shape[1], 128)

        # Verify internal tensors match direct slicing
        # Data is stored transposed (K x N), so logical dim 1 -> data dim 0
        # packed_weight dim 0 is packed by elements_per_sample
        eps = weight.gemlite_kwargs["elements_per_sample"]
        self.assertEqual(
            sliced.packed_weight,
            weight.packed_weight.narrow(0, 0, 128 // eps),
        )
        # scale dim 0 corresponds to groups along in_features
        scale_ratio = 128 // 64  # in_features_slice / group_size
        self.assertEqual(
            sliced.scale,
            weight.scale.narrow(0, 0, scale_ratio),
        )

        # Verify forward pass with sliced weight produces valid output
        model_sliced = torch.nn.Linear(128, 256, bias=False).to(
            device="cuda", dtype=torch.float16
        )
        model_sliced.weight = torch.nn.Parameter(sliced, requires_grad=False)
        x_half = torch.randn(2, 128, device="cuda", dtype=torch.float16)
        sliced_out = model_sliced(x_half)
        self.assertEqual(sliced_out.shape, (2, 256))
        self.assertFalse(torch.isnan(sliced_out).any())
        self.assertFalse(torch.isinf(sliced_out).any())

    def test_fqn_to_config_non_weight_param(self):
        """Test that UIntx configs quantize a non-weight parameter via FqnToConfig."""
        from torchao.prototype.quantization.quant_api import (
            Int8DynamicActivationUIntxWeightConfig,
            UIntxWeightOnlyConfig,
        )
        from torchao.prototype.quantization.uintx.uintx_bit_packed_tensor import (
            UIntxBitPackedTensor,
        )
        from torchao.quantization.quant_api import FqnToConfig

        configs = [
            UIntxWeightOnlyConfig(group_size=128, bit_width=4, packing_bitwidth=32),
            Int8DynamicActivationUIntxWeightConfig(
                group_size=128, bit_width=4, packing_bitwidth=32
            ),
        ]
        for config in configs:
            with self.subTest(config=type(config).__name__):
                model = torch.nn.Sequential(
                    torch.nn.Linear(128, 128, bias=False).to(
                        device="cuda", dtype=torch.float16
                    )
                )
                model[0].register_parameter(
                    "custom_param",
                    torch.nn.Parameter(
                        torch.randn(128, 128, dtype=torch.float16, device="cuda")
                    ),
                )
                original_custom_param = model[0].custom_param
                original_weight = model[0].weight

                quantize_(
                    model, FqnToConfig({"0.custom_param": config}), filter_fn=None
                )

                self.assertIsInstance(model[0].custom_param, UIntxBitPackedTensor)
                self.assertIsNot(model[0].custom_param, original_custom_param)
                self.assertIs(model[0].weight, original_weight)

    def test_non_standard_shapes(self):
        """Test shapes not divisible by 128 but divisible by 32 (gemlite requirement)."""
        from torchao.prototype.quantization.quant_api import UIntxWeightOnlyConfig

        # gemlite requires in_features divisible by 32 or group_size
        model = torch.nn.Linear(1024, 1025, bias=False).to(
            device="cuda", dtype=torch.float16
        )
        config = UIntxWeightOnlyConfig(
            group_size=None, bit_width=4, packing_bitwidth=32
        )
        quantize_(model, config)

        x = torch.randn(1, 1024, device="cuda", dtype=torch.float16)
        out = model(x)
        self.assertEqual(out.shape, (1, 1025))

    def test_dequantize_8bit(self):
        """Test that dequantize() is correct for 8-bit symmetric quantization."""
        from torchao.prototype.quantization.quant_api import UIntxWeightOnlyConfig

        model = torch.nn.Linear(512, 256, bias=False).to(
            device="cuda", dtype=torch.float16
        )
        original_weight = model.weight.clone()
        quantize_(
            model,
            UIntxWeightOnlyConfig(group_size=None, bit_width=8, packing_bitwidth=32),
        )

        dequantized = model.weight.dequantize()
        self.assertEqual(dequantized.shape, original_weight.shape)
        self.assertEqual(dequantized.dtype, original_weight.dtype)

        sqnr = compute_error(original_weight, dequantized)
        self.assertGreater(
            sqnr,
            30,
            f"8-bit dequantize SQNR {sqnr:.1f} dB is too low",
        )

    def test_dequantize_4bit(self):
        """Test that dequantize() is correct for 4-bit asymmetric quantization."""
        from torchao.prototype.quantization.quant_api import UIntxWeightOnlyConfig

        model = torch.nn.Linear(512, 256, bias=False).to(
            device="cuda", dtype=torch.float16
        )
        original_weight = model.weight.clone()
        quantize_(
            model,
            UIntxWeightOnlyConfig(group_size=64, bit_width=4, packing_bitwidth=32),
        )

        dequantized = model.weight.dequantize()
        self.assertEqual(dequantized.shape, original_weight.shape)
        self.assertEqual(dequantized.dtype, original_weight.dtype)

        sqnr = compute_error(original_weight, dequantized)
        self.assertGreater(
            sqnr,
            20,
            f"4-bit dequantize SQNR {sqnr:.1f} dB is too low",
        )

    def test_save_load_roundtrip(self):
        """Test torch.save / torch.load round-trip preserves quantized weights."""
        from torchao.prototype.quantization.quant_api import UIntxWeightOnlyConfig
        from torchao.prototype.quantization.uintx.uintx_bit_packed_tensor import (
            UIntxBitPackedTensor,
        )

        model = torch.nn.Linear(512, 256, bias=False).to(
            device="cuda", dtype=torch.float16
        )
        x = torch.randn(2, 512, device="cuda", dtype=torch.float16)
        quantize_(
            model,
            UIntxWeightOnlyConfig(group_size=64, bit_width=4, packing_bitwidth=32),
        )
        ref_out = model(x)

        # Save and load state_dict
        with tempfile.NamedTemporaryFile() as f:
            torch.save(model.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f, weights_only=True)

        # Load into a fresh model
        model2 = torch.nn.Linear(512, 256, bias=False).to(
            device="cuda", dtype=torch.float16
        )
        model2.load_state_dict(state_dict, assign=True)

        self.assertIsInstance(model2.weight, UIntxBitPackedTensor)
        out = model2(x)
        self.assertTrue(torch.equal(ref_out, out))

    def test_compile(self):
        """Test torch.compile compatibility."""
        from torchao.prototype.quantization.quant_api import UIntxWeightOnlyConfig

        model = torch.nn.Linear(512, 256, bias=False).to(
            device="cuda", dtype=torch.float16
        )
        x = torch.randn(2, 512, device="cuda", dtype=torch.float16)
        quantize_(
            model,
            UIntxWeightOnlyConfig(group_size=64, bit_width=4, packing_bitwidth=32),
        )
        ref_out = model(x)

        compiled_model = torch.compile(model)
        compiled_out = compiled_model(x)

        sqnr = compute_error(ref_out, compiled_out)
        self.assertGreater(
            sqnr,
            35,
            f"Compiled output SQNR {sqnr:.1f} dB vs eager is too low",
        )


if __name__ == "__main__":
    run_tests()
