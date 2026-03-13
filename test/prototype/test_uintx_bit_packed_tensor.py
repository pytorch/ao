# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.quantization import quantize_

try:
    import gemlite  # noqa: F401

    has_gemlite = True
except ModuleNotFoundError:
    has_gemlite = False


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not has_gemlite, "gemlite not available")
class TestUIntxBitPackedTensor(TestCase):
    def _test_quantize_and_linear(
        self, bit_width, group_size, packing_bitwidth, mode="weight_only"
    ):
        """Helper: quantize a linear layer and verify forward pass produces valid output."""
        from torchao.prototype.quantization.quant_api import UIntxWeightOnlyConfig

        in_features = 512
        out_features = 256
        model = torch.nn.Linear(in_features, out_features, bias=False).to(
            device="cuda", dtype=torch.float16
        )

        config = UIntxWeightOnlyConfig(
            group_size=group_size,
            bit_width=bit_width,
            packing_bitwidth=packing_bitwidth,
            mode=mode,
        )
        quantize_(model, config)

        # Verify weight is now UIntxBitPackedTensor
        from torchao.prototype.quantization.uintx.uintx_bit_packed_tensor import (
            UIntxBitPackedTensor,
        )

        self.assertIsInstance(model.weight, UIntxBitPackedTensor)

        # Verify forward pass works
        x = torch.randn(2, in_features, device="cuda", dtype=torch.float16)
        out = model(x)
        self.assertEqual(out.shape, (2, out_features))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

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

    def test_4bit_dynamic_mode(self):
        self._test_quantize_and_linear(
            bit_width=4, group_size=64, packing_bitwidth=32, mode="dynamic"
        )

    def test_slice_dim0(self):
        """Test narrow/slice on dim 0 (out_features) for tensor parallelism."""
        from torchao.prototype.quantization.quant_api import UIntxWeightOnlyConfig

        model = torch.nn.Linear(512, 256, bias=False).to(
            device="cuda", dtype=torch.float16
        )
        quantize_(
            model,
            UIntxWeightOnlyConfig(group_size=64, bit_width=4, packing_bitwidth=32),
        )

        sliced = model.weight.narrow(0, 0, 64)
        self.assertEqual(sliced.shape[0], 64)

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

        sliced = model.weight.narrow(1, 0, 128)
        self.assertEqual(sliced.shape[1], 128)

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

    def test_import_from_public_api(self):
        """Verify UIntxWeightOnlyConfig is available from the public API."""
        from torchao.quantization import UIntxWeightOnlyConfig  # noqa: F401


if __name__ == "__main__":
    run_tests()
