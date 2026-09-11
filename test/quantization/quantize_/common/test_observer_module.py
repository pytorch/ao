# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchao.quantization.granularity import PerTensor
from torchao.quantization.observer import AffineQuantizedMinMaxObserver
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quantize_.common.observer_module import ObservedLinear


class TestObservedLinear(unittest.TestCase):
    def setUp(self):
        self.obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.int8,
            granularity=PerTensor(),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
        )
        self.linear = nn.Linear(64, 128, bias=True)
        self.observed_weight = ObservedLinear.from_hp(self.linear, self.obs)

    def test_subclass_creation(self):
        """Test that ObservedLinear can be created from a hp linear"""
        self.assertIsInstance(self.observed_weight, torch.Tensor)
        self.assertIsInstance(self.observed_weight, ObservedLinear)
        self.assertEqual(self.observed_weight.shape, self.linear.weight.shape)
        self.assertEqual(self.observed_weight.device, self.linear.weight.device)
        self.assertEqual(self.observed_weight.dtype, self.linear.weight.dtype)
        self.assertTrue(
            torch.equal(self.observed_weight.original_weight_tensor, self.linear.weight)
        )
        self.assertIs(self.observed_weight.act_obs, self.obs)

    def test_multiple_forward_passes(self):
        """Test multiple forward passes accumulate observations correctly"""
        self.linear.weight = nn.Parameter(self.observed_weight)

        # First pass with small range
        self.linear(torch.randn(32, 64) * 0.01)
        first_max = self.obs.max_val.clone()

        # Second pass with larger range — max should expand
        self.linear(torch.randn(32, 64) * 100)
        self.assertGreater(self.obs.max_val.item(), first_max.item())

        # Verify qparams are computable
        scale, zero_point = self.obs.calculate_qparams()
        self.assertIsNotNone(scale)
        self.assertTrue(torch.isfinite(scale).all())

    def test_tensor_operations(self):
        """Test tensor operations (detach, clone) preserve ObservedLinear"""
        detached = self.observed_weight.detach()
        self.assertIsInstance(detached, ObservedLinear)
        self.assertIs(detached.act_obs, self.obs)

        cloned = self.observed_weight.clone()
        self.assertIsInstance(cloned, ObservedLinear)
        self.assertIs(cloned.act_obs, self.obs)


if __name__ == "__main__":
    unittest.main()
