# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run

import copy
import unittest

import torch
from torch.ao.quantization import (
    FakeQuantize,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization._qat_quant_api import SimpleFakeQuantize


class TestQATQuantAPI(unittest.TestCase):
    SEED = 123

    def test_simple_fake_quantize(self):
        """
        Test that `SimpleFakeQuantize` produces the exact same behavior as
        `torch.ao.quantization.FakeQuantize` by default.
        """
        observer_ctr = MinMaxObserver.with_args(quant_min=0, quant_max=127)
        old_fq = FakeQuantize(observer=observer_ctr)
        simple_fq = SimpleFakeQuantize(observer=observer_ctr())

        # Test numerics match exactly
        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256)
        x2 = copy.deepcopy(x)
        y = old_fq(x)
        y2 = simple_fq(x2)
        torch.testing.assert_allclose(y, y2, rtol=0, atol=0)

        # Test get observer attributes
        self.assertEquals(simple_fq.quant_min, 0)
        self.assertEquals(simple_fq.quant_max, 127)
        self.assertEquals(simple_fq.qscheme, torch.per_tensor_affine)
        self.assertEquals(simple_fq.is_dynamic, False)
        self.assertEquals(simple_fq.eps, torch.finfo(torch.float32).eps)

    def test_simple_fake_quantize_per_channel(self):
        """
        Test that `SimpleFakeQuantize` produces the exact same behavior as
        `torch.ao.quantization.FakeQuantize` in the per channel case.
        """

        def _fake_quantize_per_channel_affine(x, scale, zp, qmin, qmax):
            return torch.fake_quantize_per_channel_affine(x, scale, zp, 0, qmin, qmax)

        observer_ctr = PerChannelMinMaxObserver.with_args(
            ch_axis=0,
            quant_min=0,
            quant_max=127,
            qscheme=torch.per_channel_affine,
        )
        old_fq = FakeQuantize(observer=observer_ctr)
        simple_fq = SimpleFakeQuantize(
            observer=observer_ctr(),
            fake_quant_op=_fake_quantize_per_channel_affine,
        )

        # Test numerics match exactly
        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256)
        x2 = copy.deepcopy(x)
        y = old_fq(x)
        y2 = simple_fq(x2)
        torch.testing.assert_allclose(y, y2, rtol=0, atol=0)

        # Test get observer attributes
        self.assertEquals(simple_fq.ch_axis, 0)
        self.assertEquals(simple_fq.quant_min, 0)
        self.assertEquals(simple_fq.quant_max, 127)
        self.assertEquals(simple_fq.qscheme, torch.per_channel_affine)
        self.assertEquals(simple_fq.is_dynamic, False)
        self.assertEquals(simple_fq.eps, torch.finfo(torch.float32).eps)


if __name__ == "__main__":
    unittest.main()
