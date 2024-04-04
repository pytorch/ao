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
from torchao.quantization._qat_quant_api import (
    SimpleFakeQuantize,
    SymmetricPerChannelGroupMinMaxObserver,
)
from torchao.quantization.quant_primitives import get_group_qparams_symmetric


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

        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256)
        x2 = copy.deepcopy(x)
        y = old_fq(x)
        y2 = simple_fq(x2)
        torch.testing.assert_allclose(y, y2, rtol=0, atol=0)

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

        torch.manual_seed(self.SEED)
        x = torch.randn(100, 256)
        x2 = copy.deepcopy(x)
        y = old_fq(x)
        y2 = simple_fq(x2)
        torch.testing.assert_allclose(y, y2, rtol=0, atol=0)

    def test_symmetric_per_channel_group_observer(self):
        """
        Test that `SymmetricPerChannelGroupObserver` returns the same
        qparams as `get_group_qparams_symmetric` for 4-bit quant.
        """
        n_bit = 4
        qmin = -(2 ** (n_bit - 1))
        qmax = 2 ** (n_bit - 1) - 1
        group_size = 2

        observer = SymmetricPerChannelGroupMinMaxObserver(
            ch_axis=0,
            quant_min=qmin,
            quant_max=qmax,
            group_size=group_size,
        )

        torch.manual_seed(self.SEED)
        x = torch.randn(3, 4)
        x2 = copy.deepcopy(x)
        observer(x)
        (scale_obs, zp_obs) = observer.calculate_qparams()
        (scale_orig, zp_orig) = get_group_qparams_symmetric(x2, n_bit, group_size)
        print("SCALE OBS", scale_obs)
        print("SCALE ORIG", scale_orig)
        torch.testing.assert_allclose(scale_obs, scale_orig, rtol=0, atol=0)
        torch.testing.assert_allclose(zp_obs, zp_orig, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
