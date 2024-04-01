# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run
import unittest
import torch
from torchao.quantization.quant_primitives import get_group_qparams_symmetric
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_4

class TestQuantPrimitives(unittest.TestCase):
    SEED = 123

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "skipping when torch verion is 2.3 or lower")
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
        (scale_ao, _) = get_group_qparams_symmetric(weight, n_bit, groupsize)
        torch.testing.assert_allclose(scale_obs, scale_ao, rtol=0, atol=0)

if __name__ == "__main__":
    unittest.main()
