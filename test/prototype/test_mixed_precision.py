# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
import torch.nn as nn

from torchao.quantization import quantize_
from torchao.quantization.utils import compute_error


class TestWeightOnlyQuantNaive(unittest.TestCase):
    # TODO: the previous intN_weight_only test covered bit widths [2, 3, 5, 6] on CPU
    # using AQT + PlainLayout which has been removed. UIntxWeightOnlyConfig from
    # torchao.prototype supports bit widths 4 and 8 but requires gemlite + CUDA.
    # When a CPU-compatible arbitrary-bitwidth v2 path is available, expand this
    # test to cover [2, 3, 5, 6] bit widths again.
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "UIntxWeightOnlyConfig requires CUDA",
    )
    def test_quantization_intNwo(self):
        try:
            import gemlite  # noqa: F401
        except ImportError:
            self.skipTest("gemlite not available")

        from torchao.prototype.quantization.quant_api import UIntxWeightOnlyConfig

        for quantization_bit in [4]:
            with self.subTest(quantization_bit=quantization_bit):
                for x_shape in [[64, 32], [80, 80, 80, 32], [16, 64, 32]]:
                    x = torch.randn(*x_shape, dtype=torch.bfloat16, device="cuda")
                    m = nn.Sequential(nn.Linear(32, 80)).bfloat16().cuda()
                    y_ref = m(x)
                    quantize_(
                        m,
                        UIntxWeightOnlyConfig(
                            bit_width=quantization_bit,
                            group_size=32,
                        ),
                    )
                    y_wo = m(x)
                    sqnr = compute_error(y_ref, y_wo)
                    expected_sqnr_threshold = 44.0 - (8 - quantization_bit) * 6.02
                    self.assertGreater(
                        sqnr, expected_sqnr_threshold, f"sqnr: {sqnr} is too low"
                    )


if __name__ == "__main__":
    unittest.main()
