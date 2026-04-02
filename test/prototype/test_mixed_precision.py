# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
import torch.nn as nn

from torchao.experimental.quant_api import UIntxWeightOnlyConfig
from torchao.quantization import quantize_
from torchao.quantization.utils import compute_error


class TestWeightOnlyQuantNaive(unittest.TestCase):
    # TODO: the previous intN_weight_only test covered bit widths [2, 3, 5, 6] on CPU
    # using AQT + PlainLayout which has been removed. UIntxWeightOnlyConfig from
    # torchao.experimental supports bit widths 1-7 but requires torchao C++ ops
    # (currently MPS-only). When a CPU-compatible arbitrary-bitwidth v2 path is
    # available, expand this test to cover [2, 3, 5, 6] bit widths again.
    @unittest.skipIf(
        not hasattr(torch.ops, "torchao")
        or not hasattr(torch.ops.torchao, "_pack_weight_4bit"),
        "torchao experimental C++ ops not available",
    )
    def test_quantization_intNwo(self):
        for quantization_bit in [4]:
            for symmetric in [False]:
                with self.subTest(
                    quantization_bit=quantization_bit, symmetric=symmetric
                ):
                    for x_shape in [[64, 32], [80, 80, 80, 32], [16, 64, 32]]:
                        x = torch.randn(*x_shape, dtype=torch.bfloat16)
                        m = nn.Sequential(nn.Linear(32, 80)).bfloat16()
                        y_ref = m(x)
                        quantize_(
                            m,
                            UIntxWeightOnlyConfig(
                                bitwidth=quantization_bit,
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
