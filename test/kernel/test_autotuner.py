# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run
import logging
import os
import unittest

import torch
from parameterized import parameterized

from torchao.utils import is_sm_at_least_90

logging.basicConfig(level=logging.INFO)


class TestQuantFlow(unittest.TestCase):
    def setUp(self):
        os.environ["TORCHAO_AUTOTUNER_ENABLE"] = "1"

    def tearDown(self):
        del os.environ["TORCHAO_AUTOTUNER_ENABLE"]

    @parameterized.expand(
        [
            ("cuda", torch.bfloat16),
            # TODO: ("cpu", torch.bfloat16),
            ("cuda", torch.float16),
            # TODO: ("cpu", torch.float16),
        ]
    )
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_int_mm(self, device, dtype):
        from torchao.kernel import intmm

        dtype = torch.bfloat16
        m, k, n = (128, 64, 16)
        x = torch.randn(m, k, dtype=dtype, device=device)
        w = torch.randn(n, k, dtype=dtype, device=device).t()
        x_int = x.to(dtype=torch.int8)
        w_int = w.to(dtype=torch.int8)
        out32_1 = intmm.safe_int_mm(x_int, w_int)
        assert out32_1.dtype == torch.int32
        out32_2 = intmm.int_matmul(x_int, w_int)
        assert out32_2.dtype == out32_1.dtype
        torch.testing.assert_allclose(out32_1, out32_2)

    @parameterized.expand(
        [
            ("cuda", torch.bfloat16),
            ("cuda", torch.float16),
        ]
    )
    @unittest.skipIf(not is_sm_at_least_90(), "Needs H100")
    def test_int_mm_float8(self, device, dtype):
        from torchao.kernel import intmm

        dtype = torch.bfloat16
        m, k, n = (128, 64, 16)
        x = torch.randn(m, k, dtype=dtype, device=device)
        w = torch.randn(n, k, dtype=dtype, device=device).t()
        x_float8 = x.to(dtype=torch.float8_e4m3fn)
        w_float8 = w.to(dtype=torch.float8_e4m3fn)
        out32_1 = intmm.safe_int_mm(x_float8, w_float8)
        assert out32_1.dtype == torch.int32

    @parameterized.expand(
        [
            ("cuda", torch.bfloat16),
            ("cpu", torch.bfloat16),
            ("cuda", torch.float16),
            ("cpu", torch.float16),
        ]
    )
    def test_int_scaled_mm(self, device, dtype):
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest(f"{device} not available")

        from torchao.kernel import intmm

        dtype = torch.bfloat16
        m, k, n = (128, 64, 16)
        x = torch.randn(m, k, dtype=dtype, device=device)
        scales = x.sum(-1, keepdim=True)
        w = torch.randn(n, k, dtype=dtype, device=device).t()
        x_int = x.to(dtype=torch.int8)
        w_int = w.to(dtype=torch.int8)
        out32_1 = intmm.safe_int_mm(x_int, w_int) * scales
        assert out32_1.dtype == torch.bfloat16
        out32_2 = intmm.int_scaled_matmul(x_int, w_int, scales)
        assert out32_2.dtype == out32_1.dtype
        torch.testing.assert_allclose(out32_1, out32_2)


if __name__ == "__main__":
    unittest.main()
