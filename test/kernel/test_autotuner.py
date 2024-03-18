# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run
import unittest
import torch
import pytest
from parameterized import parameterized

from torchao.kernel import intmm_triton

class TestQuantFlow(unittest.TestCase):

    @parameterized.expand(
            [
                ("cuda", torch.bfloat16),
                ("cpu", torch.bfloat16),
                ("cuda", torch.float16),
                ("cpu", torch.float16),
            ])
    def test_int_mm(self, device, dtype):
        dtype = torch.bfloat16
        device = 'cuda'
        m, k, n = (128, 64, 16)
        x = torch.randn(m, k, dtype=dtype, device=device)
        w = torch.randn(n, k, dtype=dtype, device=device).t()
        x_int = x.to(dtype=torch.int8)
        w_int = w.to(dtype=torch.int8)
        out32_1 = intmm_triton.safe_int_mm(x_int, w_int)
        assert out32_1.dtype == torch.int32
        out32_2 = intmm_triton.int_matmul(x_int, w_int)
        assert out32_2.dtype == out32_1.dtype
        import pdb; pdb.set_trace()
        torch.testing.assert_allclose(out32_1, out32_2)

if __name__ == "__main__":
    unittest.main()
