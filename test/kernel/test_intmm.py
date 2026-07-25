# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.kernel.intmm import int_matmul, int_scaled_matmul, safe_int_mm


class TestIntMM(TestCase):
    def test_op_schemas_registered(self):
        """Verify that torch.ops.torchao custom op schemas exist."""
        self.assertTrue(hasattr(torch.ops.torchao, "int_matmul"))
        self.assertTrue(hasattr(torch.ops.torchao, "int_scaled_matmul"))

    def test_int_matmul_fallback(self):
        """Test int_matmul fallback path and op invocation."""
        a = torch.randint(-128, 127, (16, 32), dtype=torch.int8)
        b = torch.randint(-128, 127, (32, 16), dtype=torch.int8)

        expected = safe_int_mm(a, b)
        res_direct = int_matmul(a, b)
        res_op = torch.ops.torchao.int_matmul(a, b)

        self.assertEqual(res_direct, expected)
        self.assertEqual(res_op, expected)

    def test_int_scaled_matmul_fallback(self):
        """Test int_scaled_matmul fallback path and op invocation."""
        a = torch.randint(-128, 127, (16, 32), dtype=torch.int8)
        b = torch.randint(-128, 127, (32, 16), dtype=torch.int8)
        scales = torch.randn(16, 1, dtype=torch.float32)

        expected = safe_int_mm(a, b).float() * scales
        res_direct = int_scaled_matmul(a, b, scales)
        res_op = torch.ops.torchao.int_scaled_matmul(a, b, scales.expand((16, 16)))

        self.assertEqual(res_direct, expected)
        self.assertEqual(res_op, expected)

    def test_meta_impl(self):
        """Test Meta implementation for int_matmul and int_scaled_matmul."""
        a_meta = torch.empty((16, 32), device="meta", dtype=torch.int8)
        b_meta = torch.empty((32, 64), device="meta", dtype=torch.int8)
        scales_meta = torch.empty((16, 1), device="meta", dtype=torch.float32)

        res_meta = torch.ops.torchao.int_matmul(a_meta, b_meta)
        self.assertEqual(res_meta.shape, (16, 64))
        self.assertEqual(res_meta.device.type, "meta")
        self.assertEqual(res_meta.dtype, torch.int32)

        res_scaled_meta = torch.ops.torchao.int_scaled_matmul(
            a_meta, b_meta, scales_meta.expand((16, 64))
        )
        self.assertEqual(res_scaled_meta.shape, (16, 64))
        self.assertEqual(res_scaled_meta.device.type, "meta")
        self.assertEqual(res_scaled_meta.dtype, torch.float32)


if __name__ == "__main__":
    run_tests()
