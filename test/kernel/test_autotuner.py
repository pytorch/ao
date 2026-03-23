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

from torchao.utils import is_sm_at_least_90, torch_version_at_least

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


class TestIntScaledMatmulCPUPaths(unittest.TestCase):
    """
    Tests for the CPU-specific paths inside _int_scaled_matmul_cpu.
    Because the u8s8 VNNI branch is gated on runtime CPU feature detection,
    CI machines are unlikely to exercise it naturally.  We monkeypatch the
    two helper functions so each branch can be tested on any machine.
    """

    def _make_inputs(self, m=64, k=32, n=16, dtype=torch.bfloat16):
        a = torch.randint(-128, 127, (m, k), dtype=torch.int8)
        b = torch.randint(-128, 127, (k, n), dtype=torch.int8)
        scales = torch.randn(m, 1, dtype=dtype)
        return a, b, scales

    def _reference(self, a, b, scales):
        from torchao.kernel.intmm import safe_int_mm

        return safe_int_mm(a, b).to(scales.dtype) * scales

    @unittest.skipIf(not torch_version_at_least("2.12.0.dev"), "Need torch 2.12+")
    def test_vnni_path_via_monkeypatch(self):
        """Force the u8s8 VNNI branch and verify against the reference result."""
        import torchao.kernel.intmm as intmm_mod

        a, b, scales = self._make_inputs()
        expected = self._reference(a, b, scales)

        orig_amx = intmm_mod._cpu_is_amx_tile_supported
        orig_vnni = intmm_mod._cpu_is_vnni_supported
        try:
            # Simulate: no AMX, but VNNI present → u8s8 compensation path
            intmm_mod._cpu_is_amx_tile_supported = lambda: False
            intmm_mod._cpu_is_vnni_supported = lambda: True
            result = intmm_mod._int_scaled_matmul_cpu(a, b, scales)
        finally:
            intmm_mod._cpu_is_amx_tile_supported = orig_amx
            intmm_mod._cpu_is_vnni_supported = orig_vnni

        torch.testing.assert_close(result, expected)

    @unittest.skipIf(not torch_version_at_least("2.12.0.dev"), "Need torch 2.12+")
    def test_amx_path_via_monkeypatch(self):
        """Force the s8s8 AMX/fallback branch and verify against the reference result."""
        import torchao.kernel.intmm as intmm_mod

        a, b, scales = self._make_inputs()
        expected = self._reference(a, b, scales)

        orig_amx = intmm_mod._cpu_is_amx_tile_supported
        orig_vnni = intmm_mod._cpu_is_vnni_supported
        try:
            # Simulate: AMX present → s8s8 direct path (no compensation)
            intmm_mod._cpu_is_amx_tile_supported = lambda: True
            intmm_mod._cpu_is_vnni_supported = lambda: False
            result = intmm_mod._int_scaled_matmul_cpu(a, b, scales)
        finally:
            intmm_mod._cpu_is_amx_tile_supported = orig_amx
            intmm_mod._cpu_is_vnni_supported = orig_vnni

        torch.testing.assert_close(result, expected)

    @unittest.skipIf(not torch_version_at_least("2.12.0.dev"), "Need torch 2.12+")
    def test_no_simd_path_via_monkeypatch(self):
        """Force the no-AMX/no-VNNI branch and verify against the reference result."""
        import torchao.kernel.intmm as intmm_mod

        a, b, scales = self._make_inputs()
        expected = self._reference(a, b, scales)

        orig_amx = intmm_mod._cpu_is_amx_tile_supported
        orig_vnni = intmm_mod._cpu_is_vnni_supported
        try:
            # Simulate: neither AMX nor VNNI → s8s8 reference path
            intmm_mod._cpu_is_amx_tile_supported = lambda: False
            intmm_mod._cpu_is_vnni_supported = lambda: False
            result = intmm_mod._int_scaled_matmul_cpu(a, b, scales)
        finally:
            intmm_mod._cpu_is_amx_tile_supported = orig_amx
            intmm_mod._cpu_is_vnni_supported = orig_vnni

        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
