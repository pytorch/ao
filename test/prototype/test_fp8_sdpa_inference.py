# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FP8 SDPA inference prototype.
"""

import unittest

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)


def _is_fp8_sdpa_available():
    """Check if FP8 SDPA is available (requires CUDA and FA3 support)."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (9, 0):
        return False
    try:
        # Check for FA3 support
        from torch.nn.attention import activate_flash_attention_impl

        activate_flash_attention_impl("FA3")
        return True
    except Exception:
        return False


@common_utils.instantiate_parametrized_tests
class TestFP8SDPAInference(TestCase):
    """Test cases for FP8 SDPA inference."""

    @unittest.skipIf(not _is_fp8_sdpa_available(), "FP8 SDPA requires CUDA and FA3")
    @common_utils.parametrize(
        "shape",
        [
            (2, 8, 1024, 64),
            (1, 8, 4096, 64),
            (1, 16, 1024, 128),
        ],
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_numerical_accuracy(self, shape, dtype):
        """Test FP8 SDPA matches regular SDPA within acceptable SQNR."""
        from torch.nn.attention import (
            activate_flash_attention_impl,
            restore_flash_attention_impl,
        )

        from torchao.prototype.fp8_sdpa_inference import fp8_sdpa_parallel

        B, H, S, D = shape

        q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        with torch.no_grad():
            # Regular SDPA
            out_regular = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False
            )

            # FP8 SDPA
            activate_flash_attention_impl("FA3")
            out_fp8 = fp8_sdpa_parallel(q, k, v, is_causal=False)
            restore_flash_attention_impl()

        # Compute SQNR
        sqnr = 10 * torch.log10(
            torch.mean(out_regular.pow(2)) / torch.mean((out_regular - out_fp8).pow(2))
        )

        self.assertGreater(
            sqnr.item(),
            25.0,
            f"SQNR {sqnr.item():.2f} dB is below threshold of 25 dB",
        )

    @unittest.skipIf(not _is_fp8_sdpa_available(), "FP8 SDPA requires CUDA and FA3")
    def test_different_qkv_shapes(self):
        """Test cross-attention case where Q sequence length differs from K/V."""
        from torch.nn.attention import (
            activate_flash_attention_impl,
            restore_flash_attention_impl,
        )

        from torchao.prototype.fp8_sdpa_inference import fp8_sdpa_parallel

        B, H, D = 2, 8, 64
        S_q = 256  # Query sequence length
        S_kv = 1024  # Key/Value sequence length (different from Q)

        q = torch.randn(B, H, S_q, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S_kv, D, device="cuda", dtype=torch.bfloat16)

        activate_flash_attention_impl("FA3")
        with torch.no_grad():
            out_fp8 = fp8_sdpa_parallel(q, k, v, is_causal=False)

        # Check output shape is correct
        self.assertEqual(out_fp8.shape, (B, H, S_q, D))

        # Check output is not NaN
        self.assertFalse(torch.isnan(out_fp8).any())
        restore_flash_attention_impl()

    @unittest.skipIf(not _is_fp8_sdpa_available(), "FP8 SDPA requires CUDA and FA3")
    def test_wrap_module_with_fp8_sdpa(self):
        """Test that wrap_module_with_fp8_sdpa correctly wraps a module."""
        from torchao.prototype.fp8_sdpa_inference import wrap_module_with_fp8_sdpa

        class SimpleAttentionModule(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        module = SimpleAttentionModule().cuda()
        wrapped_module = wrap_module_with_fp8_sdpa(module)

        # Check that module is returned
        self.assertIs(wrapped_module, module)

        # Check that forward still works
        B, H, S, D = 1, 4, 128, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            out = wrapped_module(q, k, v)

        self.assertEqual(out.shape, (B, H, S, D))
        self.assertFalse(torch.isnan(out).any())

    @unittest.skipIf(not _is_fp8_sdpa_available(), "FP8 SDPA requires CUDA and FA3")
    def test_unsupported_attn_mask_raises(self):
        """Test that providing attn_mask raises ValueError."""
        from torchao.prototype.fp8_sdpa_inference import fp8_sdpa_parallel

        B, H, S, D = 1, 4, 128, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        attn_mask = torch.ones(S, S, device="cuda", dtype=torch.bfloat16)

        with self.assertRaises(ValueError) as context:
            fp8_sdpa_parallel(q, k, v, attn_mask=attn_mask)

        self.assertIn("attn_mask", str(context.exception))

    @unittest.skipIf(not _is_fp8_sdpa_available(), "FP8 SDPA requires CUDA and FA3")
    def test_unsupported_dropout_raises(self):
        """Test that providing dropout_p != 0.0 raises ValueError."""
        from torchao.prototype.fp8_sdpa_inference import fp8_sdpa_parallel

        B, H, S, D = 1, 4, 128, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

        with self.assertRaises(ValueError) as context:
            fp8_sdpa_parallel(q, k, v, dropout_p=0.1)

        self.assertIn("dropout_p", str(context.exception))

    @unittest.skipIf(not _is_fp8_sdpa_available(), "FP8 SDPA requires CUDA and FA3")
    def test_causal_attention(self):
        """Test FP8 SDPA with causal masking."""
        from torch.nn.attention import (
            activate_flash_attention_impl,
            restore_flash_attention_impl,
        )

        from torchao.prototype.fp8_sdpa_inference import fp8_sdpa_parallel

        B, H, S, D = 2, 8, 512, 64

        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

        activate_flash_attention_impl("FA3")
        try:
            with torch.no_grad():
                # Regular SDPA with causal
                out_regular = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )

                # FP8 SDPA with causal
                out_fp8 = fp8_sdpa_parallel(q, k, v, is_causal=True)

            # Compute SQNR
            sqnr = 10 * torch.log10(
                torch.mean(out_regular.pow(2))
                / torch.mean((out_regular - out_fp8).pow(2))
            )

            self.assertGreater(
                sqnr.item(),
                25.0,
                f"SQNR {sqnr.item():.2f} dB is below threshold of 25 dB for causal attention",
            )
        finally:
            restore_flash_attention_impl()

    @unittest.skipIf(not _is_fp8_sdpa_available(), "FP8 SDPA requires CUDA and FA3")
    def test_wrap_module_with_compile(self):
        """Test that wrap_module_with_fp8_sdpa correctly wraps a module."""
        from torchao.prototype.fp8_sdpa_inference import wrap_module_with_fp8_sdpa

        class SimpleAttentionModule(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        module = SimpleAttentionModule().cuda()
        wrapped_module = wrap_module_with_fp8_sdpa(module)

        compiled_module = torch.compile(wrapped_module)

        # Check that forward still works
        B, H, S, D = 1, 4, 128, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            out = compiled_module(q, k, v)

        self.assertEqual(out.shape, (B, H, S, D))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    run_tests()
