# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FP8 FA3 low-precision attention.
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

# FA3 activation is needed when calling _fp8_fa3_sdpa directly (outside the
# model-level API which handles it via the context manager in wrappers.py).
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.attention import (
    AttentionBackend,
    LowPrecisionAttentionConfig,
    apply_low_precision_attention,
)
from torchao.attention.fp8_fa3.attention import _fp8_fa3_sdpa
from torchao.attention.fp8_fa3.quantization import _fp8_sdpa_quantize
from torchao.attention.utils import _is_hopper
from torchao.quantization.utils import compute_error

_FP8_FA3_SKIP_MSG = "FP8 FA3 requires CUDA with Hopper (SM 9.x)"


# ---------------------------------------------------------------------------
# Numerical accuracy tests
# ---------------------------------------------------------------------------
@common_utils.instantiate_parametrized_tests
class TestFP8FA3NumericalAccuracy(TestCase):
    """SQNR-based numerical accuracy tests for FP8 FA3 attention."""

    def setUp(self):
        if _is_hopper():
            activate_flash_attention_impl("FA3")

    def tearDown(self):
        if _is_hopper():
            restore_flash_attention_impl()

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    @common_utils.parametrize(
        "shape",
        [
            (2, 8, 1024, 64),
            (1, 8, 4096, 64),
            (1, 16, 1024, 128),
            (4, 32, 512, 128),
        ],
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_sdpa_accuracy_non_causal(self, shape, dtype):
        """FP8 FA3 SDPA output matches regular SDPA within acceptable SQNR."""
        B, H, S, D = shape
        q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, S, D, device="cuda", dtype=dtype)

        with torch.no_grad():
            out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            out_fp8 = _fp8_fa3_sdpa(q, k, v, is_causal=False)

        sqnr = compute_error(out_ref, out_fp8)
        self.assertGreater(
            sqnr.item(),
            25.0,
            f"SQNR {sqnr.item():.2f} dB below threshold of 25 dB "
            f"for shape={shape}, dtype={dtype}",
        )

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    @common_utils.parametrize(
        "shape",
        [
            (2, 8, 1024, 64),
            (1, 8, 4096, 64),
            (1, 16, 1024, 128),
            (4, 32, 512, 128),
        ],
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_sdpa_accuracy_causal(self, shape, dtype):
        """FP8 FA3 SDPA with causal mask matches regular SDPA."""
        B, H, S, D = shape
        q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, S, D, device="cuda", dtype=dtype)

        with torch.no_grad():
            out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out_fp8 = _fp8_fa3_sdpa(q, k, v, is_causal=True)

        sqnr = compute_error(out_ref, out_fp8)
        self.assertGreater(
            sqnr.item(),
            25.0,
            f"SQNR {sqnr.item():.2f} dB below threshold of 25 dB "
            f"for causal, shape={shape}, dtype={dtype}",
        )

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_sdpa_accuracy_cross_attention(self, dtype):
        """FP8 FA3 SDPA with different Q and K/V sequence lengths."""
        B, H, D = 2, 8, 64
        S_q = 256
        S_kv = 1024

        q = torch.randn(B, H, S_q, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, S_kv, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, S_kv, D, device="cuda", dtype=dtype)

        with torch.no_grad():
            out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            out_fp8 = _fp8_fa3_sdpa(q, k, v, is_causal=False)

        self.assertEqual(out_fp8.shape, (B, H, S_q, D))
        sqnr = compute_error(out_ref, out_fp8)
        self.assertGreater(
            sqnr.item(),
            25.0,
            f"SQNR {sqnr.item():.2f} dB below threshold for cross-attention",
        )

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_sdpa_output_dtype_matches_input(self, dtype):
        """Output dtype should match the input dtype."""
        B, H, S, D = 1, 4, 256, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, S, D, device="cuda", dtype=dtype)

        with torch.no_grad():
            out = _fp8_fa3_sdpa(q, k, v)

        self.assertEqual(out.dtype, dtype)

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    def test_sdpa_output_no_nan(self):
        """Output should not contain NaN values."""
        B, H, S, D = 2, 8, 512, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            out = _fp8_fa3_sdpa(q, k, v)

        self.assertFalse(torch.isnan(out).any())


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------
@common_utils.instantiate_parametrized_tests
class TestFP8FA3InputValidation(TestCase):
    """Input validation and error handling tests."""

    def setUp(self):
        if _is_hopper():
            activate_flash_attention_impl("FA3")

    def tearDown(self):
        if _is_hopper():
            restore_flash_attention_impl()

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    def test_attn_mask_raises(self):
        """attn_mask is not supported and should raise ValueError."""
        B, H, S, D = 1, 4, 128, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(S, S, device="cuda", dtype=torch.bfloat16)

        with self.assertRaises(ValueError, msg="attn_mask"):
            _fp8_fa3_sdpa(q, k, v, attn_mask=mask)

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    def test_dropout_raises(self):
        """dropout_p != 0.0 should raise ValueError."""
        B, H, S, D = 1, 4, 128, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

        with self.assertRaises(ValueError, msg="dropout_p"):
            _fp8_fa3_sdpa(q, k, v, dropout_p=0.1)

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    def test_wrong_dimensions(self):
        """Non-4D tensors should raise ValueError."""
        k_4d = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        v_4d = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16)

        # 3D tensor
        q_3d = torch.randn(8, 128, 64, device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(ValueError, msg="4D"):
            _fp8_sdpa_quantize(q_3d, k_4d, v_4d)

        # 2D tensor
        q_2d = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(ValueError, msg="4D"):
            _fp8_sdpa_quantize(q_2d, k_4d, v_4d)

        # 5D tensor
        q_5d = torch.randn(1, 1, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(ValueError, msg="4D"):
            _fp8_sdpa_quantize(q_5d, k_4d, v_4d)

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    def test_kv_shape_mismatch(self):
        """K and V must have the same shape."""
        B, H, D = 1, 8, 64
        q = torch.randn(B, H, 128, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, 128, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, 256, D, device="cuda", dtype=torch.bfloat16)

        with self.assertRaises(ValueError, msg="K and V shape mismatch"):
            _fp8_sdpa_quantize(q, k, v)

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    def test_batch_size_mismatch(self):
        """Q and K must have the same batch size."""
        H, S, D = 8, 128, 64
        q = torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(2, H, S, D, device="cuda", dtype=torch.bfloat16)

        with self.assertRaises(ValueError, msg="Batch size mismatch"):
            _fp8_sdpa_quantize(q, k, v)

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    def test_head_count_mismatch(self):
        """Q and K must have the same number of heads."""
        B, S, D = 1, 128, 64
        q = torch.randn(B, 8, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, 16, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, 16, S, D, device="cuda", dtype=torch.bfloat16)

        with self.assertRaises(ValueError, msg="Head count mismatch"):
            _fp8_sdpa_quantize(q, k, v)

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    def test_head_dim_mismatch(self):
        """Q and K must have the same head dimension."""
        B, H, S = 1, 8, 128
        q = torch.randn(B, H, S, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, 128, device="cuda", dtype=torch.bfloat16)

        with self.assertRaises(ValueError, msg="Head dim mismatch"):
            _fp8_sdpa_quantize(q, k, v)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_quantize_output_shapes(self):
        """Quantize should return correct shapes for fp8 tensors and descales."""
        B, H, S, D = 2, 8, 256, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

        q_fp8, k_fp8, v_fp8, dq, dk, dv = _fp8_sdpa_quantize(q, k, v)

        # FP8 tensors should have same shape as input
        self.assertEqual(q_fp8.shape, (B, H, S, D))
        self.assertEqual(k_fp8.shape, (B, H, S, D))
        self.assertEqual(v_fp8.shape, (B, H, S, D))

        # FP8 dtype
        self.assertEqual(q_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(k_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(v_fp8.dtype, torch.float8_e4m3fn)

        # Descales should be per-head: (B, H)
        self.assertEqual(dq.shape, (B, H))
        self.assertEqual(dk.shape, (B, H))
        self.assertEqual(dv.shape, (B, H))


# ---------------------------------------------------------------------------
# API-level model tests
# ---------------------------------------------------------------------------
class SimpleAttentionModel(nn.Module):
    """A minimal model that calls F.scaled_dot_product_attention."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(attn_out)


@common_utils.instantiate_parametrized_tests
class TestFP8FA3ModelAPI(TestCase):
    """API-level tests using apply_low_precision_attention on a model."""

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_apply_to_model_accuracy(self, dtype):
        """apply_low_precision_attention produces output close to original model."""
        embed_dim, num_heads = 256, 8
        model = SimpleAttentionModel(embed_dim, num_heads).to(
            device="cuda", dtype=dtype
        )
        model.eval()

        x = torch.randn(2, 128, embed_dim, device="cuda", dtype=dtype)

        with torch.no_grad():
            out_ref = model(x)

        config = LowPrecisionAttentionConfig(backend=AttentionBackend.FP8_FA3)
        apply_low_precision_attention(model, config)

        with torch.no_grad():
            out_fp8 = model(x)

        sqnr = compute_error(out_ref, out_fp8)
        self.assertGreater(
            sqnr.item(),
            20.0,
            f"SQNR {sqnr.item():.2f} dB below threshold for model-level test",
        )

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    def test_apply_returns_same_model(self):
        """apply_low_precision_attention should return the same model object."""
        model = SimpleAttentionModel(128, 4).to(device="cuda", dtype=torch.bfloat16)
        config = LowPrecisionAttentionConfig(backend=AttentionBackend.FP8_FA3)
        result = apply_low_precision_attention(model, config)
        self.assertIs(result, model)

    @unittest.skipIf(not _is_hopper(), _FP8_FA3_SKIP_MSG)
    def test_apply_does_not_modify_weights(self):
        """apply_low_precision_attention should not change model parameters."""
        embed_dim, num_heads = 128, 4
        model = SimpleAttentionModel(embed_dim, num_heads).to(
            device="cuda", dtype=torch.bfloat16
        )

        params_before = {name: p.clone() for name, p in model.named_parameters()}

        config = LowPrecisionAttentionConfig(backend=AttentionBackend.FP8_FA3)
        apply_low_precision_attention(model, config)

        for name, p in model.named_parameters():
            torch.testing.assert_close(
                p,
                params_before[name],
                atol=0,
                rtol=0,
                msg=f"Parameter {name} was modified by apply_low_precision_attention",
            )


if __name__ == "__main__":
    run_tests()
