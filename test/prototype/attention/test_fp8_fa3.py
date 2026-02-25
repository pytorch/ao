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

# FA3 activation is needed when calling fp8_fa3_sdpa directly (outside the
# model-level API which handles it via the context manager in wrappers.py).
# These APIs were added in a recent PyTorch version, so guard the import
# following the same try/except pattern used in test_nf4.py (bitsandbytes)
# and test_integration.py (gemlite).
_has_fa3_activation_api = False
try:
    from torch.nn.attention import (
        activate_flash_attention_impl,
        restore_flash_attention_impl,
    )

    _has_fa3_activation_api = True
except ImportError:
    pass

from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.prototype.attention import (
    AttentionBackend,
    LowPrecisionAttentionConfig,
    apply_low_precision_attention,
)
from torchao.prototype.attention.utils import _is_fa3_available, _is_hopper

_FP8_FA3_SKIP_MSG = (
    "FP8 FA3 requires CUDA with Hopper (SM 9.x), flash-attn installed, "
    "and a PyTorch version with FA3 activation APIs"
)

_FP8_FA3_AVAILABLE = _has_fa3_activation_api and _is_hopper() and _is_fa3_available()

# Only import internal modules that depend on new PyTorch APIs when available.
if _FP8_FA3_AVAILABLE:
    from torchao.prototype.attention.fp8_fa3.attention import (
        fp8_fa3_rope_sdpa,
        fp8_fa3_sdpa,
    )
    from torchao.quantization.utils import compute_error


# ---------------------------------------------------------------------------
# Numerical accuracy tests
# ---------------------------------------------------------------------------
@common_utils.instantiate_parametrized_tests
class TestFP8FA3NumericalAccuracy(TestCase):
    """SQNR-based numerical accuracy tests for FP8 FA3 attention."""

    def setUp(self):
        if _FP8_FA3_AVAILABLE:
            activate_flash_attention_impl("FA3")

    def tearDown(self):
        if _FP8_FA3_AVAILABLE:
            restore_flash_attention_impl()

    @unittest.skipIf(not _FP8_FA3_AVAILABLE, _FP8_FA3_SKIP_MSG)
    @common_utils.parametrize(
        "shape",
        [
            (2, 8, 1024, 64),
            (1, 16, 1024, 128),
        ],
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_sdpa_accuracy(self, shape, dtype):
        """FP8 FA3 SDPA output matches regular SDPA within acceptable SQNR."""
        B, H, S, D = shape
        q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, S, D, device="cuda", dtype=dtype)

        with torch.no_grad():
            out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            out_fp8 = fp8_fa3_sdpa(q, k, v, is_causal=False)

        sqnr = compute_error(out_ref, out_fp8)
        self.assertGreater(
            sqnr.item(),
            25.0,
            f"SQNR {sqnr.item():.2f} dB below threshold of 25 dB "
            f"for shape={shape}, dtype={dtype}",
        )


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------
def _generate_rope_cos_sin(S, D, device, dtype=torch.float32):
    """Generate cos/sin frequencies for RoPE testing (NeoX half-split format)."""
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    positions = torch.arange(S, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # [S, D/2]
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos = torch.cat([cos_half, cos_half], dim=-1).to(device=device, dtype=dtype)
    sin = torch.cat([sin_half, sin_half], dim=-1).to(device=device, dtype=dtype)
    return cos, sin


def _apply_rope_ref(x, cos, sin):
    """Reference NeoX half-split RoPE: x is [B, S, H, D], cos/sin are [S, D]."""
    D = x.shape[-1]
    D_HALF = D // 2
    x_first = x[..., :D_HALF].float()
    x_second = x[..., D_HALF:].float()
    # [S, D_HALF] -> [1, S, 1, D_HALF] for broadcasting with [B, S, H, D_HALF]
    cos_half = cos[:, :D_HALF].float().unsqueeze(0).unsqueeze(2)
    sin_half = sin[:, :D_HALF].float().unsqueeze(0).unsqueeze(2)
    out_first = x_first * cos_half - x_second * sin_half
    out_second = x_second * cos_half + x_first * sin_half
    return torch.cat([out_first, out_second], dim=-1).to(x.dtype)


# ---------------------------------------------------------------------------
# RoPE SDPA numerical accuracy tests
# ---------------------------------------------------------------------------
@common_utils.instantiate_parametrized_tests
class TestFP8FA3RopeNumericalAccuracy(TestCase):
    """SQNR-based numerical accuracy tests for FP8 FA3 attention with fused RoPE."""

    def setUp(self):
        if _FP8_FA3_AVAILABLE:
            activate_flash_attention_impl("FA3")

    def tearDown(self):
        if _FP8_FA3_AVAILABLE:
            restore_flash_attention_impl()

    @unittest.skipIf(not _FP8_FA3_AVAILABLE, _FP8_FA3_SKIP_MSG)
    @common_utils.parametrize(
        "shape",
        [
            (2, 1024, 8, 64),
            (1, 1024, 16, 128),
        ],
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_rope_sdpa_accuracy(self, shape, dtype):
        """FP8 FA3 RoPE SDPA output matches ref RoPE + SDPA within acceptable SQNR."""
        B, S, H, D = shape
        q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        v = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        cos, sin = _generate_rope_cos_sin(S, D, device="cuda")

        with torch.no_grad():
            # Reference: apply RoPE, transpose to [B, H, S, D], run SDPA
            q_rope = _apply_rope_ref(q, cos, sin).transpose(1, 2)
            k_rope = _apply_rope_ref(k, cos, sin).transpose(1, 2)
            v_ref = v.transpose(1, 2)
            out_ref = F.scaled_dot_product_attention(
                q_rope, k_rope, v_ref, is_causal=False
            )

            out_fp8 = fp8_fa3_rope_sdpa(q, k, v, cos, sin, is_causal=False)

        sqnr = compute_error(out_ref, out_fp8)
        self.assertGreater(
            sqnr.item(),
            25.0,
            f"SQNR {sqnr.item():.2f} dB below threshold of 25 dB "
            f"for shape={shape}, dtype={dtype}",
        )


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

    @unittest.skipIf(not _FP8_FA3_AVAILABLE, _FP8_FA3_SKIP_MSG)
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


if __name__ == "__main__":
    run_tests()
