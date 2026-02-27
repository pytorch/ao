# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FP8 low-precision attention (FA3 and FA4 backends).

Tests are parametrized over available backends. On Hopper (SM 9.x) with
flash-attn installed, FA3 tests run. On Hopper or Blackwell (SM 10.x)
with flash_attn.cute.interface installed, FA4 tests run. Backends that
are not available on the current hardware are automatically skipped.
"""

import unittest
from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Flash attention activation APIs are needed when calling fp8 sdpa directly
# (outside the model-level API which handles it internally).
_has_flash_activation_api = False
try:
    from torch.nn.attention import (
        activate_flash_attention_impl,
        restore_flash_attention_impl,
    )

    _has_flash_activation_api = True
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
from torchao.prototype.attention.utils import (
    _is_blackwell,
    _is_fa3_available,
    _is_fa4_available,
    _is_hopper,
)


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------
@dataclass
class BackendConfig:
    """Configuration for a single backend under test."""

    name: str
    flash_impl: str  # "FA3" or "FA4"
    attention_backend: AttentionBackend
    sdpa_fn: Callable  # fp8_fa3_sdpa
    rope_sdpa_fn: Callable  # fp8_fa3_rope_sdpa, or None if not yet available
    available_eager: bool  # Can run direct sdpa calls
    available_compiled: bool  # Can run via apply_low_precision_attention
    skip_msg: str


def _probe_eager_quantized_sdpa(sdpa_fn, flash_impl: str) -> bool:
    """Try a tiny quantized SDPA call to verify the backend works in eager mode.

    FA3 uses _scaled_dot_product_attention_quantized internally,
    which requires FA3 activation. This probe catches mismatches.
    """
    try:
        activate_flash_attention_impl(flash_impl)
        try:
            q = torch.randn(1, 1, 4, 64, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                sdpa_fn(q, q, q, is_causal=False)
            return True
        except RuntimeError:
            return False
        finally:
            restore_flash_attention_impl()
    except Exception:
        return False


def _build_backend_configs() -> List[BackendConfig]:
    """Build backend configs, lazily importing functions only when available."""
    configs = []

    # FA3: Hopper only
    fa3_available = _has_flash_activation_api and _is_hopper() and _is_fa3_available()
    if fa3_available:
        from torchao.prototype.attention.fp8_fa3.attention import (
            fp8_fa3_rope_sdpa,
            fp8_fa3_sdpa,
        )

        sdpa_fn, rope_sdpa_fn = fp8_fa3_sdpa, fp8_fa3_rope_sdpa
        eager_ok = _probe_eager_quantized_sdpa(sdpa_fn, "FA3")
    else:
        sdpa_fn = rope_sdpa_fn = None
        eager_ok = False

    configs.append(
        BackendConfig(
            name="FA3",
            flash_impl="FA3",
            attention_backend=AttentionBackend.FP8_FA3,
            sdpa_fn=sdpa_fn,
            rope_sdpa_fn=rope_sdpa_fn,
            available_eager=eager_ok,
            available_compiled=eager_ok,
            skip_msg=(
                "FP8 FA3 requires Hopper (SM 9.x), flash-attn installed, "
                "and PyTorch with FA3 activation APIs"
            ),
        )
    )

    # FA4: Hopper or Blackwell
    fa4_available = (
        _has_flash_activation_api
        and (_is_hopper() or _is_blackwell())
        and _is_fa4_available()
    )
    if fa4_available:
        from torchao.prototype.attention.fp8_fa4.attention import fp8_fa4_sdpa

        sdpa_fn = fp8_fa4_sdpa
        eager_ok = _probe_eager_quantized_sdpa(sdpa_fn, "FA4")
    else:
        sdpa_fn = None
        eager_ok = False

    configs.append(
        BackendConfig(
            name="FA4",
            flash_impl="FA4",
            attention_backend=AttentionBackend.FP8_FA4,
            sdpa_fn=sdpa_fn,
            rope_sdpa_fn=None,  # FA4 rope not yet available
            available_eager=eager_ok,
            available_compiled=eager_ok,
            skip_msg=(
                "FP8 FA4 requires Hopper (SM 9.x) or Blackwell (SM 10.x), "
                "flash-attn with FA4 support installed, "
                "and PyTorch with flash activation APIs"
            ),
        )
    )

    return configs


_BACKEND_CONFIGS = _build_backend_configs()
_EAGER_BACKENDS = [c for c in _BACKEND_CONFIGS if c.available_eager]
_COMPILED_BACKENDS = [c for c in _BACKEND_CONFIGS if c.available_compiled]
_ANY_EAGER_AVAILABLE = len(_EAGER_BACKENDS) > 0
_ANY_COMPILED_AVAILABLE = len(_COMPILED_BACKENDS) > 0
_NO_EAGER_SKIP_MSG = "No FP8 attention backend available for eager mode"
_NO_COMPILED_SKIP_MSG = "No FP8 attention backend available for compiled mode"

if _ANY_EAGER_AVAILABLE or _ANY_COMPILED_AVAILABLE:
    from torchao.quantization.utils import compute_error


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
# Simple model for API-level tests
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


# ---------------------------------------------------------------------------
# Numerical accuracy tests
# ---------------------------------------------------------------------------
@common_utils.instantiate_parametrized_tests
class TestFP8SDPANumericalAccuracy(TestCase):
    """SQNR-based numerical accuracy tests for FP8 SDPA."""

    def setUp(self):
        self._active_backend = None

    def tearDown(self):
        if self._active_backend is not None:
            restore_flash_attention_impl()

    def _activate(self, backend: BackendConfig):
        activate_flash_attention_impl(backend.flash_impl)
        self._active_backend = backend

    @unittest.skipIf(not _ANY_EAGER_AVAILABLE, _NO_EAGER_SKIP_MSG)
    @common_utils.parametrize(
        "shape",
        [
            (2, 8, 1024, 64),
            (1, 16, 1024, 128),
        ],
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_sdpa_accuracy(self, shape, dtype):
        """FP8 SDPA output matches regular SDPA within acceptable SQNR."""
        B, H, S, D = shape
        q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, S, D, device="cuda", dtype=dtype)

        with torch.no_grad():
            out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        for backend in _EAGER_BACKENDS:
            self._activate(backend)
            with torch.no_grad():
                out_fp8 = backend.sdpa_fn(q, k, v, is_causal=False)
            restore_flash_attention_impl()
            self._active_backend = None

            sqnr = compute_error(out_ref, out_fp8)
            self.assertGreater(
                sqnr.item(),
                25.0,
                f"[{backend.name}] SQNR {sqnr.item():.2f} dB below threshold "
                f"of 25 dB for shape={shape}, dtype={dtype}",
            )


# ---------------------------------------------------------------------------
# RoPE SDPA numerical accuracy tests
# ---------------------------------------------------------------------------
@common_utils.instantiate_parametrized_tests
class TestFP8RopeSDPANumericalAccuracy(TestCase):
    """SQNR-based numerical accuracy tests for FP8 attention with fused RoPE."""

    def setUp(self):
        self._active_backend = None

    def tearDown(self):
        if self._active_backend is not None:
            restore_flash_attention_impl()

    def _activate(self, backend: BackendConfig):
        activate_flash_attention_impl(backend.flash_impl)
        self._active_backend = backend

    @unittest.skipIf(not _ANY_EAGER_AVAILABLE, _NO_EAGER_SKIP_MSG)
    @common_utils.parametrize(
        "shape",
        [
            (2, 1024, 8, 64),
            (1, 1024, 16, 128),
        ],
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_rope_sdpa_accuracy(self, shape, dtype):
        """FP8 RoPE SDPA output matches ref RoPE + SDPA within acceptable SQNR."""
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

        for backend in _EAGER_BACKENDS:
            if backend.rope_sdpa_fn is None:
                continue  # Backend doesn't support fused RoPE yet
            self._activate(backend)
            with torch.no_grad():
                out_fp8 = backend.rope_sdpa_fn(q, k, v, cos, sin, is_causal=False)
            restore_flash_attention_impl()
            self._active_backend = None

            sqnr = compute_error(out_ref, out_fp8)
            self.assertGreater(
                sqnr.item(),
                25.0,
                f"[{backend.name}] SQNR {sqnr.item():.2f} dB below threshold "
                f"of 25 dB for shape={shape}, dtype={dtype}",
            )


# ---------------------------------------------------------------------------
# API-level model tests
# ---------------------------------------------------------------------------
@common_utils.instantiate_parametrized_tests
class TestFP8ModelAPI(TestCase):
    """API-level tests using apply_low_precision_attention on a model."""

    @unittest.skipIf(not _ANY_COMPILED_AVAILABLE, _NO_COMPILED_SKIP_MSG)
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    @common_utils.parametrize("fuse_rope", [True, False])
    def test_apply_to_model_accuracy(self, dtype, fuse_rope):
        """apply_low_precision_attention produces output close to original model."""
        embed_dim, num_heads = 256, 8
        model = SimpleAttentionModel(embed_dim, num_heads).to(
            device="cuda", dtype=dtype
        )
        model.eval()

        x = torch.randn(2, 128, embed_dim, device="cuda", dtype=dtype)

        with torch.no_grad():
            out_ref = model(x)

        for backend in _COMPILED_BACKENDS:
            # Need a fresh model for each backend since
            # apply_low_precision_attention modifies the model.
            test_model = SimpleAttentionModel(embed_dim, num_heads).to(
                device="cuda", dtype=dtype
            )
            test_model.load_state_dict(model.state_dict())
            test_model.eval()

            config = LowPrecisionAttentionConfig(
                backend=backend.attention_backend,
                fuse_rope=fuse_rope,
            )
            test_model = apply_low_precision_attention(test_model, config)

            with torch.no_grad():
                out_fp8 = test_model(x)

            sqnr = compute_error(out_ref, out_fp8)
            self.assertGreater(
                sqnr.item(),
                20.0,
                f"[{backend.name}, fuse_rope={fuse_rope}] SQNR "
                f"{sqnr.item():.2f} dB below threshold "
                f"for model-level test, dtype={dtype}",
            )


if __name__ == "__main__":
    run_tests()
