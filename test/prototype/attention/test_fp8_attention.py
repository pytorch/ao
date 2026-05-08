# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for FP8 low-precision attention (FA3 backend on Hopper)."""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.quantization.utils import compute_error
from torchao.utils import torch_version_at_least

if torch_version_at_least("2.11.0"):
    from torchao.prototype.attention.utils import _is_fa3_available, _is_hopper

    if _is_hopper() and _is_fa3_available():
        from torch.nn.attention import (
            activate_flash_attention_impl,
            restore_flash_attention_impl,
        )

        from torchao.prototype.attention import (
            AttentionBackend,
            HadamardMode,
            apply_low_precision_attention,
        )
        from torchao.prototype.attention.fp8_fa3.attention import (
            fp8_fa3_rope_sdpa,
            fp8_fa3_sdpa,
        )
        from torchao.prototype.attention.quantization import (
            _fp8_hadamard_rope_sdpa_quantize,
            _fp8_hadamard_sdpa_quantize,
            _fp8_rope_sdpa_quantize,
            _fp8_sdpa_quantize,
            _inverse_hadamard_transform,
        )


def _rope_cos_sin(S, D, device):
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    angles = torch.outer(torch.arange(S, dtype=torch.float32), freqs)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos = torch.cat([cos_half, cos_half], dim=-1).to(device)
    sin = torch.cat([sin_half, sin_half], dim=-1).to(device)
    return cos, sin


def _apply_rope(x, cos, sin):
    """NeoX rotate-half RoPE. x: [B, S, H, D], cos/sin: [S, D]."""
    D_HALF = x.shape[-1] // 2
    rotate = torch.cat([-x[..., D_HALF:], x[..., :D_HALF]], dim=-1)
    return (
        x * cos.unsqueeze(0).unsqueeze(2) + rotate * sin.unsqueeze(0).unsqueeze(2)
    ).to(x.dtype)


class SimpleAttentionModel(nn.Module):
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
        return self.out_proj(attn_out.transpose(1, 2).contiguous().view(B, S, -1))


class SimpleRoPEAttentionModel(nn.Module):
    """Applies RoPE to Q and K immediately before SDPA (Pattern A: RoPE → transpose → SDPA)."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim)
        q = _apply_rope(q, cos, sin).transpose(1, 2)
        k = _apply_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(attn_out.transpose(1, 2).contiguous().view(B, S, -1))


@common_utils.instantiate_parametrized_tests
class TestFP8FA3Attention(TestCase):
    @unittest.skipUnless(
        torch_version_at_least("2.11.0") and _is_hopper() and _is_fa3_available(),
        "Requires PyTorch >= 2.11, Hopper GPU, and FA3",
    )
    @common_utils.parametrize("shape", [(2, 8, 1024, 64), (1, 16, 1024, 128)])
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_sdpa_accuracy(self, shape, dtype):
        B, H, S, D = shape
        q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, S, D, device="cuda", dtype=dtype)

        with torch.no_grad():
            out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        activate_flash_attention_impl("FA3")
        try:
            with torch.no_grad():
                out_fp8 = fp8_fa3_sdpa(q, k, v, is_causal=False)
        finally:
            restore_flash_attention_impl()

        sqnr = compute_error(out_ref, out_fp8)
        self.assertGreater(
            sqnr.item(),
            25.0,
            f"SQNR {sqnr.item():.2f} dB below 25 dB for shape={shape}, dtype={dtype}",
        )

    @unittest.skipUnless(
        torch_version_at_least("2.11.0") and _is_hopper() and _is_fa3_available(),
        "Requires PyTorch >= 2.11, Hopper GPU, and FA3",
    )
    @common_utils.parametrize("shape", [(2, 1024, 8, 64), (1, 1024, 16, 128)])
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_rope_sdpa_accuracy(self, shape, dtype):
        B, S, H, D = shape
        q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        v = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        cos, sin = _rope_cos_sin(S, D, "cuda")

        with torch.no_grad():
            out_ref = F.scaled_dot_product_attention(
                _apply_rope(q, cos, sin).transpose(1, 2),
                _apply_rope(k, cos, sin).transpose(1, 2),
                v.transpose(1, 2),
                is_causal=False,
            )

        activate_flash_attention_impl("FA3")
        try:
            with torch.no_grad():
                out_fp8 = fp8_fa3_rope_sdpa(q, k, v, cos, sin, is_causal=False)
        finally:
            restore_flash_attention_impl()

        sqnr = compute_error(out_ref, out_fp8)
        self.assertGreater(
            sqnr.item(),
            25.0,
            f"SQNR {sqnr.item():.2f} dB below 25 dB for shape={shape}, dtype={dtype}",
        )

    @unittest.skipUnless(
        torch_version_at_least("2.11.0") and _is_hopper() and _is_fa3_available(),
        "Requires PyTorch >= 2.11, Hopper GPU, and FA3",
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    @common_utils.parametrize("hadamard", ["NONE", "QKV", "V_ONLY"])
    def test_monkey_patch_model(self, dtype, hadamard):
        embed_dim, num_heads = 512, 8
        model = (
            SimpleAttentionModel(embed_dim, num_heads)
            .to(device="cuda", dtype=dtype)
            .eval()
        )
        x = torch.randn(2, 128, embed_dim, device="cuda", dtype=dtype)

        with torch.no_grad():
            out_ref = model(x)

        fp8_model = (
            SimpleAttentionModel(embed_dim, num_heads)
            .to(device="cuda", dtype=dtype)
            .eval()
        )
        fp8_model.load_state_dict(model.state_dict())
        fp8_model = apply_low_precision_attention(
            fp8_model,
            backend=AttentionBackend.FP8_FA3,
            hadamard=HadamardMode(hadamard),
        )

        with torch.no_grad():
            out_fp8 = fp8_model(x)

        sqnr = compute_error(out_ref, out_fp8)
        self.assertGreater(
            sqnr.item(),
            20.0,
            f"SQNR {sqnr.item():.2f} dB below 20 dB for dtype={dtype}, hadamard={hadamard}",
        )

    @unittest.skipUnless(
        torch_version_at_least("2.11.0") and _is_hopper() and _is_fa3_available(),
        "Requires PyTorch >= 2.11, Hopper GPU, and FA3",
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    @common_utils.parametrize("hadamard", ["NONE", "QKV", "V_ONLY"])
    def test_rope_fusion_model(self, dtype, hadamard):
        embed_dim, num_heads = 512, 8
        model = (
            SimpleRoPEAttentionModel(embed_dim, num_heads)
            .to(device="cuda", dtype=dtype)
            .eval()
        )
        S = 128
        x = torch.randn(2, S, embed_dim, device="cuda", dtype=dtype)
        cos, sin = _rope_cos_sin(S, embed_dim // num_heads, "cuda")

        with torch.no_grad():
            out_ref = model(x, cos, sin)

        fp8_model = (
            SimpleRoPEAttentionModel(embed_dim, num_heads)
            .to(device="cuda", dtype=dtype)
            .eval()
        )
        fp8_model.load_state_dict(model.state_dict())
        fp8_model = apply_low_precision_attention(
            fp8_model,
            backend=AttentionBackend.FP8_FA3,
            hadamard=HadamardMode(hadamard),
        )
        fp8_model = torch.compile(fp8_model)

        with torch.no_grad():
            out_fp8 = fp8_model(x, cos, sin)

        sqnr = compute_error(out_ref, out_fp8)
        self.assertGreater(
            sqnr.item(),
            20.0,
            f"SQNR {sqnr.item():.2f} dB below 20 dB for dtype={dtype}, hadamard={hadamard}",
        )


def _make_outlier_tensor(shape, dtype, outlier_channels=(0,), outlier_scale=20.0):
    """Create a tensor with outliers in specific head-dim channels."""
    x = torch.randn(shape, device="cuda", dtype=dtype)
    for ch in outlier_channels:
        x[..., ch] *= outlier_scale
    return x


@common_utils.instantiate_parametrized_tests
class TestHadamardAccuracy(TestCase):
    """Tests that Hadamard improves FP8 quantization quality on outlier-heavy inputs."""

    # -- QKV Hadamard: full SDPA SQNR improvement (nonlinear softmax) --

    @unittest.skipUnless(
        torch_version_at_least("2.11.0") and _is_hopper() and _is_fa3_available(),
        "Requires PyTorch >= 2.11, Hopper GPU, and FA3",
    )
    @common_utils.parametrize("shape", [(2, 8, 1024, 64), (1, 16, 1024, 128)])
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_hadamard_qkv_sdpa_improves_sqnr(self, shape, dtype):
        """QKV Hadamard improves full SDPA output SQNR via nonlinear softmax."""
        B, H, S, D = shape
        outlier_channels = (0, D // 2)
        q = _make_outlier_tensor((B, H, S, D), dtype, outlier_channels)
        k = _make_outlier_tensor((B, H, S, D), dtype, outlier_channels)
        v = _make_outlier_tensor((B, H, S, D), dtype, outlier_channels)

        with torch.no_grad():
            out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        activate_flash_attention_impl("FA3")
        try:
            with torch.no_grad():
                out_no_had = fp8_fa3_sdpa(q, k, v, is_causal=False, hadamard="NONE")
                out_had = fp8_fa3_sdpa(q, k, v, is_causal=False, hadamard="QKV")
        finally:
            restore_flash_attention_impl()

        sqnr_no_had = compute_error(out_ref, out_no_had).item()
        sqnr_had = compute_error(out_ref, out_had).item()
        self.assertGreater(
            sqnr_had,
            sqnr_no_had,
            f"Hadamard(QKV) SQNR ({sqnr_had:.2f} dB) should exceed baseline "
            f"({sqnr_no_had:.2f} dB) for shape={shape}, dtype={dtype}",
        )

    @unittest.skipUnless(
        torch_version_at_least("2.11.0") and _is_hopper() and _is_fa3_available(),
        "Requires PyTorch >= 2.11, Hopper GPU, and FA3",
    )
    @common_utils.parametrize("shape", [(2, 1024, 8, 64), (1, 1024, 16, 128)])
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_hadamard_qkv_rope_sdpa_improves_sqnr(self, shape, dtype):
        """QKV Hadamard improves full RoPE SDPA output SQNR."""
        B, S, H, D = shape
        outlier_channels = (0, D // 2)
        q = _make_outlier_tensor((B, S, H, D), dtype, outlier_channels)
        k = _make_outlier_tensor((B, S, H, D), dtype, outlier_channels)
        v = _make_outlier_tensor((B, S, H, D), dtype, outlier_channels)
        cos, sin = _rope_cos_sin(S, D, "cuda")

        with torch.no_grad():
            out_ref = F.scaled_dot_product_attention(
                _apply_rope(q, cos, sin).transpose(1, 2),
                _apply_rope(k, cos, sin).transpose(1, 2),
                v.transpose(1, 2),
                is_causal=False,
            )

        activate_flash_attention_impl("FA3")
        try:
            with torch.no_grad():
                out_no_had = fp8_fa3_rope_sdpa(
                    q, k, v, cos, sin, is_causal=False, hadamard="NONE"
                )
                out_had = fp8_fa3_rope_sdpa(
                    q, k, v, cos, sin, is_causal=False, hadamard="QKV"
                )
        finally:
            restore_flash_attention_impl()

        sqnr_no_had = compute_error(out_ref, out_no_had).item()
        sqnr_had = compute_error(out_ref, out_had).item()
        self.assertGreater(
            sqnr_had,
            sqnr_no_had,
            f"Hadamard(QKV) SQNR ({sqnr_had:.2f} dB) should exceed baseline "
            f"({sqnr_no_had:.2f} dB) for shape={shape}, dtype={dtype}",
        )

    # -- V_ONLY Hadamard: SDPA non-regression + outlier dim improvement --
    #
    # V_ONLY cannot improve overall SDPA SQNR because V's contribution to the
    # attention output is linear and Hadamard preserves L2 norms, so FP8
    # quantization noise power is unchanged. The real benefit (confirmed in
    # end-to-end model benchmarks) is that outlier-channel quantization error
    # is reduced, which compounds across layers.

    @unittest.skipUnless(
        torch_version_at_least("2.11.0") and _is_hopper() and _is_fa3_available(),
        "Requires PyTorch >= 2.11, Hopper GPU, and FA3",
    )
    @common_utils.parametrize("shape", [(2, 8, 1024, 64), (1, 16, 1024, 128)])
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_hadamard_v_only_improves_outlier_dims(self, shape, dtype):
        """V_ONLY Hadamard reduces quantization error on V's outlier channels."""
        B, H, S, D = shape
        outlier_channels = (0, D // 2)
        q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
        v = _make_outlier_tensor(
            (B, H, S, D), dtype, outlier_channels, outlier_scale=20.0
        )

        # Quantize V without Hadamard
        _, _, v_fp8_none, _, _, v_descale_none = _fp8_sdpa_quantize(q, k, v)
        v_recon_none = (v_fp8_none.float() * v_descale_none[:, :, None, None]).to(dtype)

        # Quantize V with Hadamard (v_only=True)
        _, _, v_fp8_had, _, _, v_descale_had = _fp8_hadamard_sdpa_quantize(
            q, k, v, v_only=True
        )
        v_recon_had = _inverse_hadamard_transform(
            (v_fp8_had.float() * v_descale_had[:, :, None, None]).to(dtype)
        )

        for ch in outlier_channels:
            err_none = (v[..., ch].float() - v_recon_none[..., ch].float()).abs().mean()
            err_had = (v[..., ch].float() - v_recon_had[..., ch].float()).abs().mean()
            self.assertLess(
                err_had.item(),
                err_none.item(),
                f"V_ONLY Hadamard should reduce error on outlier dim {ch}: "
                f"had={err_had.item():.6f} vs none={err_none.item():.6f} "
                f"for shape={shape}, dtype={dtype}",
            )

    @unittest.skipUnless(
        torch_version_at_least("2.11.0") and _is_hopper() and _is_fa3_available(),
        "Requires PyTorch >= 2.11, Hopper GPU, and FA3",
    )
    @common_utils.parametrize("shape", [(2, 1024, 8, 64), (1, 1024, 16, 128)])
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_hadamard_v_only_rope_improves_outlier_dims(self, shape, dtype):
        """V_ONLY Hadamard reduces RoPE-path quantization error on outlier channels."""
        B, S, H, D = shape
        outlier_channels = (0, D // 2)
        q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        v = _make_outlier_tensor(
            (B, S, H, D), dtype, outlier_channels, outlier_scale=20.0
        )
        cos, sin = _rope_cos_sin(S, D, "cuda")

        # Quantize V without Hadamard (RoPE path)
        _, _, v_fp8_none, _, _, v_descale_none = _fp8_rope_sdpa_quantize(
            q, k, v, cos, sin
        )
        v_recon_none = (v_fp8_none.float() * v_descale_none[:, :, None, None]).to(dtype)

        # Quantize V with Hadamard (RoPE path, v_only=True)
        _, _, v_fp8_had, _, _, v_descale_had = _fp8_hadamard_rope_sdpa_quantize(
            q, k, v, cos, sin, v_only=True
        )
        v_recon_had = _inverse_hadamard_transform(
            (v_fp8_had.float() * v_descale_had[:, :, None, None]).to(dtype)
        )

        # V in the RoPE path is transposed to [B, H, S, D] by the kernel
        v_bhsd = v.transpose(1, 2)

        for ch in outlier_channels:
            err_none = (
                (v_bhsd[..., ch].float() - v_recon_none[..., ch].float()).abs().mean()
            )
            err_had = (
                (v_bhsd[..., ch].float() - v_recon_had[..., ch].float()).abs().mean()
            )
            self.assertLess(
                err_had.item(),
                err_none.item(),
                f"V_ONLY Hadamard should reduce error on outlier dim {ch}: "
                f"had={err_had.item():.6f} vs none={err_none.item():.6f} "
                f"for shape={shape}, dtype={dtype}",
            )


if __name__ == "__main__":
    run_tests()
