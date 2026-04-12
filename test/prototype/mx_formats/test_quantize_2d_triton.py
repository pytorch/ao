"""Tests for triton_weight_quantize_2d (SM100+ kernel).

test_triton_weight_quantize_2d_scales_vs_reference:
  FP8 scale factors must match the PyTorch reference bitwise for both
  non-swizzled (M, N//16) and swizzled (M//128, N//64, 32, 16) layouts.

test_triton_weight_quantize_2d_sqnr:
  Dequantized output must reconstruct A with SQNR >= 20 dB.
"""

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.float8.float8_utils import compute_error
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    per_tensor_amax_to_scale,
)
from torchao.utils import is_sm_at_least_100, torch_version_at_least

if has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0"):
    from torchao.prototype.mx_formats.quantize_2d_triton import (
        triton_weight_quantize_2d,
    )


# BLOCK_M minimum is 128; N must be a multiple of BLOCK_N=256.
_M_VALUES = [128, 256, 512, 1024]
_N_VALUES = [256, 512, 1024, 2048]


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _weight_quantize_2d_reference_scales(A: torch.Tensor) -> torch.Tensor:
    """PyTorch oracle: per-16×16-block FP8 scale factors expanded to (M, N//16).

    Mirrors the two-level scaling in _nvfp4_2d_quantize:
      1. global encode scale from the tensor-wide amax.
      2. per-block FP8 scale clamped to [-FP8_MAX, FP8_MAX].
      3. Expand each per-block scale to cover 16 consecutive rows.

    Returns:
        (M, N//16) float8_e4m3fn — the same layout as the kernel's non-swizzled output.
    """
    FP8_MAX = 448.0
    FP4_MAX = 6.0
    M, N = A.shape
    x = A.float()
    global_amax = x.abs().max()

    blocks = x.reshape(M // 16, 16, N // 16, 16)
    block_amax = blocks.abs().amax(dim=(1, 3))  # (M//16, N//16)

    enc_g = (FP8_MAX * FP4_MAX / global_amax).clamp(max=torch.finfo(torch.float32).max)
    pvscale = (block_amax / FP4_MAX) * enc_g
    pvscale = pvscale.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)  # (M//16, N//16)

    # Expand: each block-row scale repeated 16 times → (M, N//16)
    return pvscale.repeat_interleave(16, dim=0)


def _swizzle_py(scales_expanded: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """Python equivalent of the kernel's _swizzle_scales(expand_sf, BLOCK_M, BLOCK_N).

    Transforms (M, N//16) float8_e4m3fn → (M//128, N//64, 32, 16).
    """
    u8 = scales_expanded.view(torch.uint8)
    swizzled = (
        u8.reshape(M // 128, 4, 32, N // 64, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(M // 128, N // 64, 32, 16)
    )
    return swizzled.view(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Tests — scale factors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_triton_weight_quantize_2d_scales_vs_reference(M, N):
    """FP8 scale factors must match the PyTorch reference bitwise."""
    if M % 128 != 0 or N % 128 != 0:
        pytest.skip("swizzled scales require M % 128 == 0 and N % 128 == 0")

    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    ref_scales_expanded = _weight_quantize_2d_reference_scales(A)  # (M, N//16)

    _, tri_scales, _, tri_t_scales = triton_weight_quantize_2d(A, A.float().abs().max())

    ref_scales = _swizzle_py(ref_scales_expanded, M, N)
    torch.testing.assert_close(tri_scales, ref_scales, atol=0, rtol=0)

    ref_t_scales_expanded = _weight_quantize_2d_reference_scales(
        A.T.contiguous()
    )  # (N, M//16)
    ref_t_scales = _swizzle_py(ref_t_scales_expanded, N, M)  # (N//128, M//64, 32, 16)
    torch.testing.assert_close(tri_t_scales, ref_t_scales, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Tests — quantization quality (SQNR)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_triton_weight_quantize_2d_sqnr(M, N):
    """Dequantized output must reconstruct A with SQNR >= 20 dB."""
    if M % 128 != 0 or N % 128 != 0:
        pytest.skip("swizzled scales require M % 128 == 0 and N % 128 == 0")

    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    global_amax = A.float().abs().max()
    tri_codes, tri_scales, tri_t_codes, tri_t_scales = triton_weight_quantize_2d(
        A, global_amax
    )

    # NVFP4Tensor interprets (M, N//16) scales as rowwise block_size=16 scales,
    # which matches the 2D expanded layout since every 16 rows share the same scale.
    dequant = (
        NVFP4Tensor(
            tri_codes,
            tri_scales,
            16,
            torch.bfloat16,
            per_tensor_scale=per_tensor_amax_to_scale(global_amax),
            is_swizzled_scales=True,
        )
        .dequantize()
        .float()
    )

    sqnr = compute_error(A.float(), dequant)
    assert sqnr >= 15.0, f"Rowwise SQNR {sqnr:.2f} dB < 15.0 dB for M={M} N={N}"

    dequant_t = (
        NVFP4Tensor(
            tri_t_codes,
            tri_t_scales,
            16,
            torch.bfloat16,
            per_tensor_scale=per_tensor_amax_to_scale(global_amax),
            is_swizzled_scales=True,
        )
        .dequantize()
        .float()
    )

    sqnr_t = compute_error(A.T.float(), dequant_t)
    assert sqnr_t >= 15.0, f"Colwise SQNR {sqnr_t:.2f} dB < 15.0 dB for M={M} N={N}"
