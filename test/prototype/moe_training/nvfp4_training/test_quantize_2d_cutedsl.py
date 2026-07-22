# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for cutedsl_weight_quantize_2d (SM100+ CuteDSL plain 2D weight quantize, no RHT).

The CuteDSL weight quantize reuses the fused RHT row/col kernel with an identity Hadamard operand:
the rowwise output is plain NVFP4(W) and the colwise output is plain NVFP4(W.T).
"""

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.quantize_2d_cutedsl import (
    cutedsl_weight_quantize_2d,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    per_tensor_amax_to_scale,
)

# Kernel requires out_features (dim 0) % 256 == 0, in_features (dim 1) % 128 == 0.
_M_VALUES = [256, 512, 1024]
_N_VALUES = [128, 256, 512]

_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)


def _dequantize(codes, scales, global_amax):
    return (
        NVFP4Tensor(
            codes,
            scales,
            16,
            torch.bfloat16,
            per_tensor_scale=per_tensor_amax_to_scale(global_amax),
            is_swizzled_scales=True,
        )
        .dequantize()
        .float()
    )


@_skip_no_cutedsl
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_cutedsl_weight_quantize_sqnr(M, N):
    """Rowwise dequant reconstructs W and colwise reconstructs W.T (default 2D 16x16 scaling,
    SQNR >= 18 dB)."""
    torch.manual_seed(0)
    W = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    amax = W.float().abs().max()
    w_fp4, w_sf, wt_fp4, wt_sf = cutedsl_weight_quantize_2d(W, amax)

    row_sqnr = compute_error(W.float(), _dequantize(w_fp4, w_sf, amax))
    col_sqnr = compute_error(W.float().t(), _dequantize(wt_fp4, wt_sf, amax))
    assert row_sqnr >= 18.0, f"rowwise SQNR {row_sqnr:.1f} dB < 18 for M={M} N={N}"
    assert col_sqnr >= 18.0, f"colwise SQNR {col_sqnr:.1f} dB < 18 for M={M} N={N}"


@_skip_no_cutedsl
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_cutedsl_weight_quantize_layout(M, N):
    """Output 4-tuple matches triton_weight_quantize_2d's contract (shapes + dtypes)."""
    torch.manual_seed(1)
    W = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    amax = W.float().abs().max()
    w_fp4, w_sf, wt_fp4, wt_sf = cutedsl_weight_quantize_2d(W, amax)

    assert w_fp4.shape == (M, N // 2) and w_fp4.dtype == torch.uint8
    assert wt_fp4.shape == (N, M // 2) and wt_fp4.dtype == torch.uint8
    assert (
        w_sf.shape == (M // 128, N // 64, 32, 16) and w_sf.dtype == torch.float8_e4m3fn
    )
    assert (
        wt_sf.shape == (N // 128, M // 64, 32, 16)
        and wt_sf.dtype == torch.float8_e4m3fn
    )


@_skip_no_cutedsl
@pytest.mark.skipif(
    not has_triton(), reason="parity check needs the Triton weight kernel"
)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_cutedsl_weight_quantize_16x16_matches_triton(M, N):
    from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
        triton_weight_quantize_2d,
    )

    torch.manual_seed(3)
    W = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    amax = W.float().abs().max()
    _, c_rsf, _, c_csf = cutedsl_weight_quantize_2d(W, amax)
    _, t_rsf, _, t_csf = triton_weight_quantize_2d(W, amax)
    for c_sf, t_sf in ((c_rsf, t_rsf), (c_csf, t_csf)):
        got = c_sf.flatten().float()
        ref = t_sf.flatten().float()
        mism = got != ref
        frac = mism.float().mean().item()
        assert frac < 0.02, f"{100 * frac:.2f}% of SF differ from Triton 16x16 (>2%)"
        if mism.any():
            ratio = got[mism] / ref[mism]
            assert (ratio <= 1.15).all() and (ratio >= 1 / 1.15).all(), (
                "16x16-SF mismatches vs Triton exceed one E4M3 step (not a rounding tie)"
            )


@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_weight_quantize_requires_out_features_256():
    """out_features (dim 0) must be divisible by 256 (stricter than the Triton kernel's 128)."""
    W = torch.randn(384, 256, dtype=torch.bfloat16, device="cuda")  # 384 % 256 == 128
    amax = W.float().abs().max()
    with pytest.raises(ValueError, match="out_features"):
        cutedsl_weight_quantize_2d(W, amax)


@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_weight_quantize_zero_input():
    """All-zero weight packs to zero codes; each block scale clamps to E4M3 eps (not 0, matching
    Triton), and both layouts dequantize back to zero."""
    eps = torch.finfo(torch.float8_e4m3fn).tiny
    M, N = 256, 256
    W = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    amax = W.float().abs().max()  # 0.0
    w_fp4, w_sf, wt_fp4, wt_sf = cutedsl_weight_quantize_2d(W, amax)
    for codes, sf in ((w_fp4, w_sf), (wt_fp4, wt_sf)):
        assert torch.count_nonzero(codes).item() == 0, "zero weight must pack to zero"
        assert (sf.to(torch.float32) == eps).all(), (
            "zero block scale must clamp to E4M3 eps"
        )
        dq = _dequantize(codes, sf, amax)
        assert torch.isfinite(dq).all()
        torch.testing.assert_close(dq, torch.zeros_like(dq), atol=0, rtol=0)
