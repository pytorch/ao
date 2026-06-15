# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for cutedsl_rht_quantize_row_col (SM100+ CuteDSL kernel)."""

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.float8.float8_utils import compute_error
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    nvfp4_quantize,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_cutedsl import (
    cutedsl_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_cutedsl import (
    cutedsl_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import get_rht_matrix

# Kernel requires M % 256 == 0, N % 128 == 0.
_M_VALUES = [256, 512, 1024]
_N_VALUES = [128, 256, 512, 1024]
_HARDCODED_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)

_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)


# ---------------------------------------------------------------------------
# Reference implementations (plain PyTorch)
# ---------------------------------------------------------------------------


def _rht_reference(A: torch.Tensor) -> torch.Tensor:
    """PyTorch reference RHT: returns (N, M) bfloat16."""
    M_A, N_A = A.shape
    B = get_rht_matrix(_HARDCODED_SIGN_VECTOR, A.device, torch.bfloat16, 16)
    return (A.t().reshape(-1, 16) @ B).reshape(N_A, M_A).to(torch.bfloat16)


def _rht_quantize_rowwise_reference(A: torch.Tensor):
    """NVFP4 E2M1 rowwise quantization via nvfp4_quantize (no RHT)."""
    global_amax = A.float().abs().max()
    scale_inv, codes = nvfp4_quantize(
        A, per_tensor_scale=per_tensor_amax_to_scale(global_amax)
    )
    return codes, scale_inv, global_amax


def _dequantize(codes, scales, global_amax):
    """Decode packed FP4 codes via NVFP4Tensor.dequantize()."""
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


def _quantize_row_col(A: torch.Tensor):
    sign_vector = list(_HARDCODED_SIGN_VECTOR)
    col_amax, row_amax = cutedsl_rht_amax(A, sign_vector)
    col_codes, col_sf, row_codes, row_sf = cutedsl_rht_quantize_row_col(
        A, col_amax, row_amax, sign_vector
    )
    return col_codes, col_sf, row_codes, row_sf, col_amax, row_amax


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_skip_no_cutedsl
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_cutedsl_rht_quantize_rtne_sqnr(M, N):
    """Dequantized output reconstructs post-RHT (col) / raw-A (row) with SQNR >= 20 dB."""
    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    col_codes, col_sf, row_codes, row_sf, col_amax, row_amax = _quantize_row_col(A)

    ref_rht = _rht_reference(A).float()
    col_sqnr = compute_error(ref_rht, _dequantize(col_codes, col_sf, col_amax))
    assert col_sqnr >= 20.0, f"Col SQNR {col_sqnr:.2f} dB < 20.0 dB for M={M} N={N}"

    row_sqnr = compute_error(A.float(), _dequantize(row_codes, row_sf, row_amax))
    assert row_sqnr >= 20.0, f"Row SQNR {row_sqnr:.2f} dB < 20.0 dB for M={M} N={N}"


@_skip_no_cutedsl
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_cutedsl_rht_quantize_row_sf_vs_reference(M, N):
    """Rowwise FP8 scale factors match the PyTorch reference bitwise (plain A quantize)."""
    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    _, _, row_codes, row_sf, _, _ = _quantize_row_col(A)
    _, ref_row_sf, _ = _rht_quantize_rowwise_reference(A)
    torch.testing.assert_close(row_sf.flatten(), to_blocked(ref_row_sf), atol=0, rtol=0)


@pytest.mark.skipif(not has_triton(), reason="cross-check needs triton")
@_skip_no_cutedsl
@pytest.mark.parametrize("N", [256, 512], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", [256, 512], ids=lambda m: f"M{m}")
@torch.no_grad()
def test_cutedsl_vs_triton_interchangeable(M, N):
    """Fed the SAME global amaxes, CuteDSL and Triton outputs reconstruct each other to
    high SQNR (they implement the same RHT + NVFP4 quantize)."""
    from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
        triton_rht_amax,
    )
    from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_triton import (
        triton_rht_quantize_row_col,
    )

    torch.manual_seed(0)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    sign_vector = list(_HARDCODED_SIGN_VECTOR)

    col_amax, row_amax = triton_rht_amax(A, sign_vector=sign_vector)
    c_col, c_col_sf, c_row, c_row_sf = cutedsl_rht_quantize_row_col(
        A, col_amax, row_amax, sign_vector
    )
    t_col, t_col_sf, t_row, t_row_sf = triton_rht_quantize_row_col(
        A,
        col_global_amax=col_amax,
        row_global_amax=row_amax,
        sign_vector=sign_vector,
        stochastic_rounding=False,
    )

    col_sqnr = compute_error(
        _dequantize(t_col, t_col_sf, col_amax), _dequantize(c_col, c_col_sf, col_amax)
    )
    row_sqnr = compute_error(
        _dequantize(t_row, t_row_sf, row_amax), _dequantize(c_row, c_row_sf, row_amax)
    )
    assert col_sqnr >= 28.0, f"cutedsl-vs-triton col SQNR {col_sqnr:.1f} dB < 28"
    assert row_sqnr >= 35.0, f"cutedsl-vs-triton row SQNR {row_sqnr:.1f} dB < 35"


@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_rht_quantize_zero_input():
    """All-zero input packs to zero codes and dequantizes to zero (finite).

    An all-zero block emits a scale of 0, which dequantizes cleanly back to 0.
    """
    M, N = 256, 256
    A = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    col_codes, col_sf, row_codes, row_sf, col_amax, row_amax = _quantize_row_col(A)
    assert col_amax.item() == 0.0 and row_amax.item() == 0.0

    for codes, sf, amax in (
        (col_codes, col_sf, col_amax),
        (row_codes, row_sf, row_amax),
    ):
        assert torch.count_nonzero(codes).item() == 0, "zero input must pack to zero"
        assert torch.isfinite(sf.to(torch.float32)).all()
        dq = _dequantize(codes, sf, amax)
        assert torch.isfinite(dq).all()
        torch.testing.assert_close(dq, torch.zeros_like(dq), atol=0, rtol=0)


@_skip_no_cutedsl
@pytest.mark.parametrize("N", [256, 512], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", [256, 512], ids=lambda m: f"M{m}")
@torch.no_grad()
def test_cutedsl_rht_quantize_plain_sf_and_no_rowwise(M, N):
    """The internal swizzle=False / compute_rowwise=False branches (the op always swizzles
    and computes both) produce the plain (unblocked) SF layout and skip the rowwise output.
    The FP4 codes match the swizzled path, and the plain SF is the pre-blocked form of the
    swizzled SF: to_blocked(plain) == swizzled.flatten()."""
    from torchao.prototype.moe_training.nvfp4_training._cutedsl_kernels_impl import (
        _cutedsl_rht_quantize_row_col_impl,
    )

    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    sign_vector = list(_HARDCODED_SIGN_VECTOR)
    col_amax, row_amax = cutedsl_rht_amax(A, sign_vector)

    # Swizzled reference (the op's default path).
    col_sw, col_sf_sw, row_sw, row_sf_sw = cutedsl_rht_quantize_row_col(
        A, col_amax, row_amax, sign_vector
    )
    # Plain (unblocked) SF path.
    col_p, col_sf_p, row_p, row_sf_p = _cutedsl_rht_quantize_row_col_impl(
        A, col_amax, row_amax, tuple(sign_vector), swizzle_scale_factors=False
    )

    assert col_sf_p.shape == (N, M // 16) and row_sf_p.shape == (M, N // 16)
    # FP4 codes are layout-independent.
    assert torch.equal(col_p, col_sw) and torch.equal(row_p, row_sw)
    # Plain SF is the pre-blocked form of the swizzled SF.
    torch.testing.assert_close(to_blocked(col_sf_p), col_sf_sw.flatten(), atol=0, rtol=0)
    torch.testing.assert_close(to_blocked(row_sf_p), row_sf_sw.flatten(), atol=0, rtol=0)

    # compute_rowwise=False suppresses the rowwise *return* (the kernel still computes it).
    c_fp4, c_sf, r_fp4, r_sf = _cutedsl_rht_quantize_row_col_impl(
        A, col_amax, row_amax, tuple(sign_vector), compute_rowwise=False
    )
    assert r_fp4 is None and r_sf is None
    assert c_fp4 is not None and c_sf is not None


@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_rht_quantize_stochastic_rounding_unsupported():
    A = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    col_amax, row_amax = cutedsl_rht_amax(A, list(_HARDCODED_SIGN_VECTOR))
    with pytest.raises(NotImplementedError):
        cutedsl_rht_quantize_row_col(
            A,
            col_amax,
            row_amax,
            list(_HARDCODED_SIGN_VECTOR),
            stochastic_rounding=True,
        )


@_skip_no_cutedsl
@pytest.mark.parametrize(
    "M,N",
    [(128, 256), (256, 200), (384, 256)],
    ids=["M_not_mult_256", "N_not_mult_128", "M_not_mult_256_b"],
)
@torch.no_grad()
def test_cutedsl_rht_quantize_invalid_shape_raises(M, N):
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    col_amax = torch.zeros((), dtype=torch.float32, device="cuda")
    row_amax = torch.zeros((), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        cutedsl_rht_quantize_row_col(
            A, col_amax, row_amax, list(_HARDCODED_SIGN_VECTOR)
        )


@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_ops_registered_and_fake():
    """The custom ops register under torchao:: and propagate shapes via register_fake."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    assert hasattr(torch.ops.torchao, "cutedsl_rht_amax")
    assert hasattr(torch.ops.torchao, "cutedsl_rht_quantize_row_col")

    sign_vector = list(_HARDCODED_SIGN_VECTOR)
    with FakeTensorMode():
        A = torch.empty(512, 256, dtype=torch.bfloat16, device="cuda")
        ca = torch.empty((), dtype=torch.float32, device="cuda")
        ra = torch.empty((), dtype=torch.float32, device="cuda")
        fca, fra = torch.ops.torchao.cutedsl_rht_amax(A, sign_vector)
        assert fca.shape == () and fra.shape == ()
        cf, csf, rf, rsf = torch.ops.torchao.cutedsl_rht_quantize_row_col(
            A, ca, ra, sign_vector
        )
        assert cf.shape == (256, 256) and rf.shape == (512, 128)
        assert csf.shape == (2, 8, 32, 16) and rsf.shape == (4, 4, 32, 16)
