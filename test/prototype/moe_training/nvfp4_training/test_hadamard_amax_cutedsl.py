# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for cutedsl_rht_amax (SM100+ CuteDSL kernel)."""

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_cutedsl import (
    cutedsl_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import get_rht_matrix

_HARDCODED_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)

# Kernel requires M % 256 == 0, N % 128 == 0.
_M_VALUES = [256, 512, 1024]
_N_VALUES = [128, 256, 384, 512, 1024]

_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)


@_skip_no_cutedsl
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_cutedsl_rht_amax_vs_reference(M, N):
    """col_amax = max|RHT(A.t())| (post-Hadamard), row_amax = max|A| (plain).

    col amax is reduced in float32, so it matches the float32 reference only to a small
    tolerance; row amax is exact.
    """
    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    sign_vector = list(_HARDCODED_SIGN_VECTOR)

    get_rht_matrix.cache_clear()
    B = get_rht_matrix(_HARDCODED_SIGN_VECTOR, "cuda", torch.bfloat16, 16)
    ref_col_amax = (A.t().reshape(-1, 16).float() @ B.float()).abs().max()
    ref_row_amax = A.float().abs().max()

    col_amax, row_amax = cutedsl_rht_amax(A, sign_vector)

    torch.testing.assert_close(col_amax, ref_col_amax, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(row_amax, ref_row_amax, rtol=0, atol=0)


@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_rht_amax_returns_scalars():
    """Both amaxes are scalar float32 tensors."""
    A = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    col_amax, row_amax = cutedsl_rht_amax(A, list(_HARDCODED_SIGN_VECTOR))
    assert col_amax.shape == () and col_amax.dtype == torch.float32
    assert row_amax.shape == () and row_amax.dtype == torch.float32


@_skip_no_cutedsl
@pytest.mark.parametrize(
    "M,N",
    [(128, 256), (256, 200), (384, 256)],
    ids=["M_not_mult_256", "N_not_mult_128", "M_not_mult_256_b"],
)
@torch.no_grad()
def test_cutedsl_rht_amax_invalid_shape_raises(M, N):
    """M % 256 / N % 128 violations must raise. M=128 is the subtle one: it passes an
    M % 128 check but is invalid here, and without the M % 256 check it silently returns
    zero amaxes (empty grid, no-op launch)."""
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        cutedsl_rht_amax(A, list(_HARDCODED_SIGN_VECTOR))


@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_rht_amax_rejects_non_bf16():
    A = torch.randn(256, 256, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        cutedsl_rht_amax(A, list(_HARDCODED_SIGN_VECTOR))


@pytest.mark.skipif(not has_triton(), reason="parity check needs triton")
@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_rht_amax_propagates_nan():
    """A NaN input propagates to both amaxes (max.NaN.f32 reduction), matching triton_rht_amax."""
    from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
        triton_rht_amax,
    )

    A = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    A.view(-1)[123] = float("nan")
    sign_vector = list(_HARDCODED_SIGN_VECTOR)
    col_c, row_c = cutedsl_rht_amax(A, sign_vector)
    col_t, row_t = triton_rht_amax(A, sign_vector=sign_vector)
    assert torch.isnan(col_c) and torch.isnan(row_c)
    assert torch.isnan(col_t) and torch.isnan(row_t)


@pytest.mark.skipif(not has_triton(), reason="cross-check needs triton")
@_skip_no_cutedsl
@pytest.mark.parametrize("N", [256, 512], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", [256, 512], ids=lambda m: f"M{m}")
@torch.no_grad()
def test_cutedsl_rht_amax_matches_triton(M, N):
    """CuteDSL and Triton amaxes agree closely (row is exact; col differs only by RHT
    reduction precision)."""
    from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
        triton_rht_amax,
    )

    torch.manual_seed(0)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    sign_vector = list(_HARDCODED_SIGN_VECTOR)
    col_c, row_c = cutedsl_rht_amax(A, sign_vector)
    col_t, row_t = triton_rht_amax(A, sign_vector=sign_vector)
    torch.testing.assert_close(row_c, row_t, rtol=0, atol=0)
    torch.testing.assert_close(col_c, col_t, rtol=5e-3, atol=5e-3)
