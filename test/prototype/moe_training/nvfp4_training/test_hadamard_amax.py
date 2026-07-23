# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the RHT global-amax kernels (triton and CuteDSL, SM100+)."""

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_cutedsl import (
    cutedsl_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    get_hadamard_matrix,
    get_rht_matrix,
    get_wgrad_sign_vector,
)
from torchao.utils import is_sm_at_least_100, torch_version_at_least

_HARDCODED_SIGN_VECTOR = (
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
)

_skip_no_triton = pytest.mark.skipif(
    not (has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0")),
    reason="requires triton, SM100+, and PyTorch 2.10+ (torch.compile)",
)
_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)

_KERNELS = [
    pytest.param("triton", marks=_skip_no_triton, id="triton"),
    pytest.param("cutedsl", marks=_skip_no_cutedsl, id="cutedsl"),
]

# Both kernels require N % 128 == 0 (triton_rht_amax feeds triton_rht_quantize_row_col,
# whose swizzled scales require it). triton additionally handles M % 256 != 0, so the
# sweep is the union and _skip_if_unsupported_shape drops the sub-256 M for cutedsl.
# M=32 excluded: all BLOCK_M configs (64, 128) exceed M=32 → all autotune configs fail.
_M_VALUES = [64, 96, 128, 160, 256, 512, 1024]
_N_VALUES = [128, 256, 384, 512, 1024]


def _skip_if_unsupported_shape(kernel: str, M: int, N: int) -> None:
    """Skip shapes the selected backend cannot handle."""
    if kernel == "cutedsl" and M % 256 != 0:
        pytest.skip("cutedsl amax kernel requires M % 256 == 0")


def _rht_amax(kernel, A, sign_vector):
    """Dispatch to a backend's RHT amax kernel; returns ``(col_amax, row_amax)``."""
    if kernel == "triton":
        from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
            triton_rht_amax,
        )

        return triton_rht_amax(A, sign_vector=list(sign_vector))
    return cutedsl_rht_amax(A, list(sign_vector))


@torch.no_grad()
def test_get_rht_matrix_with_hardcoded_sign_vector():
    rht_matrix = get_rht_matrix(_HARDCODED_SIGN_VECTOR, "cpu", torch.bfloat16, 16)

    expected_signs = torch.tensor(_HARDCODED_SIGN_VECTOR, dtype=torch.bfloat16)
    expected = torch.diag(expected_signs) @ get_hadamard_matrix(16, device="cpu")
    torch.testing.assert_close(rht_matrix, expected, atol=0, rtol=0)


@torch.no_grad()
def test_get_rht_matrix_with_generated_sign_matches_sampled_signs():
    get_rht_matrix.cache_clear()
    torch.manual_seed(42)
    expected_signs = get_wgrad_sign_vector(16, device="cpu")
    sign_vector = tuple(int(v) for v in expected_signs.tolist())

    get_rht_matrix.cache_clear()
    rht_matrix = get_rht_matrix(sign_vector, "cpu", torch.bfloat16, 16)

    expected = torch.diag(expected_signs) @ get_hadamard_matrix(16, device="cpu")
    torch.testing.assert_close(rht_matrix, expected, atol=0, rtol=0)


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_rht_amax_vs_reference(kernel, M, N):
    """col_amax = max|RHT(A.t())| (post-Hadamard), row_amax = max|A| (plain).

    triton reduces the RHT output in bfloat16, so it must match the bf16-rounded
    reference bitwise; CuteDSL reduces in float32, so it matches the float32 reference
    only to a small tolerance. The plain amax is exact for both.
    """
    _skip_if_unsupported_shape(kernel, M, N)
    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    get_rht_matrix.cache_clear()
    B = get_rht_matrix(_HARDCODED_SIGN_VECTOR, "cuda", torch.bfloat16, 16)
    if kernel == "triton":
        ref_col_amax = (
            (A.t().reshape(N * M // 16, 16) @ B).to(torch.bfloat16).abs().max().float()
        )
        col_tol = {"atol": 0, "rtol": 0}
    else:
        ref_col_amax = (A.t().reshape(N * M // 16, 16).float() @ B.float()).abs().max()
        col_tol = {"atol": 2e-3, "rtol": 2e-3}
    ref_row_amax = A.abs().max().float()

    get_rht_matrix.cache_clear()

    col_amax, row_amax = _rht_amax(kernel, A, _HARDCODED_SIGN_VECTOR)
    torch.testing.assert_close(col_amax, ref_col_amax, **col_tol)
    torch.testing.assert_close(row_amax, ref_row_amax, atol=0, rtol=0)


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


@_skip_no_triton
@_skip_no_cutedsl
@torch.no_grad()
def test_rht_amax_propagates_nan():
    """A NaN input propagates to both amaxes (max.NaN.f32 reduction), in both kernels."""
    A = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    A.view(-1)[123] = float("nan")
    for kernel in ("triton", "cutedsl"):
        col_amax, row_amax = _rht_amax(kernel, A, _HARDCODED_SIGN_VECTOR)
        assert torch.isnan(col_amax), kernel
        assert torch.isnan(row_amax), kernel


@_skip_no_triton
@_skip_no_cutedsl
@pytest.mark.parametrize("N", [256, 512], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", [256, 512], ids=lambda m: f"M{m}")
@torch.no_grad()
def test_cutedsl_rht_amax_matches_triton(M, N):
    """CuteDSL and Triton amaxes agree closely (row is exact; col differs only by RHT
    reduction precision)."""
    torch.manual_seed(0)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    col_c, row_c = _rht_amax("cutedsl", A, _HARDCODED_SIGN_VECTOR)
    col_t, row_t = _rht_amax("triton", A, _HARDCODED_SIGN_VECTOR)
    torch.testing.assert_close(row_c, row_t, rtol=0, atol=0)
    torch.testing.assert_close(col_c, col_t, rtol=5e-3, atol=5e-3)
