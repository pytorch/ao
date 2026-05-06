"""Tests for triton_rht_amax (SM100+ kernel)."""

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.prototype.mx_formats.hadamard_utils import (
    get_hadamard_matrix,
    get_rht_matrix,
    get_wgrad_sign_vector,
)
from torchao.utils import is_sm_at_least_100

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


# M=32 excluded: all BLOCK_M configs (64, 128) exceed M=32 → all autotune configs fail.
_M_VALUES = [64, 96, 128, 160, 256, 512]
# N=100 excluded: TMA TensorDescriptor requires stride % 16 bytes == 0;
# for bf16 this means N % 8 == 0. N=100 (100*2=200 bytes, 200%16=8) fails.
_N_VALUES = [128, 200, 256, 384, 512, 1024]


@torch.no_grad()
def test_get_rht_matrix_with_hardcoded_sign_vector():
    rht_matrix = get_rht_matrix(sign_vector=_HARDCODED_SIGN_VECTOR, device="cpu")

    expected_signs = torch.tensor(_HARDCODED_SIGN_VECTOR, dtype=torch.bfloat16)
    expected = torch.diag(expected_signs) @ get_hadamard_matrix(16, device="cpu")
    torch.testing.assert_close(rht_matrix, expected, atol=0, rtol=0)


@torch.no_grad()
def test_get_rht_matrix_with_generated_sign_matches_sampled_signs():
    get_rht_matrix.cache_clear()
    torch.manual_seed(42)
    expected_signs = get_wgrad_sign_vector(16, device="cpu")

    get_rht_matrix.cache_clear()
    torch.manual_seed(42)
    rht_matrix = get_rht_matrix(sign_vector=None, device="cpu")

    expected = torch.diag(expected_signs) @ get_hadamard_matrix(16, device="cpu")
    torch.testing.assert_close(rht_matrix, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "sign_vector",
    [None, _HARDCODED_SIGN_VECTOR],
    ids=["generated_sign_vector", "hardcoded_sign_vector"],
)
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_triton_rht_amax_vs_reference(M, N, sign_vector):
    """triton_rht_amax must match the reference RHT matmul amax exactly (bitwise)."""
    from torchao.prototype.mx_formats.hadamard_amax_triton import triton_rht_amax

    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    get_rht_matrix.cache_clear()
    if sign_vector is None:
        torch.manual_seed(42)
    B = get_rht_matrix(sign_vector=sign_vector, device="cuda")
    ref_rht_amax = (
        (A.t().reshape(N * M // 16, 16) @ B).to(torch.bfloat16).abs().max().float()
    )
    ref_amax = A.abs().max().float()

    get_rht_matrix.cache_clear()
    if sign_vector is None:
        torch.manual_seed(42)

    # Check RHT amax and regular amax are bitwise identical to reference.
    triton_rht_amax_val, triton_amax_val = triton_rht_amax(A, sign_vector=sign_vector)
    torch.testing.assert_close(triton_rht_amax_val, ref_rht_amax, atol=0, rtol=0)
    torch.testing.assert_close(triton_amax_val, ref_amax, atol=0, rtol=0)
