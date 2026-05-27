"""Tests for triton_rht_amax (SM100+ kernel)."""

import pytest
import torch
from torch.utils._triton import has_triton

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


# M=32 excluded: all BLOCK_M configs (64, 128) exceed M=32 → all autotune configs fail.
_M_VALUES = [64, 96, 128, 160, 256, 512]
# triton_rht_amax feeds triton_rht_quantize_row_col, whose swizzled scales
# require N % 128 == 0.
_N_VALUES = [128, 256, 384, 512, 1024]


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


@pytest.mark.parametrize(
    "sign_vector",
    [_HARDCODED_SIGN_VECTOR],
    ids=["hardcoded_sign_vector"],
)
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_triton_rht_amax_vs_reference(M, N, sign_vector):
    """triton_rht_amax must match the reference RHT matmul amax exactly (bitwise)."""
    from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
        triton_rht_amax,
    )

    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    get_rht_matrix.cache_clear()
    B = get_rht_matrix(sign_vector, "cuda", torch.bfloat16, 16)
    ref_rht_amax = (
        (A.t().reshape(N * M // 16, 16) @ B).to(torch.bfloat16).abs().max().float()
    )
    ref_amax = A.abs().max().float()

    get_rht_matrix.cache_clear()

    # Check RHT amax and regular amax are bitwise identical to reference.
    triton_rht_amax_val, triton_amax_val = triton_rht_amax(
        A, sign_vector=list(sign_vector)
    )
    torch.testing.assert_close(triton_rht_amax_val, ref_rht_amax, atol=0, rtol=0)
    torch.testing.assert_close(triton_amax_val, ref_amax, atol=0, rtol=0)
