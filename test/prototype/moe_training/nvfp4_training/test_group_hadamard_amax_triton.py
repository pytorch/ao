"""Tests for triton_group_rht_amax (SM100+ grouped RHT amax).

These tests compare the grouped kernel against per-expert calls to the non-grouped
triton_rht_amax and verify the grouped custom op's fake output shapes and shared
input validation.

Semantics match the non-grouped triton_rht_amax (single sign vector):
  col_amax[g] = max|RHT(A_g.T)|, row_amax[g] = max|A_g|.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.utils import is_sm_at_least_100, torch_version_at_least

if has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0"):
    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
        triton_group_rht_amax,
    )
    from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
        triton_rht_amax,
    )
    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
        get_rht_matrix,
    )

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

requires_sm100 = [
    pytest.mark.skipif(not has_triton(), reason="unsupported without triton"),
    pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+"),
    pytest.mark.skipif(
        not torch_version_at_least("2.10.0"),
        reason="requires PyTorch 2.10+",
    ),
]


def _maybe_sm100(fn):
    for mark in requires_sm100:
        fn = mark(fn)
    return fn


def _build_packed(groups, hidden_size, device, seed):
    """Return (A, offsets, group_tensors) for row-concatenated expert groups."""
    torch.manual_seed(seed)
    group_tensors = [
        torch.randn((m, hidden_size), dtype=torch.bfloat16, device=device)
        for m in groups
    ]
    A = torch.cat(group_tensors, dim=0)
    offsets = torch.empty((len(groups) + 1,), dtype=torch.int64, device=device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(
        torch.tensor(groups, dtype=torch.int64, device=device) * hidden_size, dim=0
    )
    return A, offsets, group_tensors


def _group_rht_amax_reference(A, offsets, num_tensors, sign_vector):
    """Grouped oracle: per-expert single-GPU triton_rht_amax over packed rows."""
    col = A.new_empty((num_tensors,), dtype=torch.float32)
    row = A.new_empty((num_tensors,), dtype=torch.float32)
    for g in range(num_tensors):
        row_start = offsets[g] // A.shape[1]
        row_end = offsets[g + 1] // A.shape[1]
        Ag = A[row_start:row_end]
        c, r = triton_rht_amax(Ag, list(sign_vector))
        col[g] = c
        row[g] = r
    return col, row


@_maybe_sm100
@torch.no_grad()
def test_group_rht_amax_matches_per_group_kernel_bitwise():
    """Grouped outputs exactly match per-expert triton_rht_amax outputs."""
    device = torch.device("cuda", 0)
    groups = (128, 256)
    hidden_size = 256
    A, offsets, _ = _build_packed(groups, hidden_size, device, seed=223)

    expected_col, expected_row = _group_rht_amax_reference(
        A, offsets, len(groups), _HARDCODED_SIGN_VECTOR
    )

    B = get_rht_matrix(_HARDCODED_SIGN_VECTOR, device, torch.bfloat16, 16)
    actual_col, actual_row = triton_group_rht_amax(
        A, B, offsets, len(groups), A.shape[0], hidden_size, 1
    )

    assert torch.equal(actual_col, expected_col)
    assert torch.equal(actual_row, expected_row)


@_maybe_sm100
@torch.no_grad()
def test_group_rht_amax_register_fake_shapes():
    """register_fake yields (num_tensors,) float32 outputs under fake mode."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    num_tensors = 3
    with FakeTensorMode():
        A = torch.empty((512, 256), dtype=torch.bfloat16, device="cuda")
        B = torch.empty((16, 16), dtype=torch.bfloat16, device="cuda")
        offsets = torch.empty((num_tensors + 1,), dtype=torch.int64, device="cuda")
        col, row = triton_group_rht_amax(A, B, offsets, num_tensors, 512, 256, 1)
    assert col.shape == (num_tensors,)
    assert row.shape == (num_tensors,)
    assert col.dtype == torch.float32
    assert row.dtype == torch.float32


@_maybe_sm100
@torch.no_grad()
def test_group_rht_amax_validates_packed_shape():
    """The grouped amax wrapper applies shared packed-shape validation."""
    device = torch.device("cuda", 0)
    groups = (128,)
    A, offsets, _ = _build_packed(groups, 256, device, seed=7)
    B = get_rht_matrix(_HARDCODED_SIGN_VECTOR, device, torch.bfloat16, 16)
    with pytest.raises(ValueError, match="packed_sequence_length must match"):
        triton_group_rht_amax(A, B, offsets, len(groups), 256, 256, 1)
