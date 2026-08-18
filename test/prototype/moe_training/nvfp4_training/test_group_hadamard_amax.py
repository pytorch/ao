# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the CuteDSL per-group RHT amax over a row-packed ragged tensor (SM100+).

The grouped kernel processes each group's 128-aligned row range with the same
supertiles the dense amax kernel uses on that slice, so per-group results are
bitwise equal to ``cutedsl_rht_amax`` on ``A[start:end]``.
"""

import pytest
import torch

from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_cutedsl import (
    cutedsl_group_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_cutedsl import (
    cutedsl_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)

_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)

_SIGN = [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1]
VARYING_FIRST_DIM = 1

# (group row counts, hidden size): ragged 128-aligned groups, both supertile-relevant K.
_CONFIGS = [
    pytest.param([128, 384, 256], 256, id="g3_K256"),
    pytest.param([256, 128, 512, 128, 384], 512, id="g5_K512"),
    pytest.param([128], 128, id="g1_K128"),
    pytest.param([1024], 384, id="g1_large_K384"),
]


def _packed(groups, K, scale_per_group=True):
    torch.manual_seed(0)
    parts = []
    for i, rows in enumerate(groups):
        scale = float(i + 1) if scale_per_group else 1.0
        parts.append(
            scale * torch.randn(rows, K, dtype=torch.bfloat16, device="cuda")
        )
    A = torch.cat(parts, dim=0).contiguous()
    offsets = torch.tensor(
        [sum(groups[: i + 1]) for i in range(len(groups))],
        dtype=torch.int32,
        device="cuda",
    )
    return A, offsets


@_skip_no_cutedsl
@pytest.mark.parametrize("groups,K", _CONFIGS)
@torch.no_grad()
def test_group_rht_amax_matches_dense_per_group(groups, K):
    A, offsets = _packed(groups, K)
    T = A.shape[0]
    col, row = cutedsl_group_rht_amax(
        A, _SIGN, offsets, len(groups), T, K, VARYING_FIRST_DIM
    )
    assert col.shape == (len(groups),) and row.shape == (len(groups),)
    start = 0
    for g, rows in enumerate(groups):
        d_col, d_row = cutedsl_rht_amax(A[start : start + rows].contiguous(), _SIGN)
        torch.testing.assert_close(col[g], d_col.reshape(()), atol=0, rtol=0)
        torch.testing.assert_close(row[g], d_row.reshape(()), atol=0, rtol=0)
        start += rows


@_skip_no_cutedsl
@torch.no_grad()
def test_group_rht_amax_skips_capacity_rows():
    """Rows in [logical, capacity) are storage only; poison them and expect no effect."""
    groups, K = [256, 384], 256
    A, offsets = _packed(groups, K)
    logical = A.shape[0]
    capacity = logical + 512
    A_cap = torch.full(
        (capacity, K), 3.0e38, dtype=torch.bfloat16, device="cuda"
    )
    A_cap[:logical] = A
    logical_t = torch.tensor([logical], dtype=torch.int32, device="cuda")
    col, row = cutedsl_group_rht_amax(
        A_cap, _SIGN, offsets, len(groups), capacity, K, VARYING_FIRST_DIM,
        logical_packed_length=logical_t,
    )
    ref_col, ref_row = cutedsl_group_rht_amax(
        A, _SIGN, offsets, len(groups), logical, K, VARYING_FIRST_DIM
    )
    torch.testing.assert_close(col, ref_col, atol=0, rtol=0)
    torch.testing.assert_close(row, ref_row, atol=0, rtol=0)
    assert torch.isfinite(col).all() and torch.isfinite(row).all()


@_skip_no_cutedsl
@torch.no_grad()
def test_group_rht_amax_zero_group():
    """An all-zero (fully padded) group reports zero amaxes."""
    groups, K = [128, 128, 256], 256
    A, offsets = _packed(groups, K)
    A[128:256] = 0.0  # zero out group 1
    col, row = cutedsl_group_rht_amax(
        A, _SIGN, offsets, len(groups), A.shape[0], K, VARYING_FIRST_DIM
    )
    assert col[1].item() == 0.0 and row[1].item() == 0.0
    assert col[0].item() > 0.0 and col[2].item() > 0.0


@_skip_no_cutedsl
@torch.no_grad()
def test_group_rht_amax_invalid_inputs_raise():
    A, offsets = _packed([128, 128], 256)
    T = A.shape[0]
    with pytest.raises(ValueError, match="VARYING_FIRST_DIM"):
        cutedsl_group_rht_amax(A, _SIGN, offsets, 2, T, 256, 0)
    with pytest.raises(ValueError, match="TensorWise"):
        cutedsl_group_rht_amax(
            A, _SIGN, offsets, 2, T, 256, VARYING_FIRST_DIM, scaling_type=1
        )
    with pytest.raises(ValueError, match="entries"):
        cutedsl_group_rht_amax(A, _SIGN, offsets, 3, T, 256, VARYING_FIRST_DIM)
    with pytest.raises(ValueError, match="int32"):
        cutedsl_group_rht_amax(
            A, _SIGN, offsets.to(torch.int64), 2, T, 256, VARYING_FIRST_DIM
        )


@_skip_no_cutedsl
@torch.no_grad()
def test_group_rht_amax_fake_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        A = torch.empty(512, 256, dtype=torch.bfloat16, device="cuda")
        offsets = torch.empty(4, dtype=torch.int32, device="cuda")
        col, row = cutedsl_group_rht_amax(
            A, _SIGN, offsets, 4, 512, 256, VARYING_FIRST_DIM
        )
    assert col.shape == (4,) and row.shape == (4,)
    assert col.dtype == torch.float32 and row.dtype == torch.float32
