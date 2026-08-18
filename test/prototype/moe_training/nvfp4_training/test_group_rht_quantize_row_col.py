# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the CuteDSL grouped fused RHT row+col quantize (SM100+).

128-aligned group boundaries never split a 16-token RHT segment or scale vector,
so group g's outputs equal the dense op's outputs on its row range when fed the
same per-group amaxes -- the primary contract here is bitwise equality with
``cutedsl_rht_quantize_row_col`` per group slice.
"""

import pytest
import torch

from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_cutedsl import (
    cutedsl_group_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_cutedsl import (
    cutedsl_group_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_cutedsl import (
    cutedsl_rht_quantize_row_col,
)

_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)

_SIGN = [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1]
VARYING_FIRST_DIM = 1

_CONFIGS = [
    pytest.param([128, 384, 256], 256, id="g3_K256"),
    pytest.param([256, 128, 512, 128, 384], 512, id="g5_K512"),
    pytest.param([1024], 384, id="g1_K384"),
]


def _packed(groups, K):
    torch.manual_seed(0)
    parts = [
        float(i + 1) * torch.randn(rows, K, dtype=torch.bfloat16, device="cuda")
        for i, rows in enumerate(groups)
    ]
    A = torch.cat(parts, dim=0).contiguous()
    offsets = torch.tensor(
        [sum(groups[: i + 1]) for i in range(len(groups))],
        dtype=torch.int32,
        device="cuda",
    )
    return A, offsets


def _grouped(A, offsets, E, row_amax, col_amax, sr=False, rng=None, T=None, logical=None):
    T = A.shape[0] if T is None else T
    return cutedsl_group_rht_quantize_row_col(
        A, _SIGN, offsets, E, T, A.shape[1], VARYING_FIRST_DIM,
        row_amax, col_amax, rng, sr, logical_packed_length=logical,
    )


@_skip_no_cutedsl
@pytest.mark.parametrize("groups,K", _CONFIGS)
@torch.no_grad()
def test_group_quantize_matches_dense_per_group(groups, K):
    A, offsets = _packed(groups, K)
    E, T = len(groups), A.shape[0]
    col_amax, row_amax = cutedsl_group_rht_amax(
        A, _SIGN, offsets, E, T, K, VARYING_FIRST_DIM
    )
    qa, sfa, qd, sfd = _grouped(A, offsets, E, row_amax, col_amax)
    assert qa.shape == (T, K // 2) and sfa.shape == (T, K // 16)
    assert qd.shape == (K, T // 2) and sfd.shape == (K, T // 16)

    sfa4 = sfa.reshape(T // 128, K // 64, 32, 16)
    sfd4 = sfd.reshape(K // 128, T // 64, 32, 16)
    start = 0
    for g, rows in enumerate(groups):
        end = start + rows
        d_col, d_col_sf, d_row, d_row_sf = cutedsl_rht_quantize_row_col(
            A[start:end].contiguous(),
            col_amax[g].reshape(1),
            row_amax[g].reshape(1),
            _SIGN,
        )
        # Row grain: rows [start, end) of the packed outputs.
        torch.testing.assert_close(qa[start:end], d_row, atol=0, rtol=0)
        torch.testing.assert_close(
            sfa4[start // 128 : end // 128].view(torch.uint8),
            d_row_sf.view(torch.uint8),
            atol=0,
            rtol=0,
        )
        # Col grain: byte-columns [start//2, end//2) / SF blocks [start//64, end//64).
        torch.testing.assert_close(qd[:, start // 2 : end // 2], d_col, atol=0, rtol=0)
        torch.testing.assert_close(
            sfd4[:, start // 64 : end // 64].contiguous().view(torch.uint8),
            d_col_sf.view(torch.uint8),
            atol=0,
            rtol=0,
        )
        start = end


@_skip_no_cutedsl
@torch.no_grad()
def test_group_quantize_skips_capacity_rows():
    groups, K = [256, 384], 256
    A, offsets = _packed(groups, K)
    E, logical = len(groups), A.shape[0]
    capacity = logical + 384
    A_cap = torch.full((capacity, K), 3.0e38, dtype=torch.bfloat16, device="cuda")
    A_cap[:logical] = A
    logical_t = torch.tensor([logical], dtype=torch.int32, device="cuda")
    col_amax, row_amax = cutedsl_group_rht_amax(
        A_cap, _SIGN, offsets, E, capacity, K, VARYING_FIRST_DIM,
        logical_packed_length=logical_t,
    )
    qa_c, sfa_c, qd_c, sfd_c = _grouped(
        A_cap, offsets, E, row_amax, col_amax, T=capacity, logical=logical_t
    )
    qa, sfa, qd, sfd = _grouped(A, offsets, E, row_amax, col_amax)
    torch.testing.assert_close(qa_c[:logical], qa, atol=0, rtol=0)
    torch.testing.assert_close(
        sfa_c.reshape(capacity // 128, -1)[: logical // 128],
        sfa.reshape(logical // 128, -1),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(qd_c[:, : logical // 2], qd, atol=0, rtol=0)
    torch.testing.assert_close(
        sfd_c.reshape(K // 128, capacity // 64, 32, 16)[:, : logical // 64]
        .contiguous()
        .view(torch.uint8),
        sfd.reshape(K // 128, logical // 64, 32, 16).view(torch.uint8),
        atol=0,
        rtol=0,
    )


@_skip_no_cutedsl
@torch.no_grad()
def test_group_quantize_stochastic_rounding():
    """SR is deterministic for a fixed rng_state, differs across states, and leaves
    the scale factors identical to RTNE (rounding affects codes only)."""
    groups, K = [256, 256], 256
    A, offsets = _packed(groups, K)
    E, T = len(groups), A.shape[0]
    col_amax, row_amax = cutedsl_group_rht_amax(
        A, _SIGN, offsets, E, T, K, VARYING_FIRST_DIM
    )
    rng1 = torch.tensor([7, 1234, 8, 999], dtype=torch.int64, device="cuda")
    rng2 = torch.tensor([7, 5678, 8, 111], dtype=torch.int64, device="cuda")
    out_a = _grouped(A, offsets, E, row_amax, col_amax, sr=True, rng=rng1)
    out_b = _grouped(A, offsets, E, row_amax, col_amax, sr=True, rng=rng1.clone())
    out_c = _grouped(A, offsets, E, row_amax, col_amax, sr=True, rng=rng2)
    rtne = _grouped(A, offsets, E, row_amax, col_amax)
    for a, b in zip(out_a, out_b):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not torch.equal(out_a[0], out_c[0])
    torch.testing.assert_close(
        out_a[1].view(torch.uint8), rtne[1].view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        out_a[3].view(torch.uint8), rtne[3].view(torch.uint8), atol=0, rtol=0
    )


@_skip_no_cutedsl
@torch.no_grad()
def test_group_quantize_invalid_inputs_raise():
    A, offsets = _packed([128, 128], 256)
    E, T = 2, A.shape[0]
    amax = torch.ones(E, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="VARYING_FIRST_DIM"):
        cutedsl_group_rht_quantize_row_col(
            A, _SIGN, offsets, E, T, 256, 0, amax, amax, None, False
        )
    with pytest.raises(ValueError, match="rng_state"):
        cutedsl_group_rht_quantize_row_col(
            A, _SIGN, offsets, E, T, 256, VARYING_FIRST_DIM, amax, amax, None, True
        )
    with pytest.raises(ValueError, match="float32"):
        cutedsl_group_rht_quantize_row_col(
            A, _SIGN, offsets, E, T, 256, VARYING_FIRST_DIM,
            amax.to(torch.float64), amax, None, False,
        )


@_skip_no_cutedsl
@torch.no_grad()
def test_group_quantize_fake_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        A = torch.empty(512, 256, dtype=torch.bfloat16, device="cuda")
        offsets = torch.empty(4, dtype=torch.int32, device="cuda")
        amax = torch.empty(4, dtype=torch.float32, device="cuda")
        qa, sfa, qd, sfd = cutedsl_group_rht_quantize_row_col(
            A, _SIGN, offsets, 4, 512, 256, VARYING_FIRST_DIM, amax, amax, None, False
        )
    assert qa.shape == (512, 128) and sfa.shape == (512, 16)
    assert qd.shape == (256, 256) and sfd.shape == (256, 32)
