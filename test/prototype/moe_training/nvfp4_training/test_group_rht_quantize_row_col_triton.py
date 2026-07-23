"""Tests for triton_group_rht_quantize_row_col (SM100+ grouped RHT kernel).

Pure-torch oracle (no TransformerEngine), mirroring
test_hadamard_quantize_row_col_triton.py:

  correctness (RTNE):
    - per group, for both columnwise and rowwise swizzled outputs:
        * FP8 scale factors match the mx_formats nvfp4_quantize reference within
          1 fp8 ULP (the kernel uses TE-exact div_rn for the scale; the reference
          multiplies by a reciprocal, so bytes are equal or adjacent).
        * dequantized output reconstructs the post-RHT (col) / raw-A (row) values
          with SQNR >= 20 dB.

  stochastic rounding (oracle-free):
    - launches and produces correctly-shaped outputs.
    - rng_state drives SR: identical state -> identical codes, advanced -> differ.
    - rng_state type/size validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch.utils._triton import has_triton

from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
)
from torchao.float8.float8_utils import compute_error
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    nvfp4_quantize,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.utils import from_blocked, to_blocked
from torchao.utils import is_sm_at_least_100, torch_version_at_least

if has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0"):
    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
        triton_group_rht_amax,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_triton import (
        triton_group_rht_quantize_row_col,
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


@dataclass(frozen=True)
class GraphShapeSpec:
    seed: int
    groups: tuple[int, ...]
    hidden_size: int
    shape_rep: int
    label: str = ""


SHAPE_SPECS = (
    GraphShapeSpec(seed=223, groups=(128,), hidden_size=128, shape_rep=1),
    GraphShapeSpec(seed=224, groups=(128, 256), hidden_size=512, shape_rep=1),
    GraphShapeSpec(
        seed=225, groups=(128, 256, 384, 128), hidden_size=1024, shape_rep=1
    ),
    GraphShapeSpec(seed=226, groups=(128, 128, 128, 128), hidden_size=512, shape_rep=0),
)

DEEPSEEK_SHAPE_SPECS = tuple(
    GraphShapeSpec(
        seed=300 + index,
        groups=(shape.m,) * shape.experts,
        hidden_size=shape.n,
        shape_rep=0,
        label=f"{shape.model}-{shape.projection}",
    )
    for index, shape in enumerate(get_deepseek_v3_weight_shapes(factorized_experts=2))
)


def _rht_reference(A_group: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch reference RHT of A.T in 16-blocks: returns (hidden, m) bfloat16."""
    m, hidden = A_group.shape
    return (A_group.t().reshape(-1, 16) @ B).reshape(hidden, m).to(torch.bfloat16)


def _dequantize_plain(
    codes: torch.Tensor, scales: torch.Tensor, global_amax: torch.Tensor
) -> torch.Tensor:
    """Decode packed FP4 codes + plain (non-swizzled) scales via NVFP4Tensor."""
    return (
        NVFP4Tensor(
            codes.contiguous(),
            scales.contiguous(),
            16,
            torch.bfloat16,
            per_tensor_scale=per_tensor_amax_to_scale(global_amax),
            is_swizzled_scales=False,
        )
        .dequantize()
        .float()
    )


def _assert_scales_adjacent(got: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    """Kernel scale (TE-exact div_rn) vs mx_formats reference (reciprocal multiply):
    equal or adjacent fp8 bytes (positive e4m3 bytes are magnitude-monotonic)."""
    got_b = got.flatten().contiguous().view(torch.uint8).to(torch.int16)
    ref_b = ref.flatten().contiguous().view(torch.uint8).to(torch.int16)
    assert got_b.shape == ref_b.shape, (
        f"{label}: shape mismatch {tuple(got_b.shape)} vs {tuple(ref_b.shape)}"
    )
    diff = (got_b - ref_b).abs()
    assert (diff <= 1).all(), (
        f"{label}: {(diff > 1).sum().item()}/{diff.numel()} fp8 scale bytes "
        f"differ by >1 ULP (max {diff.max().item()})"
    )


def _make_rng_state(device, values=(1, 2, 3, 4)) -> torch.Tensor:
    """[col_seed, col_offset, row_seed, row_offset] caller-owned Philox state."""
    return torch.tensor(list(values), dtype=torch.int64, device=device)


def _build_graph_case(spec):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GroupRHT correctness")

    device = torch.device("cuda", 0)
    torch.manual_seed(spec.seed)

    group_tensors = [
        torch.randn((m, spec.hidden_size), dtype=torch.bfloat16, device=device)
        for m in spec.groups
    ]
    A = torch.cat(group_tensors, dim=0)
    B = get_rht_matrix(_HARDCODED_SIGN_VECTOR, device, torch.bfloat16, 16)

    first_dims = torch.tensor(spec.groups, dtype=torch.int32, device=device)
    offsets = torch.cumsum(first_dims, dim=0, dtype=torch.int32)

    num_groups = len(spec.groups)
    amax_row = torch.empty((num_groups,), dtype=torch.float32, device=device)
    amax_col = torch.empty((num_groups,), dtype=torch.float32, device=device)
    rht_groups = []
    for g, A_g in enumerate(group_tensors):
        rht_g = _rht_reference(A_g, B)
        rht_groups.append(rht_g)
        amax_row[g] = A_g.float().abs().max()
        amax_col[g] = rht_g.float().abs().max()

    return spec, A, B, offsets, amax_row, amax_col, group_tensors, rht_groups


@pytest.fixture(scope="module", params=SHAPE_SPECS, ids=lambda s: f"seed{s.seed}")
def graph_case(request):
    return _build_graph_case(request.param)


@pytest.fixture(
    scope="module",
    params=DEEPSEEK_SHAPE_SPECS,
    ids=lambda spec: spec.label,
)
def deepseek_graph_case(request):
    return _build_graph_case(request.param)


def _check_output_shapes(spec, qa, sfa, qd, sfd):
    psl = sum(spec.groups)
    hs = spec.hidden_size
    assert qa.shape == (psl, hs // 2)
    assert sfa.shape == (psl, hs // 16)
    assert qa.dtype == torch.uint8 and sfa.dtype == torch.float8_e4m3fn

    assert qd.shape == (hs, psl // 2)
    assert sfd.shape == (hs, psl // 16)
    assert qd.dtype == torch.uint8 and sfd.dtype == torch.float8_e4m3fn


def _assert_group_rht_correctness(graph_case):
    spec, A, B, offsets, amax_row, amax_col, group_tensors, rht_groups = graph_case
    psl, hs = A.shape
    num_groups = len(spec.groups)

    qa, sfa, qd, sfd = triton_group_rht_quantize_row_col(
        A,
        list(_HARDCODED_SIGN_VECTOR),
        offsets,
        num_groups,
        psl,
        hs,
        spec.shape_rep,
        amax_row,
        amax_col,
        None,
        False,
    )
    _check_output_shapes(spec, qa, sfa, qd, sfd)

    expected_col_sf = torch.empty(
        (hs, psl // 16), dtype=torch.float8_e4m3fn, device=A.device
    )
    expected_row_sf = torch.empty(
        (psl, hs // 16), dtype=torch.float8_e4m3fn, device=A.device
    )
    col_sf_plain = from_blocked(sfd, hs, psl // 16)
    row_sf_plain = from_blocked(sfa, psl, hs // 16)

    row_offset = 0
    for g, (m, A_g, rht_g) in enumerate(zip(spec.groups, group_tensors, rht_groups)):
        ref_col_sf, _ = nvfp4_quantize(
            rht_g, per_tensor_scale=per_tensor_amax_to_scale(amax_col[g])
        )
        code_slice = slice(row_offset // 2, (row_offset + m) // 2)
        sf_slice = slice(row_offset // 16, (row_offset + m) // 16)
        expected_col_sf[:, sf_slice] = ref_col_sf
        _assert_scales_adjacent(
            col_sf_plain[:, sf_slice], ref_col_sf, f"group {g} col sf"
        )
        dq = _dequantize_plain(
            qd[:, code_slice], col_sf_plain[:, sf_slice], amax_col[g]
        )
        sqnr = compute_error(rht_g.float(), dq)
        assert sqnr >= 20.0, f"group {g} col SQNR {sqnr:.2f} dB < 20"

        ref_row_sf, _ = nvfp4_quantize(
            A_g, per_tensor_scale=per_tensor_amax_to_scale(amax_row[g])
        )
        row_slice = slice(row_offset, row_offset + m)
        expected_row_sf[row_slice] = ref_row_sf
        _assert_scales_adjacent(
            row_sf_plain[row_slice], ref_row_sf, f"group {g} row sf"
        )
        dq = _dequantize_plain(qa[row_slice], row_sf_plain[row_slice], amax_row[g])
        sqnr = compute_error(A_g.float(), dq)
        assert sqnr >= 20.0, f"group {g} row SQNR {sqnr:.2f} dB < 20"

        row_offset += m

    _assert_scales_adjacent(sfd, to_blocked(expected_col_sf), "col sf swizzled")
    _assert_scales_adjacent(sfa, to_blocked(expected_row_sf), "row sf swizzled")


@_maybe_sm100
@torch.no_grad()
def test_group_rht_correctness(graph_case):
    _assert_group_rht_correctness(graph_case)


@_maybe_sm100
@torch.no_grad()
def test_group_rht_deepseek_dimensions_correctness(deepseek_graph_case):
    """Real TorchTitan M/N dimensions with E factorized to two experts."""
    _assert_group_rht_correctness(deepseek_graph_case)


@_maybe_sm100
@torch.no_grad()
def test_group_rht_padded_capacity_masks_spare_rows():
    """Capacity rows do not affect amax and flush to zero during quantization."""
    device = torch.device("cuda", 0)
    logical_rows, capacity_rows, hidden_size = 128, 256, 128
    torch.manual_seed(227)
    valid = torch.randn(
        (logical_rows, hidden_size), dtype=torch.bfloat16, device=device
    )
    capacity = torch.empty(
        (capacity_rows, hidden_size), dtype=torch.bfloat16, device=device
    )
    capacity[:logical_rows].copy_(valid)
    capacity[logical_rows:].fill_(1000.0)
    offsets = torch.tensor([logical_rows], dtype=torch.int32, device=device)
    logical_packed_length = offsets[-1:]

    expected_col_amax, expected_row_amax = triton_group_rht_amax(
        valid,
        list(_HARDCODED_SIGN_VECTOR),
        offsets,
        1,
        logical_rows,
        hidden_size,
        1,
    )
    actual_col_amax, actual_row_amax = triton_group_rht_amax(
        capacity,
        list(_HARDCODED_SIGN_VECTOR),
        offsets,
        1,
        capacity_rows,
        hidden_size,
        1,
        logical_packed_length=logical_packed_length,
    )
    assert torch.equal(actual_col_amax, expected_col_amax)
    assert torch.equal(actual_row_amax, expected_row_amax)

    expected = triton_group_rht_quantize_row_col(
        valid,
        list(_HARDCODED_SIGN_VECTOR),
        offsets,
        1,
        logical_rows,
        hidden_size,
        1,
        expected_row_amax,
        expected_col_amax,
        None,
        False,
    )
    actual = triton_group_rht_quantize_row_col(
        capacity,
        list(_HARDCODED_SIGN_VECTOR),
        offsets,
        1,
        capacity_rows,
        hidden_size,
        1,
        actual_row_amax,
        actual_col_amax,
        None,
        False,
        logical_packed_length=logical_packed_length,
    )
    expected_qa, expected_sfa, expected_qd, expected_sfd = expected
    actual_qa, actual_sfa, actual_qd, actual_sfd = actual
    actual_sfa_plain = from_blocked(actual_sfa, capacity_rows, hidden_size // 16)
    actual_sfd_plain = from_blocked(actual_sfd, hidden_size, capacity_rows // 16)

    assert torch.equal(actual_qa[:logical_rows], expected_qa)
    assert torch.equal(
        actual_sfa_plain[:logical_rows],
        from_blocked(expected_sfa, logical_rows, hidden_size // 16),
    )
    assert torch.equal(actual_qd[:, : logical_rows // 2], expected_qd)
    assert torch.equal(
        actual_sfd_plain[:, : logical_rows // 16],
        from_blocked(expected_sfd, hidden_size, logical_rows // 16),
    )
    assert torch.count_nonzero(actual_qa[logical_rows:]) == 0
    assert torch.count_nonzero(actual_sfa_plain[logical_rows:]) == 0
    assert torch.count_nonzero(actual_qd[:, logical_rows // 2 :]) == 0
    assert torch.count_nonzero(actual_sfd_plain[:, logical_rows // 16 :]) == 0


@_maybe_sm100
@torch.no_grad()
def test_group_rht_stochastic_rounding_launches(graph_case):
    spec, A, B, offsets, amax_row, amax_col, _, _ = graph_case
    psl, hs = A.shape
    num_groups = len(spec.groups)

    qa, sfa, qd, sfd = triton_group_rht_quantize_row_col(
        A,
        list(_HARDCODED_SIGN_VECTOR),
        offsets,
        num_groups,
        psl,
        hs,
        spec.shape_rep,
        amax_row,
        amax_col,
        _make_rng_state(A.device),
        True,
    )
    _check_output_shapes(spec, qa, sfa, qd, sfd)
    assert torch.isfinite(sfa.float()).all()
    assert torch.isfinite(sfd.float()).all()


def _run_sr(graph_case, rng_state):
    spec, A, B, offsets, amax_row, amax_col, _, _ = graph_case
    psl, hs = A.shape
    return triton_group_rht_quantize_row_col(
        A,
        list(_HARDCODED_SIGN_VECTOR),
        offsets,
        len(spec.groups),
        psl,
        hs,
        spec.shape_rep,
        amax_row,
        amax_col,
        rng_state,
        True,
    )


@_maybe_sm100
@torch.no_grad()
def test_group_rht_rng_state_controls_stochastic_rounding(graph_case):
    """Same rng_state -> identical packed codes; advanced state -> codes differ."""
    qa1, _, qd1, _ = _run_sr(
        graph_case, _make_rng_state(graph_case[1].device, (11, 22, 33, 44))
    )
    qa2, _, qd2, _ = _run_sr(
        graph_case, _make_rng_state(graph_case[1].device, (11, 22, 33, 44))
    )
    assert torch.equal(qa1, qa2), "Same rng_state must yield identical row FP4 codes"
    assert torch.equal(qd1, qd2), "Same rng_state must yield identical col FP4 codes"

    qa3, _, qd3, _ = _run_sr(
        graph_case, _make_rng_state(graph_case[1].device, (11, 9999, 33, 8888))
    )
    assert not torch.equal(qa1, qa3), "Advanced rng_state must change row FP4 codes"
    assert not torch.equal(qd1, qd3), "Advanced rng_state must change col FP4 codes"


@_maybe_sm100
@torch.no_grad()
def test_group_rht_rng_state_validation(graph_case):
    """SR enabled requires a valid int64 rng_state; SR disabled ignores it."""
    spec, A, B, offsets, amax_row, amax_col, _, _ = graph_case
    psl, hs = A.shape
    num_groups = len(spec.groups)

    with pytest.raises(TypeError, match="rng_state must be a torch.Tensor"):
        _run_sr(graph_case, None)

    with pytest.raises(ValueError, match="at least 4 elements"):
        _run_sr(graph_case, _make_rng_state(A.device, (1, 2)))

    # SR disabled: rng_state is ignored, so None is accepted.
    triton_group_rht_quantize_row_col(
        A,
        list(_HARDCODED_SIGN_VECTOR),
        offsets,
        num_groups,
        psl,
        hs,
        spec.shape_rep,
        amax_row,
        amax_col,
        None,
        False,
    )


@_maybe_sm100
@pytest.mark.parametrize(
    ("invalid_amax", "error"),
    [
        ("2d", "a_global_amax must be 1D"),
        ("noncontiguous", "a_global_amax must be contiguous"),
    ],
)
def test_group_rht_amax_storage_validation(graph_case, invalid_amax, error):
    spec, A, _, offsets, _, amax_col, _, _ = graph_case
    if invalid_amax == "2d":
        amax_row = torch.empty(
            (1, len(spec.groups)), dtype=torch.float32, device=A.device
        )
    else:
        storage_size = max(2, len(spec.groups))
        amax_row = torch.empty(
            (storage_size * 2,), dtype=torch.float32, device=A.device
        )[::2]

    with pytest.raises(ValueError, match=error):
        triton_group_rht_quantize_row_col(
            A,
            list(_HARDCODED_SIGN_VECTOR),
            offsets,
            len(spec.groups),
            A.shape[0],
            A.shape[1],
            spec.shape_rep,
            amax_row,
            amax_col,
            None,
            False,
        )
