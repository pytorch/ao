"""Tests for the grouped RHT quantize kernels (SM100+), triton and cutedsl.

Two oracles: the TE-derived reference in ``nvfp4_reference`` (scales and RTNE codes
bitwise) and an independent mx_formats cross-check at 1 fp8 ULP. Mirrors
test_hadamard_quantize_row_col.py:

  correctness (RTNE):
    - per group, for both columnwise and rowwise swizzled outputs:
        * FP8 scale factors match the mx_formats nvfp4_quantize reference within
          1 fp8 ULP (the kernel uses TE-exact div_rn for the scale; the reference
          multiplies by a reciprocal, so bytes are equal or adjacent).
        * dequantized output reconstructs the post-RHT (col) / raw-A (row) values
          with SQNR >= 20 dB.

  stochastic rounding (oracle-free):
    - launches and produces correctly-shaped outputs.
    - reconstructs its inputs at SQNR >= 15 dB through the RTNE reference.
    - unbiased: averaging SR draws of an exactly-halfway value converges to it, with a
      ~50/50 split across the two neighbouring grid points.
    - rng_state drives SR: identical state -> identical codes, advanced -> differ.
    - rng_state type/size validation.

  The two backends are byte-for-byte interchangeable under RTNE only. The grouped
  CuteDSL kernel draws one Philox counter per 16-element block and consumes all four
  words, rather than reproducing triton's per-packed-byte counter stride, so its SR
  stream is a different one and is judged on the statistical properties above instead.
  The linear kernels diverge from triton the same way and for the same reason; they
  draw through the same ``philox4_all``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch.utils._triton import has_triton

from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
)
from test.prototype.moe_training.nvfp4_training._assertions import (
    assert_codes_bitwise,
    assert_scales_bitwise,
)
from test.prototype.moe_training.nvfp4_training.nvfp4_reference import (
    reference_group_rht_quantize_row_col,
    to_blocked_grouped,
)
from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    DEFAULT_SIGN_VECTOR,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    nvfp4_quantize,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.utils import from_blocked, to_blocked
from torchao.utils import is_sm_at_least_100, torch_version_at_least

_TILE_ELEMS = 32 * 16  # elements in one swizzled scale tile

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
if cutedsl_nvfp4_kernels_available():
    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_cutedsl import (
        cutedsl_group_rht_amax,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_cutedsl import (
        cutedsl_group_rht_quantize_row_col,
    )

_HARDCODED_SIGN_VECTOR = DEFAULT_SIGN_VECTOR

requires_sm100 = [
    pytest.mark.skipif(not has_triton(), reason="unsupported without triton"),
    pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+"),
    pytest.mark.skipif(
        not torch_version_at_least("2.10.0"),
        reason="requires PyTorch 2.10+",
    ),
]


_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)

_KERNELS = [
    pytest.param("triton", id="triton"),
    pytest.param("cutedsl", marks=_skip_no_cutedsl, id="cutedsl"),
]


def _maybe_sm100(fn):
    for mark in requires_sm100:
        fn = mark(fn)
    return fn


def _group_quantize(kernel, *args, **kwargs):
    """Dispatch to a backend's grouped RHT quantize op."""
    op = (
        triton_group_rht_quantize_row_col
        if kernel == "triton"
        else cutedsl_group_rht_quantize_row_col
    )
    return op(*args, **kwargs)


def _skip_if_unsupported_groups(kernel: str, num_tensors: int) -> None:
    """The cutedsl group lookup is a fixed-depth binary search capped at 64 groups."""
    if kernel == "cutedsl" and num_tensors > 64:
        pytest.skip("cutedsl grouped kernel supports at most 64 groups")


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


def _from_blocked_grouped(sfd, hidden, group_sizes):
    """De-swizzle a columnwise scale buffer, whose groups are blocked separately.

    The columnwise scales put the grouped token axis on the 64-blocked inner
    side, so each group is blocked on its own extent and the buffers are
    concatenated flat -- one whole-extent de-swizzle would read the wrong tiles
    for every group. The rowwise buffer needs no equivalent: there the grouped
    axis is the outer one, where a group is already contiguous.
    """
    out, base = [], 0
    for m in group_sizes:
        span = (hidden // 128) * (m // 64) * _TILE_ELEMS
        chunk = sfd.reshape(-1)[base : base + span].reshape(hidden, m // 16)
        out.append(from_blocked(chunk, hidden, m // 16))
        base += span
    return torch.cat(out, dim=1)


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


def triton_group_rht_quantize_row_col_ref(
    spec,
    A,
    amax_row,
    amax_col,
    group_tensors,
    rht_groups,
    qa,
    sfa,
    qd,
    sfd,
    sqnr_floor=20.0,
):
    """Compare Triton outputs with the per-group PyTorch NVFP4 reference.

    ``sqnr_floor`` is lowered for stochastic rounding, whose error is uniform over the
    quantization interval rather than bounded by half of it -- about twice the variance,
    i.e. roughly 3 dB. That is the price paid for unbiasedness, not a defect, and it is
    measured identically on both backends.
    """
    psl, hs = A.shape
    expected_col_sf = torch.empty(
        (hs, psl // 16), dtype=torch.float8_e4m3fn, device=A.device
    )
    expected_row_sf = torch.empty(
        (psl, hs // 16), dtype=torch.float8_e4m3fn, device=A.device
    )
    col_sf_plain = _from_blocked_grouped(sfd, hs, spec.groups)
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
        assert sqnr >= sqnr_floor, f"group {g} col SQNR {sqnr:.2f} dB < {sqnr_floor}"

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
        assert sqnr >= sqnr_floor, f"group {g} row SQNR {sqnr:.2f} dB < {sqnr_floor}"

        row_offset += m

    _assert_scales_adjacent(
        sfd, to_blocked_grouped(expected_col_sf, spec.groups), "col sf swizzled"
    )
    _assert_scales_adjacent(sfa, to_blocked(expected_row_sf), "row sf swizzled")


def _assert_group_rht_correctness(graph_case, kernel):
    spec, A, _, offsets, amax_row, amax_col, group_tensors, rht_groups = graph_case
    psl, hs = A.shape
    num_groups = len(spec.groups)
    _skip_if_unsupported_groups(kernel, num_groups)

    qa, sfa, qd, sfd = _group_quantize(
        kernel,
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

    # TE-derived reference: packed scale buffers and RTNE codes are bitwise.
    ref_qa, ref_sfa, ref_qd, ref_sfd = reference_group_rht_quantize_row_col(
        A, offsets, num_groups, amax_col, amax_row, _HARDCODED_SIGN_VECTOR
    )
    assert_scales_bitwise(sfa, ref_sfa, "row sf")
    assert_scales_bitwise(sfd, ref_sfd, "col sf")
    assert_codes_bitwise(qa, ref_qa, "row codes")
    assert_codes_bitwise(qd, ref_qd, "col codes")

    # Independent mx_formats cross-check (reciprocal + E4M3_EPS floor, hence 1 ULP).
    triton_group_rht_quantize_row_col_ref(
        spec,
        A,
        amax_row,
        amax_col,
        group_tensors,
        rht_groups,
        qa,
        sfa,
        qd,
        sfd,
    )


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_rht_correctness(graph_case, kernel):
    _assert_group_rht_correctness(graph_case, kernel)


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_rht_deepseek_dimensions_correctness(deepseek_graph_case, kernel):
    """Real TorchTitan M/N dimensions with E factorized to two experts."""
    _assert_group_rht_correctness(deepseek_graph_case, kernel)


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_rht_fast_math_sqnr(graph_case, kernel):
    """Fast math costs little against the exact path it replaces.

    The grouped twin of test_rht_quantize_fast_math_sqnr; see that test for why both
    paths share one loose floor. Columnwise is stable here at 30.4-31.9 dB. Rowwise is
    bitwise identical for two of these four fixtures and 41-49 dB for the other two, and
    the flips are not spread evenly -- for seed224 all 185 land in group 1 and none in
    group 0, because each group's decode scale gives its own small set of denominators
    and only some of them are ones where ``rcp.approx`` and ``div_rn`` disagree.

    The exact path is bitwise against nvfp4_reference, so bounding fast against exact
    transitively grounds fast math in the PyTorch oracle, which cannot host
    ``rcp.approx.ftz.f32`` itself. Compare NVFP4's own quantization noise, floored at
    20 dB by the RTNE reconstruction test.
    """
    spec, A, _, offsets, amax_row, amax_col, _, _ = graph_case
    psl, hs = A.shape
    num_groups = len(spec.groups)
    _skip_if_unsupported_groups(kernel, num_groups)

    def run(use_fast_math):
        return _group_quantize(
            kernel,
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
            use_fast_math=use_fast_math,
        )

    e_qa, e_sfa, e_qd, e_sfd = run(False)
    f_qa, f_sfa, f_qd, f_sfd = run(True)

    # One group's global amax stands in for all of them, unlike the per-group slicing in
    # _assert_group_rht_correctness. Both sides are dequantized identically, so a group
    # whose true scale differs is off by the same constant in each and the comparison
    # stays a valid fast-vs-exact bound -- it just weights the groups uniformly rather
    # than by their amax. Correctness of the per-group scales is covered elsewhere.
    e_row = _dequantize_plain(e_qa, from_blocked(e_sfa, psl, hs // 16), amax_row[0])
    f_row = _dequantize_plain(f_qa, from_blocked(f_sfa, psl, hs // 16), amax_row[0])
    row_sqnr = compute_error(e_row, f_row)
    assert row_sqnr >= 25.0, f"Row fast-vs-exact SQNR {row_sqnr:.2f} dB < 25.0 dB"

    e_col_sf = _from_blocked_grouped(e_sfd, hs, spec.groups)
    f_col_sf = _from_blocked_grouped(f_sfd, hs, spec.groups)
    col_sqnr = compute_error(
        _dequantize_plain(e_qd, e_col_sf, amax_col[0]),
        _dequantize_plain(f_qd, f_col_sf, amax_col[0]),
    )
    assert col_sqnr >= 25.0, f"Col fast-vs-exact SQNR {col_sqnr:.2f} dB < 25.0 dB"


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_fast_math_matches_transformer_engine(kernel, monkeypatch):
    """Both grouped fast paths are byte-identical to actual TE fast math.

    The triton half is what lets triton stand in as the fast-path oracle: TE's fast
    encode scale is ``rcp.approx.ftz.f32`` and no ATen op lowers to that instruction,
    so ``nvfp4_reference`` models the exact path only.
    """
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "1")
    te = pytest.importorskip("transformer_engine.pytorch")
    tex = pytest.importorskip("transformer_engine_torch")

    num_groups, rows, hidden = 4, 128, 128
    packed_rows = num_groups * rows
    torch.manual_seed(123)
    A = torch.randn((packed_rows, hidden), dtype=torch.bfloat16, device="cuda")
    offsets = torch.arange(
        rows, packed_rows + 1, rows, dtype=torch.int32, device="cuda"
    )
    logical_packed_length = torch.tensor(
        [packed_rows], dtype=torch.int32, device="cuda"
    )

    quantizers = []
    for _ in range(num_groups):
        quantizer = te.NVFP4Quantizer(
            fp4_dtype=te.DType.kFloat4E2M1,
            rowwise=True,
            columnwise=True,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=True,
            with_post_rht_amax=True,
            with_random_sign_mask=True,
            stochastic_rounding=False,
        )
        quantizer.optimize_for_gemm = True
        quantizers.append(quantizer)
    expected = tex.split_quantize(A, [rows] * num_groups, quantizers)

    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_cutedsl import (
        cutedsl_group_rht_amax,
    )

    # The amax kernels have no fast-math variant (neither do TE's).
    group_amax = triton_group_rht_amax if kernel == "triton" else cutedsl_group_rht_amax
    col_amax, row_amax = group_amax(
        A,
        list(_HARDCODED_SIGN_VECTOR),
        offsets,
        num_groups,
        packed_rows,
        hidden,
        0,
        logical_packed_length=logical_packed_length,
    )
    qa, sfa, qd, sfd = _group_quantize(
        kernel,
        A,
        list(_HARDCODED_SIGN_VECTOR),
        offsets,
        num_groups,
        packed_rows,
        hidden,
        0,
        row_amax,
        col_amax,
        None,
        False,
        logical_packed_length=logical_packed_length,
        use_fast_math=True,
    )

    expected_qa = torch.cat(
        [tensor._rowwise_data.view(torch.uint8) for tensor in expected], dim=0
    )
    expected_qd = torch.cat(
        [tensor._columnwise_data.view(torch.uint8) for tensor in expected], dim=1
    )
    expected_sfa = torch.cat(
        [tensor._rowwise_scale_inv.flatten() for tensor in expected]
    )
    expected_sfd = torch.cat(
        [tensor._columnwise_scale_inv.flatten() for tensor in expected]
    )
    assert_codes_bitwise(qa, expected_qa, "row codes")
    assert_scales_bitwise(sfa, expected_sfa, "row sf")
    assert_codes_bitwise(qd, expected_qd, "col codes")
    assert_scales_bitwise(sfd, expected_sfd, "col sf")


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_rht_padded_capacity_masks_spare_rows(kernel):
    """Poisoned allocation tail cannot affect any group-addressable output."""
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
    group_amax = triton_group_rht_amax if kernel == "triton" else cutedsl_group_rht_amax

    expected_col_amax, expected_row_amax = group_amax(
        valid,
        list(_HARDCODED_SIGN_VECTOR),
        offsets,
        1,
        logical_rows,
        hidden_size,
        1,
    )
    actual_col_amax, actual_row_amax = group_amax(
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

    expected = _group_quantize(
        kernel,
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
    actual = _group_quantize(
        kernel,
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
    # Only the one group's extent: the capacity tail lies past every group's
    # blocked buffer, so it is not part of the columnwise scale layout at all.
    actual_sfd_plain = _from_blocked_grouped(actual_sfd, hidden_size, (logical_rows,))

    assert torch.equal(actual_qa[:logical_rows], expected_qa)
    assert torch.equal(
        actual_sfa_plain[:logical_rows],
        from_blocked(expected_sfa, logical_rows, hidden_size // 16),
    )
    assert torch.equal(actual_qd[:, : logical_rows // 2], expected_qd)
    assert torch.equal(
        actual_sfd_plain,
        from_blocked(expected_sfd, hidden_size, logical_rows // 16),
    )
    # Tail storage is deliberately unspecified and inaccessible to grouped consumers.


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_rht_stochastic_rounding_launches(graph_case, kernel):
    spec, A, B, offsets, amax_row, amax_col, _, _ = graph_case
    psl, hs = A.shape
    num_groups = len(spec.groups)
    _skip_if_unsupported_groups(kernel, num_groups)

    qa, sfa, qd, sfd = _group_quantize(
        kernel,
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


@_maybe_sm100
@_skip_no_cutedsl
@pytest.mark.parametrize("use_fast_math", [False, True], ids=["exact", "fast"])
@torch.no_grad()
def test_cutedsl_group_quantize_matches_triton_bitwise(graph_case, use_fast_math):
    """The two grouped backends are byte-for-byte interchangeable under RTNE.

    Stochastic rounding is deliberately out of scope here. The grouped CuteDSL kernel
    draws one Philox counter per 16-element block and consumes all four output words
    instead of reproducing triton's per-packed-byte counter stride, so its SR stream is
    a different -- equally valid -- one. SR is held to the properties that actually
    matter for a stochastic kernel: ``test_group_rht_sr_reconstructs`` (same SQNR bar as
    RTNE), ``test_group_rht_sr_unbiased`` (converges in expectation), and
    ``test_group_rht_rng_state_controls_stochastic_rounding`` (determinism). The linear
    kernels diverge from triton under SR the same way, and are held to the same kind of
    structural bound; see ``test_rht_quantize_rs_at_most_one_fp4_step_from_rtne`` in
    ``test_hadamard_quantize_row_col.py``.
    """
    spec, A, B, offsets, amax_row, amax_col, _, _ = graph_case
    psl, hs = A.shape
    num_groups = len(spec.groups)
    _skip_if_unsupported_groups("cutedsl", num_groups)

    args = (
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
    cutedsl = _group_quantize("cutedsl", *args, use_fast_math=use_fast_math)
    triton_out = _group_quantize("triton", *args, use_fast_math=use_fast_math)
    for name, c, t in zip(("qa", "sfa", "qd", "sfd"), cutedsl, triton_out):
        assert torch.equal(c, t), f"{name} differs between backends"


def _run_sr(graph_case, rng_state, kernel="triton", use_fast_math=False):
    spec, A, B, offsets, amax_row, amax_col, _, _ = graph_case
    psl, hs = A.shape
    return _group_quantize(
        kernel,
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
        use_fast_math=use_fast_math,
    )


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_rht_sr_reconstructs(graph_case, kernel):
    """SR output reconstructs its inputs through the same reference RTNE is checked against.

    Bitwise agreement is the wrong contract for a stochastic kernel, so the SR codes go
    through the same per-group helper the RTNE correctness test uses. Its scale
    assertions carry over unchanged -- block scales come from the amax, not the codes,
    so they are rounding-mode independent -- but the SQNR bar drops from 20 dB to 15.
    SR measures 17.2 dB here against RTNE's 20+, on both backends and all four fixtures
    within 0.2 dB, which is the expected ~3 dB variance cost of unbiased rounding rather
    than a defect. 15 dB leaves margin over that spread while still failing hard on the
    corruption this test exists to catch: a wrong nibble order, scale, or block index
    collapses SQNR toward zero, not to 16.
    """
    spec, A, _, offsets, amax_row, amax_col, group_tensors, rht_groups = graph_case
    _skip_if_unsupported_groups(kernel, len(spec.groups))

    qa, sfa, qd, sfd = _run_sr(
        graph_case, _make_rng_state(A.device, (0x5EED, 0x0FF5, 0xBEEF, 0xCAFE)), kernel
    )
    _check_output_shapes(spec, qa, sfa, qd, sfd)
    triton_group_rht_quantize_row_col_ref(
        spec,
        A,
        amax_row,
        amax_col,
        group_tensors,
        rht_groups,
        qa,
        sfa,
        qd,
        sfd,
        sqnr_floor=15.0,
    )


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("use_fast_math", [False, True], ids=["exact", "fast"])
@torch.no_grad()
def test_group_rht_sr_unbiased(kernel, use_fast_math):
    """Hardware stochastic rounding recovers the FP32 value in expectation.

    Feed the rowwise path elements that land EXACTLY halfway between FP4 grid points
    (1.25, between 1.0 and 1.5) -- the maximal-bias case RTNE always pins to one side --
    and confirm averaging many SR draws converges to 1.25 with a ~50/50 grid split. One
    6.0 anchor per 1x16 row block sets the block amax; a global amax of 2688 gives an
    identity global scale, so every other element passes through as its raw 1.25.

    This is the guard SQNR cannot provide: a degenerate or position-correlated stream
    still reconstructs well on average but fails to split the halfway case evenly. It is
    what makes the grouped counter derivation safe to change.
    """
    _skip_if_unsupported_groups(kernel, 2)
    dev = "cuda"
    groups = (128, 256)
    hidden = 512
    psl = sum(groups)

    A = torch.full((psl, hidden), 1.25, dtype=torch.bfloat16, device=dev)
    A[:, ::16] = 6.0  # block amax anchor
    offsets = torch.cumsum(
        torch.tensor(groups, dtype=torch.int32, device=dev), dim=0, dtype=torch.int32
    )
    # Identity global scale, one entry per group.
    amax = torch.full((len(groups),), 2688.0, dtype=torch.float32, device=dev)
    halfway = torch.arange(hidden, device=dev) % 16 != 0  # the 1.25 positions

    def quantize(rng_state, sr):
        qa, sfa, _, _ = _group_quantize(
            kernel,
            A,
            list(_HARDCODED_SIGN_VECTOR),
            offsets,
            len(groups),
            psl,
            hidden,
            1,
            amax,
            amax,
            rng_state,
            sr,
            use_fast_math=use_fast_math,
        )
        sf_plain = from_blocked(sfa, psl, hidden // 16)
        return _dequantize_plain(qa, sf_plain, amax[0])[:, halfway]

    # RTNE pins every halfway element to a single side (no spread).
    assert quantize(None, False).unique().numel() == 1

    K = 32
    acc = torch.zeros(psl, int(halfway.sum()), device=dev)
    n_lo = n_hi = n_other = 0
    for k in range(K):
        offset = k * 2654435761 + 7
        vals = quantize(
            _make_rng_state(dev, (0x12345678, offset, 0xC0FFEE, offset)), True
        )
        acc += vals
        n_lo += int((vals == 1.0).sum())
        n_hi += int((vals == 1.5).sum())
        n_other += int(((vals != 1.0) & (vals != 1.5)).sum())

    mean_sr = (acc / K).mean().item()
    tot = n_lo + n_hi + n_other
    assert n_other == 0, "SR produced off-grid values"
    assert abs(mean_sr - 1.25) < 0.01, f"SR mean {mean_sr:.4f} != 1.25 (biased)"
    assert 0.45 < n_lo / tot < 0.55, f"SR grid split {n_lo / tot:.3f} not ~50/50"


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("use_fast_math", [False, True], ids=["exact", "fast"])
@torch.no_grad()
def test_group_rht_rng_state_controls_stochastic_rounding(
    graph_case, kernel, use_fast_math
):
    """Same rng_state -> identical packed codes; advanced state -> codes differ."""
    _skip_if_unsupported_groups(kernel, len(graph_case[0].groups))
    qa1, _, qd1, _ = _run_sr(
        graph_case,
        _make_rng_state(graph_case[1].device, (11, 22, 33, 44)),
        kernel,
        use_fast_math,
    )
    qa2, _, qd2, _ = _run_sr(
        graph_case,
        _make_rng_state(graph_case[1].device, (11, 22, 33, 44)),
        kernel,
        use_fast_math,
    )
    assert torch.equal(qa1, qa2), "Same rng_state must yield identical row FP4 codes"
    assert torch.equal(qd1, qd2), "Same rng_state must yield identical col FP4 codes"

    qa3, _, qd3, _ = _run_sr(
        graph_case,
        _make_rng_state(graph_case[1].device, (11, 9999, 33, 8888)),
        kernel,
        use_fast_math,
    )
    assert not torch.equal(qa1, qa3), "Advanced rng_state must change row FP4 codes"
    assert not torch.equal(qd1, qd3), "Advanced rng_state must change col FP4 codes"


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_rht_rng_state_validation(graph_case, kernel):
    """SR enabled requires a valid int64 rng_state; SR disabled ignores it."""
    spec, A, B, offsets, amax_row, amax_col, _, _ = graph_case
    psl, hs = A.shape
    num_groups = len(spec.groups)
    _skip_if_unsupported_groups(kernel, num_groups)

    with pytest.raises(TypeError, match="rng_state must be a torch.Tensor"):
        _run_sr(graph_case, None, kernel)

    with pytest.raises(ValueError, match="at least 4 elements"):
        _run_sr(graph_case, _make_rng_state(A.device, (1, 2)), kernel)

    # SR disabled: rng_state is ignored, so None is accepted.
    _group_quantize(
        kernel,
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
@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize(
    ("invalid_amax", "error"),
    [
        ("2d", "a_global_amax must be 1D"),
        ("noncontiguous", "a_global_amax must be contiguous"),
    ],
)
def test_group_rht_amax_storage_validation(graph_case, invalid_amax, error, kernel):
    spec, A, _, offsets, _, amax_col, _, _ = graph_case
    _skip_if_unsupported_groups(kernel, len(spec.groups))
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
        _group_quantize(
            kernel,
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
