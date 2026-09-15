# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the fused RHT + NVFP4 columnwise/rowwise quantize kernels (SM100+).

The two backends implement the same op and are selected by the ``kernel``
parametrization (see ``_KERNELS``):
  - ``triton``  -> ``triton_rht_quantize_row_col``
  - ``cutedsl`` -> ``cutedsl_rht_quantize_row_col``

  RTNE (stochastic_rounding=False), both backends:
    - test_rht_quantize_rtne_scales_vs_transformer_engine_reference: FP8 scale
      factors match a plain-PyTorch reference derived directly from
      TransformerEngine bitwise in swizzled layout.
    - test_rht_quantize_rtne_scales_vs_mx_formats_reference: FP8 scale factors
      match the independent mx_formats reference within 1 fp8 ULP. Not bitwise:
      mx_formats uses a reciprocal and an E4M3_EPS floor.
    - test_rht_quantize_rtne_sqnr: Dequantized output reconstructs post-RHT / raw-A
      values with SQNR >= 20 dB for both col and row paths.
    - test_rht_quantize_row_col_zero_and_near_zero_no_nan_or_saturation: zero and
      near-zero inputs stay finite and dequantize to bounded values, and a zero
      block stores a zero scale on both backends (TE applies no lower clamp).
      Raw FP4 magnitudes are not checked, since a zero decode scale makes them
      meaningless.

  RS (stochastic_rounding=True):
    - test_rht_quantize_row_col_zero_and_near_zero_no_nan_or_saturation (both backends).
    - test_triton_rht_quantize_rs_midpoint_distribution (triton): Values at the FP4
      [1.0, 1.5] midpoint (1.25) round to each neighbor ~50% of the time for both
      columnwise and rowwise paths. Columnwise input is constructed via inverse RHT so
      post-RHT values are exactly 1.25; rowwise input has 1.25 placed directly in A.
    - test_rht_quantize_rs_at_most_one_fp4_step_from_rtne (both backends): RS code is at
      most 1 FP4 magnitude index step from the RTNE code for every element, for both
      columnwise and rowwise paths, over one tile, several tiles and a short trailing
      column group. This is the structural guard on SR: it pins every SR code against an
      RTNE code that IS bitwise-checked against triton and TE.
    - test_cutedsl_rht_quantize_sr_unbiased (cutedsl): the HW ``cvt.rs`` path is unbiased
      at the same 1.25 midpoint.

  Fast math (use_fast_math=True, the nvfp4_linear default), both backends:
    - test_fast_math_matches_transformer_engine: byte-identical to real TE under
      NVTE_USE_FAST_MATH=1, over the full shape sweep. This is why triton can act as the
      fast-path oracle where the PyTorch reference cannot -- TE's fast encode scale is
      ``rcp.approx.ftz.f32`` and no ATen op lowers to that instruction, so
      ``nvfp4_reference`` models the exact path only.
    - test_rht_quantize_fast_math_sqnr: fast reconstructs the exact path (which IS
      bitwise-checked against nvfp4_reference) at 30+ dB, floored at 25 -- above NVFP4's
      own 20 dB quantization noise. Nearly all of the cost is the columnwise bfloat16
      round-through that fast math skips; the reciprocal swap alone leaves the rowwise
      output bitwise identical on most shapes.
    - the SR and interchangeability tests below run under both math modes.

  The two backends are byte-for-byte interchangeable under RTNE, in both math modes
  (test_cutedsl_vs_triton_interchangeable). Neither reproduces the other's Philox
  stream: both CuteDSL kernels draw one counter per 16-element block and consume all
  four output words, rather than triton's per-packed-byte stride. SR is therefore judged
  on the structural and statistical properties above.

Coverage:
  RS=F, RW=F  — TE bitwise scales + mx_formats scale compatibility + rtne_sqnr
  RS=F, RW=T  — TE bitwise scales + mx_formats scale compatibility + rtne_sqnr
  RS=T, RW=F  — rs_midpoint_distribution (col) + rs_at_most_one_fp4_step_from_rtne (col+row)
  RS=T, RW=T  — rs_midpoint_distribution (row) + rs_at_most_one_fp4_step_from_rtne (col+row)
                + sr_unbiased (row)
"""

import pytest
import torch
from torch.utils._triton import has_triton

from test.prototype.moe_training.nvfp4_training._assertions import (
    assert_codes_bitwise,
    assert_scales_adjacent,
    assert_scales_bitwise,
    assert_scales_finite,
    assert_zero_quantized,
    unpack_fp4_magnitudes,
    unpack_fp4_nibbles,
)
from test.prototype.moe_training.nvfp4_training._assertions import (
    dequantize as _dequantize,
)
from test.prototype.moe_training.nvfp4_training.nvfp4_reference import (
    reference_rht,
    reference_rht_quantize_row_col,
)
from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_cutedsl import (
    cutedsl_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_cutedsl import (
    cutedsl_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    DEFAULT_SIGN_VECTOR,
    get_rht_matrix,
    prepare_for_cuda_graph,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    nvfp4_quantize,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.utils import is_sm_at_least_100, torch_version_at_least

if has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0"):
    from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
        triton_rht_amax,
    )
    from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_triton import (
        triton_rht_quantize_row_col,
    )

# Shapes swept by the kernel-parametrized tests. Both kernels need M % 128 == 0 and
# N % 128 == 0 for the swizzled scale layout. M=128/384 exercise the CuteDSL 128-row
# supertile (M % 128 but not % 256); the other M run its tuned 256-row supertile.
_M_VALUES = [128, 256, 384, 512, 1024]
_N_VALUES = [128, 256, 384, 512, 1024]
# Shape both kernels accept, for the tests that do not sweep shapes.
_M_BOTH, _N_BOTH = 256, 256
_NEAR_ZERO = 1.0e-10
_HARDCODED_SIGN_VECTOR = DEFAULT_SIGN_VECTOR

_skip_no_triton = pytest.mark.skipif(
    not (has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0")),
    reason="requires triton + SM100+ + PyTorch 2.10 (custom op / torch.compile support)",
)
_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)

# Kernel backends under test: every test that is the same computation modulo the kernel
# selector is parametrized over this list.
_KERNELS = [
    pytest.param("triton", marks=_skip_no_triton, id="triton"),
    pytest.param("cutedsl", marks=_skip_no_cutedsl, id="cutedsl"),
]


# ---------------------------------------------------------------------------
# Reference implementations (plain PyTorch, backend independent)
# ---------------------------------------------------------------------------


def _input_from_rht_target(target: torch.Tensor) -> torch.Tensor:
    """Build A (M, N) bfloat16 whose post-RHT value RHT(A.t()) is ``target`` (N, M).

    The RHT matrix is orthogonal (B @ B.T = I in bfloat16), so A.t() = target @ B.T.
    """
    N, M = target.shape
    B = get_rht_matrix(
        _HARDCODED_SIGN_VECTOR, target.device, torch.bfloat16, 16
    ).float()
    A_t = (target.reshape(N * M // 16, 16) @ B.t()).reshape(N, M)
    return A_t.t().contiguous().to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Kernel dispatch helpers
# ---------------------------------------------------------------------------


def _random_i64(device: torch.device) -> torch.Tensor:
    return torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device)


def _skip_if_unsupported_shape(kernel: str, M: int, N: int) -> None:
    """Skip shapes the selected backend cannot handle."""
    if M % 128 != 0 or N % 128 != 0:
        pytest.skip("swizzled scales require M % 128 == 0 and N % 128 == 0")


def _rht_amax(
    kernel: str, x: torch.Tensor, *, sign_vector
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the selected backend's fused RHT col/row global amax.

    Returns (col_amax, row_amax) = (max|RHT(x.t())|, max|x|) as scalar float32 tensors.
    """
    if kernel == "triton":
        return triton_rht_amax(x, sign_vector=list(sign_vector))
    return cutedsl_rht_amax(x, list(sign_vector))


def _quantize_row_col(
    kernel: str,
    x: torch.Tensor,
    *,
    col_amax: torch.Tensor,
    row_amax: torch.Tensor,
    sign_vector,
    stochastic_rounding: bool = False,
    seed: torch.Tensor | None = None,
    offset: torch.Tensor | None = None,
    use_fast_math: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch to the selected backend's fused RHT row/col quantize.

    Returns (col_codes, col_sf, row_codes, row_sf). The two ops take the same four
    Philox bases and the same ``use_fast_math`` flag, so the only difference here is
    which one is called. With ``stochastic_rounding=True``, ``seed``/``offset`` default
    to fresh random int64 buffers.
    """
    quantize = (
        triton_rht_quantize_row_col
        if kernel == "triton"
        else cutedsl_rht_quantize_row_col
    )
    sr_kwargs = {}
    if stochastic_rounding:
        col_seed = _random_i64(x.device) if seed is None else seed
        col_offset = _random_i64(x.device) if offset is None else offset
        sr_kwargs = {
            "col_seed_base": col_seed,
            "col_offset_base": col_offset,
            # ^1 to decorrelate the row stream from the col one, as nvfp4_linear does.
            "row_seed_base": col_seed ^ 1,
            "row_offset_base": col_offset,
        }
    return quantize(
        x,
        col_amax,
        row_amax,
        list(sign_vector),
        stochastic_rounding=stochastic_rounding,
        use_fast_math=use_fast_math,
        **sr_kwargs,
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests — both backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize(
    "M,N", [(256, 256), (128, 512), (384, 256)], ids=["256x256", "128x512", "384x256"]
)
@pytest.mark.parametrize(
    "input_kind",
    [
        pytest.param("bounded_integer", id="bounded_integer"),
        pytest.param("zeros", id="zeros"),
        pytest.param("anchored_near_zero", id="anchored_near_zero"),
    ],
)
@torch.no_grad()
def test_rht_quantize_rtne_vs_transformer_engine_reference(kernel, M, N, input_kind):
    """Both backends must reproduce TransformerEngine's arithmetic.

    Scales and codes are bitwise. ``bounded_integer`` is adversarial because integer data
    under a power-of-two amax places scaled values exactly on E2M1 midpoints.
    """
    _skip_if_unsupported_shape(kernel, M, N)
    if input_kind == "bounded_integer":
        torch.manual_seed(42)
        A = torch.randint(-8, 9, (M, N), dtype=torch.int16, device="cuda").to(
            torch.bfloat16
        )
    elif input_kind == "zeros":
        A = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    else:
        A = torch.full((M, N), _NEAR_ZERO, dtype=torch.bfloat16, device="cuda")
        A[0, 0] = 1.0

    col_amax = reference_rht(A).float().abs().max()
    row_amax = A.float().abs().max()
    ref_col, ref_row = reference_rht_quantize_row_col(A, col_amax, row_amax)

    col_codes, col_sf, row_codes, row_sf = _quantize_row_col(
        kernel,
        A,
        col_amax=col_amax,
        row_amax=row_amax,
        sign_vector=_HARDCODED_SIGN_VECTOR,
    )
    assert_scales_bitwise(col_sf, ref_col.scales, "col sf")
    assert_scales_bitwise(row_sf, ref_row.scales, "row sf")
    assert_codes_bitwise(col_codes, ref_col.codes, "col codes")
    assert_codes_bitwise(row_codes, ref_row.codes, "row codes")

    if input_kind == "zeros":
        assert torch.count_nonzero(col_sf.float()).item() == 0
        assert torch.count_nonzero(row_sf.float()).item() == 0
    elif input_kind == "anchored_near_zero":
        for label, scales in (("col sf", col_sf), ("row sf", row_sf)):
            scales_f32 = scales.float()
            assert torch.count_nonzero(scales_f32).item() > 0, (
                f"{label}: anchor must produce a nonzero scale"
            )
            assert torch.count_nonzero(scales_f32 == 0).item() > 0, (
                f"{label}: near-zero blocks must underflow to E4M3 zero"
            )


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_fast_math_matches_transformer_engine(kernel, M, N, monkeypatch):
    """Both fast paths are byte-identical to actual TE fast math.

    This is what lets Triton stand in as the fast-path oracle. The PyTorch reference
    cannot: TE's fast encode scale is ``rcp.approx.ftz.f32`` and no ATen op lowers to
    that instruction, so ``nvfp4_reference`` models the exact path only.
    """
    _skip_if_unsupported_shape(kernel, M, N)
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "1")
    te = pytest.importorskip("transformer_engine.pytorch")

    torch.manual_seed(123)
    A = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
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
    expected = quantizer(A)

    col_amax, row_amax = _rht_amax(kernel, A, sign_vector=_HARDCODED_SIGN_VECTOR)
    col_codes, col_sf, row_codes, row_sf = _quantize_row_col(
        kernel,
        A,
        col_amax=col_amax,
        row_amax=row_amax,
        sign_vector=_HARDCODED_SIGN_VECTOR,
        use_fast_math=True,
    )

    assert_codes_bitwise(
        row_codes, expected._rowwise_data.view(torch.uint8), "row codes"
    )
    assert_scales_bitwise(row_sf, expected._rowwise_scale_inv, "row sf")
    assert_codes_bitwise(
        col_codes, expected._columnwise_data.view(torch.uint8), "col codes"
    )
    assert_scales_bitwise(col_sf, expected._columnwise_scale_inv, "col sf")


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_rht_quantize_rtne_scales_vs_mx_formats_reference(kernel, M, N):
    """Compare FP8 scale factors with the mx_formats implementation within 1 fp8 ULP.

    Columnwise: RHT + quantize of A.T. Rowwise: quantize raw A.

    This is a compatibility comparison, not the bitwise TransformerEngine
    conformance test. The kernels follow TE's correctly-rounded div_rn and do not
    lower-clamp per-vector scales. The mx_formats implementation instead multiplies
    by a reciprocal and applies an E4M3_EPS floor as part of its NVFP4 quantization.
    The epsilon floor belongs specifically to mx_formats and is not part of the
    TE-derived reference above. For the ordinary, non-underflowing scales exercised
    here, the two formulations are mathematically equivalent but can land on adjacent
    E4M3 values because of FP32 rounding order.

    Packed FP4 codes are not checked here because the TE-derived test above already
    asserts them bitwise; this test isolates compatibility with mx_formats' scale recipe.
    """
    _skip_if_unsupported_shape(kernel, M, N)

    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    col_amax, row_amax = _rht_amax(kernel, A, sign_vector=_HARDCODED_SIGN_VECTOR)
    _, col_sf, _, row_sf = _quantize_row_col(
        kernel,
        A,
        col_amax=col_amax,
        row_amax=row_amax,
        sign_vector=_HARDCODED_SIGN_VECTOR,
    )

    # Rowwise compatibility check (plain mx_formats NVFP4 quantize of A).
    ref_row_sf, _ = nvfp4_quantize(
        A, per_tensor_scale=per_tensor_amax_to_scale(A.float().abs().max())
    )
    assert_scales_adjacent(row_sf.flatten(), to_blocked(ref_row_sf), "row sf")

    if kernel == "triton":
        # bf16 in: nvfp4_quantize widens to f32 losslessly, so the block amax matches
        # the kernel's bf16 reduction exactly.
        rht = reference_rht(A)
        ref_col_sf, _ = nvfp4_quantize(
            rht, per_tensor_scale=per_tensor_amax_to_scale(rht.float().abs().max())
        )
        assert_scales_adjacent(col_sf.flatten(), to_blocked(ref_col_sf), "col sf")


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_rht_quantize_rtne_sqnr(kernel, M, N):
    """Dequantized output must reconstruct post-RHT / raw-A values with SQNR ≥ 20 dB.

    Scale factors are always swizzled; layout does not affect quantization error.
    """
    _skip_if_unsupported_shape(kernel, M, N)

    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    col_amax, row_amax = _rht_amax(kernel, A, sign_vector=_HARDCODED_SIGN_VECTOR)
    col_codes, col_sf, row_codes, row_sf = _quantize_row_col(
        kernel,
        A,
        col_amax=col_amax,
        row_amax=row_amax,
        sign_vector=_HARDCODED_SIGN_VECTOR,
    )

    # Columnwise SQNR: dequantized should reconstruct RHT(A.T)
    ref_rht = reference_rht(A).float()
    col_sqnr = compute_error(ref_rht, _dequantize(col_codes, col_sf, col_amax))
    assert col_sqnr >= 20.0, f"Col SQNR {col_sqnr:.2f} dB < 20.0 dB for M={M} N={N}"

    # Rowwise SQNR: dequantized should reconstruct raw A
    row_sqnr = compute_error(A.float(), _dequantize(row_codes, row_sf, row_amax))
    assert row_sqnr >= 20.0, f"Row SQNR {row_sqnr:.2f} dB < 20.0 dB for M={M} N={N}"


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_rht_quantize_fast_math_sqnr(kernel, M, N):
    """Fast math costs little against the exact path it replaces.

    This is the bound that grounds fast math in the PyTorch oracle, which cannot host it
    directly: the exact path is bitwise-checked against ``nvfp4_reference`` (and against
    TE), and fast math is held to a tight SQNR of that exact path.

    Columnwise carries essentially all of the cost and is stable: 30.2-32.6 dB across
    this sweep, ~1.2% of packed code bytes differing. Skipping the bfloat16 round-through
    perturbs the pre-quantization value by only ~2**-9, but an element sitting near an
    E2M1 midpoint then flips a whole FP4 step, so the error is large-and-rare rather than
    small-and-everywhere. Compare NVFP4's own quantization noise, floored at 20 dB by
    test_rht_quantize_rtne_sqnr: fast math moves the result less than quantizing it did.

    Rowwise has no accumulator to skip, so only the reciprocal changes -- but it shares
    the same loose floor rather than a tighter one, deliberately. Block scales are
    identical between the modes, so a rowwise flip requires ``rcp.approx`` and ``div_rn``
    to disagree on ``pvscale_fp8 * global_decode_scale``, and that product takes only a
    handful of distinct values per tensor. Either none of them disagrees and the output is
    bitwise identical (SQNR inf, 24 of 25 shapes here), or one does and every element
    sharing that scale becomes eligible at once. The result is lumpy and data-dependent
    -- inf, inf, 49 dB, 41 dB across the grouped fixtures -- so a tight rowwise bound
    would be flaky by construction rather than more sensitive.
    """
    _skip_if_unsupported_shape(kernel, M, N)

    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    # The amax kernels have no fast-math variant (neither do TE's), so both modes are
    # fed the identical global amaxes and only the quantize arithmetic differs.
    col_amax, row_amax = _rht_amax(kernel, A, sign_vector=_HARDCODED_SIGN_VECTOR)
    kwargs = dict(
        col_amax=col_amax, row_amax=row_amax, sign_vector=_HARDCODED_SIGN_VECTOR
    )
    e_col_codes, e_col_sf, e_row_codes, e_row_sf = _quantize_row_col(
        kernel, A, use_fast_math=False, **kwargs
    )
    f_col_codes, f_col_sf, f_row_codes, f_row_sf = _quantize_row_col(
        kernel, A, use_fast_math=True, **kwargs
    )

    col_sqnr = compute_error(
        _dequantize(e_col_codes, e_col_sf, col_amax),
        _dequantize(f_col_codes, f_col_sf, col_amax),
    )
    assert col_sqnr >= 25.0, f"Col fast-vs-exact SQNR {col_sqnr:.2f} dB < 25.0 dB"

    row_sqnr = compute_error(
        _dequantize(e_row_codes, e_row_sf, row_amax),
        _dequantize(f_row_codes, f_row_sf, row_amax),
    )
    assert row_sqnr >= 25.0, f"Row fast-vs-exact SQNR {row_sqnr:.2f} dB < 25.0 dB"


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize(
    "stochastic_rounding",
    [pytest.param(False, id="rtne"), pytest.param(True, id="stochastic_rounding")],
)
@pytest.mark.parametrize(
    "input_kind",
    [pytest.param("zeros", id="zeros"), pytest.param("near_zero", id="near_zero")],
)
@torch.no_grad()
def test_rht_quantize_row_col_zero_and_near_zero_no_nan_or_saturation(
    kernel, input_kind, stochastic_rounding
):
    M, N = _M_BOTH, _N_BOTH

    if input_kind == "zeros":
        A = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        col_amax, row_amax = _rht_amax(kernel, A, sign_vector=_HARDCODED_SIGN_VECTOR)
        assert col_amax.item() == 0.0 and row_amax.item() == 0.0
        col_codes, col_sf, row_codes, row_sf = _quantize_row_col(
            kernel,
            A,
            col_amax=col_amax,
            row_amax=row_amax,
            sign_vector=_HARDCODED_SIGN_VECTOR,
            stochastic_rounding=stochastic_rounding,
        )

        # All-zero input packs to zero codes, stores a zero block scale, and
        # dequantizes to exactly zero on both backends (TE semantics).
        assert_zero_quantized(
            col_codes, col_sf, _dequantize(col_codes, col_sf, col_amax)
        )
        assert_zero_quantized(
            row_codes, row_sf, _dequantize(row_codes, row_sf, row_amax)
        )
        return

    A_row = torch.full((M, N), _NEAR_ZERO, dtype=torch.bfloat16, device="cuda")
    A_row[0, 0] = 1.0
    row_col_amax, row_row_amax = _rht_amax(
        kernel, A_row, sign_vector=_HARDCODED_SIGN_VECTOR
    )
    _, _, row_codes, row_sf = _quantize_row_col(
        kernel,
        A_row,
        col_amax=row_col_amax,
        row_amax=row_row_amax,
        sign_vector=_HARDCODED_SIGN_VECTOR,
        stochastic_rounding=stochastic_rounding,
    )
    assert_scales_finite(row_sf)
    row_dequant = _dequantize(row_codes, row_sf, row_row_amax)
    assert torch.isfinite(row_dequant).all(), (
        "rowwise dequantized values must be finite"
    )
    assert row_dequant.abs().max() <= 1.0

    col_target = torch.full((N, M), _NEAR_ZERO, dtype=torch.float32, device="cuda")
    col_target[0, 0] = 1.0
    A_col = _input_from_rht_target(col_target)
    col_col_amax, col_row_amax = _rht_amax(
        kernel, A_col, sign_vector=_HARDCODED_SIGN_VECTOR
    )
    col_codes, col_sf, _, _ = _quantize_row_col(
        kernel,
        A_col,
        col_amax=col_col_amax,
        row_amax=col_row_amax,
        sign_vector=_HARDCODED_SIGN_VECTOR,
        stochastic_rounding=stochastic_rounding,
    )
    assert_scales_finite(col_sf)
    col_dequant = _dequantize(col_codes, col_sf, col_col_amax)
    assert torch.isfinite(col_dequant).all(), (
        "colwise dequantized values must be finite"
    )
    assert col_dequant.abs().max() <= 1.0


# ---------------------------------------------------------------------------
# Tests — cross-backend (requires both kernels)
# ---------------------------------------------------------------------------


@_skip_no_triton
@_skip_no_cutedsl
@pytest.mark.parametrize("N", [256, 512], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", [256, 384, 512], ids=lambda m: f"M{m}")
@pytest.mark.parametrize("use_fast_math", [False, True], ids=["exact", "fast"])
@torch.no_grad()
def test_cutedsl_vs_triton_interchangeable(M, N, use_fast_math):
    """Fed the same global amaxes, the two backends are byte-for-byte interchangeable.

    Both the amaxes themselves and all four quantize outputs. The backends share the
    RHT rounding (bf16 in exact mode, none in fast), the scale chain (the one-rounding
    association, and div_rn or rcp.approx.ftz per mode) and the FP4 encode PTX, so
    nothing is left to differ. Covering both modes is what keeps AUTO and TRITON
    interchangeable now that fast math is the default.
    """
    torch.manual_seed(0)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    t_amax = _rht_amax("triton", A, sign_vector=_HARDCODED_SIGN_VECTOR)
    c_amax = _rht_amax("cutedsl", A, sign_vector=_HARDCODED_SIGN_VECTOR)
    for name, t, c in zip(("col_amax", "row_amax"), t_amax, c_amax):
        assert torch.equal(t, c), f"{name} differs between backends"

    col_amax, row_amax = t_amax
    kwargs = dict(
        col_amax=col_amax,
        row_amax=row_amax,
        sign_vector=_HARDCODED_SIGN_VECTOR,
        use_fast_math=use_fast_math,
    )
    cutedsl = _quantize_row_col("cutedsl", A, **kwargs)
    triton_out = _quantize_row_col("triton", A, **kwargs)
    for name, c, t in zip(("qd", "sfd", "qa", "sfa"), cutedsl, triton_out):
        assert torch.equal(c, t), f"{name} differs between backends"


# ---------------------------------------------------------------------------
# Tests — triton only (RS statistics, CUDA graph capture)
# ---------------------------------------------------------------------------


@_skip_no_triton
@torch.no_grad()
def test_triton_rht_quantize_rs_midpoint_distribution():
    """RS of a value exactly at the FP4 midpoint (1.25) must round each direction ~50% of the time.

    Columnwise path: Constructs input A via inverse RHT so that post-RHT values are exactly:
      - 6.0 at the first element of each 16-group (anchors vec_max = global_amax = 6.0,
        so encode_scale = 1.0 exactly).
      - 1.25 everywhere else (exactly at the midpoint of the FP4 [1.0, 1.5] interval).
    The RHT matrix is orthogonal (B @ B.T = I in bfloat16), so the round-trip is exact.

    Rowwise path: A_row has 6.0 at the first element of each 16-group along N (anchors)
      and 1.25 everywhere else. Since the rowwise path quantizes A directly (no RHT),
      vec_max = 6.0 gives encode_scale = 1.0 exactly, so scaled values are 1.25 at midpoints.

    RTNE rounds 1.25 to code 2 (1.0) — the even neighbor — by round-to-nearest-even.
    RS must round to code 2 (1.0) or code 3 (1.5) with equal probability (~50% each).
    """
    N_RHT, M_RHT = 128, 128  # post-RHT shape (N_RHT = N_A, M_RHT = M_A)
    N_SAMPLES = 32

    # Build A such that RHT(A.T) has 1.25 at non-anchor positions and 6.0 at anchors.
    target = torch.full((N_RHT, M_RHT), 1.25, dtype=torch.float32, device="cuda")
    target[:, ::16] = 6.0  # one anchor per 16-group along M
    A = _input_from_rht_target(target)  # kernel expects (M_A, N_A) contiguous

    # Build A_row: 1.25 everywhere, 6.0 at first element of each 16-group along N.
    # Row path quantizes A directly; vec_max=6.0 → encode_scale=1.0 → scaled=1.25 exactly.
    A_row = torch.full((128, 128), 1.25, dtype=torch.bfloat16, device="cuda")
    A_row[:, ::16] = 6.0

    col_count_lo = 0  # code 2 = 1.0
    col_count_hi = 0  # code 3 = 1.5
    row_count_lo = 0
    row_count_hi = 0

    for _ in range(N_SAMPLES):
        col_amax, row_amax = _rht_amax("triton", A, sign_vector=_HARDCODED_SIGN_VECTOR)
        col_codes, _, _, _ = _quantize_row_col(
            "triton",
            A,
            col_amax=col_amax,
            row_amax=row_amax,
            sign_vector=_HARDCODED_SIGN_VECTOR,
            stochastic_rounding=True,
        )
        # Unpack col_codes (N_RHT, M_RHT//2) uint8 → (N_RHT, M_RHT) magnitudes
        mag_codes = unpack_fp4_magnitudes(col_codes)

        # Exclude anchor positions (m % 16 == 0 → scaled=6.0 → code 7)
        col_idx = torch.arange(M_RHT, device="cuda")
        target_mags = mag_codes[:, (col_idx % 16) != 0]  # (N_RHT, 15 * M_RHT//16)

        col_count_lo += (target_mags == 2).sum().item()  # rounded to 1.0
        col_count_hi += (target_mags == 3).sum().item()  # rounded to 1.5

        # Rowwise path check using A_row
        row_col_amax, row_row_amax = _rht_amax(
            "triton", A_row, sign_vector=_HARDCODED_SIGN_VECTOR
        )
        _, _, row_codes, _ = _quantize_row_col(
            "triton",
            A_row,
            col_amax=row_col_amax,
            row_amax=row_row_amax,
            sign_vector=_HARDCODED_SIGN_VECTOR,
            stochastic_rounding=True,
        )
        # Unpack row_codes (128, 64) uint8 → (128, 128) magnitudes
        r_mag = unpack_fp4_magnitudes(row_codes)
        # Exclude anchor positions (n % 16 == 0 → code 7)
        n_idx = torch.arange(128, device="cuda")
        r_target = r_mag[:, (n_idx % 16) != 0]
        row_count_lo += (r_target == 2).sum().item()
        row_count_hi += (r_target == 3).sum().item()

    col_total = col_count_lo + col_count_hi
    col_frac_hi = col_count_hi / col_total
    assert 0.40 <= col_frac_hi <= 0.60, (
        f"Col RS at midpoint 1.25: expected ~50% round to code 3 (1.5), "
        f"got {col_frac_hi:.4f} over {col_total} samples"
    )

    row_total = row_count_lo + row_count_hi
    row_frac_hi = row_count_hi / row_total
    assert 0.40 <= row_frac_hi <= 0.60, (
        f"Row RS at midpoint 1.25: expected ~50% round to code 3 (1.5), "
        f"got {row_frac_hi:.4f} over {row_total} samples"
    )


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize(
    "M,N",
    [(128, 128), (256, 512), (1024, 2432)],
    ids=["single-tile", "multi-tile", "short-group"],
)
@pytest.mark.parametrize("use_fast_math", [False, True], ids=["exact", "fast"])
@torch.no_grad()
def test_rht_quantize_rs_at_most_one_fp4_step_from_rtne(kernel, M, N, use_fast_math):
    """RS code must be at most 1 FP4 magnitude index step from the RTNE code.

    RS picks the floor or ceil of the scaled value on the FP4 magnitude grid.
    RTNE also picks floor or ceil (nearest). Therefore |rs_mag_idx - rtne_mag_idx| <= 1
    must hold for every element, and signs must agree.

    Both columnwise and rowwise paths are tested, on both backends. This is the
    structural guard on the SR path now that neither backend reproduces the other's
    Philox stream: it pins every SR code against the RTNE code that IS bitwise-checked
    against triton and TE, so a wrong nibble order, scale or block index fails here even
    though no bitwise SR oracle exists. The shapes span one tile, several tiles, and a
    short trailing column group, which is where a tile-derived counter is most likely to
    collide or run off the end.

    Each math mode is compared against its own RTNE codes: fast math shifts both, so
    crossing the modes would measure the arithmetic change rather than the rounding.
    """
    _skip_if_unsupported_shape(kernel, M, N)
    N_SAMPLES = 16
    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    col_amax_rtne, row_amax_rtne = _rht_amax(
        kernel, A, sign_vector=_HARDCODED_SIGN_VECTOR
    )
    col_rn, _, row_rn, _ = _quantize_row_col(
        kernel,
        A,
        col_amax=col_amax_rtne,
        row_amax=row_amax_rtne,
        sign_vector=_HARDCODED_SIGN_VECTOR,
        use_fast_math=use_fast_math,
    )
    col_rn_nibs = unpack_fp4_nibbles(col_rn)
    col_rn_sign = col_rn_nibs >> 3
    col_rn_mag = col_rn_nibs & 0x7

    row_rn_nibs = unpack_fp4_nibbles(row_rn)
    row_rn_sign = row_rn_nibs >> 3
    row_rn_mag = row_rn_nibs & 0x7

    for _ in range(N_SAMPLES):
        col_amax_rs, row_amax_rs = _rht_amax(
            kernel, A, sign_vector=_HARDCODED_SIGN_VECTOR
        )
        col_rs, _, row_rs, _ = _quantize_row_col(
            kernel,
            A,
            col_amax=col_amax_rs,
            row_amax=row_amax_rs,
            sign_vector=_HARDCODED_SIGN_VECTOR,
            stochastic_rounding=True,
            use_fast_math=use_fast_math,
        )

        col_rs_nibs = unpack_fp4_nibbles(col_rs)
        col_rs_sign = col_rs_nibs >> 3
        col_rs_mag = col_rs_nibs & 0x7

        # Columnwise: sign must match RTNE and magnitude must be at most 1 step away
        col_nonzero = (col_rs_mag != 0) | (col_rn_mag != 0)
        assert ((col_rs_sign == col_rn_sign) | ~col_nonzero).all(), (
            "Col RS changed sign relative to RTNE"
        )
        col_mag_diff = (col_rs_mag - col_rn_mag).abs()
        assert (col_mag_diff <= 1).all(), (
            f"Col RS magnitude index differs by {col_mag_diff.max().item()} from RTNE (must be ≤1)"
        )

        row_rs_nibs = unpack_fp4_nibbles(row_rs)
        row_rs_sign = row_rs_nibs >> 3
        row_rs_mag = row_rs_nibs & 0x7

        # Rowwise: same invariants
        row_nonzero = (row_rs_mag != 0) | (row_rn_mag != 0)
        assert ((row_rs_sign == row_rn_sign) | ~row_nonzero).all(), (
            "Row RS changed sign relative to RTNE"
        )
        row_mag_diff = (row_rs_mag - row_rn_mag).abs()
        assert (row_mag_diff <= 1).all(), (
            f"Row RS magnitude index differs by {row_mag_diff.max().item()} from RTNE (must be ≤1)"
        )


@_skip_no_triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_rht_quantize_row_col_cuda_graph_compile():
    """triton_rht_quantize_row_col (SR) under reduce-overhead CUDA graphs.

    Verifies:
      1. torch.compile(fullgraph=True, mode='reduce-overhead') succeeds without
         pool-allocation errors or FakeTensor errors.
      2. SR output varies between replays when seed_buf is mutated between calls
         (mutates_args=("seed_buf",) gives the tensor a stable static-buffer address
         so the kernel's captured seed pointer sees updated values on replay).
      3. SR is being applied: compiled output differs from RTNE reference.
    """
    # NOTE: this test fails when run in the same process after the RHT amax tests.
    # Root cause: triton_rht_amax (a custom_op) calls triton.set_allocator() eagerly in
    # the amax tests, leaving a process-global Triton allocator active during CUDA graph
    # capture here. torch._dynamo.reset() does not clear Triton's global allocator state.
    # TODO: fix by one of:
    #   (a) move compile tests to a dedicated file run in an isolated subprocess / --forked
    #   (b) add a conftest.py autouse fixture that calls triton.set_allocator(None) between files
    #   (c) guard triton.set_allocator() in triton_rht_amax so it only fires inside compiled regions
    shape = (128, 256)
    A = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    prepare_for_cuda_graph(
        A.device, sign_vectors=(_HARDCODED_SIGN_VECTOR,)
    )  # pre-allocate TMA scratch + RHT matrix outside pool context

    # Pre-allocate seed bufs OUTSIDE torch.compile so their addresses are stable.
    col_seed_buf = _random_i64(A.device)
    row_seed_buf = _random_i64(A.device)

    def run(data):
        # Distinct col/row offsets keep the two RS streams uncorrelated, so this calls
        # the op directly rather than through the _quantize_row_col dispatch helper.
        col_offset = _random_i64(A.device)
        row_offset = _random_i64(A.device)
        col_amax, row_amax = triton_rht_amax(
            data, sign_vector=list(_HARDCODED_SIGN_VECTOR)
        )
        col_fp4, _, row_fp4, _ = triton_rht_quantize_row_col(
            data,
            stochastic_rounding=True,
            sign_vector=list(_HARDCODED_SIGN_VECTOR),
            col_seed_base=col_seed_buf,
            col_offset_base=col_offset,
            row_offset_base=row_offset,
            row_seed_base=row_seed_buf,
            col_global_amax=col_amax,
            row_global_amax=row_amax,
        )
        return col_fp4, row_fp4

    compiled = torch.compile(run, mode="reduce-overhead", fullgraph=True)
    for _ in range(3):
        compiled(A)  # warmup

    # SR output should differ when seed bufs are updated between replays
    col_r1, row_r1 = [x.clone() for x in compiled(A)]
    col_r2, row_r2 = [x.clone() for x in compiled(A)]
    assert not torch.equal(col_r1, col_r2), (
        "Col SR outputs should differ with different seeds"
    )
    assert not torch.equal(row_r1, row_r2), (
        "Row SR outputs should differ with different seeds"
    )

    # SR IS applied: output should differ from round-to-nearest reference
    rtne_col_amax, rtne_row_amax = _rht_amax(
        "triton", A, sign_vector=_HARDCODED_SIGN_VECTOR
    )
    rtne_col_ref, _, rtne_row_ref, _ = _quantize_row_col(
        "triton",
        A,
        col_amax=rtne_col_amax,
        row_amax=rtne_row_amax,
        sign_vector=_HARDCODED_SIGN_VECTOR,
    )
    assert not torch.equal(col_r1, rtne_col_ref), (
        "Col SR should produce different output than RTNE"
    )
    assert not torch.equal(row_r1, rtne_row_ref), (
        "Row SR should produce different output than RTNE"
    )


# ---------------------------------------------------------------------------
# Tests — cutedsl only (internal impl flags, arg validation, op registration)
# ---------------------------------------------------------------------------


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
    col_amax, row_amax = _rht_amax("cutedsl", A, sign_vector=_HARDCODED_SIGN_VECTOR)

    # Swizzled reference (the op's default path).
    col_sw, col_sf_sw, row_sw, row_sf_sw = _quantize_row_col(
        "cutedsl",
        A,
        col_amax=col_amax,
        row_amax=row_amax,
        sign_vector=_HARDCODED_SIGN_VECTOR,
    )
    # Plain (unblocked) SF path.
    col_p, col_sf_p, row_p, row_sf_p = _cutedsl_rht_quantize_row_col_impl(
        A, col_amax, row_amax, tuple(sign_vector), swizzle_scale_factors=False
    )

    assert col_sf_p.shape == (N, M // 16) and row_sf_p.shape == (M, N // 16)
    # FP4 codes are layout-independent.
    assert torch.equal(col_p, col_sw) and torch.equal(row_p, row_sw)
    # Plain SF is the pre-blocked form of the swizzled SF.
    torch.testing.assert_close(
        to_blocked(col_sf_p), col_sf_sw.flatten(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        to_blocked(row_sf_p), row_sf_sw.flatten(), atol=0, rtol=0
    )

    # compute_rowwise=False suppresses the rowwise *return* (the kernel still computes it).
    c_fp4, c_sf, r_fp4, r_sf = _cutedsl_rht_quantize_row_col_impl(
        A, col_amax, row_amax, tuple(sign_vector), compute_rowwise=False
    )
    assert r_fp4 is None and r_sf is None
    assert c_fp4 is not None and c_sf is not None


@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_rht_quantize_sr_requires_seed_offset():
    """stochastic_rounding=True needs both seed and offset tensors; omitting either raises."""
    A = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    col_amax, row_amax = _rht_amax("cutedsl", A, sign_vector=_HARDCODED_SIGN_VECTOR)
    with pytest.raises(ValueError):
        cutedsl_rht_quantize_row_col(
            A,
            col_amax,
            row_amax,
            list(_HARDCODED_SIGN_VECTOR),
            stochastic_rounding=True,
        )


@_skip_no_cutedsl
@pytest.mark.parametrize("use_fast_math", [False, True], ids=["exact", "fast"])
@torch.no_grad()
def test_cutedsl_rht_quantize_sr_unbiased(use_fast_math):
    """Hardware stochastic rounding (cvt.rs) is unbiased. Feed the row path elements that land
    EXACTLY halfway between FP4 grid points (1.25, between 1.0 and 1.5) -- the maximal-bias case
    RTNE always pins to one side -- and confirm averaging many SR draws converges to 1.25 with a
    ~50/50 grid split. Per 1x16 row-block one 6.0 anchor sets the block amax; global amax 2688
    gives an identity global scale, so every other element passes through as its raw value 1.25."""
    M, N = 256, 256
    dev = "cuda"
    A = torch.full((M, N), 1.25, dtype=torch.bfloat16, device=dev)
    A[:, ::16] = 6.0  # block amax anchor
    amax = torch.tensor(
        2688.0, dtype=torch.float32, device=dev
    )  # identity global scale (0-dim)
    halfway = torch.arange(N, device=dev) % 16 != 0  # the 1.25 positions

    # RTNE pins every halfway element to a single side (no spread).
    _, _, rtne_codes, rtne_sf = _quantize_row_col(
        "cutedsl",
        A,
        col_amax=amax,
        row_amax=amax,
        sign_vector=_HARDCODED_SIGN_VECTOR,
        use_fast_math=use_fast_math,
    )
    assert _dequantize(rtne_codes, rtne_sf, amax)[:, halfway].unique().numel() == 1

    seed = torch.tensor([0x12345678], dtype=torch.int64, device=dev)
    K = 32
    acc = torch.zeros(M, N, device=dev)
    n_lo = n_hi = n_other = 0
    for k in range(K):
        offset = torch.tensor([k * 2654435761 + 7], dtype=torch.int64, device=dev)
        _, _, codes, sf = _quantize_row_col(
            "cutedsl",
            A,
            col_amax=amax,
            row_amax=amax,
            sign_vector=_HARDCODED_SIGN_VECTOR,
            stochastic_rounding=True,
            seed=seed,
            offset=offset,
            use_fast_math=use_fast_math,
        )
        vals = _dequantize(codes, sf, amax)[:, halfway]
        acc[:, halfway] += vals
        n_lo += int((vals == 1.0).sum())
        n_hi += int((vals == 1.5).sum())
        n_other += int(((vals != 1.0) & (vals != 1.5)).sum())

    mean_sr = (acc / K)[:, halfway].mean().item()
    tot = n_lo + n_hi + n_other
    assert n_other == 0, "SR produced off-grid values"
    assert abs(mean_sr - 1.25) < 0.01, f"SR mean {mean_sr:.4f} != 1.25 (biased)"
    assert 0.45 < n_lo / tot < 0.55, f"SR grid split {n_lo / tot:.3f} not ~50/50"


@_skip_no_cutedsl
@pytest.mark.parametrize(
    "M,N",
    [(64, 256), (256, 200), (192, 256)],
    ids=["M_not_mult_128", "N_not_mult_128", "M_not_mult_128_b"],
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
