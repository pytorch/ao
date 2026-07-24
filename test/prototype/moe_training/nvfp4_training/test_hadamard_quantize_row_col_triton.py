"""Tests for triton_rht_quantize_row_col (SM100+ kernel).

  RTNE (stochastic_rounding=False):
    - test_triton_rht_quantize_rtne_scales_vs_reference: FP8 scale factors match the PyTorch
      reference bitwise for both col and row paths in swizzled layout.
    - test_triton_rht_quantize_rtne_sqnr: Dequantized output reconstructs post-RHT / raw-A
      values with SQNR ≥ 20 dB for both col and row paths.

  RS (stochastic_rounding=True):
    - test_triton_rht_quantize_rs_midpoint_distribution: Values at the FP4 [1.0, 1.5]
      midpoint (1.25) round to each neighbor ~50% of the time for both columnwise and
      rowwise paths. Columnwise input is constructed via inverse RHT so post-RHT values
      are exactly 1.25; rowwise input has 1.25 placed directly in A.
    - test_triton_rht_quantize_rs_at_most_one_fp4_step_from_rtne: RS code is at most 1
      FP4 magnitude index step from the RTNE code for every element, for both columnwise
      and rowwise paths.

Coverage:
  RS=F, RW=F  — rtne_scales_vs_reference + rtne_sqnr
  RS=F, RW=T  — rtne_scales_vs_reference + rtne_sqnr
  RS=T, RW=F  — rs_midpoint_distribution (col) + rs_at_most_one_fp4_step_from_rtne (col+row)
  RS=T, RW=T  — rs_midpoint_distribution (row) + rs_at_most_one_fp4_step_from_rtne (col+row)
"""

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.float8.float8_utils import compute_error
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
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
    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
        get_rht_matrix,
        prepare_for_cuda_graph,
    )

# M must be ≥ 128 (BLOCK_M minimum). M=32/64/96 excluded.
_M_VALUES = [128, 160, 256, 512]
# N must be ≥ 128 (BLOCK_N fixed=128). N=100 excluded.
_N_VALUES = [128, 200, 256, 384, 512, 1024]
_NEAR_ZERO = 1.0e-10
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


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _rht_reference(A: torch.Tensor) -> torch.Tensor:
    """PyTorch reference RHT: returns (N, M) bfloat16."""
    M_A, N_A = A.shape
    B = get_rht_matrix(_HARDCODED_SIGN_VECTOR, A.device, torch.bfloat16, 16)
    return (A.t().reshape(-1, 16) @ B).reshape(N_A, M_A).to(torch.bfloat16)


def _rht_quantize_reference(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RHT + NVFP4 E2M1 columnwise quantization via nvfp4_quantize (RTNE).

    Returns:
        codes:       (N, M//2) uint8 packed FP4 codes.
        scale_inv:   (N, M//16) float8_e4m3fn per-vector decode scales.
        global_amax: scalar float32.
    """
    # Pass bfloat16 output of _rht_reference directly: nvfp4_quantize converts bf16→f32
    # losslessly, so block amax matches the kernel's tl.max(bf16) exactly.
    x_t_rht = _rht_reference(A)  # (N, M) bfloat16
    global_amax = x_t_rht.float().abs().max()
    scale_inv, codes = nvfp4_quantize(
        x_t_rht, per_tensor_scale=per_tensor_amax_to_scale(global_amax)
    )
    return codes, scale_inv, global_amax


def _rht_quantize_rowwise_reference(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 E2M1 rowwise quantization via nvfp4_quantize (RTNE, no RHT applied).

    Returns:
        codes:       (M, N//2) uint8 packed FP4 codes.
        scale_inv:   (M, N//16) float8_e4m3fn per-vector decode scales.
        global_amax: scalar float32 (max(abs(A))).
    """
    global_amax = A.float().abs().max()
    scale_inv, codes = nvfp4_quantize(
        A, per_tensor_scale=per_tensor_amax_to_scale(global_amax)
    )
    return codes, scale_inv, global_amax


def _dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_amax: torch.Tensor,
) -> torch.Tensor:
    """Decode packed FP4 codes via NVFP4Tensor.dequantize()."""
    # orig_dtype=bfloat16: all test inputs are bfloat16; affects only the default
    # output dtype of dequantize(), overridden by the explicit .float() call below.
    return (
        NVFP4Tensor(
            codes,
            scales,
            16,
            torch.bfloat16,
            per_tensor_scale=per_tensor_amax_to_scale(global_amax),
            is_swizzled_scales=True,
        )
        .dequantize()
        .float()
    )


def _random_i64(device: torch.device) -> torch.Tensor:
    return torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device)


def _sr_kwargs(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "col_seed_base": _random_i64(device),
        "col_offset_base": _random_i64(device),
        "row_offset_base": _random_i64(device),
        "row_seed_base": _random_i64(device),
    }


def _quantize_row_col(
    A: torch.Tensor, stochastic_rounding: bool
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    col_amax, row_amax = triton_rht_amax(A, sign_vector=list(_HARDCODED_SIGN_VECTOR))
    kwargs = _sr_kwargs(A.device) if stochastic_rounding else {}
    col_codes, col_sf, row_codes, row_sf = triton_rht_quantize_row_col(
        A,
        stochastic_rounding=stochastic_rounding,
        sign_vector=list(_HARDCODED_SIGN_VECTOR),
        col_global_amax=col_amax,
        row_global_amax=row_amax,
        **kwargs,
    )
    return col_codes, col_sf, row_codes, row_sf, col_amax, row_amax


def _assert_scales_adjacent(got: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    """Kernel uses TE-exact div_rn for the per-vector scale; the mx_formats
    reference multiplies by a reciprocal and applies an E4M3_EPS floor. The two
    agree mathematically, so fp8 scale bytes are equal or land on adjacent
    representable values (|Δ raw uint8| <= 1, since positive e4m3 bytes are
    magnitude-monotonic)."""
    got_b = got.flatten().contiguous().view(torch.uint8).to(torch.int16)
    ref_b = ref.flatten().contiguous().view(torch.uint8).to(torch.int16)
    assert got_b.shape == ref_b.shape, f"{label}: shape mismatch"
    diff = (got_b - ref_b).abs()
    assert (diff <= 1).all(), (
        f"{label}: {(diff > 1).sum().item()}/{diff.numel()} fp8 scale bytes "
        f"differ by >1 ULP (max {diff.max().item()})"
    )


def _assert_scales_finite(scales: torch.Tensor) -> None:
    scales_f32 = scales.to(torch.float32)
    assert torch.isfinite(scales_f32).all(), "scale factors must be finite"


def _assert_zero_quantized(
    codes: torch.Tensor,
    scales: torch.Tensor,
    dequantized: torch.Tensor,
) -> None:
    assert torch.count_nonzero(codes).item() == 0, "all-zero input must pack to zero"
    scales_f32 = scales.to(torch.float32)
    # TE emits a zero per-vector scale for zero/near-zero vectors (no lower clamp).
    torch.testing.assert_close(
        scales_f32,
        torch.zeros_like(scales_f32),
        atol=0,
        rtol=0,
    )
    assert torch.isfinite(dequantized).all(), "dequantized zero input must be finite"
    torch.testing.assert_close(
        dequantized, torch.zeros_like(dequantized), atol=0, rtol=0
    )


def _input_from_rht_target(target: torch.Tensor) -> torch.Tensor:
    N, M = target.shape
    B = get_rht_matrix(
        _HARDCODED_SIGN_VECTOR, target.device, torch.bfloat16, 16
    ).float()
    A_t = (target.reshape(N * M // 16, 16) @ B.t()).reshape(N, M)
    return A_t.t().contiguous().to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Tests — RTNE (stochastic_rounding=False)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
@torch.no_grad()
def test_triton_rht_amax_propagates_nan():
    A = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
    A[0, 0] = float("nan")

    col_amax, row_amax = triton_rht_amax(A, sign_vector=list(_HARDCODED_SIGN_VECTOR))

    assert torch.isnan(col_amax)
    assert torch.isnan(row_amax)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_triton_rht_quantize_rtne_scales_vs_reference(M, N):
    """FP8 scale factors must match the PyTorch reference within 1 fp8 ULP.

    Columnwise: RHT + quantize of A.T. Rowwise: quantize raw A.

    The kernel computes the per-vector scale with TE-exact correctly-rounded
    division (tl.div_rn) and no lower clamp, while the mx_formats reference uses a
    reciprocal multiply and an E4M3_EPS floor. The two agree mathematically, so the
    fp8 scale bytes are equal or adjacent. Packed FP4 codes are NOT checked bitwise
    (the encode-scale reciprocal causes ~0.2% nibble differences at FP4 midpoints);
    use the SQNR test for quantization-quality validation.
    """
    if M % 128 != 0 or N % 128 != 0:
        pytest.skip("swizzled scales require M % 128 == 0 and N % 128 == 0")

    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    _, ref_col_sf, _ = _rht_quantize_reference(A)
    col_amax, row_amax = triton_rht_amax(A, sign_vector=list(_HARDCODED_SIGN_VECTOR))
    tri_col_codes, tri_col_sf, tri_row_codes, tri_row_sf = triton_rht_quantize_row_col(
        A,
        stochastic_rounding=False,
        sign_vector=list(_HARDCODED_SIGN_VECTOR),
        col_global_amax=col_amax,
        row_global_amax=row_amax,
    )

    # Columnwise scale check
    _assert_scales_adjacent(tri_col_sf, to_blocked(ref_col_sf), "col scale")

    # Rowwise scale check
    _, ref_row_sf, _ = _rht_quantize_rowwise_reference(A)
    _assert_scales_adjacent(tri_row_sf, to_blocked(ref_row_sf), "row scale")


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_triton_rht_quantize_rtne_sqnr(M, N):
    """Dequantized output must reconstruct post-RHT / raw-A values with SQNR ≥ 20 dB.

    Scale factors are always swizzled; layout does not affect quantization error.
    """
    if M % 128 != 0 or N % 128 != 0:
        pytest.skip("swizzled scales require M % 128 == 0 and N % 128 == 0")

    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    col_amax, row_amax = triton_rht_amax(A, sign_vector=list(_HARDCODED_SIGN_VECTOR))
    tri_col_codes, tri_col_sf, tri_row_codes, tri_row_sf = triton_rht_quantize_row_col(
        A,
        stochastic_rounding=False,
        sign_vector=list(_HARDCODED_SIGN_VECTOR),
        col_global_amax=col_amax,
        row_global_amax=row_amax,
    )

    # Columnwise SQNR: dequantized should reconstruct RHT(A.T)
    ref_rht = _rht_reference(A).float()
    col_sqnr = compute_error(ref_rht, _dequantize(tri_col_codes, tri_col_sf, col_amax))
    assert col_sqnr >= 20.0, f"Col SQNR {col_sqnr:.2f} dB < 20.0 dB for M={M} N={N}"

    # Rowwise SQNR: dequantized should reconstruct raw A
    row_sqnr = compute_error(
        A.float(), _dequantize(tri_row_codes, tri_row_sf, row_amax)
    )
    assert row_sqnr >= 20.0, f"Row SQNR {row_sqnr:.2f} dB < 20.0 dB for M={M} N={N}"


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
@pytest.mark.parametrize(
    "stochastic_rounding",
    [pytest.param(False, id="rtne"), pytest.param(True, id="stochastic_rounding")],
)
@pytest.mark.parametrize(
    "input_kind",
    [pytest.param("zeros", id="zeros"), pytest.param("near_zero", id="near_zero")],
)
@torch.no_grad()
def test_triton_rht_quantize_row_col_zero_and_near_zero_no_nan_or_saturation(
    input_kind, stochastic_rounding
):
    M, N = 128, 128

    if input_kind == "zeros":
        A = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        col_codes, col_sf, row_codes, row_sf, col_amax, row_amax = _quantize_row_col(
            A, stochastic_rounding
        )

        _assert_zero_quantized(
            col_codes, col_sf, _dequantize(col_codes, col_sf, col_amax)
        )
        _assert_zero_quantized(
            row_codes, row_sf, _dequantize(row_codes, row_sf, row_amax)
        )
        return

    A_row = torch.full((M, N), _NEAR_ZERO, dtype=torch.bfloat16, device="cuda")
    A_row[0, 0] = 1.0
    _, _, row_codes, row_sf, _, row_amax = _quantize_row_col(A_row, stochastic_rounding)
    _assert_scales_finite(row_sf)
    row_dequant = _dequantize(row_codes, row_sf, row_amax)
    assert torch.isfinite(row_dequant).all(), (
        "rowwise dequantized values must be finite"
    )
    # No lower scale clamp (TE semantics): near-zero blocks flush to a zero scale,
    # so their dequantized output is zero and cannot saturate.
    assert row_dequant.abs().max() <= 1.0

    col_target = torch.full((N, M), _NEAR_ZERO, dtype=torch.float32, device="cuda")
    col_target[0, 0] = 1.0
    A_col = _input_from_rht_target(col_target)
    col_codes, col_sf, _, _, col_amax, _ = _quantize_row_col(A_col, stochastic_rounding)
    _assert_scales_finite(col_sf)
    col_dequant = _dequantize(col_codes, col_sf, col_amax)
    assert torch.isfinite(col_dequant).all(), (
        "colwise dequantized values must be finite"
    )
    assert col_dequant.abs().max() <= 1.0


# ---------------------------------------------------------------------------
# Tests — RS (stochastic_rounding=True)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
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
    # Since B is orthogonal, A.T = target @ B^{-1} = target @ B.T.
    B = get_rht_matrix(_HARDCODED_SIGN_VECTOR, "cuda", torch.bfloat16, 16).float()
    target = torch.full((N_RHT, M_RHT), 1.25, dtype=torch.float32, device="cuda")
    target[:, ::16] = 6.0  # one anchor per 16-group along M
    A_t = (target.reshape(N_RHT * M_RHT // 16, 16) @ B.t()).reshape(N_RHT, M_RHT)
    A = A_t.t().contiguous().to(torch.bfloat16)  # kernel expects (M_A, N_A) contiguous

    # Build A_row: 1.25 everywhere, 6.0 at first element of each 16-group along N.
    # Row path quantizes A directly; vec_max=6.0 → encode_scale=1.0 → scaled=1.25 exactly.
    A_row = torch.full((128, 128), 1.25, dtype=torch.bfloat16, device="cuda")
    A_row[:, ::16] = 6.0

    col_count_lo = 0  # code 2 = 1.0
    col_count_hi = 0  # code 3 = 1.5
    row_count_lo = 0
    row_count_hi = 0

    for _ in range(N_SAMPLES):
        _col_seed_buf = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        _col_offset_buf = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        _row_offset_buf = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        _row_seed_buf = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        col_amax, row_amax = triton_rht_amax(
            A, sign_vector=list(_HARDCODED_SIGN_VECTOR)
        )
        col_codes, _, _, _ = triton_rht_quantize_row_col(
            A,
            stochastic_rounding=True,
            sign_vector=list(_HARDCODED_SIGN_VECTOR),
            col_seed_base=_col_seed_buf,
            col_offset_base=_col_offset_buf,
            row_offset_base=_row_offset_buf,
            row_seed_base=_row_seed_buf,
            col_global_amax=col_amax,
            row_global_amax=row_amax,
        )
        # Unpack col_codes (N_RHT, M_RHT//2) uint8 → (N_RHT, M_RHT) nibbles
        lo = (col_codes & 0xF).long()
        hi = (col_codes >> 4).long()
        all_nibs = torch.empty(N_RHT, M_RHT, dtype=torch.long, device="cuda")
        all_nibs[:, ::2] = lo
        all_nibs[:, 1::2] = hi
        mag_codes = all_nibs & 0x7

        # Exclude anchor positions (m % 16 == 0 → scaled=6.0 → code 7)
        col_idx = torch.arange(M_RHT, device="cuda")
        target_mags = mag_codes[:, (col_idx % 16) != 0]  # (N_RHT, 15 * M_RHT//16)

        col_count_lo += (target_mags == 2).sum().item()  # rounded to 1.0
        col_count_hi += (target_mags == 3).sum().item()  # rounded to 1.5

        # Rowwise path check using A_row
        _col_seed_r = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        _col_off_r = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        _row_off_r = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        _row_seed_r = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        row_col_amax, row_row_amax = triton_rht_amax(
            A_row, sign_vector=list(_HARDCODED_SIGN_VECTOR)
        )
        _, _, row_codes, _ = triton_rht_quantize_row_col(
            A_row,
            stochastic_rounding=True,
            sign_vector=list(_HARDCODED_SIGN_VECTOR),
            col_seed_base=_col_seed_r,
            col_offset_base=_col_off_r,
            row_offset_base=_row_off_r,
            row_seed_base=_row_seed_r,
            col_global_amax=row_col_amax,
            row_global_amax=row_row_amax,
        )
        # Unpack row_codes (128, 64) uint8 → (128, 128) nibbles
        r_lo = (row_codes & 0xF).long()
        r_hi = (row_codes >> 4).long()
        r_nibs = torch.empty(128, 128, dtype=torch.long, device="cuda")
        r_nibs[:, ::2] = r_lo
        r_nibs[:, 1::2] = r_hi
        r_mag = r_nibs & 0x7
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


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
@torch.no_grad()
def test_triton_rht_quantize_rs_at_most_one_fp4_step_from_rtne():
    """RS code must be at most 1 FP4 magnitude index step from the RTNE code.

    RS picks the floor or ceil of the scaled value on the FP4 magnitude grid.
    RTNE also picks floor or ceil (nearest). Therefore |rs_mag_idx - rtne_mag_idx| <= 1
    must hold for every element, and signs must agree.

    Both columnwise and rowwise paths are tested.
    """
    M, N = 128, 128
    N_SAMPLES = 16
    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    def _unpack(codes: torch.Tensor) -> torch.Tensor:
        """Unpack (R, C//2) uint8 → (R, C) nibble values."""
        lo = (codes & 0xF).long()
        hi = (codes >> 4).long()
        out = torch.empty(
            codes.shape[0], codes.shape[1] * 2, dtype=torch.long, device=codes.device
        )
        out[:, ::2] = lo
        out[:, 1::2] = hi
        return out

    col_amax_rtne, row_amax_rtne = triton_rht_amax(
        A, sign_vector=list(_HARDCODED_SIGN_VECTOR)
    )
    col_rn, _, row_rn, _ = triton_rht_quantize_row_col(
        A,
        stochastic_rounding=False,
        sign_vector=list(_HARDCODED_SIGN_VECTOR),
        col_global_amax=col_amax_rtne,
        row_global_amax=row_amax_rtne,
    )
    col_rn_nibs = _unpack(col_rn)
    col_rn_sign = col_rn_nibs >> 3
    col_rn_mag = col_rn_nibs & 0x7

    row_rn_nibs = _unpack(row_rn)
    row_rn_sign = row_rn_nibs >> 3
    row_rn_mag = row_rn_nibs & 0x7

    for _ in range(N_SAMPLES):
        _col_seed_buf = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        _col_offset_buf = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        _row_offset_buf = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        _row_seed_buf = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        col_amax_rs, row_amax_rs = triton_rht_amax(
            A, sign_vector=list(_HARDCODED_SIGN_VECTOR)
        )
        col_rs, _, row_rs, _ = triton_rht_quantize_row_col(
            A,
            stochastic_rounding=True,
            sign_vector=list(_HARDCODED_SIGN_VECTOR),
            col_seed_base=_col_seed_buf,
            col_offset_base=_col_offset_buf,
            row_offset_base=_row_offset_buf,
            row_seed_base=_row_seed_buf,
            col_global_amax=col_amax_rs,
            row_global_amax=row_amax_rs,
        )

        col_rs_nibs = _unpack(col_rs)
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

        row_rs_nibs = _unpack(row_rs)
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


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
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
    # NOTE: this test fails when run in the same process after test_hadamard_amax_triton.py.
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
    col_seed_buf = torch.randint(
        -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
    )
    row_seed_buf = torch.randint(
        -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
    )

    def run(data):
        col_offset = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
        row_offset = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=A.device
        )
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
    rtne_col_amax, rtne_row_amax = triton_rht_amax(
        A, sign_vector=list(_HARDCODED_SIGN_VECTOR)
    )
    rtne_col_ref, _, rtne_row_ref, _ = triton_rht_quantize_row_col(
        A,
        stochastic_rounding=False,
        sign_vector=list(_HARDCODED_SIGN_VECTOR),
        col_global_amax=rtne_col_amax,
        row_global_amax=rtne_row_amax,
    )
    assert not torch.equal(col_r1, rtne_col_ref), (
        "Col SR should produce different output than RTNE"
    )
    assert not torch.equal(row_r1, rtne_row_ref), (
        "Row SR should produce different output than RTNE"
    )
