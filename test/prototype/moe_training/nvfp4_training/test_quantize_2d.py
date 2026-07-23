# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from torch.utils._triton import has_triton

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    prepare_for_cuda_graph,
)
from torchao.prototype.moe_training.nvfp4_training.quantize_2d_cutedsl import (
    cutedsl_weight_quantize_2d,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    per_tensor_amax_to_scale,
)
from torchao.utils import is_sm_at_least_100, torch_version_at_least

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

# The CuteDSL kernel requires out_features (dim 0) % 256 == 0 and in_features (dim 1)
# % 128 == 0; the triton kernel's BLOCK_M minimum is 128 and its swizzled scales require
# both dims % 128. The sweep is the union, with _skip_if_unsupported_shape dropping
# M=128 for cutedsl so triton keeps its BLOCK_M-minimum coverage.
_M_VALUES = [128, 256, 512, 1024]
_N_VALUES = [128, 256, 512, 1024, 2048]


def _skip_if_unsupported_shape(kernel: str, M: int, N: int) -> None:
    """Skip shapes the selected backend cannot handle."""
    if kernel == "cutedsl" and M % 256 != 0:
        pytest.skip("cutedsl weight quantize requires out_features % 256 == 0")


# Minimum reconstruction SQNR (dB) per backend; both land around 19 dB on the grid above.
_MIN_SQNR_DB = {"triton": 15.0, "cutedsl": 18.0}

_FP8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
_NEAR_ZERO = 1.0e-10


def _weight_quantize_2d(kernel, W, amax):
    """Dispatch to a backend's 2D weight-quantize kernel.

    Returns ``(codes, scales, t_codes, t_scales)``: the rowwise NVFP4(W) codes + swizzled
    FP8 scale factors, then the colwise NVFP4(W.T) codes + swizzled scale factors.
    """
    if kernel == "triton":
        from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
            triton_weight_quantize_2d,
        )

        return triton_weight_quantize_2d(W, amax)
    return cutedsl_weight_quantize_2d(W, amax)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _weight_quantize_2d_reference_scales(A: torch.Tensor) -> torch.Tensor:
    """PyTorch oracle: per-16×16-block FP8 scale factors expanded to (M, N//16).

    Mirrors the two-level scaling in _nvfp4_2d_quantize:
      1. global encode scale from the tensor-wide amax.
      2. per-block FP8 scale clamped to [FP8_EPS, FP8_MAX].
      3. Expand each per-block scale to cover 16 consecutive rows.

    Returns:
        (M, N//16) float8_e4m3fn — the same layout as the kernel's non-swizzled output.
    """
    FP8_MAX = 448.0
    FP8_EPS = torch.finfo(torch.float8_e4m3fn).tiny
    FP4_MAX = 6.0
    M, N = A.shape
    x = A.float()
    global_amax = x.abs().max()

    blocks = x.reshape(M // 16, 16, N // 16, 16)
    block_amax = blocks.abs().amax(dim=(1, 3))  # (M//16, N//16)

    is_global_amax_zero = global_amax == 0
    safe_global_amax = torch.where(
        is_global_amax_zero, torch.ones_like(global_amax), global_amax
    )
    enc_g = (FP8_MAX * FP4_MAX / safe_global_amax).clamp(
        max=torch.finfo(torch.float32).max
    )
    enc_g = torch.where(is_global_amax_zero, torch.ones_like(enc_g), enc_g)
    pvscale = (block_amax / FP4_MAX) * enc_g
    pvscale = pvscale.clamp(FP8_EPS, FP8_MAX).to(torch.float8_e4m3fn)  # (M//16, N//16)

    # Expand: each block-row scale repeated 16 times → (M, N//16)
    return pvscale.repeat_interleave(16, dim=0)


def _swizzle_py(scales_expanded: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """Python equivalent of the kernel's _swizzle_scales(expand_sf, BLOCK_M, BLOCK_N).

    Transforms (M, N//16) float8_e4m3fn → (M//128, N//64, 32, 16).
    """
    u8 = scales_expanded.view(torch.uint8)
    swizzled = (
        u8.reshape(M // 128, 4, 32, N // 64, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(M // 128, N // 64, 32, 16)
    )
    return swizzled.view(torch.float8_e4m3fn)


def _dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_amax: torch.Tensor,
) -> torch.Tensor:
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


def _unpack_fp4_magnitudes(codes: torch.Tensor) -> torch.Tensor:
    lo = (codes & 0xF).long()
    hi = (codes >> 4).long()
    out = torch.empty(
        codes.shape[0], codes.shape[1] * 2, dtype=torch.long, device=codes.device
    )
    out[:, ::2] = lo
    out[:, 1::2] = hi
    return out & 0x7


def _assert_scales_finite_and_nonzero(scales: torch.Tensor) -> None:
    scales_f32 = scales.to(torch.float32)
    assert torch.isfinite(scales_f32).all(), "scale factors must be finite"
    assert (scales_f32 >= _FP8_E4M3_EPS).all(), (
        f"scale factors must be clamped to at least {_FP8_E4M3_EPS}"
    )


def _assert_zero_quantized(
    codes: torch.Tensor,
    scales: torch.Tensor,
    dequantized: torch.Tensor,
) -> None:
    assert torch.count_nonzero(codes).item() == 0, "all-zero input must pack to zero"
    scales_f32 = scales.to(torch.float32)
    torch.testing.assert_close(
        scales_f32,
        torch.full_like(scales_f32, _FP8_E4M3_EPS),
        atol=0,
        rtol=0,
    )
    assert torch.isfinite(dequantized).all(), "dequantized zero input must be finite"
    torch.testing.assert_close(
        dequantized, torch.zeros_like(dequantized), atol=0, rtol=0
    )


def _assert_near_zero_values_do_not_saturate(
    codes: torch.Tensor, near_zero_mask: torch.Tensor
) -> None:
    magnitudes = _unpack_fp4_magnitudes(codes)
    near_zero_magnitudes = magnitudes[near_zero_mask]
    assert (near_zero_magnitudes <= 1).all(), (
        "near-zero values must not saturate to large FP4 magnitudes"
    )


def _assert_scales_match_up_to_rounding_ties(
    scales: torch.Tensor, reference: torch.Tensor, what: str
) -> None:
    """Allow the small fraction of blocks whose FP8 scale rounds the tie the other way.

    Every mismatch must be within one E4M3 step (ratio in [1/1.15, 1.15]), so a real recipe
    divergence (e.g. 1D instead of 2D block scaling) still fails.
    """
    got = scales.flatten().float()
    ref = reference.flatten().float()
    mism = got != ref
    frac = mism.float().mean().item()
    assert frac < 0.02, f"{100 * frac:.2f}% of {what} differ (>2%)"
    if mism.any():
        ratio = got[mism] / ref[mism]
        assert (ratio <= 1.15).all() and (ratio >= 1 / 1.15).all(), (
            f"{what} mismatches exceed one E4M3 step (not a rounding tie)"
        )


def _assert_scales_vs_reference(
    kernel: str, scales: torch.Tensor, reference: torch.Tensor, what: str
) -> None:
    """triton reproduces the PyTorch oracle bitwise; CuteDSL up to FP8 rounding ties."""
    if kernel == "triton":
        torch.testing.assert_close(scales, reference, atol=0, rtol=0)
    else:
        _assert_scales_match_up_to_rounding_ties(scales, reference, what)


# ---------------------------------------------------------------------------
# Tests — scale factors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_weight_quantize_2d_scales_vs_reference(kernel, M, N):
    """Swizzled FP8 scale factors must match the PyTorch 16x16 reference."""
    _skip_if_unsupported_shape(kernel, M, N)
    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    ref_scales_expanded = _weight_quantize_2d_reference_scales(A)  # (M, N//16)

    _, scales, _, t_scales = _weight_quantize_2d(kernel, A, A.float().abs().max())

    ref_scales = _swizzle_py(ref_scales_expanded, M, N)
    _assert_scales_vs_reference(kernel, scales, ref_scales, "rowwise SF")

    ref_t_scales_expanded = _weight_quantize_2d_reference_scales(
        A.T.contiguous()
    )  # (N, M//16)
    ref_t_scales = _swizzle_py(ref_t_scales_expanded, N, M)  # (N//128, M//64, 32, 16)
    _assert_scales_vs_reference(kernel, t_scales, ref_t_scales, "colwise SF")


@_skip_no_triton
@_skip_no_cutedsl
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_cutedsl_weight_quantize_2d_matches_triton(M, N):
    """Both backends emit the same 2D 16x16 scale factors, modulo FP8 rounding ties."""
    _skip_if_unsupported_shape("cutedsl", M, N)
    torch.manual_seed(3)
    W = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    amax = W.float().abs().max()
    _, c_rsf, _, c_csf = _weight_quantize_2d("cutedsl", W, amax)
    _, t_rsf, _, t_csf = _weight_quantize_2d("triton", W, amax)
    _assert_scales_match_up_to_rounding_ties(c_rsf, t_rsf, "rowwise SF vs Triton 16x16")
    _assert_scales_match_up_to_rounding_ties(c_csf, t_csf, "colwise SF vs Triton 16x16")


# ---------------------------------------------------------------------------
# Tests — quantization quality (SQNR) and output contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_weight_quantize_2d_sqnr(kernel, M, N):
    """Rowwise dequant must reconstruct A and colwise dequant A.T."""
    _skip_if_unsupported_shape(kernel, M, N)
    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    global_amax = A.float().abs().max()
    codes, scales, t_codes, t_scales = _weight_quantize_2d(kernel, A, global_amax)

    # NVFP4Tensor interprets (M, N//16) scales as rowwise block_size=16 scales,
    # which matches the 2D expanded layout since every 16 rows share the same scale.
    min_sqnr = _MIN_SQNR_DB[kernel]

    sqnr = compute_error(A.float(), _dequantize(codes, scales, global_amax))
    assert sqnr >= min_sqnr, (
        f"Rowwise SQNR {sqnr:.2f} dB < {min_sqnr} dB for M={M} N={N}"
    )

    sqnr_t = compute_error(A.T.float(), _dequantize(t_codes, t_scales, global_amax))
    assert sqnr_t >= min_sqnr, (
        f"Colwise SQNR {sqnr_t:.2f} dB < {min_sqnr} dB for M={M} N={N}"
    )


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_weight_quantize_2d_layout(kernel, M, N):
    """Output 4-tuple shapes + dtypes, the contract both backends share."""
    _skip_if_unsupported_shape(kernel, M, N)
    torch.manual_seed(1)
    W = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    amax = W.float().abs().max()
    w_fp4, w_sf, wt_fp4, wt_sf = _weight_quantize_2d(kernel, W, amax)

    assert w_fp4.shape == (M, N // 2) and w_fp4.dtype == torch.uint8
    assert wt_fp4.shape == (N, M // 2) and wt_fp4.dtype == torch.uint8
    assert (
        w_sf.shape == (M // 128, N // 64, 32, 16) and w_sf.dtype == torch.float8_e4m3fn
    )
    assert (
        wt_sf.shape == (N // 128, M // 64, 32, 16)
        and wt_sf.dtype == torch.float8_e4m3fn
    )


# ---------------------------------------------------------------------------
# Tests — degenerate inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize(
    "input_kind",
    [pytest.param("zeros", id="zeros"), pytest.param("near_zero", id="near_zero")],
)
@torch.no_grad()
def test_weight_quantize_2d_zero_and_near_zero_no_nan_or_saturation(kernel, input_kind):
    M, N = 256, 256

    if input_kind == "zeros":
        A = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    else:
        A = torch.full((M, N), _NEAR_ZERO, dtype=torch.bfloat16, device="cuda")
        A[0, 0] = 1.0

    global_amax = A.float().abs().max()
    row_codes, row_sf, col_codes, col_sf = _weight_quantize_2d(kernel, A, global_amax)

    if input_kind == "zeros":
        # Zero input packs to zero codes, every block scale clamps to E4M3 eps (not 0),
        # and both layouts dequantize back to zero.
        _assert_zero_quantized(
            row_codes, row_sf, _dequantize(row_codes, row_sf, global_amax)
        )
        _assert_zero_quantized(
            col_codes, col_sf, _dequantize(col_codes, col_sf, global_amax)
        )
        return

    _assert_scales_finite_and_nonzero(row_sf)
    row_dequant = _dequantize(row_codes, row_sf, global_amax)
    assert torch.isfinite(row_dequant).all(), (
        "rowwise dequantized values must be finite"
    )
    assert row_dequant.abs().max() <= 1.0

    row_near_zero_mask = torch.ones(M, N, dtype=torch.bool, device="cuda")
    row_near_zero_mask[0, 0] = False
    _assert_near_zero_values_do_not_saturate(row_codes, row_near_zero_mask)

    _assert_scales_finite_and_nonzero(col_sf)
    col_dequant = _dequantize(col_codes, col_sf, global_amax)
    assert torch.isfinite(col_dequant).all(), (
        "colwise dequantized values must be finite"
    )
    assert col_dequant.abs().max() <= 1.0

    col_near_zero_mask = torch.ones(N, M, dtype=torch.bool, device="cuda")
    col_near_zero_mask[0, 0] = False
    _assert_near_zero_values_do_not_saturate(col_codes, col_near_zero_mask)


# ---------------------------------------------------------------------------
# Tests — backend-specific
# ---------------------------------------------------------------------------


@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_weight_quantize_2d_requires_out_features_256():
    """out_features (dim 0) must be divisible by 256 (stricter than the Triton kernel's 128)."""
    W = torch.randn(384, 256, dtype=torch.bfloat16, device="cuda")  # 384 % 256 == 128
    amax = W.float().abs().max()
    with pytest.raises(ValueError, match="out_features"):
        cutedsl_weight_quantize_2d(W, amax)


@_skip_no_triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_weight_quantize_2d_cuda_graph_compile():
    """triton_weight_quantize_2d under reduce-overhead CUDA graphs.

    Weight quantization is deterministic (no SR), so consecutive calls should produce
    identical outputs. Primarily verifies fullgraph=True compilation succeeds via the
    registered custom_op + register_fake. Also the only M % 256 != 0 coverage, a shape
    the CuteDSL kernel rejects.
    """
    shape = (128, 256)
    W = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    prepare_for_cuda_graph(
        W.device
    )  # pre-allocate TMA scratch + SR bufs outside pool context

    def run(w):
        amax = w.float().abs().max()
        codes, sf, t_codes, t_sf = _weight_quantize_2d("triton", w, amax)
        return codes.clone()

    compiled = torch.compile(run, mode="reduce-overhead", fullgraph=True)
    for _ in range(3):
        compiled(W)  # warmup

    r1 = compiled(W)
    r2 = compiled(W)
    torch.testing.assert_close(r1, r2)
