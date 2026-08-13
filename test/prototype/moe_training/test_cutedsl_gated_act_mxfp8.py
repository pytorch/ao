# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F


def _is_sm_10x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


if not _is_sm_10x():
    pytest.skip("MXFP8 requires CUDA SM 10.x", allow_module_level=True)

from torchao.prototype.moe_training.kernels.mxfp8 import (
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cutedsl_kernels_available,
)

if not _mxfp8_cutedsl_kernels_available:
    pytest.skip("MXFP8 cutedsl kernels not available", allow_module_level=True)

from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_gated_act_mxfp8 import (
    _gemm_swizzled_scale_idx,
    _launch_gated_act_mxfp8,
    _validate_inputs,
    gated_act_mxfp8_cutedsl_backward,
    gated_act_mxfp8_cutedsl_forward,
)
from torchao.prototype.mx_formats.utils import from_blocked

# Irregular, multi-chunk shape for the edge-input pattern axis.
_PATTERN_SHAPE = (256, 384)


# Forward data and all forward scales must match the eager reference bitwise.
# Backward data may differ by one E4M3 code in a bounded fraction of elements:
# the kernel's fast sigmoid and d_silu FMA contraction have no bit-exact eager
# equivalent (measured rate 5.7e-7). A backward scale byte may additionally
# flip by one code where such a difference lands on a block amax at the RCEIL
# mantissa-carry boundary; the affected block's codes then shift by ~2x, so
# those elements are excluded from the data compare and budgeted by the scales
# check instead. Keep in sync with MAX_DIFFERING_FRACTION in
# benchmarks/prototype/moe_training/mxfp8/bench_cutedsl_gated_act_mxfp8.py
# (whose opt-in validation holds scales fully bitwise).
_MAX_DIFFERING_FRACTION = 1e-5

# Absolute floor on the backward diff budgets: at small shapes the fractional
# bound rounds to zero, which would demand a bitwise-exact backward. Matches
# the bench's floor.
_COUNT_FLOOR = 8


def _eager_reference(gated_input, grad_h, act_kind):
    """Compute the bf16 tensor the kernel is expected to quantize.

    Keep in sync with eager_reference() in benchmarks/prototype/moe_training/
    mxfp8/bench_cutedsl_gated_act_mxfp8.py: both mirror the kernel's
    evaluation order.
    """
    assert act_kind == "silu", f"no eager reference wired up for {act_kind!r}"
    K = gated_input.shape[1] // 2
    gate = gated_input[:, :K].float()
    up = gated_input[:, K:].float()
    if grad_h is None:
        return (F.silu(gate) * up).bfloat16()
    grad_h_f = grad_h.float()
    # Mirror the kernel's evaluation order (silu path); the kernel contracts
    # dact into a single FMA, which eager cannot reproduce bit for bit.
    sigmoid_gate = torch.sigmoid(gate)
    silu = gate * sigmoid_gate
    dact = silu * (1.0 - sigmoid_gate) + sigmoid_gate
    dgate = ((dact * grad_h_f) * up).bfloat16()
    dup = (silu * grad_h_f).bfloat16()
    return torch.cat([dgate, dup], dim=1)


def _assert_layout(actual, ref, msg):
    assert actual.shape == ref.shape, f"{msg}: shape {actual.shape} vs {ref.shape}"
    assert actual.stride() == ref.stride(), (
        f"{msg}: stride {actual.stride()} vs {ref.stride()}"
    )
    assert actual.dtype == ref.dtype, f"{msg}: dtype {actual.dtype} vs {ref.dtype}"


def _assert_bitwise(actual, ref, msg):
    # Raw-byte compare: NaN codes are expected content in the edge patterns.
    _assert_layout(actual, ref, msg)
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        ref.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
        msg=msg,
    )


def _e4m3_ordinal(codes):
    # Map sign-magnitude E4M3 bytes onto a signed number line so adjacent
    # codes differ by 1 across the +/-0 boundary (raw byte distance jumps to
    # 128 there). Keep in sync with _e4m3_ordinal in the bench.
    c = codes.int()
    return torch.where(c >= 0x80, 0x80 - c, c)


def _assert_scales_match(actual, ref, msg, exact, logical_rows, logical_cols):
    """Forward scales are bitwise. A backward scale byte may flip by one code
    in a bounded count of blocks (a one-ulp amax difference at the RCEIL
    mantissa-carry boundary). Returns the logical (rows, cols) mask of
    differing blocks for the data compare to exclude, or None when exact."""
    if exact:
        _assert_bitwise(actual, ref, msg)
        return None
    _assert_layout(actual, ref, msg)
    a = from_blocked(
        actual.contiguous().view(torch.uint8), logical_rows, logical_cols
    ).int()
    r = from_blocked(
        ref.contiguous().view(torch.uint8), logical_rows, logical_cols
    ).int()
    gap = (a - r).abs()
    max_gap = int(gap.max())
    assert max_gap <= 1, f"{msg}: max E8M0 code gap {max_gap} > 1"
    mismatch = gap != 0
    count = int(mismatch.sum())
    limit = max(_COUNT_FLOOR, int(_MAX_DIFFERING_FRACTION * gap.numel()))
    assert count <= limit, f"{msg}: {count} scale bytes differ, limit {limit}"
    return mismatch


def _assert_qdata_matches(actual, ref, msg, exact, exclude_mask=None):
    """Forward data is bitwise; backward data within one E4M3 code in a
    bounded count of elements. exclude_mask marks elements of blocks whose
    scale byte differs: their codes legitimately shift ~2x and are budgeted
    by the scales check."""
    if exact:
        assert exclude_mask is None
        _assert_bitwise(actual, ref, msg)
        return
    _assert_layout(actual, ref, msg)
    a = _e4m3_ordinal(actual.contiguous().view(torch.uint8))
    r = _e4m3_ordinal(ref.contiguous().view(torch.uint8))
    gap = (a - r).abs()
    numel = gap.numel()
    if exclude_mask is not None:
        gap = gap[~exclude_mask]
    max_gap = int(gap.max()) if gap.numel() else 0
    assert max_gap <= 1, f"{msg}: max E4M3 code gap {max_gap} > 1"
    count = int((gap != 0).sum())
    limit = max(_COUNT_FLOOR, int(_MAX_DIFFERING_FRACTION * numel))
    assert count <= limit, f"{msg}: {count} codes differ, limit {limit}"


def _run_and_check(gated_input, grad_h, rowwise, colwise, act_kind, tag):
    """Run the public op and check all four outputs against the eager reference."""
    M = gated_input.shape[0]
    K = gated_input.shape[1] // 2
    if grad_h is not None:
        outputs = gated_act_mxfp8_cutedsl_backward(
            grad_h, gated_input, rowwise=rowwise, colwise=colwise
        )
    else:
        outputs = gated_act_mxfp8_cutedsl_forward(
            gated_input, rowwise=rowwise, colwise=colwise
        )

    # Fixed four-output tuple regardless of which directions are enabled.
    assert isinstance(outputs, tuple) and len(outputs) == 4, f"{tag}: arity"
    output_rowwise, output_colwise, scales_rowwise, scales_colwise = outputs

    # Forward emits h (width K); backward emits [dGate | dUp] (width 2K).
    expected_width = K if grad_h is None else 2 * K
    reference = _eager_reference(gated_input, grad_h, act_kind)
    assert reference.shape == (M, expected_width)
    exact = grad_h is None

    if rowwise:
        ref_q, ref_s = mxfp8_quantize_2d_1x32_cutedsl(reference, scaling_mode="rceil")
        assert output_rowwise.shape == (M, expected_width), f"{tag}: rowwise width"
        assert output_rowwise.stride() == (expected_width, 1), (
            f"{tag}: rowwise qdata must be row-major, got {output_rowwise.stride()}"
        )
        mismatch = _assert_scales_match(
            scales_rowwise,
            ref_s,
            f"{tag}: rowwise scales",
            exact,
            M,
            expected_width // 32,
        )
        exclude = None if mismatch is None else mismatch.repeat_interleave(32, dim=1)
        _assert_qdata_matches(
            output_rowwise, ref_q, f"{tag}: rowwise qdata", exact, exclude
        )
    else:
        assert output_rowwise.numel() == 0, f"{tag}: rowwise output should be empty"
        assert scales_rowwise.numel() == 0, f"{tag}: rowwise scales should be empty"
        assert output_rowwise.dtype == torch.float8_e4m3fn
        assert scales_rowwise.dtype == torch.float8_e8m0fnu
        assert output_rowwise.device == gated_input.device

    if colwise:
        ref_q, ref_s = mxfp8_quantize_2d_32x1_cutedsl(reference, scaling_mode="rceil")
        assert output_colwise.shape == (M, expected_width), f"{tag}: colwise width"
        assert output_colwise.stride() == (1, M), (
            f"{tag}: colwise qdata must have stride (1, M), got {output_colwise.stride()}"
        )
        assert scales_colwise.ndim == 1, f"{tag}: colwise scales should be flat"
        # Colwise scales use transposed blocked coordinates: rows are output
        # columns, columns are 32-row blocks.
        mismatch = _assert_scales_match(
            scales_colwise,
            ref_s,
            f"{tag}: colwise scales",
            exact,
            expected_width,
            M // 32,
        )
        exclude = None if mismatch is None else mismatch.repeat_interleave(32, dim=1).T
        _assert_qdata_matches(
            output_colwise, ref_q, f"{tag}: colwise qdata", exact, exclude
        )
    else:
        assert output_colwise.numel() == 0, f"{tag}: colwise output should be empty"
        assert scales_colwise.numel() == 0, f"{tag}: colwise scales should be empty"
        assert output_colwise.dtype == torch.float8_e4m3fn
        assert scales_colwise.dtype == torch.float8_e8m0fnu
        assert output_colwise.device == gated_input.device

    return outputs


def _make_gated_act_edge_input(
    M: int, K: int, pattern: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (gated_input, grad_h) for one input pattern. Pinned rows use
    gate = 20 (sigmoid saturates to 1.0) with power-of-two `up` values so h is
    bitwise stable; rowwise patterns sit in rows 0..8, colwise patterns in
    columns 100..103 over rows 0..31."""
    if pattern == "normal":
        torch.manual_seed(42)
        gated = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        return gated, grad

    if pattern == "boundary":
        # Block amaxes at and above the E4M3 max (448), plus large/tiny mixes.
        torch.manual_seed(13)
        gated = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        gate, up = gated[:, :K], gated[:, K:]
        gate[:6, :] = 20.0
        up[0, :] = 22.375  # h = bf16(447.5) = 448: amax exactly at E4M3 max
        up[1, :] = -22.375  # negative boundary
        up[2, :] = 23.0  # h = 460: amax in (448, 512), RCEIL boundary region
        up[3, ::2] = 16.0  # h = 320 mixed with ...
        up[3, 1::2] = 2.0**-12  # ... tiny values in the same 1x32 blocks
        up[4, 0] = 22.375  # block amax 448 with everything else ~448 * 2^-9,
        up[4, 1:] = 2.0**-4  # landing at the E4M3 subnormal boundary
        gate[:32, 100:104] = 20.0
        up[:32, 100] = 22.375  # colwise variants of the same boundaries
        up[:32, 101] = -22.375
        up[:32, 102] = 23.0
        up[:32:2, 103] = 16.0
        up[1:32:2, 103] = 2.0**-12
        return gated, grad

    if pattern == "zeros":
        # Zero rows/blocks: amax 0 takes the byte-0 scale and must still
        # produce zero codes.
        torch.manual_seed(5)
        gated = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        gate, up = gated[:, :K], gated[:, K:]
        gate[:2, :] = 20.0
        up[:2, :] = 0.5
        up[0, :64] = 0.0  # zero rowwise blocks amid a nonzero row
        gated[1, :] = 0.0  # fully zero row
        gate[:, 128:160] = 0.0  # silu(0) * up = 0: zero colwise stripes
        gate[:32, 100] = 20.0
        up[:32, 100] = 0.0  # zero colwise block
        grad[2, :] = 0.0  # zero gradient row (backward)
        grad[:32, 7] = 0.0  # zero gradient column block
        return gated, grad

    if pattern == "subnormal_tiny":
        # Tiny amaxes: the byte-0 scale must pair with the 2^127 reciprocal so
        # the blocks do not collapse to zero codes.
        torch.manual_seed(7)
        gated = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        gate, up = gated[:, :K], gated[:, K:]
        gate[:4, :] = 20.0
        up[:4, :] = 0.5
        up[0, :64] = 2.0**-125  # tiny amax: scale byte clamps to 0
        up[1, :] = 2.0**-130  # bf16 subnormal inputs
        up[2, ::2] = 2.0**-126  # bf16 min-normal mixed with zeros
        up[2, 1::2] = 0.0
        gate[:32, 100:102] = 20.0
        up[:32, 100] = 2.0**-125  # tiny colwise block
        up[:32, 101] = 2.0**-130
        grad[:32, 9] = 2.0**-120  # tiny gradients (backward)
        return gated, grad

    if pattern == "nan_inf":
        # NaN or Inf amax invalidates the block: scale byte 255 and every
        # element quantizes to the E4M3 NaN code.
        torch.manual_seed(11)
        gated = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        gate, up = gated[:, :K], gated[:, K:]
        gate[0, 0] = float("nan")  # NaN input propagates through the activation
        up[1, 5] = float("nan")
        gate[2, :] = 88.0  # sigmoid saturates: silu(x) == x
        up[2, :] = 3.0e38  # products overflow bf16 to Inf: Inf-amax blocks
        gate[3:5, :] = 20.0
        up[3:5, :] = 0.5
        up[3, 0] = float("inf")  # Inf element
        up[3, 1] = -3.0e38  # h overflows f32 to -Inf: mixed-sign Inf block
        gate[4, 0] = float("inf")
        up[4, 0] = 0.0  # silu(inf) * 0 -> NaN element
        gate[:32, 100] = float("inf")
        up[:32, 100] = 0.0  # all-NaN colwise block
        gate[:32, 101] = 88.0
        up[:32, 101] = 3.0e38  # Inf colwise block
        grad[5, 7] = float("nan")  # NaN gradient (backward)
        return gated, grad

    if pattern == "mixed_extreme":
        # Every special pattern in one tensor on a random background.
        torch.manual_seed(17)
        gated = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        gate, up = gated[:, :K], gated[:, K:]
        gate[:4, :] = 20.0
        up[:4, :] = 0.5
        up[0, :] = 22.375  # boundary row (amax 448)
        up[1, ::2] = 16.0  # large/tiny mix
        up[1, 1::2] = 2.0**-12
        up[2, :64] = 0.0  # zero blocks
        up[3, :64] = 2.0**-125  # tiny-amax blocks
        gate[4, 0] = float("nan")  # NaN input
        gate[5, :] = 88.0
        up[5, :] = 3.0e38  # Inf-amax row
        gate[6, :] = 20.0
        up[6, :] = 0.5
        gate[6, 0] = float("inf")
        up[6, 0] = 0.0  # silu(inf) * 0 -> NaN element
        gate[:32, 100:103] = 20.0
        up[:32, 100] = 0.0  # zero colwise block
        up[:32, 101] = 2.0**-125  # tiny colwise block
        up[:32, 102] = 22.375  # boundary colwise block
        gate[:32, 103] = float("inf")
        up[:32, 103] = 0.0  # all-NaN colwise block
        grad[7, 3] = float("nan")  # NaN gradient (backward)
        grad[8, :] = 0.0  # zero gradient row
        return gated, grad

    raise AssertionError(f"unknown pattern: {pattern}")


@pytest.mark.parametrize(
    "M,K",
    (
        (128, 128),
        (256, 256),
        (640, 384),  # one non-square, irregular shape
        (512, 2048),
        (1024, 7168),
    ),
)
@pytest.mark.parametrize("is_backward", (False, True))
@pytest.mark.parametrize(
    "rowwise,colwise", ((True, False), (False, True), (True, True))
)
# The public ops hard-wire silu today; new activation kinds slot into this
# axis once the ops expose a selector.
@pytest.mark.parametrize("act_kind", ("silu",))
def test_gated_act_mxfp8_numerics(M, K, is_backward, rowwise, colwise, act_kind):
    gated_input, grad = _make_gated_act_edge_input(M, K, "normal")
    tag = (
        f"{act_kind} bwd={is_backward} rowwise={rowwise} colwise={colwise} M={M} K={K}"
    )
    _run_and_check(
        gated_input, grad if is_backward else None, rowwise, colwise, act_kind, tag
    )


@pytest.mark.parametrize(
    "pattern",
    ("normal", "boundary", "zeros", "subnormal_tiny", "nan_inf", "mixed_extreme"),
)
@pytest.mark.parametrize("is_backward", (False, True))
@pytest.mark.parametrize(
    "rowwise,colwise", ((True, False), (False, True), (True, True))
)
@pytest.mark.parametrize("act_kind", ("silu",))
def test_gated_act_mxfp8_edge_inputs(pattern, is_backward, rowwise, colwise, act_kind):
    M, K = _PATTERN_SHAPE
    gated_input, grad = _make_gated_act_edge_input(M, K, pattern)
    tag = f"{act_kind} {pattern} bwd={is_backward} rowwise={rowwise} colwise={colwise}"
    outputs = _run_and_check(
        gated_input,
        grad if is_backward else None,
        rowwise,
        colwise,
        act_kind,
        tag,
    )

    # Targeted probes at known coordinates (forward rowwise only: exact codes).
    if is_backward or not rowwise:
        return
    q = outputs[0].view(torch.uint8)
    scales_r = outputs[2].contiguous().view(torch.uint8).flatten()
    ncb = (K // 32 + 3) // 4  # 128x4 scale-column blocks in the swizzled layout
    if pattern in ("subnormal_tiny", "mixed_extreme"):
        # Tiny-amax blocks must not collapse to zero codes (byte-0 scale
        # descales by 2^127).
        row = 0 if pattern == "subnormal_tiny" else 3
        assert bool(q[row, :64].ne(0).all()), (
            f"{tag}: tiny-amax block quantized to zero codes"
        )
        assert int(scales_r[_gemm_swizzled_scale_idx(row, 0, ncb)]) == 0x00, (
            f"{tag}: tiny amax did not clamp to scale byte 0x00"
        )
    if pattern in ("nan_inf", "mixed_extreme"):
        # Invalidated (NaN-amax) blocks map to scale byte 0xFF and E4M3 NaN
        # codes.
        row = 4 if pattern == "nan_inf" else 6
        assert int(q[row, 0]) & 0x7F == 0x7F, (
            f"{tag}: NaN element did not map to the E4M3 NaN code"
        )
        assert int(scales_r[_gemm_swizzled_scale_idx(row, 0, ncb)]) == 0xFF, (
            f"{tag}: NaN-amax block did not take scale byte 0xFF"
        )


@pytest.mark.parametrize("is_backward", (False, True))
@pytest.mark.parametrize(
    "rowwise,colwise", ((True, False), (False, True), (True, True))
)
def test_gated_act_mxfp8_torch_compile(is_backward, rowwise, colwise):
    """torch.compile(fullgraph=True) must trace the fake impls and return
    bitwise the eager op's outputs (both paths run the identical kernel)."""
    M, K = _PATTERN_SHAPE
    gated_input, grad = _make_gated_act_edge_input(M, K, "normal")
    grad = grad if is_backward else None

    def run_op(gated, grad_h):
        if grad_h is None:
            return gated_act_mxfp8_cutedsl_forward(
                gated, rowwise=rowwise, colwise=colwise
            )
        return gated_act_mxfp8_cutedsl_backward(
            grad_h, gated, rowwise=rowwise, colwise=colwise
        )

    tag = f"compile bwd={is_backward} rowwise={rowwise} colwise={colwise}"
    try:
        eager_outputs = run_op(gated_input, grad)
        compiled_outputs = torch.compile(run_op, fullgraph=True)(gated_input, grad)
        for i, (c, e) in enumerate(zip(compiled_outputs, eager_outputs)):
            _assert_bitwise(c, e, f"{tag}: output {i}")
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("M,K", ((128, 128), (256, 384)))
@pytest.mark.parametrize("is_backward", (False, True))
def test_gated_act_mxfp8_mode_consistency(M, K, is_backward):
    """Both-scales mode must reproduce each single-mode run bitwise; the modes
    take different kernel paths (cached-activation vs single-orientation)."""
    gated_input, grad = _make_gated_act_edge_input(M, K, "normal")

    def run_op(rowwise, colwise):
        if is_backward:
            return gated_act_mxfp8_cutedsl_backward(
                grad, gated_input, rowwise=rowwise, colwise=colwise
            )
        return gated_act_mxfp8_cutedsl_forward(
            gated_input, rowwise=rowwise, colwise=colwise
        )

    row_only = run_op(True, False)
    col_only = run_op(False, True)
    both = run_op(True, True)
    tag = f"mode-consistency bwd={is_backward} M={M} K={K}"
    _assert_bitwise(both[0], row_only[0], f"{tag}: rowwise qdata")
    _assert_bitwise(both[2], row_only[2], f"{tag}: rowwise scales")
    _assert_bitwise(both[1], col_only[1], f"{tag}: colwise qdata")
    _assert_bitwise(both[3], col_only[3], f"{tag}: colwise scales")


def test_gated_act_mxfp8_invalid_inputs():
    ok = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="bfloat16"):
        gated_act_mxfp8_cutedsl_forward(ok.float())
    with pytest.raises(ValueError, match="multiples of 128"):
        gated_act_mxfp8_cutedsl_forward(
            torch.randn(130, 256, device="cuda", dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="multiples of 128"):
        gated_act_mxfp8_cutedsl_forward(
            torch.randn(128, 260, device="cuda", dtype=torch.bfloat16)
        )
    # Zero-size inputs satisfy every modulus but cannot form a launch grid
    # or a TMA descriptor.
    with pytest.raises(ValueError, match="nonzero"):
        gated_act_mxfp8_cutedsl_forward(
            torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="nonzero"):
        gated_act_mxfp8_cutedsl_forward(
            torch.empty(128, 0, device="cuda", dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="even"):
        gated_act_mxfp8_cutedsl_forward(
            torch.randn(128, 257, device="cuda", dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="contiguous"):
        gated_act_mxfp8_cutedsl_forward(
            torch.randn(256, 128, device="cuda", dtype=torch.bfloat16).t()
        )
    with pytest.raises(ValueError, match="rowwise/colwise"):
        gated_act_mxfp8_cutedsl_forward(ok, rowwise=False, colwise=False)
    with pytest.raises(ValueError, match="grad_h"):
        gated_act_mxfp8_cutedsl_backward(
            torch.randn(128, 256, device="cuda", dtype=torch.bfloat16), ok
        )

    # 32-byte pointer alignment: contiguous storage-offset views are legal
    # torch tensors but break the launcher's assumed_align contract.
    base = torch.randn(128 * 256 + 16, device="cuda", dtype=torch.bfloat16)
    misaligned = base[1 : 1 + 128 * 256].view(128, 256)
    assert misaligned.is_contiguous()
    with pytest.raises(ValueError, match="32-byte aligned"):
        gated_act_mxfp8_cutedsl_forward(misaligned)
    with pytest.raises(ValueError, match="32-byte aligned"):
        gated_act_mxfp8_cutedsl_backward(base[1 : 1 + 128 * 128].view(128, 128), ok)
    # A 32-byte-aligned storage offset must still run and match a fresh copy.
    offset_ok = base[16 : 16 + 128 * 256].view(128, 256)
    assert offset_ok.data_ptr() % 32 == 0
    got = gated_act_mxfp8_cutedsl_forward(offset_ok)
    want = gated_act_mxfp8_cutedsl_forward(offset_ok.clone())
    _assert_bitwise(got[0], want[0], "aligned-offset view: rowwise qdata")
    _assert_bitwise(got[2], want[2], "aligned-offset view: rowwise scales")

    # INT32 indexing bound, allocation-free: _validate_inputs only inspects
    # metadata, so a FakeTensor exercises the check without 4 GiB of HBM.
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        big = torch.empty(8388736, 256, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="32-bit indexing limit"):
        _validate_inputs(big)

    # The fakes must reject both-False too: otherwise compile/export would
    # trace a call eager rejects (returning aliased zero-size outputs).
    with FakeTensorMode():
        fake = torch.empty(128, 512, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="rowwise/colwise"):
            torch.ops.torchao.gated_act_mxfp8_cutedsl_forward(
                fake, rowwise=False, colwise=False
            )


def test_gated_act_mxfp8_invalid_geometry():
    """The private geometry override must reject values the kernel's grid and
    bit-mask thread mapping cannot represent: the floor-division grid would
    silently skip columns/rows instead of failing."""
    gated_input = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    outputs = (
        torch.empty(128, 128, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(0, device="cuda", dtype=torch.uint8),
        torch.empty(0, device="cuda", dtype=torch.uint8),
    )
    for bad in ((96, 64, True), (256, 64, True)):  # non-pow2; K % CX != 0
        with pytest.raises(ValueError, match=f"CX={bad[0]}"):
            _launch_gated_act_mxfp8(
                gated_input, None, outputs, True, False, geometry=bad
            )
    with pytest.raises(ValueError, match="CY=48"):
        _launch_gated_act_mxfp8(
            gated_input, None, outputs, True, False, geometry=(64, 48, True)
        )
    # The staged path's output smem is double-buffered with no in-loop
    # TMA-store drain, so stage counts past the double buffer must be
    # rejected rather than silently corrupting reused buffers.
    tall = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="pipeline stages"):
        _launch_gated_act_mxfp8(
            tall, None, outputs, True, False, geometry=(64, 96, False)
        )
    # Row chunks ride CUDA grid dim y (cap 65535): a small-K shape inside the
    # int32 element bound must still be rejected. The gate fires before
    # compile or launch, so only the input allocation is paid.
    if torch.cuda.get_device_properties(0).total_memory >= 8 * 2**30:
        grid_y_input = torch.empty(65536 * 32, 256, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="grid y-dimension"):
            _launch_gated_act_mxfp8(
                grid_y_input, None, outputs, True, False, geometry=(64, 32, True)
            )
        del grid_y_input
    # A valid non-default geometry must produce bit-identical results.
    ref = gated_act_mxfp8_cutedsl_forward(gated_input, rowwise=True, colwise=False)
    alt = tuple(torch.empty_like(t) for t in ref)
    _launch_gated_act_mxfp8(
        gated_input, None, alt, True, False, geometry=(64, 64, True)
    )
    _assert_bitwise(alt[0], ref[0], "geometry override: rowwise qdata")
    _assert_bitwise(alt[2], ref[2], "geometry override: rowwise scales")


def test_gated_act_mxfp8_wrappers_unavailable(monkeypatch):
    """When the CuTeDSL runtime is unavailable the public wrappers must raise
    the informative NotImplementedError, not a raw import error. The flag is
    read at call time, so monkeypatching it simulates the unavailable case."""
    from torchao.prototype.moe_training.kernels.mxfp8 import quant as _quant

    monkeypatch.setattr(_quant, "_mxfp8_cutedsl_kernels_available", False)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(
        NotImplementedError, match="gated_act_mxfp8_cutedsl_forward requires"
    ):
        gated_act_mxfp8_cutedsl_forward(x)
    with pytest.raises(
        NotImplementedError, match="gated_act_mxfp8_cutedsl_backward requires"
    ):
        gated_act_mxfp8_cutedsl_backward(
            torch.randn(128, 128, device="cuda", dtype=torch.bfloat16), x
        )


def test_gated_act_mxfp8_int32_boundary():
    """Largest legal shape: 2*K*M - K - 1 lands just under INT32_MAX; verify
    boundary rows against the standalone quantizer without materializing a
    full-size reference."""
    if torch.cuda.get_device_properties(0).total_memory < 32 * 2**30:
        pytest.skip("needs >= 32 GiB of device memory")
    M, K = 131072, 8192
    torch.manual_seed(3)
    gated_input = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
    output_rowwise = gated_act_mxfp8_cutedsl_forward(
        gated_input, rowwise=True, colwise=False
    )[0]
    # Rowwise 1x32 blocks are row-local, so row slabs compare cleanly (the
    # blocked scales' swizzle offsets are not slice-local; qdata only).
    for rows in (slice(0, 128), slice(M - 128, M)):
        reference = _eager_reference(gated_input[rows], None, "silu")
        ref_q, _ = mxfp8_quantize_2d_1x32_cutedsl(reference, scaling_mode="rceil")
        _assert_bitwise(
            output_rowwise[rows].contiguous(), ref_q, f"int32-boundary rows {rows}"
        )
    del gated_input, output_rowwise
    torch.cuda.empty_cache()
