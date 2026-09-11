# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, NVIDIA CORPORATION.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 four-over-six quantization and the linear layer that consumes it.

Four-over-six is an adaptive NVFP4 block-scaling recipe: every quantization
block is encoded twice and the candidate with the lower dequantization error
is stored.

* The **map-to-6** candidate is the standard NVFP4 encoding: the E4M3 block
  scale maps the block amax to FP4 value 6.
* The **map-to-4** candidate expands the E4M3 block scale by 1.5x, so FP4
  value 4 reaches the range that value 6 reaches in the standard encoding.
  The FP4 grid is denser around 4 than around 6, which lowers error for
  blocks whose mass sits below the amax.

Errors are compared per block with a configurable metric (mean-absolute or
mean-squared, computed in the input domain); ties select map-to-6. To leave
E4M3 headroom for the 1.5x scale expansion, the global (per-tensor) scale is
derived from a reduced E4M3 bound of 256 by default instead of 448.

Two global-scale granularities are supported for activations:

* per-tensor: one FP32 scale for the whole tensor (the default), and
* row-wise: one FP32 scale per tensor row, derived from that row's amax.

The reference arithmetic pins every rounding step, so results are
reproducible bit for bit across implementations. Two details are
load-bearing:

* The block-scale association is ``(block_amax / 6) * S_enc`` — one division
  then one multiply. The standard NVFP4 path uses
  ``block_amax * (S_enc * (1/6))``, which rounds differently on a fraction of
  blocks.
* The per-block error is accumulated sequentially in element order with FP32
  round-to-nearest adds, and 16x16 tiles reduce their 16 row-group errors in
  a pairwise halving tree. Both orders affect candidate selection on ties
  near the FP32 rounding boundary.

``four_over_six_linear`` mirrors the recipe's training semantics:

* forward GEMM: activations quantized 1x16 four-over-six (optionally
  row-scaled), weights quantized 16x16 four-over-six;
* backward with per-tensor activations: gradients use standard NVFP4
  round-to-nearest-even (four-over-six never applies to gradients), and the
  saved columnwise activation/weight codes are four-over-six;
* backward with row-scaled activations: high-precision (bf16) GEMMs. A
  row-scaled four-over-six tensor has no columnwise form — the per-row scales
  do not transpose — so the quantized wgrad operand cannot be produced.

Those backward defaults can be overridden with ``backward_override``:

* ``"quantized"``: the standard-NVFP4-gradient backward above (the
  per-tensor default; rejected for row-scaled activations);
* ``"high_precision"``: bf16 GEMMs on the saved original operands (the
  row-scaled default);
* ``"dequantized"``: bf16 GEMMs on dequantizations of the rowwise operands
  the forward GEMM consumed, so the gradients differentiate the
  quantized-forward function itself — the RL train/inference-consistency
  mode. Only 4-bit codes and scales are saved for backward, which also
  cuts activation memory.

Weights quantize with 16x16 tiles by default; ``weight_block="1x16"`` selects
rowwise blocks instead.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    f32_to_f4_unpacked,
    pack_uint4,
    unpack_uint4,
)
from torchao.prototype.mx_formats.utils import to_blocked

from .four_over_six_cutedsl import (
    _cutedsl_quantize_eligible,
    four_over_six_quantize_cutedsl,
)

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
_FP32_MAX = torch.finfo(torch.float32).max

__all__ = [
    "four_over_six_global_encode_scale",
    "four_over_six_quantize",
    "nvfp4_dequantize",
    "four_over_six_mm",
    "four_over_six_linear",
    "NVFP4FourOverSixLinear",
]


def four_over_six_global_encode_scale(
    global_amax: torch.Tensor, e4m3_scale_bound: int = 256
) -> torch.Tensor:
    """Global encode scale: bound * 6 / amax.

    ``global_amax`` may be a scalar (per-tensor) or a 1-D per-row vector.
    ``amax == 0`` gives inf and an enormous amax underflows the scale to
    zero; both fall back to the identity scale.
    """
    amax = global_amax.to(torch.float32)
    candidate = torch.full_like(amax, float(e4m3_scale_bound) * FP4_E2M1_MAX) / amax
    candidate = candidate.clamp(max=_FP32_MAX)
    return torch.where(
        (amax == 0.0) | (candidate == 0.0), torch.ones_like(candidate), candidate
    )


def _fp4_rtne(scaled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-to-nearest-even FP4 codes and their exact FP32 values.

    Reproduces ``cvt.rn.satfinite.e2m1x2.f32`` followed by
    ``cvt.rn.f16x2.e2m1x2`` (E2M1 values are exact in FP16 and FP32).
    """
    clamped = scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
    codes = f32_to_f4_unpacked(clamped)
    return codes, f4_unpacked_to_f32(codes)


def _candidate_error(
    values: torch.Tensor,
    scale_fp8: torch.Tensor,
    xf: torch.Tensor,
    global_amax: torch.Tensor,
    err_mode: str,
    e4m3_scale_bound: int,
) -> torch.Tensor:
    """Per-block dequantization error: FP32 adds in element order, per 1x16 group.

    values/xf: (rows, num_groups, 16); scale_fp8: (rows, num_groups, 1);
    global_amax broadcastable against (rows, num_groups).
    """
    sf = scale_fp8.to(torch.float32)[..., 0]
    # The denominator must be a tensor: dividing by a python scalar lowers to a
    # multiply by its (inexact) reciprocal, which double-rounds and flips
    # candidate picks near error ties. Tensor-tensor division is a true
    # correctly-rounded FP32 division.
    err_denom = torch.full(
        (),
        FP4_E2M1_MAX * float(e4m3_scale_bound),
        dtype=torch.float32,
        device=xf.device,
    )
    err = torch.zeros_like(xf[..., 0])
    for idx in range(16):
        val = ((values[..., idx] * sf) * global_amax) / err_denom
        diff = val - xf[..., idx]
        if err_mode == "mse":
            err = err + diff * diff
        else:
            err = err + diff.abs()
    return err


def _tile_error_tree_sum(err: torch.Tensor) -> torch.Tensor:
    """Reduce 16 row-group errors per 16x16 tile in the warp-shuffle tree order.

    err: (rows, num_groups) with rows % 16 == 0 -> (rows // 16, num_groups).
    """
    rows = err.view(err.shape[0] // 16, 16, err.shape[1])
    rows = rows[:, 0:8] + rows[:, 8:16]
    rows = rows[:, 0:4] + rows[:, 4:8]
    rows = rows[:, 0:2] + rows[:, 2:4]
    return rows[:, 0] + rows[:, 1]


def four_over_six_quantize(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    *,
    block: str = "1x16",
    err_mode: str = "mae",
    e4m3_scale_bound: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D tensor to NVFP4 with four-over-six block selection.

    Args:
        x: (R, C) bfloat16 or float32, C % 16 == 0 (R % 16 == 0 for 16x16).
        global_amax: scalar FP32 amax, or a (R,) per-row amax vector for the
            row-scaled variant (1x16 blocks only).
        block: "1x16" (activations/gradient operands) or "16x16" (weights).
        err_mode: "mae" or "mse" candidate-selection error metric.
        e4m3_scale_bound: 256 (default, leaves map-to-4 headroom) or 448.

    Returns:
        (codes, scales): (R, C//2) uint8 packed FP4 codes (low nibble = even
        element) and (R, C//16) float8_e4m3fn block scales.
    """
    if block not in ("1x16", "16x16"):
        raise ValueError(f"block must be '1x16' or '16x16', got {block!r}")
    if err_mode not in ("mae", "mse"):
        raise ValueError(f"err_mode must be 'mae' or 'mse', got {err_mode!r}")
    if e4m3_scale_bound not in (256, 448):
        raise ValueError(f"e4m3_scale_bound must be 256 or 448, got {e4m3_scale_bound}")
    if x.dim() != 2:
        raise ValueError(f"x must be 2D, got {x.dim()}D")
    rows, cols = x.shape
    if cols % 16 != 0:
        raise ValueError(f"C must be divisible by 16, got C={cols}")
    if block == "16x16" and rows % 16 != 0:
        raise ValueError(f"16x16 blocks require R divisible by 16, got R={rows}")
    row_scaled = global_amax.dim() == 1 and global_amax.numel() == rows
    if row_scaled and block != "1x16":
        raise ValueError("row-scaled four-over-six supports 1x16 blocks only")
    if not row_scaled and global_amax.numel() != 1:
        raise ValueError(
            f"global_amax must be a scalar or a ({rows},) row vector, "
            f"got shape {tuple(global_amax.shape)}"
        )

    # Fast path: the CuTe DSL kernel is an op-for-op reimplementation of the
    # arithmetic below with bitwise-identical codes and scales — except NaN
    # inputs, which take NaN-dropping block amaxes and +6 codes in the
    # kernel while this body's torch.amax propagates NaN into the scales;
    # see the op docstring. Ineligible shapes/dtypes and pre-SM100 devices
    # silently fall through to the pure-PyTorch body.
    if _cutedsl_quantize_eligible(x):
        return four_over_six_quantize_cutedsl(
            x, global_amax, block, err_mode, e4m3_scale_bound
        )

    xf = x.float().view(rows, cols // 16, 16)
    s_enc = four_over_six_global_encode_scale(global_amax, e4m3_scale_bound)
    if row_scaled:
        s_enc = s_enc.view(rows, 1)
        err_amax = global_amax.to(torch.float32).view(rows, 1)
    else:
        err_amax = global_amax.to(torch.float32)

    if block == "16x16":
        tiles = xf.abs().view(rows // 16, 16, cols // 16, 16)
        block_amax = tiles.amax(dim=(1, 3)).repeat_interleave(16, dim=0)
    else:
        block_amax = xf.abs().amax(dim=-1)

    # Scale-pair construction: base = (block_amax / 6) * S_enc, then the 1.5x
    # map-to-4 expansion; both capped at the full E4M3 range. The divisor is a
    # tensor for a true correctly-rounded FP32 division (a python-scalar
    # divisor lowers to a reciprocal multiply, which double-rounds).
    fp4_max = torch.full((), FP4_E2M1_MAX, dtype=torch.float32, device=xf.device)
    base = (block_amax / fp4_max) * s_enc
    scale6 = base.clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    scale4 = (base * 1.5).clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    s_dec = 1.0 / s_enc
    inv6 = (1.0 / (scale6.to(torch.float32) * s_dec)).clamp(max=_FP32_MAX)
    inv4 = (1.0 / (scale4.to(torch.float32) * s_dec)).clamp(max=_FP32_MAX)

    codes6, values6 = _fp4_rtne(xf * inv6.unsqueeze(-1))
    codes4, values4 = _fp4_rtne(xf * inv4.unsqueeze(-1))
    err6 = _candidate_error(
        values6, scale6.unsqueeze(-1), xf, err_amax, err_mode, e4m3_scale_bound
    )
    err4 = _candidate_error(
        values4, scale4.unsqueeze(-1), xf, err_amax, err_mode, e4m3_scale_bound
    )
    if block == "16x16":
        pick4 = (
            _tile_error_tree_sum(err4) < _tile_error_tree_sum(err6)
        ).repeat_interleave(16, dim=0)
    else:
        pick4 = err4 < err6

    codes = torch.where(pick4.unsqueeze(-1), codes4, codes6)
    scales = torch.where(pick4, scale4, scale6)
    return pack_uint4(codes.view(rows, cols)), scales


def nvfp4_dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_amax: torch.Tensor,
    *,
    e4m3_scale_bound: int = 256,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize packed NVFP4 codes and block scales back to high precision.

    This is the standard NVFP4 decode — four-over-six changes only the
    encode-side scale selection. It is a pure-PyTorch correctness helper,
    not an optimized kernel; a fused decode kernel is future work (TODO).
    The per-block decode scale is ``(f32(scale) * amax) *
    factor_inv`` with ``factor_inv = 1 / (6 * bound)`` a correctly-rounded
    FP32 reciprocal, and each element is ``f32(code) * decode_scale`` cast to
    ``out_dtype``. Scales from either block granularity dequantize
    identically (a 16x16 tile stores its scale byte on every row).

    Args:
        codes: (R, C//2) uint8 packed FP4 codes.
        scales: (R, C//16) float8_e4m3fn block scales.
        global_amax: scalar FP32 amax, or a (R,) per-row amax vector for the
            row-scaled variant.
        e4m3_scale_bound: the bound the codes were quantized with (this
            recipe family defaults to 256; standard NVFP4 uses 448).
        out_dtype: output dtype (the kernel's OType cast).
    """
    if e4m3_scale_bound not in (256, 448):
        raise ValueError(f"e4m3_scale_bound must be 256 or 448, got {e4m3_scale_bound}")
    rows, packed_cols = codes.shape
    cols = packed_cols * 2
    if scales.shape != (rows, cols // 16):
        raise ValueError(
            f"scales must have shape ({rows}, {cols // 16}), got {tuple(scales.shape)}"
        )
    row_scaled = global_amax.dim() == 1 and global_amax.numel() == rows
    if not row_scaled and global_amax.numel() != 1:
        raise ValueError(
            f"global_amax must be a scalar or a ({rows},) row vector, "
            f"got shape {tuple(global_amax.shape)}"
        )
    return _nvfp4_dequantize_op(codes, scales, global_amax, e4m3_scale_bound, out_dtype)


# Registered as a custom op so torch.compile keeps the eager decode: inductor
# codegen of the fused unpack + broadcast-scale graph miscompiles the
# low-nibble lane (torch 2.14 nightly), and the dequantized backward's whole
# contract is bitwise parity with the fprop operands.
# TODO: file the upstream pytorch inductor issue and cite it here as the
# removal trigger for this workaround.
@torch.library.custom_op("torchao::nvfp4_dequantize", mutates_args=())
def _nvfp4_dequantize_op(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_amax: torch.Tensor,
    e4m3_scale_bound: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    rows, packed_cols = codes.shape
    cols = packed_cols * 2
    row_scaled = global_amax.dim() == 1 and global_amax.numel() == rows
    values = f4_unpacked_to_f32(unpack_uint4(codes)).view(rows, cols // 16, 16)
    amax = global_amax.to(torch.float32)
    if row_scaled:
        amax = amax.view(rows, 1)
    # The reciprocal must come from a true FP32 division (see _candidate_error
    # on why a python-scalar denominator double-rounds).
    factor_inv = torch.ones((), dtype=torch.float32, device=codes.device) / torch.full(
        (),
        FP4_E2M1_MAX * float(e4m3_scale_bound),
        dtype=torch.float32,
        device=codes.device,
    )
    decode_scale = (scales.to(torch.float32) * amax) * factor_inv
    return (values * decode_scale.unsqueeze(-1)).to(out_dtype).view(rows, cols)


@_nvfp4_dequantize_op.register_fake
def _(codes, scales, global_amax, e4m3_scale_bound, out_dtype):
    return codes.new_empty((codes.shape[0], codes.shape[1] * 2), dtype=out_dtype)


def _standard_rtne_quantize(
    x: torch.Tensor, global_amax: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard NVFP4 1x16 round-to-nearest-even quantize for gradient operands.

    The gradient scale chain keeps the standard association
    ``block_amax * (S_enc * (1/6))`` and the full 448 E4M3 bound; only
    non-gradient four-over-six tensors use the ``(block_amax / 6) * S_enc``
    association above.
    """
    rows, cols = x.shape
    xf = x.float().view(rows, cols // 16, 16)
    s_enc = four_over_six_global_encode_scale(global_amax, e4m3_scale_bound=448)
    block_amax = xf.abs().amax(dim=-1)
    scales = (
        (block_amax * (s_enc * (1.0 / FP4_E2M1_MAX)))
        .clamp(max=FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )
    enc = (1.0 / (scales.to(torch.float32) * (1.0 / s_enc))).clamp(max=_FP32_MAX)
    codes, _ = _fp4_rtne(xf * enc.unsqueeze(-1))
    return pack_uint4(codes.view(rows, cols)), scales


def _global_decode_scale(amax: torch.Tensor, e4m3_scale_bound: int) -> torch.Tensor:
    """Per-tensor decode scale consumed by the GEMM: amax / (bound * 6).

    The scalar divisor (a reciprocal-multiply lowering) is fine here, unlike
    the encode chain's tensor divisors: this factor never enters the encode
    arithmetic — consumers reconstruct it for the GEMM's per-tensor scale
    slot — so it sits outside the div.rn encode contract, and the
    linear-level tests pin the resulting GEMM outputs.
    """
    return amax.to(torch.float32) / (float(e4m3_scale_bound) * FP4_E2M1_MAX)


def _scaled_mm_nvfp4(
    a_codes: torch.Tensor,
    a_scales: torch.Tensor,
    a_global: torch.Tensor,
    b_codes_t: torch.Tensor,
    b_scales: torch.Tensor,
    b_global: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Block-scaled FP4 GEMM with per-tensor second-level scales.

    a_codes: (M, K//2) uint8; b_codes_t: (K//2, N) transposed uint8 view;
    a_scales/b_scales: plain (rows, K//16) float8 block scales (swizzled here).
    """
    return F.scaled_mm(
        a_codes.view(torch.float4_e2m1fn_x2),
        b_codes_t.view(torch.float4_e2m1fn_x2),
        scale_a=[to_blocked(a_scales).flatten(), a_global],
        scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        scale_b=[to_blocked(b_scales).flatten(), b_global],
        scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        output_dtype=out_dtype,
    )


@torch._dynamo.allow_in_graph
class four_over_six_mm(torch.autograd.Function):
    """NVFP4 four-over-six quantized matmul.

    3 GEMMs:
      forward:   x_row @ W.T  = output       (1x16 four-over-six x, 16x16 four-over-six W)
      backward:  dy_row @ W.T = grad_input   (standard-NVFP4 dy; saved columnwise W)
      backward:  dy_col.T @ x_col = grad_weight (standard-NVFP4 dy; saved columnwise x)

    With row-scaled activations the backward runs in bf16 instead (see the
    module docstring), saving the high-precision operands.

    ``backward_override`` selects among the quantized, high-precision, and
    dequantized backwards described in the module docstring; ``None`` keeps
    the defaults above. ``weight_block`` selects the weight tile granularity.

    Requires: M % 128 == 0, K % 128 == 0, N % 128 == 0. Non-bf16 inputs are
    cast to bf16 and gradients are always bf16, so leaves that require grad
    must be bf16 (matching ``nvfp4_mm_triton``).
    """

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        bias: Optional[torch.Tensor],
        err_mode: str = "mae",
        e4m3_scale_bound: int = 256,
        row_scaled_activation: bool = False,
        backward_override: Optional[str] = None,
        weight_block: str = "16x16",
    ):
        M = input_hp.shape[:-1].numel()
        K = input_hp.shape[-1]
        N = weight_hp.shape[0]
        if input_hp.dtype != torch.bfloat16:
            input_hp = input_hp.to(torch.bfloat16)
        if weight_hp.dtype != torch.bfloat16:
            weight_hp = weight_hp.to(torch.bfloat16)
        if M % 128 != 0 or K % 128 != 0 or N % 128 != 0:
            raise ValueError(
                f"four_over_six_mm requires M, K, N all divisible by 128; "
                f"got M={M}, K={K}, N={N}"
            )
        if backward_override is None:
            backward_override = (
                "high_precision" if row_scaled_activation else "quantized"
            )
        if backward_override not in ("quantized", "high_precision", "dequantized"):
            raise ValueError(
                f"backward_override must be 'quantized', 'high_precision', or "
                f"'dequantized', got {backward_override!r}"
            )
        if backward_override == "quantized" and row_scaled_activation:
            raise ValueError(
                "row-scaled four-over-six has no quantized backward; use "
                "'high_precision' or 'dequantized'"
            )
        input_2d = input_hp.reshape(-1, K).contiguous()

        if row_scaled_activation:
            x_amax = input_2d.abs().amax(dim=1).to(torch.float32)
        else:
            x_amax = input_2d.abs().amax().to(torch.float32)
        w_amax = weight_hp.abs().amax().to(torch.float32)

        x_codes, x_scales = four_over_six_quantize(
            input_2d,
            x_amax,
            block="1x16",
            err_mode=err_mode,
            e4m3_scale_bound=e4m3_scale_bound,
        )
        w_codes, w_scales = four_over_six_quantize(
            weight_hp,
            w_amax,
            block=weight_block,
            err_mode=err_mode,
            e4m3_scale_bound=e4m3_scale_bound,
        )
        w_global = _global_decode_scale(w_amax, e4m3_scale_bound)

        if row_scaled_activation:
            # The GEMM's per-tensor slot cannot hold a per-row scale: run it
            # with the constant 1/(6*bound) factor, then apply the raw per-row
            # amaxes on the FP32 output before the bf16 cast.
            x_global = torch.full(
                (),
                1.0 / (FP4_E2M1_MAX * float(e4m3_scale_bound)),
                dtype=torch.float32,
                device=input_2d.device,
            )
            output = _scaled_mm_nvfp4(
                x_codes,
                x_scales,
                x_global,
                w_codes.t(),
                w_scales,
                w_global,
                torch.float32,
            )
            output = (output * x_amax.view(-1, 1)).to(torch.bfloat16)
        else:
            x_global = _global_decode_scale(x_amax, e4m3_scale_bound)
            output = _scaled_mm_nvfp4(
                x_codes,
                x_scales,
                x_global,
                w_codes.t(),
                w_scales,
                w_global,
                torch.bfloat16,
            )
        output = output.reshape(*input_hp.shape[:-1], N)
        if bias is not None:
            output = output + bias.to(output.dtype)

        if backward_override == "high_precision":
            ctx.save_for_backward(input_2d, weight_hp)
        elif backward_override == "dequantized":
            # The rowwise operands the forward GEMM just consumed; backward
            # dequantizes them, differentiating the quantized-forward function.
            ctx.save_for_backward(
                x_codes,
                x_scales,
                x_amax,
                w_codes,
                w_scales,
                w_amax,
            )
        else:
            x_col_codes, x_col_scales = four_over_six_quantize(
                input_2d.t().contiguous(),
                x_amax,
                block="1x16",
                err_mode=err_mode,
                e4m3_scale_bound=e4m3_scale_bound,
            )
            w_col_codes, w_col_scales = four_over_six_quantize(
                weight_hp.t().contiguous(),
                w_amax,
                block=weight_block,
                err_mode=err_mode,
                e4m3_scale_bound=e4m3_scale_bound,
            )
            ctx.save_for_backward(
                x_col_codes,
                x_col_scales,
                x_amax,
                w_col_codes,
                w_col_scales,
                w_amax,
            )
        ctx.backward_override = backward_override
        ctx.e4m3_scale_bound = e4m3_scale_bound
        ctx.input_orig_shape = input_hp.shape
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        if ctx.backward_override == "high_precision":
            input_2d, weight_hp = ctx.saved_tensors
            grad_input = (grad_output_2d @ weight_hp).reshape(ctx.input_orig_shape)
            grad_weight = grad_output_2d.t() @ input_2d
        elif ctx.backward_override == "dequantized":
            (
                x_codes,
                x_scales,
                x_amax,
                w_codes,
                w_scales,
                w_amax,
            ) = ctx.saved_tensors
            weight_dq = nvfp4_dequantize(
                w_codes, w_scales, w_amax, e4m3_scale_bound=ctx.e4m3_scale_bound
            )
            input_dq = nvfp4_dequantize(
                x_codes, x_scales, x_amax, e4m3_scale_bound=ctx.e4m3_scale_bound
            )
            grad_input = (grad_output_2d @ weight_dq).reshape(ctx.input_orig_shape)
            grad_weight = grad_output_2d.t() @ input_dq
        else:
            (
                x_col_codes,
                x_col_scales,
                x_amax,
                w_col_codes,
                w_col_scales,
                w_amax,
            ) = ctx.saved_tensors
            dy_amax = grad_output_2d.abs().amax().to(torch.float32)
            dy_row_codes, dy_row_scales = _standard_rtne_quantize(
                grad_output_2d, dy_amax
            )
            dy_col_codes, dy_col_scales = _standard_rtne_quantize(
                grad_output_2d.t().contiguous(), dy_amax
            )
            dy_global = _global_decode_scale(dy_amax, 448)
            grad_input = _scaled_mm_nvfp4(
                dy_row_codes,
                dy_row_scales,
                dy_global,
                w_col_codes.t(),
                w_col_scales,
                _global_decode_scale(w_amax, ctx.e4m3_scale_bound),
                torch.bfloat16,
            ).reshape(ctx.input_orig_shape)
            grad_weight = _scaled_mm_nvfp4(
                dy_col_codes,
                dy_col_scales,
                dy_global,
                x_col_codes.t(),
                x_col_scales,
                _global_decode_scale(x_amax, ctx.e4m3_scale_bound),
                torch.bfloat16,
            )

        grad_bias = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if ctx.has_bias
            else None
        )
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


four_over_six_linear = four_over_six_mm.apply


class NVFP4FourOverSixLinear(nn.Linear):
    """Linear layer with NVFP4 four-over-six quantized GEMMs.

    Drop-in replacement for nn.Linear implementing the four-over-six recipe:
    forward GEMM operands use four-over-six NVFP4, gradients use standard
    NVFP4 (or bf16 when ``row_scaled_activation`` is set — see the module
    docstring for why row-scaled has no quantized backward).
    ``backward_override`` and ``weight_block`` pass through to
    :class:`four_over_six_mm`. ``bias`` defaults off, matching the recipe's
    usage (unlike nn.Linear).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        err_mode: str = "mae",
        e4m3_scale_bound: int = 256,
        row_scaled_activation: bool = False,
        backward_override: Optional[str] = None,
        weight_block: str = "16x16",
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.err_mode = err_mode
        self.e4m3_scale_bound = e4m3_scale_bound
        self.row_scaled_activation = row_scaled_activation
        self.backward_override = backward_override
        self.weight_block = weight_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return four_over_six_linear(
            x,
            self.weight,
            self.bias,
            self.err_mode,
            self.e4m3_scale_bound,
            self.row_scaled_activation,
            self.backward_override,
            self.weight_block,
        )

    @classmethod
    def from_linear(
        cls,
        mod: nn.Linear,
        err_mode: str = "mae",
        e4m3_scale_bound: int = 256,
        row_scaled_activation: bool = False,
        backward_override: Optional[str] = None,
        weight_block: str = "16x16",
    ) -> "NVFP4FourOverSixLinear":
        new = cls(
            mod.in_features,
            mod.out_features,
            mod.bias is not None,
            err_mode=err_mode,
            e4m3_scale_bound=e4m3_scale_bound,
            row_scaled_activation=row_scaled_activation,
            backward_override=backward_override,
            weight_block=weight_block,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        if mod.weight.device != torch.device("meta"):
            new.weight = mod.weight
            if mod.bias is not None:
                new.bias = mod.bias
        return new
