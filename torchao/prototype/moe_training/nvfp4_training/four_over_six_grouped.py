# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, NVIDIA CORPORATION.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Differentiable NVFP4 four-over-six grouped GEMM for MoE training.

Grouped counterpart of ``four_over_six_mm`` for routed-expert layers:
``A`` holds token groups packed along dim 0, ``B`` holds one weight matrix
per expert, and ``offs`` marks each group's end row. Every token group is
quantized as its own tensor:

* per-tensor activations (the default): each token group gets its own
  global scale from that group's amax. The group amaxes are expanded to a
  per-row amax vector so the whole packed tensor quantizes in one
  ``four_over_six_quantize`` call — bitwise identical to quantizing each
  group separately, because the quantizer derives every row's scale chain
  from that row's amax entry. The forward GEMM is one
  ``F.scaled_grouped_mm`` with per-group second-level scales.
* row-scaled activations: one global scale per token row. The forward is a
  single ``F.scaled_grouped_mm`` carrying the constant per-tensor factor in
  every group's slot, its bf16 output upcast and scaled by the raw per-row
  amaxes (torch's grouped GEMM only emits bf16 high-precision output). That
  is one GEMM-output rounding away from a per-group loop of dense
  four-over-six GEMMs, and the tests pin the fused output against a
  rounding-emulated loop oracle.

Weights always quantize per expert with per-tensor scales
(``weight_block`` selects 16x16 tiles or 1x16 blocks, as in the dense op).

Gradients never quantize with four-over-six, so the grouped backward
supports only the high-precision and dequantized overrides of
``four_over_six_mm``:

* ``"high_precision"`` (the default): bf16 grouped GEMMs on the saved
  original operands;
* ``"dequantized"``: bf16 grouped GEMMs on dequantizations of the rowwise
  operands the forward consumed — the RL train/inference-consistency mode.

``"quantized"`` raises. Requires K % 128 == 0 and N % 128 == 0; token
groups must be 128-row aligned unless ``pad_token_groups_for_grouped_mm``
is set, which zero-pads each group to the next 128 multiple before
quantization (zero rows quantize to zero codes and are sliced away from the
output).
"""

from typing import Optional

import torch
import torch.nn.functional as F

from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
    FP4_E2M1_MAX,
    _global_decode_scale,
    four_over_six_quantize,
    nvfp4_dequantize,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
    _DEVICE_ASSERTS,
)
from torchao.prototype.moe_training.utils import (
    conditional_nostrict_trace,
    pad_token_groups,
    unpad_token_groups,
)
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.quantization.quantize_.common import KernelPreference
from torchao.utils import is_sm_at_least_100

_ALIGNMENT = 128
_SCALE_RECIPE = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
_SWIZZLE = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]

__all__ = ["four_over_six_grouped_mm"]


@conditional_nostrict_trace
def four_over_six_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    offs: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    err_mode: str = "mae",
    e4m3_scale_bound: int = 256,
    row_scaled_activation: bool = False,
    weight_block: str = "16x16",
    backward_override: Optional[str] = None,
    pad_token_groups_for_grouped_mm: bool = False,
) -> torch.Tensor:
    """Quantize grouped activations and expert weights with four-over-six.

    ``A`` has shape ``(M, K)``, ``B`` has shape ``(E, N, K)``, and ``offs``
    contains the cumulative row-end offset for each expert. Knobs match
    ``four_over_six_mm``; see the module docstring for the grouped-specific
    backward and alignment semantics.
    """
    output = _FourOverSixGroupedMM.apply(
        A,
        B,
        offs,
        err_mode,
        e4m3_scale_bound,
        row_scaled_activation,
        weight_block,
        backward_override,
        pad_token_groups_for_grouped_mm,
    )
    if bias is not None:
        output = output + bias.to(output.dtype)
    return output


def _expand_group_amax(
    row_amax: torch.Tensor, group_end_offsets: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Per-row amax vector holding each row's group amax.

    Rows past the final offset (the pad-helper's over-allocated tail) take
    the last group's amax; they are all-zero and never enter the GEMM.
    """
    group_idx = torch.searchsorted(
        group_end_offsets,
        torch.arange(row_amax.shape[0], device=row_amax.device, dtype=torch.int32),
        right=True,
    ).clamp_(max=num_experts - 1)
    group_amax = torch.zeros(
        num_experts, dtype=torch.float32, device=row_amax.device
    ).scatter_reduce_(
        0, group_idx, row_amax.to(torch.float32), reduce="amax", include_self=True
    )
    return group_amax[group_idx], group_amax


def _quantize_expert_weights(
    weight: torch.Tensor,
    weight_amax: torch.Tensor,
    weight_block: str,
    err_mode: str,
    e4m3_scale_bound: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert four-over-six quantization of a stacked (E, N, K) weight.

    Flattening experts along rows and expanding each expert's amax over its
    rows quantizes the whole stack in one call — bitwise identical to a
    per-expert loop, because the quantizer derives every row's scale chain
    from that row's amax entry, 1x16 blocks never cross rows, and 16x16
    tiles never cross experts (N % 128 == 0 keeps expert boundaries 16-row
    aligned, and every row of a tile carries the same expert's amax).
    """
    num_experts, N, K = weight.shape
    flat_codes, flat_scales = four_over_six_quantize(
        weight.reshape(num_experts * N, K),
        weight_amax.repeat_interleave(N),
        block=weight_block,
        err_mode=err_mode,
        e4m3_scale_bound=e4m3_scale_bound,
    )
    return (
        flat_codes.view(num_experts, N, K // 2),
        flat_scales.view(num_experts, N, K // 16),
    )


def _dequantize_expert_weights(
    codes: torch.Tensor,
    scales: torch.Tensor,
    weight_amax: torch.Tensor,
    e4m3_scale_bound: int,
) -> torch.Tensor:
    """Dequantize stacked per-expert codes back to a bf16 (E, N, K) weight.

    Flattening experts along rows and expanding each expert's amax over its
    rows reproduces the per-expert scalar dequantization exactly — the
    decode chain reads one amax entry per row either way.
    """
    num_experts, N = codes.shape[0], codes.shape[1]
    row_amax = weight_amax.to(torch.float32).repeat_interleave(N)
    flat = nvfp4_dequantize(
        codes.reshape(num_experts * N, -1),
        scales.reshape(num_experts * N, -1),
        row_amax,
        e4m3_scale_bound=e4m3_scale_bound,
    )
    return flat.view(num_experts, N, -1)


def _row_scaled_single_grouped_gemm(
    x_codes: torch.Tensor,
    x_scales: torch.Tensor,
    x_amax: torch.Tensor,
    x_global: torch.Tensor,
    w_codes: torch.Tensor,
    w_scales: torch.Tensor,
    w_global: torch.Tensor,
    padded_group_end_offsets: torch.Tensor,
) -> torch.Tensor:
    """One ``F.scaled_grouped_mm`` covering every group of the row-scaled
    forward.

    The GEMM epilogue applies the E4M3 block scales and the per-tensor
    factors — the constant 1/(6*bound) in every group's activation slot and
    each expert's amax/(6*bound) — and the raw per-row amax multiply plus
    the final bf16 cast happen on the upcast output. The GEMM emits bf16
    (torch's grouped GEMM has no FP32 output mode), which costs one
    rounding before the row scale relative to a loop of dense FP32-output
    GEMMs per group. Rows past the final offset may hold garbage; the pad
    helper's unpad drops them.
    """
    num_experts = w_codes.shape[0]
    output = F.scaled_grouped_mm(
        x_codes.view(torch.float4_e2m1fn_x2),
        w_codes.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
        # scaled_grouped_mm consumes swizzled scale bytes viewed at the
        # logical 2D shape, as in the per-tensor forward; the view needs
        # the 128-row alignment the forward enforces. One to_blocked over
        # the row-flattened expert scales equals the per-expert stack
        # bitwise because N % 128 == 0 keeps expert boundaries on swizzle
        # row-block boundaries.
        scale_a=[
            to_blocked(x_scales).view(x_scales.shape),
            x_global.expand(num_experts).contiguous(),
        ],
        scale_recipe_a=_SCALE_RECIPE,
        scale_b=[
            to_blocked(w_scales.reshape(-1, w_scales.shape[-1])).view(num_experts, -1),
            w_global,
        ],
        scale_recipe_b=_SCALE_RECIPE,
        swizzle_a=_SWIZZLE,
        swizzle_b=_SWIZZLE,
        offs=padded_group_end_offsets,
        output_dtype=torch.bfloat16,
    )
    return (output.to(torch.float32) * x_amax.view(-1, 1)).to(torch.bfloat16)


class _FourOverSixGroupedMM(torch.autograd.Function):
    """NVFP4 four-over-six grouped forward with override-only backward."""

    @staticmethod
    def forward(
        ctx,
        input_act: torch.Tensor,
        weight: torch.Tensor,
        group_end_offsets: torch.Tensor,
        err_mode: str,
        e4m3_scale_bound: int,
        row_scaled_activation: bool,
        weight_block: str,
        backward_override: Optional[str],
        pad_token_groups_for_grouped_mm: bool,
    ) -> torch.Tensor:
        if group_end_offsets.ndim != 1 or group_end_offsets.dtype != torch.int32:
            raise ValueError("offs must be a 1D int32 tensor")
        if not group_end_offsets.is_contiguous():
            raise ValueError("offs must be contiguous")
        if group_end_offsets.numel() != weight.shape[0]:
            raise ValueError("offs must contain one group-end offset per expert")
        if not is_sm_at_least_100():
            raise NotImplementedError(
                "NVFP4 four-over-six grouped GEMM requires SM100+"
            )
        if backward_override is None:
            backward_override = "high_precision"
        if backward_override not in ("high_precision", "dequantized"):
            if backward_override == "quantized":
                raise ValueError(
                    "grouped four-over-six has no quantized backward; use "
                    "'high_precision' or 'dequantized'"
                )
            raise ValueError(
                f"backward_override must be 'high_precision' or 'dequantized', "
                f"got {backward_override!r}"
            )

        num_tokens, K = input_act.shape
        num_experts, N, _ = weight.shape
        if K % _ALIGNMENT != 0 or N % _ALIGNMENT != 0:
            raise ValueError(
                f"K and N must be divisible by {_ALIGNMENT}; got K={K}, N={N}"
            )
        if _DEVICE_ASSERTS:
            group_sizes = torch.diff(
                group_end_offsets, prepend=group_end_offsets.new_zeros(1)
            )
            torch.ops.aten._assert_async.msg(
                torch.all(group_sizes >= 0), "offs must be non-decreasing"
            )
            torch.ops.aten._assert_async.msg(
                group_end_offsets[-1] == num_tokens,
                "the final group-end offset must equal A.shape[0]",
            )
            if not pad_token_groups_for_grouped_mm:
                torch.ops.aten._assert_async.msg(
                    torch.all(group_sizes % _ALIGNMENT == 0),
                    "every token group must be 128-row aligned when padding is disabled",
                )

        input_act = input_act.to(torch.bfloat16).contiguous()
        weight = weight.to(torch.bfloat16).contiguous()
        original_input = input_act

        padded_group_start_offsets = None
        if pad_token_groups_for_grouped_mm:
            # The fused pad/unpad CUDA kernels only accept alignment_size 32
            # and at most 32 groups; this op needs 128-row alignment with any
            # expert count, so it pins the pure-torch path.
            input_act, padded_group_start_offsets, padded_group_end_offsets = (
                pad_token_groups(
                    input_act,
                    group_end_offsets,
                    alignment_size=_ALIGNMENT,
                    kernel_preference=KernelPreference.EMULATED,
                )
            )
        else:
            padded_group_end_offsets = group_end_offsets

        row_amax = input_act.abs().amax(dim=1)
        group_amax = None
        if row_scaled_activation:
            x_amax = row_amax.to(torch.float32)
        else:
            x_amax, group_amax = _expand_group_amax(
                row_amax, padded_group_end_offsets, num_experts
            )
        weight_amax = weight.abs().amax(dim=(1, 2)).to(torch.float32)

        x_codes, x_scales = four_over_six_quantize(
            input_act,
            x_amax,
            block="1x16",
            err_mode=err_mode,
            e4m3_scale_bound=e4m3_scale_bound,
        )
        w_codes, w_scales = _quantize_expert_weights(
            weight, weight_amax, weight_block, err_mode, e4m3_scale_bound
        )
        w_global = _global_decode_scale(weight_amax, e4m3_scale_bound)

        if row_scaled_activation:
            # The row-scaled forward carries the constant 1/(6*bound) factor
            # in every group's per-tensor slot and scales the GEMM output by
            # the raw per-row amaxes; one F.scaled_grouped_mm covers all
            # groups.
            x_global = torch.full(
                (),
                1.0 / (FP4_E2M1_MAX * float(e4m3_scale_bound)),
                dtype=torch.float32,
                device=input_act.device,
            )
            output = _row_scaled_single_grouped_gemm(
                x_codes,
                x_scales,
                x_amax,
                x_global,
                w_codes,
                w_scales,
                w_global,
                padded_group_end_offsets,
            )
        else:
            output = F.scaled_grouped_mm(
                x_codes.view(torch.float4_e2m1fn_x2),
                w_codes.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
                # scaled_grouped_mm consumes swizzled scale bytes viewed at the
                # logical 2D shape (the layout the group quantize kernels
                # return); the view needs the 128-row alignment enforced above.
                scale_a=[
                    to_blocked(x_scales).view(x_scales.shape),
                    _global_decode_scale(group_amax, e4m3_scale_bound),
                ],
                scale_recipe_a=_SCALE_RECIPE,
                # One flattened to_blocked covers all experts, bitwise equal
                # to a per-expert stack (see _row_scaled_single_grouped_gemm).
                scale_b=[
                    to_blocked(w_scales.reshape(-1, w_scales.shape[-1])).view(
                        num_experts, -1
                    ),
                    w_global,
                ],
                scale_recipe_b=_SCALE_RECIPE,
                swizzle_a=_SWIZZLE,
                swizzle_b=_SWIZZLE,
                offs=padded_group_end_offsets,
                output_dtype=torch.bfloat16,
            )

        if pad_token_groups_for_grouped_mm:
            output = unpad_token_groups(
                output,
                group_end_offsets,
                padded_group_start_offsets,
                num_tokens,
                alignment_size=_ALIGNMENT,
                kernel_preference=KernelPreference.EMULATED,
            )

        if backward_override == "high_precision":
            ctx.save_for_backward(original_input, weight, group_end_offsets)
        else:
            if padded_group_start_offsets is None:
                padded_group_start_offsets = group_end_offsets.new_zeros(0)
            ctx.save_for_backward(
                x_codes,
                x_scales,
                x_amax,
                w_codes,
                w_scales,
                weight_amax,
                group_end_offsets,
                padded_group_start_offsets,
            )
        ctx.backward_override = backward_override
        ctx.e4m3_scale_bound = e4m3_scale_bound
        ctx.pad_token_groups_for_grouped_mm = pad_token_groups_for_grouped_mm
        ctx.num_tokens = num_tokens
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.to(torch.bfloat16).contiguous()

        if ctx.backward_override == "high_precision":
            input_act, weight, group_end_offsets = ctx.saved_tensors
        else:
            (
                x_codes,
                x_scales,
                x_amax,
                w_codes,
                w_scales,
                weight_amax,
                group_end_offsets,
                padded_group_start_offsets,
            ) = ctx.saved_tensors
            input_act = nvfp4_dequantize(
                x_codes, x_scales, x_amax, e4m3_scale_bound=ctx.e4m3_scale_bound
            )
            if ctx.pad_token_groups_for_grouped_mm:
                input_act = unpad_token_groups(
                    input_act,
                    group_end_offsets,
                    padded_group_start_offsets,
                    ctx.num_tokens,
                    alignment_size=_ALIGNMENT,
                    kernel_preference=KernelPreference.EMULATED,
                )
            weight = _dequantize_expert_weights(
                w_codes, w_scales, weight_amax, ctx.e4m3_scale_bound
            )

        grad_input = torch._grouped_mm(
            grad_output,
            weight,
            offs=group_end_offsets,
            out_dtype=torch.bfloat16,
        )
        grad_weight = torch._grouped_mm(
            grad_output.transpose(-2, -1),
            input_act,
            offs=group_end_offsets,
            out_dtype=torch.bfloat16,
        )
        return grad_input, grad_weight, None, None, None, None, None, None, None
