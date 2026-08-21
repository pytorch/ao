# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Differentiable NVFP4 grouped GEMM for MoE training."""

from typing import Optional

import torch
import torch.nn.functional as F

from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_cutedsl import (
    cutedsl_group_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
    triton_group_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
    _DEVICE_ASSERTS,
    VARYING_FIRST_DIM,
)
from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_cutedsl import (
    cutedsl_group_weight_quantize_2d,
)
from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_triton import (
    triton_group_weight_quantize_2d,
)
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_cutedsl import (
    cutedsl_group_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_triton import (
    triton_group_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.group_weight_amax_triton import (
    triton_group_weight_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    CUTEDSL_NVFP4_REQUIREMENTS,
    cutedsl_nvfp4_kernels_available,
    cutedsl_nvfp4_unavailable_reason,
)
from torchao.prototype.moe_training.utils import (
    conditional_nostrict_trace,
    pad_token_groups,
    unpad_token_groups,
)
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale
from torchao.quantization.quantize_.common import KernelPreference
from torchao.utils import is_sm_at_least_100

_ALIGNMENT = 128
_SCALE_RECIPE = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
_SWIZZLE = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]


def _resolve_backends(
    kernel_preference: KernelPreference,
    num_experts: int,
) -> tuple[bool, bool]:
    """``(use_cutedsl_rht, use_cutedsl_weight)`` for this call.

    Under AUTO each op independently takes CuteDSL where it can run and Triton
    otherwise, so a missing runtime or an unsupported shape costs speed rather than
    failing. CUTEDSL is the same choice made loudly: an explicit request that cannot
    be honored raises instead of silently running the slower path.
    """
    if kernel_preference not in (
        KernelPreference.AUTO,
        KernelPreference.TRITON,
        KernelPreference.CUTEDSL,
    ):
        raise ValueError(
            "NVFP4 grouped GEMM only supports kernel_preference AUTO, TRITON, or "
            f"CUTEDSL, got {kernel_preference!r}"
        )
    if kernel_preference == KernelPreference.TRITON:
        return False, False

    if not cutedsl_nvfp4_kernels_available():
        if kernel_preference == KernelPreference.CUTEDSL:
            raise RuntimeError(
                f"kernel_preference=CUTEDSL requires {CUTEDSL_NVFP4_REQUIREMENTS} "
                f"({cutedsl_nvfp4_unavailable_reason()})."
            )
        return False, False

    # Imported here rather than at module scope: _cutedsl_group_kernels_impl pulls in
    # cutlass at import time, so it is only reachable once availability is known.
    from torchao.prototype.moe_training.nvfp4_training._cutedsl_group_kernels_impl import (
        MAX_GROUPS,
    )

    use_cutedsl_rht = num_experts <= MAX_GROUPS
    if kernel_preference == KernelPreference.CUTEDSL and not use_cutedsl_rht:
        raise ValueError(
            f"kernel_preference=CUTEDSL requires at most {MAX_GROUPS} experts, "
            f"got {num_experts}"
        )
    # The weight quantize accepts every shape the grouped GEMM does (out_features
    # % 128, enforced by the caller), so it takes CuteDSL whenever the runtime is
    # present. Only the RHT path has a constraint of its own.
    return use_cutedsl_rht, True


@conditional_nostrict_trace
def _to_nvfp4_rht_rs_then_scaled_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    sign_vector: tuple[int, ...] | list[int],
    sr_seed: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    pad_token_groups_for_grouped_mm: bool = False,
    kernel_preference: KernelPreference = KernelPreference.AUTO,
    use_fast_math: bool = True,
) -> torch.Tensor:
    """Quantize and multiply grouped activations and expert weights.

    ``A`` has shape ``(M, K)``, ``B`` has shape ``(E, N, K)``, and ``offs``
    contains the cumulative row-end offset for each expert. The caller owns the
    persistent RHT sign vector and stochastic-rounding seed.

    ``kernel_preference`` selects the quantization backend. AUTO (default) runs each
    op on CuteDSL where it can and falls back to Triton per op; TRITON forces Triton;
    CUTEDSL demands CuteDSL and raises if it cannot run. The per-expert weight amax
    is Triton on every path -- it has no CuteDSL twin.

    ``use_fast_math`` selects the RHT quantize variant that consumes the FP32 accumulator
    directly and takes an approximate reciprocal, matching TransformerEngine under
    ``NVTE_USE_FAST_MATH=1``. It is worth about 25% of the quantize stage at production
    shapes, which is why it defaults on. Both backends implement it and both stay bitwise
    identical to TE and to each other, so AUTO and TRITON agree either way. The per-expert
    weight amax and the 2D weight quantize are unaffected: without an RHT there is no
    accumulator round-through to skip, which is also why TE has no 2D fast path.
    """
    output = _NVFP4GroupedMM.apply(
        A,
        B,
        sign_vector,
        sr_seed,
        offs,
        pad_token_groups_for_grouped_mm,
        kernel_preference,
        use_fast_math,
    )
    if bias is not None:
        output = output + bias.to(output.dtype)
    return output


class _NVFP4GroupedMM(torch.autograd.Function):
    """NVFP4 grouped forward, dgrad, and wgrad on the selected quantization backend."""

    @staticmethod
    def forward(
        ctx,
        input_act: torch.Tensor,
        weight: torch.Tensor,
        sign_vector: tuple[int, ...] | list[int],
        sr_seed: torch.Tensor,
        group_end_offsets: Optional[torch.Tensor],
        pad_token_groups_for_grouped_mm: bool,
        kernel_preference: KernelPreference,
        use_fast_math: bool,
    ) -> torch.Tensor:
        if input_act.ndim != 2:
            raise ValueError(f"input_act must be 2D, got {input_act.ndim}D")
        if weight.ndim != 3:
            raise ValueError(f"weight must be 3D, got {weight.ndim}D")
        if not isinstance(sign_vector, (tuple, list)) or len(sign_vector) != 16:
            raise ValueError("sign_vector must be a tuple or list with 16 elements")
        if any(sign not in (-1, 1) for sign in sign_vector):
            raise ValueError("sign_vector elements must be -1 or 1")
        if group_end_offsets is None:
            raise ValueError("offs is required for NVFP4 grouped GEMM")
        if group_end_offsets.ndim != 1 or group_end_offsets.dtype != torch.int32:
            raise ValueError("offs must be a 1D int32 tensor")
        if not group_end_offsets.is_contiguous():
            raise ValueError("offs must be contiguous")
        if group_end_offsets.numel() != weight.shape[0]:
            raise ValueError("offs must contain one group-end offset per expert")
        if sr_seed.ndim != 1 or sr_seed.numel() != 1:
            raise ValueError("sr_seed must be a one-element tensor")
        if sr_seed.dtype != torch.int64 or not sr_seed.is_cuda:
            raise ValueError("sr_seed must be a CUDA int64 tensor")
        if not (input_act.is_cuda and weight.is_cuda and group_end_offsets.is_cuda):
            raise ValueError("input_act, weight, and offs must be CUDA tensors")
        if not (
            input_act.device
            == weight.device
            == group_end_offsets.device
            == sr_seed.device
        ):
            raise ValueError("all tensor arguments must be on the same device")
        if not is_sm_at_least_100():
            raise NotImplementedError("NVFP4 grouped training GEMM requires SM100+")

        num_tokens, K = input_act.shape
        num_experts, N, weight_K = weight.shape
        if weight_K != K:
            raise ValueError(
                f"input and weight contraction dimensions differ: {K} and {weight_K}"
            )
        if K % _ALIGNMENT != 0 or N % _ALIGNMENT != 0:
            raise ValueError(
                f"K and N must be divisible by {_ALIGNMENT}; got K={K}, N={N}"
            )
        if _DEVICE_ASSERTS:
            group_sizes = torch.diff(
                group_end_offsets, prepend=group_end_offsets.new_zeros(1)
            )
            torch.ops.aten._assert_async.msg(
                torch.all(group_sizes > 0), "offs must describe non-empty groups"
            )
            torch.ops.aten._assert_async.msg(
                group_end_offsets[-1] <= num_tokens,
                "the final group-end offset must not exceed A.shape[0]",
            )
            if pad_token_groups_for_grouped_mm:
                torch.ops.aten._assert_async.msg(
                    group_end_offsets[-1] == num_tokens,
                    "internally padded input offsets must cover A.shape[0]",
                )
            if not pad_token_groups_for_grouped_mm:
                torch.ops.aten._assert_async.msg(
                    torch.all(group_sizes % _ALIGNMENT == 0),
                    "every token group must be 128-row aligned when padding is disabled",
                )

        use_cutedsl_rht, use_cutedsl_weight = _resolve_backends(
            kernel_preference, num_experts
        )

        input_act = input_act.to(torch.bfloat16).contiguous()
        # The 2D quantizer consumes logical W (E, N, K), then produces rowwise W
        # and rowwise W.T codes with the scale layouts required by each GEMM.
        weight = weight.to(torch.bfloat16).contiguous()
        sign_vector = tuple(sign_vector)
        sign_vector_list = list(sign_vector)

        padded_group_start_offsets = None
        if pad_token_groups_for_grouped_mm:
            input_act, padded_group_start_offsets, padded_group_end_offsets = (
                pad_token_groups(
                    input_act,
                    group_end_offsets,
                    alignment_size=_ALIGNMENT,
                    kernel_preference=KernelPreference.TRITON,
                )
            )
        else:
            padded_group_end_offsets = group_end_offsets

        packed_sequence_length = input_act.shape[0]
        logical_packed_length = padded_group_end_offsets[-1:]
        group_rht_amax = (
            cutedsl_group_rht_amax if use_cutedsl_rht else triton_group_rht_amax
        )
        group_rht_quantize_row_col = (
            cutedsl_group_rht_quantize_row_col
            if use_cutedsl_rht
            else triton_group_rht_quantize_row_col
        )
        x_col_amax, x_row_amax = group_rht_amax(
            input_act,
            sign_vector_list,
            padded_group_end_offsets,
            num_experts,
            packed_sequence_length,
            K,
            VARYING_FIRST_DIM,
            logical_packed_length=logical_packed_length,
        )
        x_row_codes, x_row_sf, x_col_codes, x_col_sf = group_rht_quantize_row_col(
            input_act,
            sign_vector_list,
            padded_group_end_offsets,
            num_experts,
            packed_sequence_length,
            K,
            VARYING_FIRST_DIM,
            x_row_amax,
            x_col_amax,
            rng_state=None,
            enable_stochastic_rounding=False,
            logical_packed_length=logical_packed_length,
            use_fast_math=use_fast_math,
        )

        weight_amax = triton_group_weight_amax(weight, num_experts)
        group_weight_quantize_2d = (
            cutedsl_group_weight_quantize_2d
            if use_cutedsl_weight
            else triton_group_weight_quantize_2d
        )
        weight_codes, weight_sf, weight_t_codes, weight_t_sf = group_weight_quantize_2d(
            weight, weight_amax, num_experts
        )
        output = F.scaled_grouped_mm(
            x_row_codes.view(torch.float4_e2m1fn_x2),
            # Transpose rowwise W codes to the grouped-GEMM RHS layout (E, K, N).
            weight_codes.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
            scale_a=[x_row_sf, per_tensor_amax_to_scale(x_row_amax)],
            scale_recipe_a=_SCALE_RECIPE,
            scale_b=[weight_sf.flatten(1), per_tensor_amax_to_scale(weight_amax)],
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
                kernel_preference=KernelPreference.TRITON,
            )

        ctx.save_for_backward(
            x_col_codes,
            x_col_sf,
            x_col_amax,
            weight_t_codes,
            weight_t_sf,
            weight_amax,
            group_end_offsets,
            padded_group_start_offsets,
            padded_group_end_offsets,
            sr_seed,
        )
        ctx.pad_token_groups_for_grouped_mm = pad_token_groups_for_grouped_mm
        ctx.num_tokens = num_tokens
        ctx.sign_vector = sign_vector
        ctx.use_cutedsl_rht = use_cutedsl_rht
        ctx.use_fast_math = use_fast_math
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x_col_codes,
            x_col_sf,
            x_col_amax,
            weight_t_codes,
            weight_t_sf,
            weight_amax,
            original_group_end_offsets,
            padded_group_start_offsets,
            padded_group_end_offsets,
            sr_seed,
        ) = ctx.saved_tensors

        grad_output = grad_output.to(torch.bfloat16).contiguous()
        if ctx.pad_token_groups_for_grouped_mm:
            grad_output, _, _ = pad_token_groups(
                grad_output,
                original_group_end_offsets,
                alignment_size=_ALIGNMENT,
                kernel_preference=KernelPreference.TRITON,
            )

        num_experts = padded_group_end_offsets.numel()
        packed_sequence_length, N = grad_output.shape
        logical_packed_length = padded_group_end_offsets[-1:]
        sign_vector_list = list(ctx.sign_vector)
        group_rht_amax = (
            cutedsl_group_rht_amax if ctx.use_cutedsl_rht else triton_group_rht_amax
        )
        group_rht_quantize_row_col = (
            cutedsl_group_rht_quantize_row_col
            if ctx.use_cutedsl_rht
            else triton_group_rht_quantize_row_col
        )
        dy_col_amax, dy_row_amax = group_rht_amax(
            grad_output,
            sign_vector_list,
            padded_group_end_offsets,
            num_experts,
            packed_sequence_length,
            N,
            VARYING_FIRST_DIM,
            logical_packed_length=logical_packed_length,
        )

        col_offset = torch.randint(
            0, 2**32, (1,), dtype=torch.int64, device=grad_output.device
        )
        row_offset = torch.randint(
            0, 2**32, (1,), dtype=torch.int64, device=grad_output.device
        )
        rng_state = torch.cat((sr_seed, col_offset, sr_seed ^ 1, row_offset))
        dy_row_codes, dy_row_sf, dy_col_codes, dy_col_sf = group_rht_quantize_row_col(
            grad_output,
            sign_vector_list,
            padded_group_end_offsets,
            num_experts,
            packed_sequence_length,
            N,
            VARYING_FIRST_DIM,
            dy_row_amax,
            dy_col_amax,
            rng_state,
            enable_stochastic_rounding=True,
            logical_packed_length=logical_packed_length,
            use_fast_math=ctx.use_fast_math,
        )

        grad_input = F.scaled_grouped_mm(
            dy_row_codes.view(torch.float4_e2m1fn_x2),
            # Transpose rowwise W.T codes to the dgrad RHS layout (E, N, K).
            weight_t_codes.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
            scale_a=[dy_row_sf, per_tensor_amax_to_scale(dy_row_amax)],
            scale_recipe_a=_SCALE_RECIPE,
            scale_b=[weight_t_sf.flatten(1), per_tensor_amax_to_scale(weight_amax)],
            scale_recipe_b=_SCALE_RECIPE,
            swizzle_a=_SWIZZLE,
            swizzle_b=_SWIZZLE,
            offs=padded_group_end_offsets,
            output_dtype=torch.bfloat16,
        )
        grad_weight = F.scaled_grouped_mm(
            dy_col_codes.view(torch.float4_e2m1fn_x2),
            x_col_codes.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
            scale_a=[dy_col_sf, per_tensor_amax_to_scale(dy_col_amax)],
            scale_recipe_a=_SCALE_RECIPE,
            scale_b=[x_col_sf, per_tensor_amax_to_scale(x_col_amax)],
            scale_recipe_b=_SCALE_RECIPE,
            swizzle_a=_SWIZZLE,
            swizzle_b=_SWIZZLE,
            offs=padded_group_end_offsets,
            output_dtype=torch.bfloat16,
        )
        if ctx.pad_token_groups_for_grouped_mm:
            grad_input = unpad_token_groups(
                grad_input,
                original_group_end_offsets,
                padded_group_start_offsets,
                ctx.num_tokens,
                alignment_size=_ALIGNMENT,
                kernel_preference=KernelPreference.TRITON,
            )

        # One gradient per forward input: input_act, weight, sign_vector, sr_seed,
        # group_end_offsets, pad_token_groups_for_grouped_mm, kernel_preference,
        # use_fast_math.
        return grad_input, grad_weight, None, None, None, None, None, None
