# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
NVFP4 training linear layer with quantized forward and backward passes.

Modeled on mx_linear.py (MXFP8 training), this implements an autograd function
that quantizes all three GEMMs in a Linear layer to NVFP4:

    Forward:  input @ weight^T = output
    Backward: grad_output @ weight = grad_input
    Backward: input^T @ grad_output = grad_weight

The implementation uses TorchAO's pure-Triton RHT and stochastic-rounding path.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_cutedsl import (
    cutedsl_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
    triton_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    CUTEDSL_NVFP4_REQUIREMENTS,
    cutedsl_nvfp4_kernels_available,
    cutedsl_nvfp4_unavailable_reason,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_cutedsl import (
    cutedsl_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_triton import (
    triton_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.quantize_2d_cutedsl import (
    cutedsl_weight_quantize_2d,
)
from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
    triton_weight_quantize_2d,
)
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


def _rht_amax(x: torch.Tensor, sign_vector_list: list[int], use_cutedsl: bool):
    """(col_amax, row_amax) global amaxes on the selected backend."""
    if use_cutedsl:
        return cutedsl_rht_amax(x, sign_vector_list)
    return triton_rht_amax(x, sign_vector=sign_vector_list)


def _rht_quantize_row_col(
    x: torch.Tensor,
    col_amax: torch.Tensor,
    row_amax: torch.Tensor,
    sign_vector_list: list[int],
    use_cutedsl: bool,
    sr_seed: Optional[torch.Tensor] = None,
):
    """Fused columnwise-RHT + rowwise NVFP4 quantize on the selected backend.

    RTNE when sr_seed is None; stochastic rounding otherwise, with fresh offsets
    drawn from the default CUDA RNG (a first-class CUDA-graph side input — see
    the nvfp4_mm_triton docstring). CuteDSL derives its col/row streams from a
    single (seed, offset) base internally; Triton takes separate col/row bases.
    """
    if sr_seed is None:
        if use_cutedsl:
            return cutedsl_rht_quantize_row_col(x, col_amax, row_amax, sign_vector_list)
        return triton_rht_quantize_row_col(
            x,
            stochastic_rounding=False,
            sign_vector=sign_vector_list,
            col_global_amax=col_amax,
            row_global_amax=row_amax,
        )
    offset_col = torch.randint(0, 2**32, (1,), dtype=torch.int64, device=x.device)
    if use_cutedsl:
        return cutedsl_rht_quantize_row_col(
            x,
            col_amax,
            row_amax,
            sign_vector_list,
            stochastic_rounding=True,
            seed=sr_seed,
            offset=offset_col,
        )
    offset_row = torch.randint(0, 2**32, (1,), dtype=torch.int64, device=x.device)
    return triton_rht_quantize_row_col(
        x,
        stochastic_rounding=True,
        sign_vector=sign_vector_list,
        col_seed_base=sr_seed,
        row_seed_base=sr_seed ^ 1,
        col_offset_base=offset_col,
        row_offset_base=offset_row,
        col_global_amax=col_amax,
        row_global_amax=row_amax,
    )


def _weight_quantize_2d(x: torch.Tensor, use_cutedsl: bool):
    """2D NVFP4 weight quantization producing both rowwise and colwise outputs.

    Returns (W_fp4_x2, W_bs, W_gs, Wt_fp4_x2, Wt_sf, W_amax) where:
      W_*  = rowwise quantized x (for forward GEMM)
      Wt_* = colwise quantized x = rowwise quantized x.T (for dgrad GEMM)

    use_cutedsl selects the CuteDSL kernel (plain transpose-quantize via an identity Hadamard,
    canonical 1x16 scales) over the Triton 2D weight kernel. Neither path applies RHT or SR.
    """
    global_amax = x.float().abs().max()
    quantize = cutedsl_weight_quantize_2d if use_cutedsl else triton_weight_quantize_2d
    codes, sf, t_codes, t_sf = quantize(x, global_amax)
    return (
        codes.view(torch.float4_e2m1fn_x2),
        sf.flatten(),
        per_tensor_amax_to_scale(global_amax),
        t_codes.view(torch.float4_e2m1fn_x2),
        t_sf,
        global_amax,
    )


@torch._dynamo.allow_in_graph
class nvfp4_mm_triton(torch.autograd.Function):
    """NVFP4 quantized matmul: pure-triton RHT + stochastic rounding path.

    3 GEMMs:
      forward:   x_row @ W.T  = output         (triton RHT rowwise + 2D weight)
      backward:  dy_sr @ W.T  = grad_input      (triton SR rowwise + 2D weight)
      backward:  dy_col.T @ x_col = grad_weight (triton col RHT + SR for dy; saved col for x)

    Requires: bfloat16 input, M % 128 == 0, K % 128 == 0, N % 128 == 0.
    Saves only FP4 codes+scales for backward (memory efficient vs full-precision activations).

    sr_seed is a single fixed buffer giving the Philox key. Backward generates fresh
    offset_base values via torch.randint (default CUDA RNG, no generator= arg). Under
    torch.compile(mode="reduce-overhead") the default CUDA generator is a first-class
    CUDA graph side input: the framework advances it between replays, giving different
    SR noise each backward step without save_for_backward or external counter plumbing.
    """

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        bias: Optional[torch.Tensor],
        sr_seed: torch.Tensor,
        sign_vector: tuple[int, ...] | list[int],
        use_cutedsl: bool = False,
    ):
        sign_vector = tuple(sign_vector)
        sign_vector_list = list(sign_vector)
        M = input_hp.shape[-2]
        K = input_hp.shape[-1]
        N = weight_hp.shape[0]
        if input_hp.dtype != torch.bfloat16:
            input_hp = input_hp.to(torch.bfloat16)
        if weight_hp.dtype != torch.bfloat16:
            weight_hp = weight_hp.to(torch.bfloat16)
        if M % 128 != 0 or K % 128 != 0 or N % 128 != 0:
            raise ValueError(
                f"nvfp4_mm_triton requires M, K, N all divisible by 128; "
                f"got M={M}, K={K}, N={N}"
            )
        if use_cutedsl and M % 256 != 0:
            # The outer M % 128 gate is too coarse for CuteDSL: M=128 reaches the amax
            # kernel and silently returns zero amaxes.
            raise ValueError(
                f"kernel_preference=CUTEDSL requires M divisible by 256, got M={M}"
            )
        if use_cutedsl and N % 256 != 0:
            # The CuteDSL weight quantize maps the weight as (out=N, in=K) into the fused kernel,
            # whose M tiler needs out_features % 256 (stricter than the Triton weight kernel's %128).
            raise ValueError(
                f"kernel_preference=CUTEDSL requires N (out_features) divisible by 256, got N={N}"
            )
        input_2d = input_hp.reshape(-1, K).contiguous()

        # Amaxes are separate ops so TP callers can all-reduce them before quantizing.
        x_col_amax, x_row_amax = _rht_amax(input_2d, sign_vector_list, use_cutedsl)
        x_col_codes, x_col_sf, x_row_codes, x_row_sf = _rht_quantize_row_col(
            input_2d, x_col_amax, x_row_amax, sign_vector_list, use_cutedsl
        )

        # Fused weight quantization: rowwise for forward GEMM, colwise saved for dgrad
        (
            W_fp4_x2,
            W_bs,
            W_gs,
            Wt_fp4_x2,
            Wt_sf,
            W_amax,
        ) = _weight_quantize_2d(weight_hp, use_cutedsl)
        x_gs = per_tensor_amax_to_scale(x_row_amax)

        output = torch.nn.functional.scaled_mm(
            x_row_codes.view(torch.float4_e2m1fn_x2),
            W_fp4_x2.t(),
            scale_a=[x_row_sf.flatten(), x_gs],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[W_bs, W_gs],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )
        output = output.reshape(*input_hp.shape[:-1], N)
        if bias is not None:
            output = output + bias

        ctx.save_for_backward(
            x_col_codes,
            x_col_sf,
            x_col_amax,
            Wt_fp4_x2,
            Wt_sf,
            W_amax,
            sr_seed,
        )
        ctx.input_orig_shape = input_hp.shape
        ctx.has_bias = bias is not None
        ctx.sign_vector = sign_vector
        ctx.use_cutedsl = use_cutedsl
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x_col_codes,
            x_col_sf,
            x_col_amax,
            Wt_fp4_x2,
            Wt_sf,
            W_amax,
            sr_seed,
        ) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        # Quantize grad_output with SR for GEMM 2 (dgrad, rowwise) and GEMM 3 (wgrad, col RHT).
        sign_vector_list = list(ctx.sign_vector)
        dy_col_amax, dy_row_amax = _rht_amax(
            grad_output_2d, sign_vector_list, ctx.use_cutedsl
        )
        dy_col_fp4, dy_col_sf, dy_row_fp4, dy_row_sf = _rht_quantize_row_col(
            grad_output_2d,
            dy_col_amax,
            dy_row_amax,
            sign_vector_list,
            ctx.use_cutedsl,
            sr_seed=sr_seed,
        )

        # -----------------------------------------------------------
        # GEMM 2: dy_sr @ W.T → grad_input  (SR rowwise; saved colwise W)
        # -----------------------------------------------------------
        dy_bs = dy_row_sf.flatten()
        Wt_bs = Wt_sf.flatten()
        dy_gs = per_tensor_amax_to_scale(dy_row_amax)
        Wt_gs = per_tensor_amax_to_scale(W_amax)
        grad_input = torch.nn.functional.scaled_mm(
            dy_row_fp4.view(torch.float4_e2m1fn_x2),
            Wt_fp4_x2.t(),
            scale_a=[dy_bs, dy_gs],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[Wt_bs, Wt_gs],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )
        grad_input = grad_input.reshape(ctx.input_orig_shape)

        # -----------------------------------------------------------
        # GEMM 3: dy_col.T @ x_col → grad_weight  (col RHT + SR)
        # -----------------------------------------------------------
        dy_row_amax_w = per_tensor_amax_to_scale(dy_col_amax)
        x_gs_w = per_tensor_amax_to_scale(x_col_amax)
        grad_weight = torch.nn.functional.scaled_mm(
            dy_col_fp4.view(torch.float4_e2m1fn_x2),
            x_col_codes.view(torch.float4_e2m1fn_x2).t(),
            scale_a=[dy_col_sf.flatten(), dy_row_amax_w],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[x_col_sf.flatten(), x_gs_w],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )

        grad_bias = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if ctx.has_bias
            else None
        )
        # Extra Nones: sr_seed, sign_vector, use_cutedsl
        return grad_input, grad_weight, grad_bias, None, None, None


def nvfp4_linear(
    input_hp: torch.Tensor,
    weight_hp: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    sign_vector: tuple[int, ...] | list[int],
    kernel_preference: KernelPreference = KernelPreference.TRITON,
    sr_seed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convenience wrapper around the nvfp4_mm_triton autograd function.

    Performs a quantized linear operation: output = input @ weight^T + bias,
    with NVFP4 quantization on forward and backward GEMMs.

    Args:
        input_hp: High precision input [..., in_features]
        weight_hp: High precision weight [out_features, in_features]
        bias: Optional bias [out_features]
        sign_vector: RHT sign vector used for amax and quantization.
        kernel_preference: Backend for quantization, TRITON (default) or CUTEDSL.
            CUTEDSL runs the full path on CuteDSL — the amax, the forward RTNE quantize, the
            backward SR (cvt.rs) quantize, and the 2D weight quantize (requires out_features % 256).
        sr_seed: Fixed int64 seed tensor (size=(1,)) for SR Philox key. Allocated
            fresh if None. For reproducibility, pass a pre-allocated module buffer.
    """
    if kernel_preference not in (KernelPreference.TRITON, KernelPreference.CUTEDSL):
        raise ValueError(
            "NVFP4 training linear only supports kernel_preference TRITON or CUTEDSL, "
            f"got {kernel_preference!r}"
        )
    use_cutedsl = kernel_preference == KernelPreference.CUTEDSL
    if use_cutedsl and not cutedsl_nvfp4_kernels_available():
        raise RuntimeError(
            f"kernel_preference=CUTEDSL requires {CUTEDSL_NVFP4_REQUIREMENTS} "
            f"({cutedsl_nvfp4_unavailable_reason()})."
        )

    if sr_seed is None:
        sr_seed = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=input_hp.device
        )
    return nvfp4_mm_triton.apply(
        input_hp,
        weight_hp,
        bias,
        sr_seed,
        sign_vector,
        use_cutedsl,
    )
