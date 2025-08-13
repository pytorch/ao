# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Tuple

import torch

from torchao.float8.config import ScalingGranularity
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.conversion_utils import MoEScalingType
from torchao.prototype.moe_training.kernels import (
    triton_fp8_per_group_colwise_scales,
    triton_fp8_per_group_rowwise_scales,
    triton_fp8_rowwise_3d_transpose_rhs,
)
from torchao.prototype.moe_training.utils import (
    _is_column_major,
    _to_mxfp8_per_group_colwise,
    _to_mxfp8_per_group_rowwise,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx

logger: logging.Logger = logging.getLogger(__name__)


def _scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    scaling_type: MoEScalingType = MoEScalingType.FP8_ROWWISE,
) -> torch.Tensor:
    """
    This function performs dynamic float8 quantization with row-wise scaling
    on the input tensors A and B, then performs a scaled grouped GEMM and returns the results.

    Args:
        A (bf16/float32 torch.Tensor): The first high-precision input tensor, which must be a 2D tensor of shape (M * num_groups, K)
            and in row-major memory layout.
        B_t (bf16/float32 torch.Tensor): The second high-precision input tensor which must be 3D, which must be shape (E, K, N)
            and in column-major memory layout.
        offs (int32 torch.Tensor): The offsets to use to mark the starting index of each group along dim0 of the A tensor.
        out_dtype (Optional[torch.dtype]): The dtype of the output tensor. Currently only torch.bfloat16 is supported.
    """
    # TODO: Remove logging once prototype is more mature. This is currently very useful for development and debugging.
    if scaling_type == MoEScalingType.FP8_ROWWISE:
        # print("Using fp8 rowwise scaled_grouped_mm")
        return _Float8GroupedMM.apply(
            A,
            B_t,
            offs,
            out_dtype,
        )
    elif scaling_type == MoEScalingType.MXFP8:
        print("Using mxfp8 scaled_grouped_mm")
        block_size = 32  # TODO: should we make this configurable? plumb it through in a config somehow?
        return _MXFP8GroupedMM.apply(
            A,
            B_t,
            offs,
            block_size,
            out_dtype,
        )
    else:
        raise ValueError(f"Unsupported scaling type {scaling_type}")


class _Float8GroupedMM(torch.autograd.Function):
    """Differentiable implementation of grouped GEMM with dynamic float8 quantization."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> torch.Tensor:
        # torchao _scaled_grouped_mm only supports A=2D|3D and B=3D.
        assert A.ndim == 2 or A.ndim == 3, "A must be 2D or 3D"
        assert B_t.ndim == 3, "B must be 3D"

        assert A.size(-1) % 16 == 0, (
            f"A must have a last dim divisible by 16, but got shape: {A.shape}"
        )
        assert B_t.size(-2) % 16 == 0 and B_t.size(-1) % 16 == 0, (
            f"B must have last 2 dims divisible by 16, but got shape: {B_t.shape}"
        )

        # Assert input tensors are in high-precision dtypes.
        assert A.dtype == torch.float32 or A.dtype == torch.bfloat16, (
            "A must be float32 or bfloat16"
        )
        assert B_t.dtype == torch.float32 or B_t.dtype == torch.bfloat16, (
            "B must be float32 or bfloat16"
        )
        assert offs is None or offs.dtype == torch.int32, (
            "offs must be int32 tensor or None"
        )

        # Assert A and B dims are compatible for a scaled grouped GEMM.
        assert A.size(-1) == B_t.size(-2), (
            f"shape {A.shape} and {B_t.shape} are not compatible for _scaled_grouped_mm"
        )

        # The left operand in the scaled grouped GEMM must be row-major due to hardware requirements.
        assert not _is_column_major(A), "A must be row-major"

        # Due to hardware requirements, the right operand in a scaled grouped GEMM must be column-major.
        assert _is_column_major(B_t), "B must be column-major"

        # Convert high precision input tensor to float8, row-major for left operand of grouped GEMM.
        # A shape: (M, K) or (B, M, K)
        # A_scales shape: (M,1) or (B, M, 1)
        A_scales = tensor_to_scale(
            A,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        A_scaled = A.to(torch.float32) * A_scales
        A_fp8_row_major = to_fp8_saturated(A_scaled, torch.float8_e4m3fn)

        # Convert B to float8, column-major for right operand of grouped GEMM.
        # B_t shape: (E, K, N)
        # B_t scales must be computed rowwise keeping the outer/final dim, so:
        # B_t_scales shape: (E, 1, N)
        B_t_scales = tensor_to_scale(
            B_t,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-2,
            round_scales_to_power_of_2=True,
        )
        B_t_scaled = B_t.to(torch.float32) * B_t_scales
        B_t_fp8_col_major = to_fp8_saturated(B_t_scaled, torch.float8_e4m3fn)

        # Store what we need for backward.
        ctx.save_for_backward(A, B_t, offs)
        ctx.out_dtype = out_dtype

        # Perform scaled grouped GEMM and return result.
        # output shape: scaled grouped mm of (M,K) @ (B,K,N) = (M,N)
        assert not _is_column_major(A_fp8_row_major), (
            "A must be row-major for output = A @ B"
        )
        assert _is_column_major(B_t_fp8_col_major), (
            "B must be column-major for output = A @ B"
        )

        # Squeeze empty dims out of scales, to comply with grouped mm API.
        # A_scales shape: (M,1) or (B, M, 1)
        # B_t_scales shape: (E, 1, N)
        A_scales = A_scales.squeeze(-1)
        B_t_scales = B_t_scales.squeeze(1)
        return torch._scaled_grouped_mm(
            A_fp8_row_major,
            B_t_fp8_col_major,
            A_scales.reciprocal(),  # Reciprocals are needed for rescaling the output.
            B_t_scales.reciprocal(),
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        A, B_t, offs = ctx.saved_tensors
        out_dtype = ctx.out_dtype

        # Convert grad_output to float8, row-major for left operand of grouped GEMM
        # needed for grad_A: grad_output @ B
        #
        # grad_output shape: (M, N)
        # grad_output_scale shape: (M, 1)
        grad_output_scales = tensor_to_scale(
            grad_output,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        grad_output_scaled = grad_output.to(torch.float32) * grad_output_scales
        grad_output_fp8_row_major = to_fp8_saturated(
            grad_output_scaled, torch.float8_e4m3fn
        )

        # Compute B fp8 column-major for right operand of grouped GEMM:
        # grad_A = grad_output @ B.
        B_fp8_col_major, B_scales = triton_fp8_rowwise_3d_transpose_rhs(
            B_t._data if hasattr(B_t, "_data") else B_t,
            output_dtype=torch.float8_e4m3fn,
            round_scales_to_power_of_2=True,
        )

        # Compute grad_A.
        # grad_A = grad_output @ B
        # grad_A = scaled grouped mm of (M,N) @ (B,N,K) = (M,K)
        assert not _is_column_major(grad_output_fp8_row_major), (
            "grad_output must be row-major for grad_A = grad_output @ B"
        )
        assert _is_column_major(B_fp8_col_major), (
            "B must be column-major for grad_A = grad_output @ B"
        )

        # Squeeze empty dims out of scales, to comply with grouped mm API.
        # grad_output_scales shape: (M,1) or (B, M, 1)
        # B_scales shape: (E, 1, N)
        grad_output_scales = grad_output_scales.squeeze(-1)
        B_scales = B_scales.squeeze(1)
        grad_A = torch._scaled_grouped_mm(
            grad_output_fp8_row_major,
            B_fp8_col_major,
            grad_output_scales.reciprocal(),
            B_scales.reciprocal(),
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

        # grad_B is a special case. both operands of the grouped gemm will be 2D with offsets determing the "groups."
        # Compute scales for grad_output_t and A, which are both 2D tensors with offsets which define the "jagged" groups.

        # Convert transpose of grad_output to float8, row-major for left operand of grouped GEMM
        # needed for grad_B: grad_output_t @ A
        grad_output_t_fp8_row_major, grad_output_t_scales = (
            triton_fp8_per_group_rowwise_scales(
                grad_output.transpose(-2, -1),
                offs,
                torch.float8_e4m3fn,
                round_scales_to_power_of_2=True,
            )
        )

        A_fp8_col_major, A_scales = triton_fp8_per_group_colwise_scales(
            A,
            offs,
            torch.float8_e4m3fn,
            round_scales_to_power_of_2=True,
        )

        # Compute grad_B = grad_output_t @ A.
        # grad_B = grad_output_t @ A
        assert not _is_column_major(grad_output_t_fp8_row_major), (
            "grad_output_t must be row-major for grad_B = grad_output_t @ A"
        )
        assert _is_column_major(A_fp8_col_major), (
            "A must be column-major for grad_B = grad_output_t @ A"
        )

        # Per-token group scales computed via triton kernels above do not have
        # the empty dim like the scales computed via tensor_to_scale, so we need
        # don't need to squeeze here.
        grad_B = torch._scaled_grouped_mm(
            grad_output_t_fp8_row_major,
            A_fp8_col_major,
            grad_output_t_scales.reciprocal(),
            A_scales.reciprocal(),
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )
        return grad_A, grad_B.transpose(-2, -1), None, None, None, None


class _MXFP8GroupedMM(torch.autograd.Function):
    """Differentiable implementation of grouped GEMM with dynamic mxpf8 quantization."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: Optional[torch.Tensor] = None,
        block_size: int = 32,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
        emulated: bool = True,
    ) -> torch.Tensor:
        # torchao _scaled_grouped_mm only supports A=2D and B=3D.
        assert A.ndim == 2, "A must be 2D"
        assert B_t.ndim == 3, "B must be 3D"
        assert block_size == 32, "Only block_size=32 is supported"
        assert emulated, "Only emulated mxfp8 grouped gemm is supported"

        # Cast to mxpf8 across dim -1.
        # A_mx shape: (M, K)
        # A_scale shape: (M, K//block_size)
        A_scale, A_mx = to_mx(A, elem_dtype=torch.float8_e4m3fn, block_size=block_size)

        # Cast B_t per-expert to mxfp8 across dim1.
        # B_t_mx shape: (E, K, N)
        # B_t_scale shape: (E, K//block_size, N)
        B_t_scale, B_t_mx = _to_mxfp8_3d_expert_weights_dim1(B_t, block_size=block_size)

        # Store what we need for backward.
        ctx.save_for_backward(A, B_t, offs)
        ctx.block_size = block_size
        ctx.out_dtype = out_dtype

        # Perform scaled grouped GEMM and return result.
        # output = input @ weight.T
        # output shape: (M, N)
        out = emulated_mxfp8_scaled_grouped_mm(
            A_mx,
            A_scale,
            B_t_mx,
            B_t_scale,
            offs=offs,
            block_size=block_size,
            out_dtype=out_dtype,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        A, B_t, offs = ctx.saved_tensors
        block_size = ctx.block_size
        out_dtype = ctx.out_dtype
        # Compute grad_A.
        # grad_A = grad_output @ B
        # grad_A = scaled grouped mm of (M,N) @ (B,N,K) = (M,K)
        grad_out_scale, grad_out_mx = to_mx(
            grad_out, elem_dtype=torch.float8_e4m3fn, block_size=block_size
        )
        B_t_scale, B_t_mx = _to_mxfp8_3d_expert_weights_dim1(
            B_t.transpose(-2, -1).contiguous(),
            block_size=block_size,
            elem_dtype=torch.float8_e4m3fn,
        )
        grad_A = emulated_mxfp8_scaled_grouped_mm(
            grad_out_mx,
            grad_out_scale,
            B_t_mx,
            B_t_scale,
            offs=offs,
            out_dtype=out_dtype,
        )
        # Compute grad_B = grad_output_t @ A
        grad_out_t_mx, grad_out_t_scale = _to_mxfp8_per_group_rowwise(
            grad_out.transpose(-2, -1).contiguous(),
            offs=offs,
            block_size=block_size,
        )
        A_mx, A_scale = _to_mxfp8_per_group_colwise(
            A,
            offs=offs,
            block_size=block_size,
        )
        grad_B = emulated_mxfp8_scaled_grouped_mm(
            grad_out_t_mx,
            grad_out_t_scale,
            A_mx,
            A_scale,
            offs=offs,
        )
        # In forward we receive pre-transposed weights B_t as input
        grad_B_t = grad_B.transpose(-2, -1)

        return grad_A, grad_B_t, None, None, None


def _to_mxfp8_3d_expert_weights_dim1(
    w_t: torch.Tensor,  # (num_experts, K, N)
    block_size: int = 32,
    elem_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a 3D tensor of shape (experts, K, N) to MXFP8 format along dim1.
    Args:
        x (torch.Tensor): Input tensor to be converted.
        block_size (int): Block size for MXFP8 quantization.
        elem_dtype (torch.dtype): Element dtype for MXFP8 quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Converted tensor and scale tensor.
            - scale shape: (expets, K // block_size, N)
            - output shape: (experts, K, N)
    """
    # To cast B_t per-expert to mxfp8 across dim1, we transpose the experts, cast along dim -1, then untranspose.
    w_scale, w_mx = to_mx(
        w_t.transpose(-2, -1).contiguous(), elem_dtype=elem_dtype, block_size=block_size
    )
    w_t_scale, w_t_mx = w_scale.transpose(-2, -1), w_mx.transpose(-2, -1)
    return w_t_scale, w_t_mx


def emulated_mxfp8_scaled_grouped_mm(
    A_mx: torch.Tensor,
    A_scale: torch.Tensor,
    B_t_mx: torch.Tensor,
    B_t_scale: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    block_size: int = 32,
) -> torch.Tensor:
    if A_mx.ndim == 2 and B_t_mx.ndim == 3:
        return _emulated_mxfp8_scaled_grouped_mm_2d_3d(
            A_mx, A_scale, B_t_mx, B_t_scale, offs, out_dtype, block_size
        )
    elif A_mx.ndim == 2 and B_t_mx.ndim == 2:
        return _emulated_mxfp8_scaled_grouped_mm_2d_2d(
            A_mx, A_scale, B_t_mx, B_t_scale, offs, out_dtype, block_size
        )
    else:
        raise NotImplementedError


def _emulated_mxfp8_scaled_grouped_mm_2d_3d(
    A_mx: torch.Tensor,
    A_scale: torch.Tensor,
    B_t_mx: torch.Tensor,
    B_t_scale: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    block_size: int = 32,
) -> torch.Tensor:
    # Dequantize input
    # A_mx shape: (M, K)
    # A_scale shape: (M, K//block_size)
    A_orig_shape = A_mx.shape

    # Reshape to be able to do per-scaling group multiplication
    # A_mx shape: (M, K//block_size, block_size)
    # A_scale shape: (M, K//block_size, 1)
    A_mx = A_mx.reshape(*A_mx.shape[:-1], A_mx.shape[-1] // block_size, block_size)
    A_scale = A_scale.unsqueeze(-1)

    # Rescale and cast to bfloat16
    A = A_mx.to(torch.bfloat16) * A_scale.to(torch.bfloat16)

    # Reshape back to original shape
    # A shape: (M, K)
    A = A.reshape(A_orig_shape)

    # Dequantize weights
    # B_t_mx shape: (E, K, N)
    # B_t_scale shape: (E, K//block_size, N)
    E, K, N = B_t_mx.shape

    # Tranpose to get block_size on rightmost dim
    # B_mx shape: (E, N, K)
    # B_scale shape: (E, N, K//block_size)
    B_mx, B_scale = B_t_mx.transpose(-2, -1), B_t_scale.transpose(-2, -1)

    # Reshape to be able to do per-scaling group multiplication
    # B_mx shape: (E, N, K//block_size, block_size)
    # B_scale shape: (E, N, K//block_size, 1)
    B_mx = B_mx.reshape(*B_mx.shape[:-1], B_mx.shape[-1] // block_size, block_size)
    B_scale = B_scale.unsqueeze(-1)

    # Rescale and cast to bfloat16
    B = B_mx.to(torch.bfloat16) * B_scale.to(torch.bfloat16)

    # Reshape back to original shape
    # B shape: (E, K, N)
    B_t = B.reshape(E, N, K).transpose(-2, -1)

    # Perform bf16 grouped GEMM.
    out = torch._grouped_mm(A, B_t, offs=offs, out_dtype=out_dtype)
    return out


def _emulated_mxfp8_scaled_grouped_mm_2d_2d(
    A_mx: torch.Tensor,  # (M, K)
    A_scale: torch.Tensor,  # (M, K//block_size)
    B_mx: torch.Tensor,  # (K, N)
    B_scale: torch.Tensor,  # (K//block_size, N)
    offs: torch.Tensor,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    block_size: int = 32,
) -> torch.Tensor:
    assert A_mx.ndim == 2, "A must be 2D"
    assert B_mx.ndim == 2, "B must be 2D"
    A = torch.zeros(
        A_mx.shape,
        dtype=torch.bfloat16,
        device=A_mx.device,
        requires_grad=A_mx.requires_grad,
    )
    B = torch.zeros(
        B_mx.shape,
        dtype=torch.bfloat16,
        device=B_mx.device,
        requires_grad=B_mx.requires_grad,
    )

    # Dequantize input per each scaling group
    scales_start_idx = 0
    group_start_idx = 0
    for group_end_idx in offs.tolist():
        group_size = group_end_idx - group_start_idx
        scale_group_size = group_size // block_size
        if group_size == 0:
            group_start_idx = group_end_idx
            continue

        # -- Dequantize A tensor
        # A_group shape: (M, group_size)
        # A_scale shape: (M, group_size//block_size)
        A_group = A_mx[:, group_start_idx:group_end_idx]
        A_group_shape = A_group.shape

        # Get scales for this group.
        # scales shape: (M, group_size//block_size)
        scales = A_scale[:, scales_start_idx : scales_start_idx + scale_group_size]

        # Reshape to be able to do per-scaling group multiplication
        # A_group shape: (M, group_size//block_size, block_size)
        # A_scale shape: (M, group_size//block_size, 1)
        A_group = A_group.reshape(
            *A_group.shape[:-1], A_group.shape[-1] // block_size, block_size
        )
        scales = scales.unsqueeze(-1)

        # Rescale and cast to bfloat16
        A_group = A_group.to(torch.bfloat16) * scales.to(torch.bfloat16)

        # Reshape back to original shape and store in dequantized A buffer
        # A shape: (M, group_size)
        A_group = A_group.reshape(A_group_shape)
        A[:, group_start_idx:group_end_idx] = A_group

        # -- Dequantize B tensor
        # B_group shape is (group_size, N)
        B_group = B_mx[group_start_idx:group_end_idx, :]
        B_group_shape = B_group.shape

        # Scales shape is (group_size//block_size, N)
        scales = B_scale[scales_start_idx : scales_start_idx + scale_group_size, :]

        # Transpose B to get scaling group on rightmost dim, to make things easier
        # B_group_shape = (N, group_size)
        # scales shape = N, group_size//block_size)
        B_group, scales = B_group.transpose(-2, -1), scales.transpose(-2, -1)

        # Reshape B to be able to do per-scaling group multiplication
        # B_group shape: (N, group_size//block_size, block_size)
        # scales shape: (N, group_size//block_size, 1)
        B_group = B_group.reshape(
            *B_group.shape[:-1], B_group.shape[-1] // block_size, block_size
        )
        scales = scales.unsqueeze(-1)

        # Cast to bf16 and perform scaling
        B_group = B_group.to(torch.bfloat16) * scales.to(torch.bfloat16)

        # Reshape B_group back to original shape and store in dequantized B buffer
        B_group = B_group.reshape(B_group_shape[1], B_group_shape[0]).transpose(-2, -1)
        B[group_start_idx:group_end_idx, :] = B_group

        # Increment group start and scale start indices
        group_start_idx = group_end_idx
        scales_start_idx += scale_group_size

    # Perform bf16 grouped GEMM using dequantized A and B.
    out = torch._grouped_mm(A, B, offs=offs, out_dtype=out_dtype)
    return out
