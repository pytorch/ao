# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch

from torchao.float8.config import ScalingGranularity
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.conversion_utils import MoEScalingType
from torchao.prototype.moe_training.kernels import (
    fbgemm_mxfp8_grouped_mm_2d_3d,
    triton_fp8_per_group_colwise_scales,
    triton_fp8_rowwise_3d_transpose_rhs,
)
from torchao.prototype.moe_training.utils import (
    _is_column_major,
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
        logger.info("Using fp8 rowwise for _scaled_grouped_mm")
        return _Float8GroupedMM.apply(
            A,
            B_t,
            offs,
            out_dtype,
        )
    elif scaling_type == MoEScalingType.MXFP8:
        logger.info("Using mxfp8 for _scaled_grouped_mm")
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
        A_data_row_major = to_fp8_saturated(A_scaled, torch.float8_e4m3fn)

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
        B_t_data_col_major = to_fp8_saturated(B_t_scaled, torch.float8_e4m3fn)

        # Store what we need for backward.
        ctx.save_for_backward(A, B_t, offs)
        ctx.out_dtype = out_dtype

        # Perform scaled grouped GEMM and return result.
        # output shape: scaled grouped mm of (M,K) @ (B,K,N) = (M,N)
        assert not _is_column_major(A_data_row_major), (
            "A must be row-major for output = A @ B"
        )
        assert _is_column_major(B_t_data_col_major), (
            "B must be column-major for output = A @ B"
        )

        # Squeeze empty dims out of scales, to comply with grouped mm API.
        # A_scales shape: (M,1) or (B, M, 1)
        # B_t_scales shape: (E, 1, N)
        A_scales = A_scales.squeeze(-1)
        B_t_scales = B_t_scales.squeeze(1)
        return torch._scaled_grouped_mm(
            A_data_row_major,
            B_t_data_col_major,
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
        # grad_output shape: (Mg, N)
        # grad_output_scale shape: (Mg, 1)
        grad_output_scales = tensor_to_scale(
            grad_output,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        grad_output_scaled = grad_output.to(torch.float32) * grad_output_scales
        grad_output_data_row_major = to_fp8_saturated(
            grad_output_scaled, torch.float8_e4m3fn
        )

        # Compute B fp8 column-major for right operand of grouped GEMM:
        # grad_A = grad_output @ B.
        B_data_col_major, B_scales = triton_fp8_rowwise_3d_transpose_rhs(
            B_t._data if hasattr(B_t, "_data") else B_t,
            output_dtype=torch.float8_e4m3fn,
            round_scales_to_power_of_2=True,
        )

        # Compute grad_A.
        # grad_A = grad_output @ B
        # grad_A = scaled grouped mm of (M,N) @ (B,N,K) = (M,K)
        assert not _is_column_major(grad_output_data_row_major), (
            "grad_output must be row-major for grad_A = grad_output @ B"
        )
        assert _is_column_major(B_data_col_major), (
            "B must be column-major for grad_A = grad_output @ B"
        )

        # Squeeze empty dims out of scales, to comply with grouped mm API.
        # grad_output_scales shape: (M,1) or (B, M, 1)
        # B_scales shape: (E, 1, N)
        grad_output_scales = grad_output_scales.squeeze(-1)
        B_scales = B_scales.squeeze(1)
        grad_A = torch._scaled_grouped_mm(
            grad_output_data_row_major,
            B_data_col_major,
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
        # Use transpose method to avoid uncoalesced memory accesses.
        grad_out_data_colwise, grad_out_scales = triton_fp8_per_group_colwise_scales(
            grad_output.t()
            .contiguous()
            .t(),  # Quantization is over 2x faster when input is col major, even with this transformation
            offs,
            torch.float8_e4m3fn,
            round_scales_to_power_of_2=True,
        )
        grad_output_t_data_row_major = grad_out_data_colwise.t()
        grad_output_t_scales = grad_out_scales.t()

        A_data_col_major, A_scales = triton_fp8_per_group_colwise_scales(
            A.t()
            .contiguous()
            .t(),  # Quantization is over 2x faster when input is col major, even with this transformation
            offs,
            torch.float8_e4m3fn,
            round_scales_to_power_of_2=True,
        )

        # Compute grad_B = grad_output_t @ A.
        # grad_B = grad_output_t @ A
        assert not _is_column_major(grad_output_t_data_row_major), (
            "grad_output_t must be row-major for grad_B = grad_output_t @ A"
        )
        assert _is_column_major(A_data_col_major), (
            "A must be column-major for grad_B = grad_output_t @ A"
        )

        # Per-token group scales computed via triton kernels above do not have
        # the empty dim like the scales computed via tensor_to_scale, so we need
        # don't need to squeeze here.
        grad_B = torch._scaled_grouped_mm(
            grad_output_t_data_row_major,
            A_data_col_major,
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
        emulated: bool = False,
    ) -> torch.Tensor:
        # torchao _scaled_grouped_mm only supports A=2D and B=3D.
        assert A.ndim == 2, "A must be 2D"
        assert B_t.ndim == 3, "B must be 3D"
        assert block_size == 32, "Only block_size=32 is supported"

        # Store what we need for backward.
        ctx.save_for_backward(A, B_t, offs)
        ctx.block_size = block_size
        ctx.out_dtype = out_dtype
        ctx.emulated = emulated

        # A_data shape: (M, K)
        # A_scale shape: (M, K//block_size)
        A_scale, A_data = to_mx(
            A, elem_dtype=torch.float8_e4m3fn, block_size=block_size
        )

        # B_data shape: (E, N, K)
        # B_scale shape: (E, N, K//block_size)
        B_scales, B_data = to_mx(
            B_t.transpose(-2, -1),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
        )

        # output = input @ weight.T
        # output shape: (M, N)
        mxfp8_2d_3d_grouped_mm = (
            _emulated_mxfp8_scaled_grouped_mm_2d_3d
            if emulated
            else fbgemm_mxfp8_grouped_mm_2d_3d
        )
        out = mxfp8_2d_3d_grouped_mm(
            A_data,
            A_scale,
            B_data,
            B_scales,
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
        emulated = ctx.emulated

        # grad_out_data shape: (M, N)
        # grad_out_scale shape: (M, N//block_size)
        grad_out_scale, grad_out_data = to_mx(
            grad_out, elem_dtype=torch.float8_e4m3fn, block_size=block_size
        )

        # B_data shape: (E, K, N)
        # B_scale shape: (E, K, N//block_size)
        B_scales, B_data = to_mx(
            # TODO: can we support non-contiguous input tensor in to_mx to eliminate this inefficiency?
            B_t.contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
        )

        # grad_A = scaled grouped mm of (M,N) @ (B,N,K) = (M,K)
        mxfp8_2d_3d_grouped_mm = (
            _emulated_mxfp8_scaled_grouped_mm_2d_3d
            if emulated
            else fbgemm_mxfp8_grouped_mm_2d_3d
        )
        grad_A = mxfp8_2d_3d_grouped_mm(
            grad_out_data,
            grad_out_scale,
            B_data,
            B_scales,
            offs=offs,
            out_dtype=out_dtype,
        )

        # grad_out_t_data shape: (N, M)
        # grad_out_t_scales shape: (N, M//block_size)
        grad_out_t_scales, grad_out_t_data = to_mx(
            # TODO: can we support non-contiguous input tensor in to_mx to eliminate this inefficiency?
            grad_out.transpose(-2, -1).contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
        )

        # Transpose A so we can scale along the M dimension, then un-transpose.
        # A_t_data shape: (K, M)
        # A_t_scales shape: (K, M//block_size)
        A_t_scales, A_t_data = to_mx(
            A.transpose(-2, -1).contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
        )

        # A_data shape = (M, K)
        A_data = A_t_data.transpose(-2, -1)

        # A_scales shape = (M//block_size, K)
        A_scales = A_t_scales.transpose(-2, -1)

        # grad_B_t = scaled grouped mm of (N,M) @ (M,K) = (E,N,K)
        grad_B = _emulated_mxfp8_scaled_grouped_mm_2d_2d(
            grad_out_t_data,
            grad_out_t_scales,
            A_data,
            A_scales,
            offs=offs,
        )

        # grad_B shape =  (E,K,N)
        grad_B_t = grad_B.transpose(-2, -1)

        return grad_A, grad_B_t, None, None, None


def _emulated_mxfp8_scaled_grouped_mm_2d_3d(
    A_data: torch.Tensor,
    A_scale: torch.Tensor,
    B_data: torch.Tensor,
    B_scale: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    block_size: int = 32,
) -> torch.Tensor:
    assert A_data.ndim == 2, f"A must be 2D, got {A_data.ndim}"
    assert B_data.ndim == 3, f"B must be 3D, got {B_data.ndim}"
    assert A_scale.shape[0] == A_data.shape[0], (
        f"A_scale must have same M dim as A_data, got A={A_data.shape} and A_scale={A_scale.shape}"
    )
    assert A_scale.shape[1] == A_data.shape[1] // block_size, (
        f"A_scale dim1 should be size K//block_size, got A={A_data.shape} and A_scale={A_scale.shape}"
    )
    assert B_scale.shape[0] == B_data.shape[0], (
        f"B_scale must have same E dim as B_data, got B={B_data.shape} and B_scale={B_scale.shape}"
    )
    assert B_scale.shape[1] == B_data.shape[1], (
        f"B_scale must have same N dim as B_data, got B={B_data.shape} and B_scale={B_scale.shape}"
    )
    assert B_scale.shape[2] == B_data.shape[2] // block_size, (
        f"B_scale dim2 should be size K//block_size, got B={B_data.shape} and B_scale={B_scale.shape}"
    )

    # Dequantize input
    # A_data shape: (M, K)
    # A_scale shape: (M, K//block_size)
    A_orig_shape = A_data.shape

    # Reshape to be able to do per-scaling group multiplication
    # A_data shape: (M, K//block_size, block_size)
    # A_scale shape: (M, K//block_size, 1)
    A_data = A_data.reshape(
        *A_data.shape[:-1], A_data.shape[-1] // block_size, block_size
    )
    A_scale = A_scale.unsqueeze(-1)

    # Rescale and cast to bfloat16
    A = A_data.to(torch.bfloat16) * A_scale.to(torch.bfloat16)

    # Reshape back to original shape
    # A shape: (M, K)
    A = A.reshape(A_orig_shape)

    # Dequantize weights
    # Tranpose to get block_size on rightmost dim
    # B_data shape: (E, N, K)
    # B_scale shape: (E, N, K//block_size)
    E, N, K = B_data.shape

    # Reshape to be able to do per-scaling group multiplication
    # B_data shape: (E, N, K//block_size, block_size)
    # B_scale shape: (E, N, K//block_size, 1)
    B_data = B_data.reshape(
        *B_data.shape[:-1], B_data.shape[-1] // block_size, block_size
    )
    B_scale = B_scale.unsqueeze(-1)

    # Rescale and cast to bfloat16
    B = B_data.to(torch.bfloat16) * B_scale.to(torch.bfloat16)

    # Reshape back to original shape
    # B shape: (E, K, N)
    B_t = B.reshape(E, N, K).transpose(-2, -1)

    # Perform bf16 grouped GEMM.
    out = torch._grouped_mm(A, B_t, offs=offs, out_dtype=out_dtype)
    return out


def _emulated_mxfp8_scaled_grouped_mm_2d_2d(
    A_data: torch.Tensor,  # (M, K)
    A_scale: torch.Tensor,  # (M, K//block_size)
    B_data: torch.Tensor,  # (K, N)
    B_scale: torch.Tensor,  # (K//block_size, N)
    offs: torch.Tensor,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    block_size: int = 32,
) -> torch.Tensor:
    assert A_data.ndim == 2, "A must be 2D"
    assert B_data.ndim == 2, "B must be 2D"
    A = torch.zeros(
        A_data.shape,
        dtype=torch.bfloat16,
        device=A_data.device,
        requires_grad=A_data.requires_grad,
    )
    B = torch.zeros(
        B_data.shape,
        dtype=torch.bfloat16,
        device=B_data.device,
        requires_grad=B_data.requires_grad,
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
        A_group = A_data[:, group_start_idx:group_end_idx]
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
        B_group = B_data[group_start_idx:group_end_idx, :]
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
