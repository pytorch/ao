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
    triton_fp8_per_group_colwise_scales,
    triton_fp8_rowwise_3d_transpose_rhs,
)
from torchao.prototype.moe_training.kernels.mxfp8 import (
    _mxfp8_cuda_kernels_available as _mxfp8_cuda_kernels_available_quant,
)
from torchao.prototype.moe_training.kernels.mxfp8 import (
    mx_block_rearrange_2d_M_groups_cuda,
    mxfp8_quantize_cuda_3d,
    triton_mx_block_rearrange_2d_K_groups,
    triton_mx_block_rearrange_per_group_3d,
)
from torchao.prototype.moe_training.utils import (
    _is_column_major,
    conditional_nostrict_trace,
)
from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.kernels import (
    _mxfp8_cuda_kernels_available as _mxfp8_cuda_kernels_available_mx,
)
from torchao.prototype.mx_formats.kernels import (
    _triton_kernels_available,
    triton_mxfp8_dequant_dim0,
    triton_to_mxfp8_dim0,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor, to_mx
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
from torchao.quantization.quantize_.common import KernelPreference

logger: logging.Logger = logging.getLogger(__name__)

# Check if SM100 kernels are available
# All SM100-dependent kernels are guarded at their definition sites
_SM100_KERNELS_AVAILABLE = (
    _mxfp8_cuda_kernels_available_quant
    and _mxfp8_cuda_kernels_available_mx
    and _triton_kernels_available
)


def _quantize_then_scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    scaling_type: MoEScalingType = MoEScalingType.FP8_ROWWISE,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
    kernel_preference: KernelPreference = KernelPreference.AUTO,
) -> torch.Tensor:
    """
    This function performs dynamic quantization with the given recipe
    on the input tensors A and B, then performs a scaled grouped GEMM and returns the results.

    Args:
        A (bf16/float32 torch.Tensor): The first high-precision input tensor, which must be a 2D tensor of shape (M * num_groups, K)
            and in row-major memory layout.
        B_t (bf16/float32 torch.Tensor): The second high-precision input tensor which must be 3D, which must be shape (E, K, N)
            and in column-major memory layout.
        offs (int32 torch.Tensor): The offsets to use to mark the starting index of each group along dim0 of the A tensor.
        out_dtype (Optional[torch.dtype]): The dtype of the output tensor. Currently only torch.bfloat16 is supported.
        scaling_type (MoEScalingType): The scaling type to use for quantization.
        float8_dtype (torch.dtype): The float8 dtype to use for quantization. Default is torch.float8_e4m3fn.
            For AMD MI300, use torch.float8_e4m3fnuz.
        kernel_preference (KernelPreference): Kernel preference for quantization and compute. Only applies to MXFP8 scaling types.
    """
    # TODO: Remove logging once prototype is more mature. This is currently very useful for development and debugging.
    if scaling_type == MoEScalingType.FP8_ROWWISE:
        return _to_fp8_rowwise_then_scaled_grouped_mm(
            A,
            B_t,
            offs,
            float8_dtype,
            out_dtype,
        )
    elif (
        scaling_type == MoEScalingType.MXFP8
        or scaling_type == MoEScalingType.MXFP8_WGRAD_WITH_HP
    ):
        block_size = 32
        wgrad_with_hp = scaling_type == MoEScalingType.MXFP8_WGRAD_WITH_HP
        return _to_mxfp8_then_scaled_grouped_mm(
            A,
            B_t,
            offs,
            block_size,
            float8_dtype,
            out_dtype,
            kernel_preference=kernel_preference,
            wgrad_with_hp=wgrad_with_hp,
            scale_calculation_mode=ScaleCalculationMode.RCEIL,
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
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> torch.Tensor:
        # torchao _quantize_then_scaled_grouped_mm only supports A=2D|3D and B=3D.
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
            f"shape {A.shape} and {B_t.shape} are not compatible for _quantize_then_scaled_grouped_mm"
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
            float8_dtype,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        A_scaled = A.to(torch.float32) * A_scales
        A_data_row_major = to_fp8_saturated(A_scaled, float8_dtype)

        # Convert B to float8, column-major for right operand of grouped GEMM.
        # B_t shape: (E, K, N)
        # B_t scales must be computed rowwise keeping the outer/final dim, so:
        # B_t_scales shape: (E, 1, N)
        B_t_scales = tensor_to_scale(
            B_t,
            float8_dtype,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-2,
            round_scales_to_power_of_2=True,
        )
        B_t_scaled = B_t.to(torch.float32) * B_t_scales
        B_t_data_col_major = to_fp8_saturated(B_t_scaled, float8_dtype)

        # Store what we need for backward.
        ctx.save_for_backward(A, B_t, offs)
        ctx.out_dtype = out_dtype
        ctx.float8_dtype = float8_dtype

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
        float8_dtype = ctx.float8_dtype

        # Convert grad_output to float8, row-major for left operand of grouped GEMM
        # needed for grad_A: grad_output @ B
        #
        # grad_output shape: (Mg, N)
        # grad_output_scale shape: (Mg, 1)
        grad_output_scales = tensor_to_scale(
            grad_output,
            float8_dtype,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        grad_output_scaled = grad_output.to(torch.float32) * grad_output_scales
        grad_output_data_row_major = to_fp8_saturated(grad_output_scaled, float8_dtype)

        # Compute B fp8 column-major for right operand of grouped GEMM:
        # grad_A = grad_output @ B.
        B_data_col_major, B_scales = triton_fp8_rowwise_3d_transpose_rhs(
            B_t._data if hasattr(B_t, "_data") else B_t,
            output_dtype=float8_dtype,
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
            float8_dtype,
            round_scales_to_power_of_2=True,
        )
        grad_output_t_data_row_major = grad_out_data_colwise.t()
        grad_output_t_scales = grad_out_scales.t()

        A_data_col_major, A_scales = triton_fp8_per_group_colwise_scales(
            A.t()
            .contiguous()
            .t(),  # Quantization is over 2x faster when input is col major, even with this transformation
            offs,
            float8_dtype,
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
        return grad_A, grad_B.transpose(-2, -1), None, None, None


class _MXFP8GroupedMM(torch.autograd.Function):
    """
    Differentiable implementation of grouped GEMM with dynamic MXFP8 quantization.

    This autograd function performs grouped matrix multiplication with MXFP8 quantization
    for efficient MoE training. It supports both pre-quantized (MXTensor) and high-precision
    inputs, with configurable quantization and layout conversion options.
    """

    @staticmethod
    def forward(
        ctx,
        input_act: torch.Tensor,
        weight_t: torch.Tensor,
        group_offsets: Optional[torch.Tensor] = None,
        block_size: int = 32,
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        wgrad_with_hp: bool = False,
        scale_calculation_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL,
    ) -> torch.Tensor:
        """
        Forward pass: Quantize inputs and perform grouped GEMM.

        Args:
            input_act: Input activations, shape (M, K) - may be MXTensor or high-precision
            weight_t: Expert weights transposed, shape (E, K, N) - always high-precision
            group_offsets: Cumulative token counts per expert, shape (E,)
            block_size: Block size for MXFP8 quantization (must be 32)
            out_dtype: Output dtype (bfloat16 or float32)
            kernel_preference: Kernel preference (AUTO uses CUDA/Triton, EMULATED uses to_mx)
            wgrad_with_hp: Compute weight gradient in high precision
            scale_calculation_mode: Mode for scale calculation (RCEIL, FLOOR, etc.)

        Returns:
            Output tensor, shape (M, N)
        """
        assert kernel_preference in (
            KernelPreference.AUTO,
            KernelPreference.EMULATED,
        ), "kernel_preference must be AUTO or EMULATED"

        # emulated mode validation
        emulated = kernel_preference == KernelPreference.EMULATED
        assert emulated or _SM100_KERNELS_AVAILABLE, (
            "SM100 kernels not available. Please use use torchao CUDA 12.8+ build on SM100/100a device(s). "
            "Otherwise, set kernel_preference=KernelPreference.EMULATED (emulated mode implements basic functionality without efficient kernels)."
        )

        # Input validation
        assert input_act.ndim == 2, "input_act must be 2D"
        assert weight_t.ndim == 3, "weight_t must be 3D"
        assert block_size == 32, "Only block_size=32 is supported"
        assert group_offsets is not None, (
            "group_offsets must be provided for 2d-3d grouped mm"
        )
        assert out_dtype in (torch.bfloat16, torch.float32), (
            "out_dtype must be bfloat16 or float32"
        )
        if isinstance(input_act, MXTensor):
            assert wgrad_with_hp, (
                "only `wgrad_with_hp` recipe is supported for pre-quantized inputs, support for other recipes is still in progress"
            )

        # Quantize input activations along dim0
        # input_act_data shape: (M, K)
        # input_act_scales shape: (M, K//block_size)
        input_act_data, input_act_scales = _extract_or_quantize_dim0(
            input_act, block_size, kernel_preference, scale_calculation_mode
        )

        # Quantize expert weights along dim0 (after transposing from (E, K, N) to (E, N, K))
        # weight_data shape: (E, N, K)
        # weight_scales shape: (E, N, K//block_size)
        weight_data, weight_scales = _extract_or_quantize_dim0(
            weight_t.transpose(-2, -1),
            block_size,
            kernel_preference,
            scale_calculation_mode,
        )

        # Perform grouped GEMM: output = input_act @ weight_t
        # output shape: (M, N)
        if emulated:
            # Use emulated BF16 path: dequantize and use regular grouped mm
            # weight_data is (E, N, K), weight_scales is (E, N, K//block_size)
            # The emulated function expects B in (E, N, K) format
            output = _emulated_mxfp8_scaled_grouped_mm_2d_3d(
                input_act_data,
                input_act_scales,
                weight_data,  # Keep as (E, N, K)
                weight_scales,  # Keep as (E, N, K//block_size)
                offs=group_offsets,
                out_dtype=out_dtype,
                block_size=block_size,
            )
        else:
            # Path using SM100 kernels.
            # Convert scales to blocked layout on a per-group basis required for tcgen05.mma for 2d-3d grouped mm.
            input_act_scales_blocked = mx_block_rearrange_2d_M_groups_cuda(
                input_act_scales, group_offsets
            )
            weight_scales_blocked = triton_mx_block_rearrange_per_group_3d(
                weight_scales
            )
            output = torch._scaled_grouped_mm(
                input_act_data,
                weight_data.transpose(-2, -1),  # Transpose back to (E, K, N)
                input_act_scales_blocked,
                weight_scales_blocked,
                offs=group_offsets,
                out_dtype=out_dtype,
            )

        # Save tensors and config for backward
        ctx.save_for_backward(input_act, weight_t, group_offsets)
        ctx.block_size = block_size
        ctx.float8_dtype = float8_dtype
        ctx.out_dtype = out_dtype
        ctx.kernel_preference = kernel_preference
        ctx.wgrad_with_hp = wgrad_with_hp
        ctx.scale_calculation_mode = scale_calculation_mode

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: Compute gradients w.r.t. input activations and weights.

        Args:
            grad_output: Gradient from upstream, shape (M, N) - may be MXTensor

        Returns:
            tuple: (grad_input, grad_weight_t, None, ...) matching forward args
        """
        # Retrieve saved tensors and config
        input_act, weight_t, group_offsets = ctx.saved_tensors
        block_size = ctx.block_size
        out_dtype = ctx.out_dtype
        kernel_preference = ctx.kernel_preference
        wgrad_with_hp = ctx.wgrad_with_hp
        scale_calculation_mode = ctx.scale_calculation_mode

        # Check SM100 kernel availability when not using emulated mode
        emulated = kernel_preference == KernelPreference.EMULATED
        assert emulated or _SM100_KERNELS_AVAILABLE, (
            "SM100 kernels not available. Please use use torchao CUDA 12.8+ build on SM100/100a device(s)."
            "Otherwise, set kernel_preference=KernelPreference.EMULATED (emulated mode implements basic functionality without efficient kernels)."
        )

        # Compute gradient w.r.t. input activations
        grad_input = _compute_dgrad(
            grad_output,
            weight_t,
            group_offsets,
            block_size,
            out_dtype,
            scale_calculation_mode,
            kernel_preference,
        )

        # Compute gradient w.r.t. weights (high-precision or quantized)
        grad_weight_t = _compute_wgrad(
            grad_output,
            input_act,
            group_offsets,
            block_size,
            out_dtype,
            scale_calculation_mode,
            wgrad_with_hp,
            kernel_preference,
        )
        return grad_input, grad_weight_t, None, None, None, None, None, None, None


def _compute_dgrad(
    grad_output: torch.Tensor,
    weight_t: torch.Tensor,
    group_offsets: torch.Tensor,
    block_size: int,
    out_dtype: torch.dtype,
    scale_calculation_mode: ScaleCalculationMode,
    kernel_preference: KernelPreference = KernelPreference.AUTO,
) -> torch.Tensor:
    """
    Compute gradient w.r.t. input activations: grad_input = grad_output @ weight.

    Args:
        grad_output: Gradient output, shape (M, N)
        weight_t: Expert weights transposed, shape (E, K, N)
        group_offsets: Group offsets for grouped mm
        block_size: Block size for quantization
        out_dtype: Output dtype
        scale_calculation_mode: Mode for scale calculation
        kernel_preference: Kernel preference (AUTO uses CUDA/Triton, EMULATED uses to_mx)

    Returns:
        grad_input, shape (M, K)
    """
    # Quantize grad_output along dim0
    # grad_output_data shape: (M, N)
    # grad_output_scales shape: (M, N//block_size)
    grad_output_data, grad_output_scales = _extract_or_quantize_dim0(
        grad_output, block_size, kernel_preference, scale_calculation_mode
    )

    if kernel_preference == KernelPreference.EMULATED:
        # No CUDA kernel in emulated mode, use torch native impl
        weight_data, weight_scales = _quantize_3d_along_dim1_native(
            weight_t.transpose(-2, -1), block_size, scale_calculation_mode
        )
        grad_input = _emulated_mxfp8_scaled_grouped_mm_2d_3d(
            grad_output_data,  # (M, N)
            grad_output_scales,  # (M, N//block_size)
            weight_data.transpose(-2, -1),  # (E, N, K)
            weight_scales.transpose(-2, -1),  # (E, K, N//block_size)
            offs=group_offsets,
            out_dtype=out_dtype,
            block_size=block_size,
        )
        return grad_input  # (M, K)

    # Path requiring SM100 kernels.
    # Use CUDA kernel for dim1 quantization
    # weight_data: (E, N, K), weight_scales: (E, N//block_size, K)
    weight = weight_t.transpose(-2, -1)
    weight_data, weight_scales = mxfp8_quantize_cuda_3d(
        weight._data if hasattr(weight, "_data") else weight,
        block_size=block_size,
        scaling_mode=scale_calculation_mode.value.lower(),
    )

    # Transpose scales to align with torch API requirement:
    # (E, N//block_size, K) -> (E, K, N//block_size)
    weight_scales = weight_scales.transpose(-2, -1)

    # Convert scales to blocked format
    grad_output_scales_blocked = mx_block_rearrange_2d_M_groups_cuda(
        grad_output_scales, group_offsets
    )
    weight_scales_blocked = triton_mx_block_rearrange_per_group_3d(weight_scales)

    # Compute grad_input = grad_output @ weight
    grad_input = torch._scaled_grouped_mm(
        grad_output_data,  # (M, N)
        weight_data,  # (E, N, K)
        grad_output_scales_blocked,  # (M, N//block_size)
        weight_scales_blocked,  # (E, K, N//block_size)
        offs=group_offsets,
        out_dtype=out_dtype,
    )
    return grad_input  # (M, K)


def _compute_wgrad(
    grad_output: torch.Tensor,
    input_act: torch.Tensor,
    group_offsets: torch.Tensor,
    block_size: int,
    out_dtype: torch.dtype,
    scale_calculation_mode: ScaleCalculationMode,
    wgrad_with_hp: bool = False,
    kernel_preference: KernelPreference = KernelPreference.AUTO,
) -> torch.Tensor:
    """
    Compute gradient w.r.t. weights with quantization.

    Args:
        grad_output: Gradient output (MXTensor or high-precision), shape (M, N)
        input_act: Input activations (MXTensor or high-precision), shape (M, K)
        group_offsets: Group offsets
        block_size: Block size for quantization
        out_dtype: Output dtype
        scale_calculation_mode: Mode for scale calculation
        wgrad_with_hp: Whether to compute weight gradient in high precision
        kernel_preference: Kernel preference for quantization and compute

    Returns:
        grad_weight_t, shape (E, K, N)
    """
    # Dequantize if needed
    grad_output = _dequantize_if_mxtensor(grad_output, block_size)
    input_act = _dequantize_if_mxtensor(input_act, block_size)

    if wgrad_with_hp:
        grad_weight = torch._grouped_mm(
            grad_output.transpose(-2, -1),
            input_act,
            offs=group_offsets,
            out_dtype=out_dtype,
        )
        return grad_weight.transpose(-2, -1)

    # Quantize grad_output and input_act transposed along dim1 (M dimension)
    if kernel_preference == KernelPreference.EMULATED:
        # Use native PyTorch quantization (works on any hardware)
        grad_output_t_scales, grad_output_t_data = to_mx(
            grad_output.transpose(
                -2, -1
            ).contiguous(),  # (M,N) -> (N,M) and quantize along M
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
            scaling_mode=scale_calculation_mode,
        )
        input_act_t_scales, input_act_t_data = to_mx(
            input_act.transpose(
                -2, -1
            ).contiguous(),  # (M,K) -> (K,M) and quantize along M
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
            scaling_mode=scale_calculation_mode,
        )

        # Dequantize and run bf16 grouped mm for emulation
        grad_weight = _emulated_mxfp8_scaled_grouped_mm_2d_2d(
            grad_output_t_data,  # (N, M)
            grad_output_t_scales,  # (N, M//block_size)
            input_act_t_data.transpose(-2, -1),  # (K, M) -> (M, K)
            input_act_t_scales.transpose(
                -2, -1
            ),  # (K, M//block_size) -> (M//block_size, K)
            offs=group_offsets,
            out_dtype=out_dtype,
            block_size=block_size,
        )
        # Transpose to match weight_t shape in forward: (E, N, K) -> (E, K, N)
        return grad_weight.transpose(-2, -1)

    # Path requiring SM100 kernels.
    # Use CUDA kernel for dim1 quant
    # (M,N) -> (M//block_size, N)^T -> (N, M//block_size)
    grad_output_t_mx = _to_mxfp8_dim1_kernel_wrapper(
        grad_output,
        block_size,
        elem_dtype=torch.float8_e4m3fn,
        hp_dtype=grad_output.dtype,
        kernel_preference=KernelPreference.AUTO,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=scale_calculation_mode,
    )
    grad_output_t_data = grad_output_t_mx.qdata
    grad_output_t_scales = grad_output_t_mx.scale

    # (M,K) -> (M//block_size, K)^T -> (K, M//block_size)
    input_act_t_mx = _to_mxfp8_dim1_kernel_wrapper(
        input_act,
        block_size,
        elem_dtype=torch.float8_e4m3fn,
        hp_dtype=input_act.dtype,
        kernel_preference=KernelPreference.AUTO,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=scale_calculation_mode,
    )
    input_act_t_data = input_act_t_mx.qdata
    input_act_t_scales = input_act_t_mx.scale

    # Convert scales to blocked layout required for tcgen05.mma on a per-group basis for 2d-2d grouped mm
    scale_group_offsets = group_offsets // block_size
    grad_output_t_scales_blocked = triton_mx_block_rearrange_2d_K_groups(
        grad_output_t_scales,
        scale_group_offsets,
    )
    input_act_t_scales_blocked = triton_mx_block_rearrange_2d_K_groups(
        input_act_t_scales,
        scale_group_offsets,
    )

    # Compute grad_weight = grad_output_t @ input_act
    # Shape: (N, M) @ (M, K) = (E, N, K)
    grad_weight = torch._scaled_grouped_mm(
        grad_output_t_data,
        input_act_t_data.transpose(-2, -1),
        grad_output_t_scales_blocked,
        input_act_t_scales_blocked,
        offs=group_offsets,
        out_dtype=out_dtype,
    )

    # Transpose to match weight_t shape in forward: (E, N, K) -> (E, K, N)
    return grad_weight.transpose(-2, -1)


def _quantize_3d_along_dim1_native(
    x: torch.Tensor,
    block_size: int,
    scale_calculation_mode: ScaleCalculationMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize 3D tensor (E, N, K) along dim1 (N dimension) using native PyTorch.
    Works on any hardware, not just SM100.

    Args:
        x: Input tensor of shape (E, N, K)
        block_size: Block size for quantization
        scale_calculation_mode: Mode for scale calculation

    Returns:
        tuple: (quantized_data, scales)
            - quantized_data: shape (E, N, K)
            - scales: shape (E, N//block_size, K)
    """
    # Transpose (E,N,K) to (E,K,N) so N is final dim,
    # since to_mx scales along that dim
    scales, qdata = to_mx(
        x.transpose(-2, -1).contiguous(),
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=scale_calculation_mode,
    )

    # Transpose tensors and scales back so we have effectively
    # quantized input shape (E, N, K) along N
    qdata = qdata.transpose(-2, -1)
    scales = scales.transpose(-2, -1)

    return qdata, scales


def _extract_or_quantize_dim0(
    tensor: torch.Tensor,
    block_size: int,
    kernel_preference: KernelPreference,
    scale_calculation_mode: ScaleCalculationMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract qdata and scales from MXTensor or quantize high-precision tensor along dim0.

    Args:
        tensor: Input tensor (MXTensor or high-precision)
        block_size: Block size for quantization
        kernel_preference: Kernel preference (AUTO uses Triton if available, EMULATED uses to_mx)
        scale_calculation_mode: Mode for scale calculation

    Returns:
        tuple: (quantized_data, scales)
    """
    if isinstance(tensor, MXTensor):
        return tensor.qdata, tensor.scale

    # Use SM100 Triton kernel if AUTO mode and kernels available
    if kernel_preference == KernelPreference.AUTO and _SM100_KERNELS_AVAILABLE:
        qdata, scale = triton_to_mxfp8_dim0(
            tensor,
            inner_block_size=block_size,
            scaling_mode=str(scale_calculation_mode.value).lower(),
        )
    else:
        # Use native PyTorch (works on any hardware)
        scale, qdata = to_mx(
            tensor,
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
            scaling_mode=scale_calculation_mode,
        )
    return qdata, scale


def _dequantize_if_mxtensor(
    tensor: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Dequantize MXTensor if needed, otherwise return tensor unchanged.

    Args:
        tensor: Input tensor (MXTensor or high-precision)
        block_size: Block size for dequantization

    Returns:
        High-precision tensor
    """
    if isinstance(tensor, MXTensor):
        return triton_mxfp8_dequant_dim0(
            tensor.qdata,
            tensor.scale.view(torch.uint8),  # Triton can't handle e8m0 directly yet
            out_dtype=tensor._orig_dtype,
            scale_block_size=block_size,
        )
    return tensor


def _to_mxfp8_dim1_3d(
    B: torch.Tensor,
    block_size: int = 32,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a 3D tensor to MXFP8 format with (block_size, 1) scaling granularity.
    """
    E, N, K = B.shape
    B_reshaped = B.reshape(E * N, K)
    B_t_mx = _to_mxfp8_dim1_kernel_wrapper(
        B_reshaped,
        block_size,
        elem_dtype=float8_dtype,
        hp_dtype=B_reshaped.dtype,
        kernel_preference=KernelPreference.AUTO,  # Not used
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=scaling_mode,
    )
    B_data = B_t_mx.qdata.t()  # (K, E*N) -> (E*N, K)
    B_data = B_data.reshape(E, N, K)  # (E*N, K) -> (E, N, K)
    B_scales = B_t_mx.scale.view(torch.uint8)  # (K, E*N//block_size)
    B_scales = B_scales.reshape(
        K, E, N // block_size
    )  # (K, E*N//block_size) -> (K, E, N//block_size)
    B_scales = B_scales.permute(
        1, 0, 2
    )  # (K, E, N//block_size) -> (E, K, N//block_size)
    B_scales = B_scales.view(torch.float8_e8m0fnu)

    # TODO: Update cutlass grouped gemm to accept NT/TN/NN/TT layouts so we can avoid this conversion to column major
    B_data = B_data.transpose(-2, -1).contiguous().transpose(-2, -1)
    return B_scales, B_data


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
    # Transpose to get block_size on rightmost dim
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


@torch.compiler.disable
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


def round_up(x, y):
    return ((x + y - 1) // y) * y


# Aliases for convenience/clarity
@conditional_nostrict_trace
def _to_mxfp8_then_scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    block_size: int = 32,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    kernel_preference: KernelPreference = KernelPreference.AUTO,
    wgrad_with_hp: bool = False,
    scale_calculation_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL,
) -> torch.Tensor:
    """
    Differentiable mxfp8 grouped gemm with dynamic mxfp8 quantization.

    Args:
        - A (bf16/float32 torch.Tensor): The first high-precision input tensor,
            which must be a 2D tensor of shape (M * num_groups, K)
            and in row-major memory layout.
        - B_t (bf16/float32 torch.Tensor): The second high-precision input tensor
            which must be 3D, which must be shape (G, K, N)
            and in "per group column-major memory" layout (i.e., strides of (N*K, 1, N)).
        - offs (int32 torch.Tensor): The offsets to use to mark the end index of each group along the dim0 of the A tensor.
        - block_size (int): The block size to use for mxpf8 quantization. Currently only 32 is supported.
        - float8_dtype (torch.dtype): The float8 dtype to use for quantization. Default is torch.float8_e4m3fn.
        - out_dtype (Optional[torch.dtype]): The dtype of the output tensor. Default is torch.bfloat16.
        - kernel_preference (KernelPreference): Kernel preference (AUTO uses CUDA/Triton, EMULATED uses to_mx).
        - wgrad_with_hp (bool): Whether to compute weight gradients in high precision.
        - scale_calculation_mode (ScaleCalculationMode): The mode to use for scale calculation.

    Returns:
        - out (torch.Tensor): The result of the mxpf8 scaled grouped gemm.
    """
    return _MXFP8GroupedMM.apply(
        A,
        B_t,
        offs,
        block_size,
        float8_dtype,
        out_dtype,
        kernel_preference,
        wgrad_with_hp,
        scale_calculation_mode,
    )


_to_fp8_rowwise_then_scaled_grouped_mm = _Float8GroupedMM.apply
