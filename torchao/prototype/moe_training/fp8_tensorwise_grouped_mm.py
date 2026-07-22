# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tensorwise FP8 grouped GEMM for MoE training.

Unlike the rowwise path (one scale per row/column), the tensorwise path uses
a single scalar scale per entire tensor, following Turbo Primus:
- Forward: one scale for all of A, one scale for all of B_t
- Backward (grad_A): one scale for grad_output, one scale for all of B
- Backward (grad_B): one scale for grad_output, one scale for A
"""

from typing import Optional, Tuple

import torch

from torchao.float8.float8_utils import amax_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.kernels.fp8_tensorwise_2d import (
    triton_fp8_tensorwise_quantize_2d,
    triton_fp8_tensorwise_quantize_2d_dual_layout,
)
from torchao.prototype.moe_training.utils import _is_column_major

_USE_TRITON_QUANTIZE_2D = triton_fp8_tensorwise_quantize_2d is not None
_USE_TRITON_QUANTIZE_2D_DUAL = triton_fp8_tensorwise_quantize_2d_dual_layout is not None

try:
    import triton as _triton

    from torchao.prototype.moe_training.kernels.fp8_tensorwise_2d import (
        triton_fp8_tensorwise_amax,
    )
    from torchao.prototype.moe_training.kernels.fp8_tensorwise_3d import (
        EPS as _EPS_3D,
    )
    from torchao.prototype.moe_training.kernels.fp8_tensorwise_3d import (
        FP8_DTYPE_MAP as _FP8_DTYPE_MAP_3D,
    )
    from torchao.prototype.moe_training.kernels.fp8_tensorwise_3d import (
        _fp8_tensorwise_3d_dual_layout_quantize_kernel,
    )

    _USE_TRITON_3D = True
except ImportError:
    _USE_TRITON_3D = False


@torch.library.custom_op("torchao::fp8_tensorwise_quantize_2d", mutates_args={})
def _fp8_tensorwise_quantize_2d(
    tensor: torch.Tensor,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tensorwise quantize a 2D tensor to FP8. Returns (fp8_data,
    inv_scale_expanded), where inv_scale_expanded has shape (M,) and holds the
    reciprocal of the single tensorwise scale broadcast to every row.

    Registered as custom_op so inductor does not fuse quantization into
    Triton kernels that may crash during autotuning on certain hardware.
    """
    if _USE_TRITON_QUANTIZE_2D and tensor.is_contiguous():
        return triton_fp8_tensorwise_quantize_2d(
            tensor, output_dtype, round_scales_to_power_of_2=True
        )

    tensor_clean = torch.nan_to_num(tensor)
    amax = tensor_clean.abs().amax().to(torch.float32)
    scale = amax_to_scale(amax, output_dtype, round_scales_to_power_of_2=True)
    fp8_data = to_fp8_saturated(
        tensor_clean * scale.to(tensor_clean.dtype), output_dtype
    )
    inv_scale = 1.0 / scale
    inv_scale_expanded = inv_scale.reshape(1).expand(tensor.size(0)).contiguous()
    return fp8_data, inv_scale_expanded


@_fp8_tensorwise_quantize_2d.register_fake
def _fake_fp8_tensorwise_quantize_2d(
    tensor: torch.Tensor,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, K = tensor.shape
    fp8_data = torch.empty(M, K, dtype=output_dtype, device=tensor.device)
    inv_scale_expanded = torch.empty(M, dtype=torch.float32, device=tensor.device)
    return fp8_data, inv_scale_expanded


@torch.library.custom_op(
    "torchao::fp8_tensorwise_quantize_2d_dual_layout", mutates_args={}
)
def _fp8_tensorwise_quantize_2d_dual_layout(
    tensor: torch.Tensor,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tensorwise quantize a 2D tensor to FP8 and return row-major and column-major
    copies plus the expanded inverse scale.
    """
    if _USE_TRITON_QUANTIZE_2D_DUAL and tensor.is_contiguous():
        return triton_fp8_tensorwise_quantize_2d_dual_layout(
            tensor, output_dtype, round_scales_to_power_of_2=True
        )

    fp8_data, inv_scale_expanded = _fp8_tensorwise_quantize_2d(tensor, output_dtype)
    M, D = fp8_data.shape
    fp8_col_major = torch.empty(D, M, dtype=fp8_data.dtype, device=fp8_data.device).t()
    fp8_col_major.copy_(fp8_data)
    return fp8_data, fp8_col_major, inv_scale_expanded


@_fp8_tensorwise_quantize_2d_dual_layout.register_fake
def _fake_fp8_tensorwise_quantize_2d_dual_layout(
    tensor: torch.Tensor,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, K = tensor.shape
    fp8_row = torch.empty(M, K, dtype=output_dtype, device=tensor.device)
    fp8_col = torch.empty(K, M, dtype=output_dtype, device=tensor.device).t()
    inv_scale_expanded = torch.empty(M, dtype=torch.float32, device=tensor.device)
    return fp8_row, fp8_col, inv_scale_expanded


@torch.library.custom_op(
    "torchao::fp8_tensorwise_quantize_3d_single_scale", mutates_args={}
)
def _fp8_tensorwise_quantize_3d_single_scale(
    tensor: torch.Tensor,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize entire 3D (E, K, N) col-major tensor with a single scalar scale
    across all experts. Materializes both FP8 layouts for forward and backward.

    Returns:
        fwd_fp8: (E, K, N) column-major
        fwd_inv_scales: (E, N) — inverse scale for _scaled_grouped_mm
        rhs_fp8: (E, N, K) column-major (transposed layout for grad_A)
        rhs_inv_scales: (E, K) — inverse scale for _scaled_grouped_mm
    """
    E, K, N = tensor.shape

    if _USE_TRITON_3D:
        tl_output_dtype = _FP8_DTYPE_MAP_3D[output_dtype]
        fp8_dtype_min = torch.finfo(output_dtype).min
        fp8_dtype_max = torch.finfo(output_dtype).max
        input_dtype_max = torch.finfo(tensor.dtype).max

        fwd_buf = torch.empty(
            (E, K, N), dtype=output_dtype, device=tensor.device
        ).as_strided((E, K, N), (K * N, 1, K))
        rhs_buf = torch.empty(
            (E, N, K), dtype=output_dtype, device=tensor.device
        ).as_strided((E, N, K), (N * K, 1, N))

        # Flat 1D amax over the entire tensor. The staged path avoids global
        # atomics for DeepSeek-sized tensors and falls back to atomic amax only
        # for tensors with too many partial blocks.
        amax_buf = triton_fp8_tensorwise_amax(tensor)
        expert_amax = amax_buf.expand(E).contiguous()

        fwd_inv_scales = torch.empty(E, N, dtype=torch.float32, device=tensor.device)
        rhs_inv_scales = torch.empty(E, K, dtype=torch.float32, device=tensor.device)
        grid = lambda meta: (E, _triton.cdiv(N, meta["BLOCK_SIZE_N"]))

        _fp8_tensorwise_3d_dual_layout_quantize_kernel[grid](
            tensor,
            tensor.stride(0),
            tensor.stride(1),
            tensor.stride(2),
            fwd_buf,
            fwd_buf.stride(0),
            fwd_buf.stride(1),
            fwd_buf.stride(2),
            rhs_buf,
            rhs_buf.stride(0),
            rhs_buf.stride(1),
            rhs_buf.stride(2),
            expert_amax,
            fwd_inv_scales,
            rhs_inv_scales,
            E,
            K,
            N,
            fp8_dtype_min,
            fp8_dtype_max,
            tl_output_dtype,
            True,
            INPUT_DTYPE_MAX=input_dtype_max,
            EPS=_EPS_3D,
        )

        return fwd_buf, fwd_inv_scales, rhs_buf, rhs_inv_scales

    # PyTorch fallback.
    tensor_clean = torch.nan_to_num(tensor)
    amax = tensor_clean.abs().amax().to(torch.float32)
    scale = amax_to_scale(amax, output_dtype, round_scales_to_power_of_2=True)
    inv_scale = 1.0 / scale

    fp8_data = to_fp8_saturated(
        tensor_clean * scale.to(tensor_clean.dtype), output_dtype
    )

    fwd_inv_scales = inv_scale.unsqueeze(0).expand(E, N).contiguous()
    rhs_inv_scales = inv_scale.unsqueeze(0).expand(E, K).contiguous()
    rhs_fp8 = fp8_data.contiguous().transpose(-2, -1)

    return fp8_data, fwd_inv_scales, rhs_fp8, rhs_inv_scales


@_fp8_tensorwise_quantize_3d_single_scale.register_fake
def _fake_fp8_tensorwise_quantize_3d_single_scale(
    tensor: torch.Tensor,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    E, K, N = tensor.shape
    fwd_fp8 = torch.empty(E, K, N, dtype=output_dtype, device=tensor.device)
    fwd_fp8 = fwd_fp8.as_strided((E, K, N), (K * N, 1, K))
    fwd_inv_scales = torch.empty(E, N, dtype=torch.float32, device=tensor.device)
    rhs_fp8 = torch.empty(E, N, K, dtype=output_dtype, device=tensor.device)
    rhs_fp8 = rhs_fp8.as_strided((E, N, K), (K * N, 1, N))
    rhs_inv_scales = torch.empty(E, K, dtype=torch.float32, device=tensor.device)
    return fwd_fp8, fwd_inv_scales, rhs_fp8, rhs_inv_scales


def _to_fp8_tensorwise_then_scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """
    Differentiable FP8 grouped matrix multiplication with dynamic FP8 tensorwise quantization.

    This function quantizes inputs A and B_t to FP8 format using tensorwise scaling
    (one scalar scale per entire tensor), then performs a scaled grouped matrix
    multiplication.

    Args:
        A: Left operand tensor of shape (M, K). Must be row-major,
            with dtype float32 or bfloat16, and K divisible by 16.
        B_t: Right operand tensor of shape (E, K, N), transposed and in per-group column-major
            format, meaning strides of (K*N, 1, K). Must have dtype float32 or bfloat16,
            with K and N divisible by 16.
        offs: Offset tensor of shape (num_groups,) with dtype int32, defining
            group boundaries for the grouped GEMM operation. Group sizes must be divisible by 16.
        out_dtype: Output dtype for the result. Defaults to torch.bfloat16.
        float8_dtype: FP8 dtype to quantize to. Defaults to torch.float8_e4m3fn.

    Returns:
        torch.Tensor: Result of grouped matrix multiplication with shape (M, N).
    """
    return _Float8TensorwiseGroupedMM.apply(A, B_t, offs, out_dtype, float8_dtype)


def _copy_to_column_major(tensor: torch.Tensor) -> torch.Tensor:
    """Return a same-shape FP8 tensor with column-major strides."""
    assert tensor.ndim == 2, "column-major copy helper expects a 2D tensor"
    M, D = tensor.shape
    out = torch.empty(D, M, dtype=tensor.dtype, device=tensor.device).t()
    out.copy_(tensor)
    return out


class _Float8TensorwiseGroupedMM(torch.autograd.Function):
    """Differentiable implementation of grouped GEMM with dynamic tensorwise float8 quantization."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> torch.Tensor:
        assert A.ndim == 2, "A must be 2D"
        assert B_t.ndim == 3, "B must be 3D"

        assert A.size(-1) % 16 == 0, (
            f"A must have a last dim divisible by 16, but got shape: {A.shape}"
        )
        assert B_t.size(-2) % 16 == 0 and B_t.size(-1) % 16 == 0, (
            f"B must have last 2 dims divisible by 16, but got shape: {B_t.shape}"
        )

        assert A.dtype == torch.float32 or A.dtype == torch.bfloat16, (
            "A must be float32 or bfloat16"
        )
        assert B_t.dtype == torch.float32 or B_t.dtype == torch.bfloat16, (
            "B must be float32 or bfloat16"
        )
        assert offs is not None and offs.dtype == torch.int32, (
            "offs must be an int32 tensor"
        )

        assert A.size(-1) == B_t.size(-2), (
            f"shape {A.shape} and {B_t.shape} are not compatible for grouped mm"
        )

        assert not _is_column_major(A), "A must be row-major"
        assert _is_column_major(B_t), "B must be column-major"

        B_t_data = B_t._data if hasattr(B_t, "_data") else B_t

        A_fp8, A_col_major, A_inv_scales = _fp8_tensorwise_quantize_2d_dual_layout(
            A, float8_dtype
        )
        (B_t_fp8, B_t_inv_scales, B_rhs_fp8, B_rhs_inv_scales) = (
            _fp8_tensorwise_quantize_3d_single_scale(B_t_data, float8_dtype)
        )

        # Pre-build col-major A and flat inv-scales for grad_B in backward.
        _, K = A.shape
        num_groups = offs.numel()
        A_inv_scales_flat = A_inv_scales[:1].expand(num_groups * K).contiguous()

        ctx.save_for_backward(
            A_col_major,
            A_inv_scales_flat,
            B_rhs_fp8,
            B_rhs_inv_scales,
            offs,
        )
        ctx.out_dtype = out_dtype
        ctx.float8_dtype = float8_dtype

        return torch._scaled_grouped_mm(
            A_fp8,
            B_t_fp8,
            A_inv_scales,
            B_t_inv_scales,
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            A_col_major,
            A_inv_scales_flat,
            B_data_col_major,
            B_inv_scales_expanded,
            offs,
        ) = ctx.saved_tensors
        out_dtype = ctx.out_dtype
        float8_dtype = ctx.float8_dtype

        (
            grad_output_fp8,
            grad_output_col_major,
            go_inv_scales,
        ) = _fp8_tensorwise_quantize_2d_dual_layout(grad_output, float8_dtype)

        # grad_A = grad_output @ B
        grad_A = torch._scaled_grouped_mm(
            grad_output_fp8,
            B_data_col_major,
            go_inv_scales,
            B_inv_scales_expanded,
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

        # grad_B = grad_output_t @ A
        _, N = grad_output.shape
        num_groups = offs.numel()

        grad_output_t_row_major = grad_output_col_major.t()
        go_inv_scales_flat = go_inv_scales[:1].expand(num_groups * N).contiguous()

        grad_B = torch._scaled_grouped_mm(
            grad_output_t_row_major,
            A_col_major,
            go_inv_scales_flat,
            A_inv_scales_flat,
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )
        return grad_A, grad_B.transpose(-2, -1), None, None, None
