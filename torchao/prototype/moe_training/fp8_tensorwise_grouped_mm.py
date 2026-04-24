# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tensorwise FP8 grouped GEMM for MoE training.

Unlike the rowwise path (one scale per row/column), the tensorwise path uses:
- Forward: one scale for all of A, one scale per expert for B_t
- Backward (grad_A): one scale for grad_output, one scale per expert for B
- Backward (grad_B): one scale per group for both grad_output and A
"""

from typing import Optional, Tuple

import torch
from torch.utils._triton import has_triton

from torchao.float8.float8_utils import amax_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.kernels.fp8_tensorwise_2d import (
    triton_fp8_tensorwise_quantize_2d,
)
from torchao.prototype.moe_training.kernels.fp8_tensorwise_per_group import (
    triton_fp8_tensorwise_per_group_quantize,
)
from torchao.prototype.moe_training.utils import _is_column_major

_USE_TRITON_QUANTIZE_2D = triton_fp8_tensorwise_quantize_2d is not None
_USE_TRITON_PER_GROUP = triton_fp8_tensorwise_per_group_quantize is not None


@torch.library.custom_op(
    "torchao::fp8_tensorwise_quantize_2d", mutates_args={}
)
def _fp8_tensorwise_quantize_2d(
    tensor: torch.Tensor,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tensorwise quantize a 2D tensor to FP8. Returns (fp8_data, scale_per_row)
    where scale_per_row is the single tensorwise scale expanded to (M,).

    Registered as custom_op so inductor does not fuse quantization into
    Triton kernels that may crash during autotuning on certain hardware.
    """
    # Fused Triton path: two sequential passes over the tensor (amax then
    # quantize) instead of ~15 separate ATen kernel launches.  Each ROCm HIP
    # kernel launch costs ~60 µs in dispatch overhead, so collapsing 15 launches
    # to 3 saves ~720 µs of pure dispatch overhead per call.
    #
    # Falls back to the PyTorch path if Triton is not available or the tensor
    # is not contiguous (the Triton kernel requires contiguous layout).
    if _USE_TRITON_QUANTIZE_2D and tensor.is_contiguous():
        return triton_fp8_tensorwise_quantize_2d(
            tensor, output_dtype, round_scales_to_power_of_2=True
        )

    # PyTorch fallback (kept for non-contiguous inputs and non-Triton builds).
    #
    # Stay in the original dtype (e.g. BF16) throughout to avoid materialising
    # a large float32 copy of the input.  The BF16→F32 promotion kernel is the
    # single largest GPU cost per call (~350 µs for a 98432×2048 tensor) so
    # keeping large tensors in BF16 cuts that overhead entirely.
    tensor_clean = torch.nan_to_num(tensor)
    amax = tensor_clean.abs().amax().to(torch.float32)
    scale = amax_to_scale(amax, output_dtype, round_scales_to_power_of_2=True)
    fp8_data = to_fp8_saturated(tensor_clean * scale.to(tensor_clean.dtype), output_dtype)
    # .contiguous() materialises the expansion so the output has stride (1,).
    # A stride-0 expanded tensor would cause _scaled_grouped_mm to read
    # incorrect memory on ROCm, producing a GPU hardware exception.
    scale_expanded = scale.reshape(1).expand(tensor.size(0)).contiguous()
    return fp8_data, scale_expanded


@_fp8_tensorwise_quantize_2d.register_fake
def _fake_fp8_tensorwise_quantize_2d(
    tensor: torch.Tensor,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, K = tensor.shape
    fp8_data = torch.empty(M, K, dtype=output_dtype, device=tensor.device)
    scale_expanded = torch.empty(M, dtype=torch.float32, device=tensor.device)
    return fp8_data, scale_expanded


@torch.library.custom_op(
    "torchao::fp8_tensorwise_quantize_3d", mutates_args={}
)
def _fp8_tensorwise_quantize_3d(
    tensor: torch.Tensor,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-expert tensorwise quantize a 3D (E, K, N) column-major tensor to FP8.
    Returns (fp8_data, scales) where scales has shape (E, N) — one scale per
    expert expanded across the N dimension.

    Registered as custom_op so inductor does not fuse quantization into
    Triton kernels that may crash during autotuning on certain hardware.
    """
    E = tensor.size(0)
    N = tensor.size(-1)
    # Sanitize before any computation so NaN cannot propagate into the scale.
    # Stay in original dtype to avoid a large float32 copy (see quantize_2d).
    tensor_clean = torch.nan_to_num(tensor)

    # For the 3D col-major tensor, amax+amin on non-contiguous strides is slow.
    # abs() produces a contiguous output (PyTorch resets strides for unary ops),
    # so abs().amax() benefits from contiguous memory access in the reduction.
    # We still avoid the F32 copy; abs() stays in the original dtype (BF16).
    amax = tensor_clean.abs().amax(dim=(-2, -1)).to(torch.float32)  # (E,)

    scale = amax_to_scale(amax, output_dtype, round_scales_to_power_of_2=True)  # (E,)
    fp8_data = to_fp8_saturated(
        tensor_clean * scale.to(tensor_clean.dtype).view(E, 1, 1), output_dtype
    )
    # scale is already float32; no extra .to() needed.
    scale_expanded = scale.unsqueeze(-1).expand(-1, N).contiguous()  # (E, N)
    return fp8_data, scale_expanded


@_fp8_tensorwise_quantize_3d.register_fake
def _fake_fp8_tensorwise_quantize_3d(
    tensor: torch.Tensor,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    E, K, N = tensor.shape
    # Preserve column-major strides for the fp8 output.
    fp8_data = torch.empty(E, K, N, dtype=output_dtype, device=tensor.device)
    fp8_data = fp8_data.as_strided((E, K, N), (K * N, 1, K))
    scale_expanded = torch.empty(E, N, dtype=torch.float32, device=tensor.device)
    return fp8_data, scale_expanded


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
    (one scale per tensor for A, one scale per expert for B_t), then performs a
    scaled grouped matrix multiplication.

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
        assert A.ndim == 2 or A.ndim == 3, "A must be 2D or 3D"
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
        assert offs is None or offs.dtype == torch.int32, (
            "offs must be int32 tensor or None"
        )

        assert A.size(-1) == B_t.size(-2), (
            f"shape {A.shape} and {B_t.shape} are not compatible for grouped mm"
        )

        assert not _is_column_major(A), "A must be row-major"
        assert _is_column_major(B_t), "B must be column-major"

        # Unwrap B_t if it's a wrapper tensor.
        B_t_data = B_t._data if hasattr(B_t, "_data") else B_t

        # Quantize A and B_t using custom ops (opaque to inductor).
        A_fp8, A_scales = _fp8_tensorwise_quantize_2d(A, float8_dtype)
        B_t_fp8, B_t_scales = _fp8_tensorwise_quantize_3d(B_t_data, float8_dtype)

        ctx.save_for_backward(A, B_t, offs)
        ctx.out_dtype = out_dtype
        ctx.float8_dtype = float8_dtype

        assert not _is_column_major(A_fp8), (
            "A must be row-major for output = A @ B"
        )
        assert _is_column_major(B_t_fp8), (
            "B must be column-major for output = A @ B"
        )

        return torch._scaled_grouped_mm(
            A_fp8,
            B_t_fp8,
            A_scales.reciprocal(),
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

        # Unwrap B_t if it's a wrapper tensor.
        B_t_data = B_t._data if hasattr(B_t, "_data") else B_t

        # =====================================================================
        # grad_A = grad_output @ B
        # =====================================================================

        # Tensorwise quantize grad_output: single scale for entire tensor.
        grad_output_fp8, grad_output_scales = _fp8_tensorwise_quantize_2d(
            grad_output, float8_dtype
        )

        # Per-expert tensorwise quantize B_t, then transpose to (E, N, K) col-major.
        B_t_fp8, B_t_scales = _fp8_tensorwise_quantize_3d(B_t_data, float8_dtype)
        # B_t_fp8: (E, K, N) col-major, B_t_scales: (E, N)
        # Transpose (E, K, N) col-major -> (E, N, K) col-major.
        # .contiguous() on col-major gives row-major (E, K, N),
        # then .transpose(-2, -1) gives (E, N, K) with strides (K*N, 1, N) = col-major.
        K = B_t_data.size(-2)
        B_data_col_major = B_t_fp8.contiguous().transpose(-2, -1)

        # Recompute B scales as (E, K) for the transposed B.
        # B_t_scales is (E, N) — extract per-expert scale (first col) and expand to (E, K).
        B_scales_expanded = B_t_scales[:, 0:1].expand(-1, K).contiguous()

        assert not _is_column_major(grad_output_fp8), (
            "grad_output must be row-major for grad_A = grad_output @ B"
        )
        assert _is_column_major(B_data_col_major), (
            "B must be column-major for grad_A = grad_output @ B"
        )

        grad_A = torch._scaled_grouped_mm(
            grad_output_fp8,
            B_data_col_major,
            grad_output_scales.reciprocal(),
            B_scales_expanded.reciprocal(),
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

        # =====================================================================
        # grad_B = grad_output_t @ A
        # =====================================================================
        # Both operands are 2D with offsets defining the "jagged" groups.
        # We use per-group tensorwise scales: one scale per group.

        M, N = grad_output.shape
        _, K = A.shape
        num_groups = offs.numel()

        # Quantize grad_output and A with per-group tensorwise scales.
        grad_out_fp8_data, grad_out_scales_flat = (
            _fp8_tensorwise_per_group_quantize(
                grad_output, offs, float8_dtype, output_scale_dim=N
            )
        )
        A_fp8_data, A_scales_flat = _fp8_tensorwise_per_group_quantize(
            A, offs, float8_dtype, output_scale_dim=K
        )

        # Convert grad_output to col-major, then transpose to get
        # grad_output_t in row-major.
        grad_out_col_major = torch.empty(
            N, M, dtype=float8_dtype, device=grad_output.device
        ).t()
        grad_out_col_major.copy_(grad_out_fp8_data)
        grad_output_t_row_major = grad_out_col_major.t()

        # Convert A to col-major for the RHS of grouped GEMM.
        A_col_major = torch.empty(
            K, M, dtype=float8_dtype, device=A.device
        ).t()
        A_col_major.copy_(A_fp8_data)

        assert not _is_column_major(grad_output_t_row_major), (
            "grad_output_t must be row-major for grad_B = grad_output_t @ A"
        )
        assert _is_column_major(A_col_major), (
            "A must be column-major for grad_B = grad_output_t @ A"
        )

        grad_B = torch._scaled_grouped_mm(
            grad_output_t_row_major,
            A_col_major,
            grad_out_scales_flat.reciprocal(),
            A_scales_flat.reciprocal(),
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )
        return grad_A, grad_B.transpose(-2, -1), None, None, None


@torch.library.custom_op(
    "torchao::fp8_tensorwise_per_group_quantize", mutates_args={}
)
def _fp8_tensorwise_per_group_quantize(
    tensor: torch.Tensor,
    offs: torch.Tensor,
    output_dtype: torch.dtype,
    output_scale_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor to FP8 with per-group tensorwise scales.

    Each group (defined by offs along dim0) gets a single tensorwise scale,
    which is then expanded to fill output_scale_dim slots per group in the
    flat scale tensor.

    Uses PyTorch ops (amax/amin + segment_reduce) for the per-group amax,
    then expands the per-group scales to the flat format expected by
    _scaled_grouped_mm.

    Args:
        tensor: (M, D) 2D input tensor.
        offs: Group boundary offsets along dim0.
        output_dtype: Target FP8 dtype.
        output_scale_dim: Number of scale slots per group in the output
            (N for colwise / grad_output, K for A).

    Returns:
        fp8_data: (M, D) FP8 tensor in row-major layout.
        scales_flat: (output_scale_dim * num_groups,) float32 scale tensor.
    """
    assert tensor.ndim == 2, "input tensor must be 2D"
    M, D = tensor.shape
    num_groups = offs.numel()

    # Fused Triton path: two Triton kernels (amax + quantize) with nan_to_num
    # fused inline, instead of ~17 separate ATen kernel launches.
    # Falls back to PyTorch if Triton unavailable or tensor non-contiguous.
    if _USE_TRITON_PER_GROUP and tensor.is_contiguous():
        return triton_fp8_tensorwise_per_group_quantize(
            tensor, offs, output_dtype, output_scale_dim,
            round_scales_to_power_of_2=True,
        )

    # PyTorch fallback (kept for non-contiguous inputs and non-Triton builds).
    tensor_clean = torch.nan_to_num(tensor)

    row_amax = tensor_clean.abs().amax(dim=1).to(torch.float32)  # (M,) float32
    group_starts = torch.cat([
        torch.zeros(1, dtype=offs.dtype, device=offs.device), offs[:-1]
    ])
    group_lengths = (offs - group_starts).to(torch.int64)
    group_amax = torch.segment_reduce(row_amax, "max", lengths=group_lengths, unsafe=True)

    row_group_ids = torch.bucketize(
        torch.arange(M, device=tensor.device), offs, right=True
    ).clamp(max=num_groups - 1)

    group_scale = amax_to_scale(
        group_amax, output_dtype, round_scales_to_power_of_2=True
    )

    per_row_scale = group_scale[row_group_ids].to(tensor_clean.dtype).unsqueeze(1)  # (M, 1)
    fp8_data = to_fp8_saturated(tensor_clean * per_row_scale, output_dtype)

    scales_flat = (
        group_scale
        .unsqueeze(1)
        .expand(-1, output_scale_dim)
        .contiguous()
        .view(-1)
    )

    return fp8_data, scales_flat


@_fp8_tensorwise_per_group_quantize.register_fake
def _fake_fp8_tensorwise_per_group_quantize(
    tensor: torch.Tensor,
    offs: torch.Tensor,
    output_dtype: torch.dtype,
    output_scale_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, D = tensor.shape
    num_groups = offs.numel()
    fp8_data = torch.empty(M, D, dtype=output_dtype, device=tensor.device)
    scales_flat = torch.empty(
        output_scale_dim * num_groups,
        dtype=torch.float32,
        device=tensor.device,
    )
    return fp8_data, scales_flat
