# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch


def ceil_div(x: int, y: int) -> int:
    """Helper function for ceiling division."""
    return (x + y - 1) // y


@torch.library.custom_op(
    "torchao::cutedsl_grouped_gemm", mutates_args={"output_tensor"}
)
def _cutedsl_grouped_gemm_custom_op(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: torch.Tensor,
    output_tensor: torch.Tensor,
    addmm: bool = False,
    a_offs: Optional[torch.Tensor] = None,
    b_offs: Optional[torch.Tensor] = None,
    out_offset: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> None:
    """Custom op wrapper for CuTe DSL grouped GEMM (in-place operation).

    Args:
        a: Input tensor A (2D or 3D)
        b: Input tensor B (2D or 3D)
        scale_a: MXFP8 scales for tensor A
        scale_b: MXFP8 scales for tensor B
        offs: Group offsets tensor
        output_tensor: Pre-allocated output tensor to write results into
        addmm: Whether to perform addmm operation (default: False)
        a_offs: Starting offset for tensor A (optional)
        b_offs: Starting offset for tensor B (optional)
        out_offset: Output offset for writing within the output buffer (optional)
        num_sms: Number of streaming multiprocessors to use (optional)
        out_dtype: Output data type (default: torch.bfloat16)

    Returns:
        None (writes results in-place to output_tensor)
    """
    from torchao.prototype.moe_training.ep.syncless.cutedsl_mxfp8_gmm import (
        grouped_gemm,
    )

    grouped_gemm(
        a=a,
        b=b,
        scale_a=scale_a,
        scale_b=scale_b,
        offs=offs,
        addmm=addmm,
        a_offs=a_offs,
        b_offs=b_offs,
        out_offset=out_offset,
        num_sms=num_sms,
        self=output_tensor,
        out_dtype=out_dtype,
    )
    # Return None - results are written in-place to output_tensor


@_cutedsl_grouped_gemm_custom_op.register_fake
def _fake_cutedsl_grouped_gemm_custom_op(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: torch.Tensor,
    output_tensor: torch.Tensor,
    addmm: bool = False,
    a_offs: Optional[torch.Tensor] = None,
    b_offs: Optional[torch.Tensor] = None,
    out_offset: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> None:
    """Fake implementation for meta/shape inference."""

    # Validate input dtypes
    assert a.dtype == b.dtype == torch.float8_e4m3fn, (
        f"Expected float8_e4m3fn, got a: {a.dtype}, b: {b.dtype}"
    )
    assert scale_a.dtype == scale_b.dtype == torch.float8_e8m0fnu, (
        f"Expected float8_e8m0fnu scales"
    )

    # Validate tensor dimensions
    assert a.ndim == 2, f"Tensor a must be 2D, got {a.ndim}D"
    assert b.ndim in (2, 3), f"Tensor b must be 2D or 3D, got {b.ndim}D"

    G = offs.shape[0]  # Number of groups

    if b.ndim == 3:
        assert b.shape[0] == G, f"b.shape[0] ({b.shape[0]}) != G ({G})"

    # Determine output dimensions
    out_3d = a.ndim == b.ndim  # Both 3D -> 3D output

    if out_3d:
        # 3D x 3D case
        M, K = a.shape[-2:]
        N = b.shape[-1]
        out_shape = (G, M, N)
    else:
        # 2D x 3D case (most common for MoE)
        M, K = a.shape
        N = b.shape[-1]
        out_shape = (M, N)

    # Validate contraction dimension
    if not out_3d:
        assert a.shape[-1] == b.shape[-2], (
            f"Contraction dim mismatch: a.shape[-1]={a.shape[-1]} != b.shape[-2]={b.shape[-2]}"
        )

    # Determine output dtype
    if out_dtype is None:
        out_dtype = torch.bfloat16

    # Validate output tensor shape and compatibility
    if out_offset is not None:
        # When using out_offset, the output_tensor is larger and we write at an offset
        assert not out_3d, "out_offset only supported for 2D output"
        assert output_tensor.shape[0] >= M and output_tensor.shape[1] == N, (
            f"output_tensor too small for offset writing"
        )
    else:
        assert output_tensor.shape == torch.Size(out_shape), (
            f"output_tensor.shape {output_tensor.shape} != expected {out_shape}"
        )

    # Validate output tensor dtype
    if out_dtype is not None:
        expected_dtype = out_dtype
    else:
        expected_dtype = torch.bfloat16

    assert output_tensor.dtype == expected_dtype, (
        f"output_tensor.dtype {output_tensor.dtype} != expected {expected_dtype}"
    )

    # In-place operation - no return value


# Convenience function that matches the original API
def cutedsl_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: torch.Tensor,
    *,
    addmm: bool = False,
    a_offs: Optional[torch.Tensor] = None,
    b_offs: Optional[torch.Tensor] = None,
    out_offset: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
    self: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
) -> None:
    """Public API for CuTe DSL grouped GEMM custom op (in-place operation).

    This is a drop-in replacement for the original grouped_gemm function
    that uses PyTorch's custom op system for better integration with
    torch.compile, distributed training, and other PyTorch features.

    Always operates in-place, writing results to the provided output tensor.
    """
    _cutedsl_grouped_gemm_custom_op(
        a=a,
        b=b,
        scale_a=scale_a,
        scale_b=scale_b,
        offs=offs,
        output_tensor=self,
        addmm=addmm,
        a_offs=a_offs,
        b_offs=b_offs,
        out_offset=out_offset,
        num_sms=num_sms,
        out_dtype=out_dtype,
    )
