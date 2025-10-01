# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch.distributed._tensor import DTensor

from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_mx_block_rearrange,
    triton_to_mxfp8_dim1,
)

Tensor = torch.Tensor


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix, use_triton_kernel: bool = False) -> Tensor:
    """
    Rearrange a large matrix by breaking it into blocks and applying the rearrangement pattern.

    See:
        https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)
        use_triton_kernel: Whether to use a triton implementation instead of relying on
            torch.compile

    Returns:
        Rearranged tensor of shape (32*ceil_div(H,128), 16*ceil_div(W,4))
    """
    if use_triton_kernel:
        return triton_mx_block_rearrange(input_matrix).flatten()

    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    # TODO This is to work around VLLM's usage of compile w/ dynamic shapes
    if torch.compiler.is_compiling() or (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def from_blocked(
    blocked_tensor: Tensor, original_rows: int, original_cols: int
) -> Tensor:
    """
    Inverse of to_blocked: convert from blocked layout back to regular row-major layout.

    Args:
        blocked_tensor: Flattened blocked tensor from to_blocked()
        original_rows: Original number of rows before blocking
        original_cols: Original number of columns before blocking

    Returns:
        Tensor of shape (original_rows, original_cols) in regular layout
    """
    n_row_blocks = ceil_div(original_rows, 128)
    n_col_blocks = ceil_div(original_cols, 4)

    rearranged = blocked_tensor.view(n_row_blocks * n_col_blocks, 32, 16)

    temp = rearranged.reshape(n_row_blocks * n_col_blocks, 32, 4, 4)

    temp = temp.transpose(1, 2)

    blocks = temp.reshape(n_row_blocks, n_col_blocks, 128, 4)

    padded_view = blocks.permute(0, 2, 1, 3)

    padded = padded_view.reshape(n_row_blocks * 128, n_col_blocks * 4)

    return padded[:original_rows, :original_cols]


def hp_data_dims_to_swizzled_scale_dims_nvfp4(
    hp_data_M,
    hp_data_K,
) -> Tuple[int, int]:
    """
    Given the `M` and `K` dimensions of a high precision contiguous tensor,
    returns a 2d tuple of the dims of the swizzled nvfp4 scale corresponding to
    that tensor.
    """
    # a 128x64 unpacked or 128x32 packed qdata tile corresponds
    # to a swizzled 32x16 scale tile
    scale_M = ceil_div(hp_data_M, 128) * 32
    scale_K = ceil_div(hp_data_K, 64) * 16
    return scale_M, scale_K


def _to_blocked_single(scales: Tensor) -> Tensor:
    """Assume that we have a 128x4 block of scales in K Major order

    To see more information on the individual tile layout:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    """
    assert scales.shape == (128, 4)
    scales_tiled = scales.view(4, 32, 4)  # view as 4 - (32, 4) tiles
    return scales_tiled.transpose(0, 1).reshape(32, 16)  # Interleave tiles


def _to_mxfp8_dim1_kernel_wrapper(
    a,
    block_size,
    elem_dtype,
    hp_dtype,
    gemm_kernel_choice,
    cast_kernel_choice,
    scale_calculation_mode: ScaleCalculationMode,
):
    # avoid circular import
    # TODO(future PR): split this utils file in two
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    if cast_kernel_choice == MXFP8Dim1CastKernelChoice.TRITON:
        assert scale_calculation_mode == ScaleCalculationMode.FLOOR
        a_data, a_scale = triton_to_mxfp8_dim1(a, block_size)
    elif cast_kernel_choice == MXFP8Dim1CastKernelChoice.CUDA:
        assert scale_calculation_mode in (
            ScaleCalculationMode.FLOOR,
            ScaleCalculationMode.RCEIL,
        )
        _, a_data, _, a_scale = mxfp8_quantize_cuda(
            a,
            rowwise=False,
            colwise=True,
            scaling_mode=scale_calculation_mode.value,
        )
    else:
        raise ValueError(f"must be one of [CUDA, TRITON], got {cast_kernel_choice}")

    if isinstance(a_data, DTensor):
        assert isinstance(a_scale, DTensor)
        a_data_local = a_data.to_local()
        a_scale_local = a_scale.to_local()
        inner = MXTensor(
            a_data_local.t(),
            a_scale_local,
            elem_dtype,
            block_size,
            hp_dtype,
            gemm_kernel_choice,
            False,
            None,
        )
        mx_tensor = DTensor.from_local(
            inner,
            a_data.device_mesh,
            a_data.placements,
            run_check=False,
            shape=a_data.t().size(),
            stride=a_data.t().stride(),
        )
    else:
        mx_tensor = MXTensor(
            a_data.t(),
            a_scale,
            elem_dtype,
            block_size,
            hp_dtype,
            gemm_kernel_choice,
            False,
            None,
        )
    return mx_tensor
