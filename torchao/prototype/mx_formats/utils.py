# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
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

aten = torch.ops.aten


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


def hp_data_dims_to_swizzled_scale_dims_mx(
    hp_data_M,
    hp_data_K,
) -> Tuple[int, int]:
    """
    Given the `M` and `K` dimensions of a high precision contiguous tensor,
    returns a 2d tuple of the dims of the swizzled mx scale corresponding to
    that tensor.
    """
    # a 128x128 unpacked or 128x64 packed qdata tile corresponds
    # to a swizzled 32x16 scale tile
    scale_M = ceil_div(hp_data_M, 128) * 32
    scale_K = ceil_div(hp_data_K, 128) * 16
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
    kernel_preference,
    cast_kernel_choice,
    scale_calculation_mode: ScaleCalculationMode,
):
    # avoid circular import
    # TODO(future PR): split this utils file in two
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    if cast_kernel_choice == MXFP8Dim1CastKernelChoice.TRITON:
        assert scale_calculation_mode in (
            ScaleCalculationMode.FLOOR,
            ScaleCalculationMode.RCEIL,
        )
        a_data, a_scale = triton_to_mxfp8_dim1(
            a, block_size, scale_calculation_mode.value
        )
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

    is_swizzled_scales = False
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
            kernel_preference,
            None,
            is_swizzled_scales,
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
            kernel_preference,
            None,
            is_swizzled_scales,
        )
    return mx_tensor


def _swizzle_aware_slice(
    x: torch.Tensor, dim, start, end, step
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input: NVFP4Tensor or MXTensor
    Output: sliced qdata and scale, does the right thing for unswizzled and swizzled scales
    """

    M, K = x.shape[0], x.shape[1]

    # The scale manipulations below assume a flattened scale. For now, we
    # flatten the scale, go through the calculations below, and then reshape
    # it back to the format which matches the shape of `qdata`.
    # TODO(future PR): update this

    if x.is_swizzled_scales:
        scale_rows = M
        scale_cols = K // x.block_size
        n_row_blocks = ceil_div(scale_rows, 128)
        n_col_blocks = ceil_div(scale_cols, 4)
        elements_per_block = 32 * 16  # 512 elements

        if dim == 0:
            # Row slicing
            # Handle sys.maxsize (default slice end)
            if end == sys.maxsize:
                end = M

            # Check if start/end align with 128-row boundaries
            if start is not None and start % 128 != 0:
                raise RuntimeError(
                    f"Row slicing of NVFP4Tensor with swizzled scales requires "
                    f"start index to be a multiple of 128, got {start}"
                )
            if end is not None and end != M and end % 128 != 0:
                raise RuntimeError(
                    f"Row slicing of NVFP4Tensor with swizzled scales requires "
                    f"end index to be a multiple of 128 or equal to tensor size {M}, got {end}"
                )

            # Calculate which row blocks to keep
            start_block = 0 if start is None else start // 128
            end_block = n_row_blocks if end is None or end >= M else end // 128

            # The swizzled tensor has shape (n_row_blocks * n_col_blocks * 32 * 16,)
            blocks_per_row = n_col_blocks
            start_idx = start_block * blocks_per_row * elements_per_block
            end_idx = (
                end_block * blocks_per_row * elements_per_block
                if end_block < n_row_blocks
                else None
            )

            sliced_scale = aten.slice.Tensor(
                x.scale.flatten(), 0, start_idx, end_idx, 1
            )
            sliced_data = aten.slice.Tensor(x.qdata, 0, start, end, step)

        elif dim == 1:
            # Column slicing
            # Handle sys.maxsize (default slice end)
            if end == sys.maxsize:
                end = K

            # Check if start/end align with 64-column boundaries (4 scale columns * 16 block_size)
            if start is not None and start % 64 != 0:
                raise RuntimeError(
                    f"Column slicing of NVFP4Tensor with swizzled scales requires "
                    f"start index to be a multiple of 64, got {start}"
                )
            if end is not None and end != K and end % 64 != 0:
                raise RuntimeError(
                    f"Column slicing of NVFP4Tensor with swizzled scales requires "
                    f"end index to be a multiple of 64 or equal to tensor size {K}, got {end}"
                )

            # TODO(future PR): use torch.float4_e2m1fn_x2 for nvfp4 and mxfp4
            if x.qdata.dtype != torch.float8_e4m3fn:
                # Also check FP4 packing alignment
                if start is not None and start % 2 != 0:
                    raise RuntimeError(
                        f"Start index {start} must be even for FP4 packing"
                    )
                if end is not None and end != K and end % 2 != 0:
                    raise RuntimeError(f"End index {end} must be even for FP4 packing")

            # Calculate which column blocks to keep
            start_scale_col = 0 if start is None else start // 16
            end_scale_col = scale_cols if end is None or end >= K else end // 16

            start_col_block = start_scale_col // 4
            end_col_block = end_scale_col // 4

            # Verify the end aligns with block boundary
            if end_scale_col % 4 != 0:
                raise RuntimeError(
                    f"Column slicing end index {end} does not align with scale block boundaries. "
                    f"End must result in a multiple of 4 scale columns (64 data columns)."
                )

            if start_col_block == 0 and end_col_block == n_col_blocks:
                # Full width - no slicing needed
                sliced_scale = x.scale
            else:
                # Extract specific column blocks from each row block
                # Each row block in swizzled format contains n_col_blocks chunks of (32, 16)
                elements_per_row_block = n_col_blocks * elements_per_block

                # Build list of slices to extract
                slices_to_extract = []
                for row_block in range(n_row_blocks):
                    row_start = row_block * elements_per_row_block
                    col_start = row_start + start_col_block * elements_per_block
                    col_end = row_start + end_col_block * elements_per_block
                    slices_to_extract.append(x.scale.flatten()[col_start:col_end])

                # Concatenate all the slices
                sliced_scale = torch.cat(slices_to_extract, dim=0)

            # Slice the data tensor
            if x.qdata.dtype != torch.float8_e4m3fn:
                packed_start = None if start is None else start // 2
                packed_end = None if end is None else end // 2
            else:
                packed_start = start
                packed_end = end
            sliced_data = aten.slice.Tensor(
                x.qdata, dim, packed_start, packed_end, step
            )

        else:
            raise ValueError(
                f"NVFP4Tensor only supports slicing along dimensions 0 and 1, got dim={dim}"
            )

    else:
        scale_shaped = x.scale.view(M, K // x.block_size)

        if dim == 0:
            sliced_scale = aten.slice.Tensor(scale_shaped, dim, start, end, step)
            sliced_data = aten.slice.Tensor(x.qdata, dim, start, end, step)

        elif dim == 1:
            if start is not None:
                assert start % x.block_size == 0, (
                    f"Start index {start} must be a multiple of block_size {x.block_size}"
                )
                assert start % 2 == 0, (
                    f"Start index {start} must be even for FP4 packing"
                )

            if end is not None and end != sys.maxsize:
                assert end % x.block_size == 0, (
                    f"End index {end} must be a multiple of block_size {x.block_size}"
                )
                assert end % 2 == 0, f"End index {end} must be even for FP4 packing"

            if x.qdata.dtype != torch.float8_e4m3fn:
                packed_start = None if start is None else start // 2
                packed_end = None if end is None else end // 2
            else:
                packed_start = start
                packed_end = end
            sliced_data = aten.slice.Tensor(
                x.qdata, dim, packed_start, packed_end, step
            )

            start_block = 0 if start is None else start // x.block_size
            end_block = None if end is None else end // x.block_size
            sliced_scale = aten.slice.Tensor(
                scale_shaped, 1, start_block, end_block, step
            )

        sliced_scale = sliced_scale.flatten()

    # reshape at the end
    sliced_M = sliced_data.shape[0]
    if x.qdata.dtype == torch.float8_e4m3fn:
        sliced_K = sliced_data.shape[1]
    else:
        # multiply by 2 to convert from bytes to num_elements
        sliced_K = sliced_data.shape[1] * 2
    if x.is_swizzled_scales:
        if x.block_size == 16:
            scale_M, scale_K = hp_data_dims_to_swizzled_scale_dims_nvfp4(
                sliced_M, sliced_K
            )
        else:
            assert x.block_size == 32, f"unexpected {x.block_size=}"
            scale_M, scale_K = hp_data_dims_to_swizzled_scale_dims_mx(
                sliced_M, sliced_K
            )
    else:
        # nvfp4: a 1x16 unpacked or 1x8 packed qdata tile corresponds to 1
        # scale element
        # mx: a 1x32 unpacked or 1x16 packed qdata tile corresponds to 1
        # scale element
        scale_M = sliced_M
        scale_K = sliced_K // x.block_size
    sliced_scale = sliced_scale.view(scale_M, scale_K)

    return sliced_data, sliced_scale
