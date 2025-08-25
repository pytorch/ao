# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.prototype.mx_formats.kernels import triton_mx_block_rearrange

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


def _to_blocked_single(scales: Tensor) -> Tensor:
    """Assume that we have a 128x4 block of scales in K Major order

    To see more information on the individual tile layout:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    """
    assert scales.shape == (128, 4)
    scales_tiled = scales.view(4, 32, 4)  # view as 4 - (32, 4) tiles
    return scales_tiled.transpose(0, 1).reshape(32, 16)  # Interleave tiles


def to_blocked_per_group_2d(
    x_scales: Tensor, group_offs: Tensor, Mg: int, K: int, block_size: int = 32
) -> Tensor:
    """
    Convert scales to blocked format for a 2D tensor (input activations / token groups)

    Args:
        x_scales: Tensor with per group scales in blocked format concatenated into one tensor.
        group_offs: Tensor of shape (num_groups,) which contains the end index of each group along the Mg dimension.
        Mg: total size of all groups summed together
        K: K dim size

    Returns:
        blocked_scales: Tensor
        start_row_after_padding: Tensor of shape (num_groups,) which contains the start row after padding for each group.
    """
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp4_quantize import _to_blocked

    assert x_scales.ndim == 2, "x_scales must be 2D"
    assert block_size == 32, "Only block_size=32 is supported for now"
    blocked_scales_list = []
    start_row_after_padding_list = [0]
    group_start_idx = 0
    for i, group_end_idx in enumerate(group_offs.tolist()):
        group_size = group_end_idx - group_start_idx
        prev_start_row_after_padding = start_row_after_padding_list[i]
        if group_size == 0:
            start_row_after_padding_list.append(prev_start_row_after_padding)
            continue

        # Convert group scales to blocked format
        group_scales = x_scales[group_start_idx:group_end_idx]
        group_scales_blocked = _to_blocked(group_scales)
        blocked_scales_list.append(group_scales_blocked)

        # Calculate the start row after padding
        scaling_groups_per_row = K // block_size
        rows_for_group = group_scales_blocked.numel() // scaling_groups_per_row
        new_start_row = prev_start_row_after_padding + rows_for_group
        start_row_after_padding_list.append(new_start_row)

        # Update next group start index
        group_start_idx = group_end_idx

    blocked_scales = torch.cat(blocked_scales_list, dim=0).contiguous()
    blocked_scales = blocked_scales.reshape(-1, K // 32)
    start_row_after_padding = torch.tensor(
        start_row_after_padding_list, device=x_scales.device, dtype=torch.int64
    )
    return blocked_scales, start_row_after_padding


def to_blocked_per_group_3d(weight_scales: Tensor) -> Tensor:
    """
    Convert scales to blocked format for each group for a 3D tensor (expert weights)

    Args:
        scales: Tensor of shape (E, N, K//block_size)
        group_offs: Tensor of shape (num_groups,) which contains the end index of each group along the
    """
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp4_quantize import _to_blocked

    blocked_scales_list = []
    num_groups = weight_scales.shape[0]
    for i in range(num_groups):
        group_scales = weight_scales[i]
        group_scales_blocked = _to_blocked(group_scales)
        blocked_scales_list.append(group_scales_blocked)
    weight_scales_blocked = torch.stack(blocked_scales_list, dim=0).contiguous()
    weight_scales_blocked = weight_scales_blocked.reshape(num_groups, -1)
    return weight_scales_blocked
