# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.prototype.mx_formats.kernels import triton_mx_block_rearrange

Tensor = torch.Tensor


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix, use_triton_kernel: bool = True) -> Tensor:
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
    # if torch.compiler.is_compiling() or (rows, cols) != (padded_rows, padded_cols):
    if (rows, cols) != (padded_rows, padded_cols):
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


def _to_blocked_single(scales: Tensor) -> Tensor:
    """Assume that we have a 128x4 block of scales in K Major order

    To see more information on the individual tile layout:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    """
    assert scales.shape == (128, 4)
    scales_tiled = scales.view(4, 32, 4)  # view as 4 - (32, 4) tiles
    return scales_tiled.transpose(0, 1).reshape(32, 16)  # Interleave tiles
