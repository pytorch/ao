# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple


def block_shape_to_group_size(block_shape, tensor_shape):
    """Calculates the total number of elements in a group from a block_shape."""
    n_group, k_group = block_shape
    n_dim, k_dim = tensor_shape

    if n_group == -1:
        n_group = n_dim
    if k_group == -1:
        k_group = k_dim

    return n_group * k_group


def group_size_to_block_shapes(
    lut_group_size: int,
    tensor_shape: Tuple[int, int],
) -> Tuple[List[int], Optional[List[int]]]:
    """
    Translates legacy integer-based group sizes into the new block_shape list format.

    This function encodes the implicit assumptions of the old system:
    - LUTs were always grouped by rows.
    - Scales were always grouped by columns.

    Args:
        lut_group_size (int): The total number of elements that shared a single LUT.
        tensor_shape (Tuple[int, int]): The shape of the weight tensor (N, K).
            This is required to calculate the number of rows for the LUT group.

    Returns:
        A tuple containing:
        - lut_block_shape (List[int]): The new block shape for LUTs (e.g., [N, -1]).
        - scale_block_shape (Optional[List[int]]): The new block shape for scales
          (e.g., [-1, K]), or None.
    """
    _n_rows, k_cols = tensor_shape

    if lut_group_size % k_cols != 0:
        raise ValueError(
            f"lut_group_size ({lut_group_size}) must be divisible by the number "
            f"of columns ({k_cols}) for legacy row-grouping."
        )
    rows_per_lut = lut_group_size // k_cols
    lut_block_shape = [rows_per_lut, -1]

    return lut_block_shape
