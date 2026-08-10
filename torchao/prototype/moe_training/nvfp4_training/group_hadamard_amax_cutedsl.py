# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL per-group RHT amax over a row-packed ragged tensor (SM100+).

Drop-in for ``triton_group_rht_amax``: per expert group of a row-concatenated packed
tensor, the post-RHT columnwise amax and the raw rowwise amax, without materializing
the transformed output. Graph-safe: the launch depends only on the allocated row
capacity; rows at or beyond ``logical_packed_length`` are skipped on device.
"""

from typing import List, Optional, Tuple

import torch

from .hadamard_cutedsl_utils import raise_if_cutedsl_nvfp4_unavailable

_TENSORWISE_SCALING = 0  # int(torch.nn.functional.ScalingType.TensorWise)


@torch.library.custom_op("torchao::cutedsl_group_rht_amax", mutates_args=())
def cutedsl_group_rht_amax(
    A: torch.Tensor,
    sign_vector: List[int],
    offsets: torch.Tensor,
    num_tensors: int,
    packed_sequence_length: int,
    hidden_size: int,
    shape_rep: int,
    scaling_type: int = _TENSORWISE_SCALING,
    logical_packed_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group RHT columnwise amax and raw rowwise amax (grouped, graph-safe).

    Args:
        A: packed (packed_sequence_length, hidden_size) bfloat16 tensor, row-major.
            Groups are concatenated along the row dimension with 128-aligned
            boundaries; hidden_size must be divisible by 128.
        sign_vector: sign vector for the cached 16x16 RHT matrix.
        offsets: (num_tensors,) int32 cumulative row-end offsets, one per group.
        num_tensors: number of expert groups.
        packed_sequence_length: allocated row capacity of A (divisible by 128).
        hidden_size: number of columns of A.
        shape_rep: grouped shape representation; only VARYING_FIRST_DIM (1).
        scaling_type: int encoding of F.ScalingType; only TensorWise is supported.
        logical_packed_length: one-element int32 CUDA tensor with the valid padded
            row count. Rows beyond it are storage capacity only. Defaults to
            ``offsets[-1:]``.

    Returns:
        (col_amax, row_amax), each (num_tensors,) float32:
          - col_amax[g] = max|RHT(A_g.t())|
          - row_amax[g] = max|A_g|
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_rht_amax")
    if scaling_type != _TENSORWISE_SCALING:
        raise ValueError("cutedsl_group_rht_amax supports only TensorWise scaling")

    from ._cutedsl_kernels_impl import _cutedsl_group_rht_amax_impl

    return _cutedsl_group_rht_amax_impl(
        A,
        sign_vector,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        logical_packed_length,
    )


@cutedsl_group_rht_amax.register_fake
def _(
    A,
    sign_vector,
    offsets,
    num_tensors,
    packed_sequence_length,
    hidden_size,
    shape_rep,
    scaling_type=_TENSORWISE_SCALING,
    logical_packed_length=None,
):
    col = A.new_empty((num_tensors,), dtype=torch.float32)
    row = A.new_empty((num_tensors,), dtype=torch.float32)
    return col, row
