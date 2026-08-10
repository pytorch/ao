# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL grouped fused RHT columnwise + rowwise NVFP4 quantize (SM100+).

Quantizes a row-packed ragged activation tensor per expert group, with
per-group scales from the (E,) global amax vectors.
"""

from typing import List, Optional, Tuple

import torch

from .hadamard_cutedsl_utils import raise_if_cutedsl_nvfp4_unavailable


@torch.library.custom_op(
    "torchao::cutedsl_group_rht_quantize_row_col", mutates_args=()
)
def cutedsl_group_rht_quantize_row_col(
    A: torch.Tensor,
    sign_vector: List[int],
    offsets: torch.Tensor,
    num_tensors: int,
    packed_sequence_length: int,
    hidden_size: int,
    shape_rep: int,
    a_global_amax: torch.Tensor,
    d_global_amax: torch.Tensor,
    rng_state: Optional[torch.Tensor],
    enable_stochastic_rounding: bool,
    logical_packed_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped fused RHT columnwise + direct rowwise NVFP4 E2M1 quantization.

    Args:
        A: packed (packed_sequence_length, hidden_size) bfloat16 capacity buffer,
            row-major, groups concatenated along dim 0 with 128-aligned boundaries.
        sign_vector: sign vector for the cached 16x16 RHT matrix.
        offsets: (num_tensors,) int32 cumulative row-end offsets, one per group.
        num_tensors: number of expert groups.
        packed_sequence_length: allocated row capacity of A (divisible by 128).
        hidden_size: number of columns of A (divisible by 128).
        shape_rep: grouped shape representation; only VARYING_FIRST_DIM (1).
        a_global_amax: (num_tensors,) float32 per-group rowwise amax (max|A_g|).
        d_global_amax: (num_tensors,) float32 per-group columnwise amax
            (max|RHT(A_g.t())|).
        rng_state: int64 CUDA tensor [col_seed, col_offset, row_seed, row_offset];
            required when enable_stochastic_rounding, else may be None. The caller
            owns its advancement; the op performs no host RNG.
        enable_stochastic_rounding: hardware cvt.rs rounding for both grains.
        logical_packed_length: one-element int32 CUDA tensor with the valid padded
            row count; rows beyond it are storage capacity only. Defaults to
            ``offsets[-1:]``.

    Returns:
        (qa_base, sfa, qd, sfd) matching ``triton_group_rht_quantize_row_col``:
          - qa_base: (packed_sequence_length, hidden_size//2) uint8 rowwise codes.
          - sfa: (packed_sequence_length, hidden_size//16) float8_e4m3fn -- swizzled
            scale bytes viewed in the logical 2D shape.
          - qd: (hidden_size, packed_sequence_length//2) uint8 columnwise codes.
          - sfd: (hidden_size, packed_sequence_length//16) float8_e4m3fn, swizzled
            bytes in the logical 2D shape.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_rht_quantize_row_col")

    from ._cutedsl_kernels_impl import _cutedsl_group_rht_quantize_row_col_impl

    return _cutedsl_group_rht_quantize_row_col_impl(
        A,
        sign_vector,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        a_global_amax,
        d_global_amax,
        rng_state,
        enable_stochastic_rounding,
        logical_packed_length,
    )


@cutedsl_group_rht_quantize_row_col.register_fake
def _(
    A,
    sign_vector,
    offsets,
    num_tensors,
    packed_sequence_length,
    hidden_size,
    shape_rep,
    a_global_amax,
    d_global_amax,
    rng_state,
    enable_stochastic_rounding,
    logical_packed_length=None,
):
    T, K = packed_sequence_length, hidden_size
    qa = A.new_empty((T, K // 2), dtype=torch.uint8)
    sfa = A.new_empty((T, K // 16), dtype=torch.float8_e4m3fn)
    qd = A.new_empty((K, T // 2), dtype=torch.uint8)
    sfd = A.new_empty((K, T // 16), dtype=torch.float8_e4m3fn)
    return qa, sfa, qd, sfd
