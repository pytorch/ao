# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL grouped fused RHT + NVFP4 quantize (SM100+).

Drop-in backend for ``triton_group_rht_quantize_row_col``: same signature, same
output contract, and byte-for-byte identical output under RTNE. Under stochastic
rounding the outputs are statistically equivalent but not bitwise equal -- this kernel
draws one Philox counter per 16-element block and consumes all four output words
instead of reproducing triton's per-packed-byte counter stride. The stream stays a pure
function of tile coordinates and ``rng_state``, so results remain reproducible.

The kernel is a structural port of TransformerEngine's
``nvte_group_hadamard_transform_cast_fusion_graph_safe``; see
``_cutedsl_group_kernels_impl``.
"""

from typing import List, Optional, Tuple

import torch

from .group_hadamard_utils import _validate_grouped_hadamard_inputs
from .group_rht_quantize_row_col_triton import _validate_graph_amax, _validate_rng_state
from .hadamard_cutedsl_utils import raise_if_cutedsl_nvfp4_unavailable
from .hadamard_utils import _device_key, get_rht_matrix


@torch.library.custom_op("torchao::cutedsl_group_rht_quantize_row_col", mutates_args=())
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
    use_fast_math: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped fused RHT columnwise + direct rowwise NVFP4 E2M1 quantization.

    Signature and returns match ``triton_group_rht_quantize_row_col``. ``A`` is
    the pre-packed ``(packed_sequence_length, hidden_size)`` capacity buffer;
    rows at or after ``logical_packed_length == offsets[-1]`` are untouched
    allocation capacity and must not be consumed. ``offsets`` is int32
    cumulative row-end offsets, one per group. Zero-valued per-group padding
    before the final offset is processed normally.

    ``shape_rep`` is validated but does not reach the kernel: group membership
    is read from ``offsets`` alone, which is correct for both SAME_BOTH_DIMS and
    VARYING_FIRST_DIM.

    Returns ``(qa_base, sfa, qd, sfd)``; both scale tensors carry swizzled bytes
    reinterpreted to their logical 2D shapes.

    ``use_fast_math=True`` selects the compile-specialized TE-fast-compatible
    arithmetic path; False preserves TE-default-exact output.

    Raises:
        NotImplementedError: pre-SM100 or a missing CuteDSL runtime.
        ValueError: bad shapes/dtypes, ``num_tensors`` above the kernel's group
            cap, or stochastic rounding without a valid ``rng_state``.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_rht_quantize_row_col")

    from ._cutedsl_group_kernels_impl import (
        MAX_GROUPS,
        _cutedsl_group_rht_quantize_row_col_impl,
    )

    B = get_rht_matrix(tuple(sign_vector), _device_key(A.device), torch.bfloat16, 16)
    _validate_grouped_hadamard_inputs(
        A,
        B,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        logical_packed_length,
    )
    if num_tensors > MAX_GROUPS:
        raise ValueError(
            f"num_tensors must be <= {MAX_GROUPS} for the CuteDSL grouped kernel, "
            f"got {num_tensors}"
        )
    row_amax = _validate_graph_amax(
        a_global_amax, "a_global_amax", num_tensors, A.device
    )
    col_amax = _validate_graph_amax(
        d_global_amax, "d_global_amax", num_tensors, A.device
    )
    rng_state = _validate_rng_state(rng_state, A.device, enable_stochastic_rounding)

    col_fp4, col_sf, row_fp4, row_sf = _cutedsl_group_rht_quantize_row_col_impl(
        A,
        offsets,
        row_amax,
        col_amax,
        num_tensors,
        tuple(sign_vector),
        logical_packed_length=logical_packed_length,
        stochastic_rounding=enable_stochastic_rounding,
        sr_rng=rng_state if enable_stochastic_rounding else None,
        use_fast_math=use_fast_math,
    )
    return (
        row_fp4,
        row_sf.view(packed_sequence_length, hidden_size // 16),
        col_fp4,
        col_sf.view(hidden_size, packed_sequence_length // 16),
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
    use_fast_math=False,
):
    qa_base = A.new_empty((packed_sequence_length, hidden_size // 2), dtype=torch.uint8)
    sfa = A.new_empty(
        (packed_sequence_length, hidden_size // 16), dtype=torch.float8_e4m3fn
    )
    qd = A.new_empty((hidden_size, packed_sequence_length // 2), dtype=torch.uint8)
    sfd = A.new_empty(
        (hidden_size, packed_sequence_length // 16), dtype=torch.float8_e4m3fn
    )
    return qa_base, sfa, qd, sfd
