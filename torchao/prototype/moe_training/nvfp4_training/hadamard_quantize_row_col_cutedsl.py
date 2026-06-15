# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL fused RHT + NVFP4 columnwise/rowwise quantize for NVFP4 training (SM100+).

A single pass over ``A`` produces both the columnwise RHT output and the rowwise plain
NVFP4 cast. The two global amaxes are taken as input: the caller computes them first via
``cutedsl_rht_amax``.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from .hadamard_cutedsl_utils import (
    CUTEDSL_NVFP4_REQUIREMENTS,
    cutedsl_nvfp4_kernels_available,
    cutedsl_nvfp4_unavailable_reason,
)

_DEFAULT_SCALING_TYPE = int(F.ScalingType.TensorWise) if hasattr(F, "ScalingType") else 0


@torch.library.custom_op("torchao::cutedsl_rht_quantize_row_col", mutates_args=())
def cutedsl_rht_quantize_row_col(
    A: torch.Tensor,
    col_global_amax: torch.Tensor,
    row_global_amax: torch.Tensor,
    sign_vector: List[int],
    stochastic_rounding: bool = False,
    hadamard_dimension: int = 16,
    scaling_type: int = _DEFAULT_SCALING_TYPE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RHT + NVFP4 E2M1 columnwise quantize fused with rowwise quantize.

    Produces both:
      - Columnwise: quantize of RHT(A.t()) scaled by ``col_global_amax`` -> (N, M//2) packed
        FP4 + (N//128, M//64, 32, 16) swizzled scales.
      - Rowwise: direct NVFP4 quantize of A scaled by ``row_global_amax`` -> (M, N//2) packed
        FP4 + (M//128, N//64, 32, 16) swizzled scales.

    Args:
        A: (M, N) bfloat16, row-major. M must be divisible by 256, N by 128.
        col_global_amax: scalar float32 ``max(abs(RHT(A.t())))``. Compute via
            ``cutedsl_rht_amax`` (and optionally all-reduce for TP) before passing in.
        row_global_amax: scalar float32 ``max(abs(A))``. Same source.
        sign_vector: RHT sign vector as a list of ints.
        stochastic_rounding: not implemented (raises if True).
        hadamard_dimension: Hadamard dimension (only 16 supported).
        scaling_type: int encoding of F.ScalingType. Only TensorWise is supported.

    Returns:
        4-tuple (col_fp4, col_sf, row_fp4, row_sf):
          - col_fp4: (N, M//2) uint8 packed FP4 codes (columnwise).
          - col_sf:  (N//128, M//64, 32, 16) float8_e4m3fn swizzled scale factors.
          - row_fp4: (M, N//2) uint8 packed FP4 codes (rowwise).
          - row_sf:  (M//128, N//64, 32, 16) float8_e4m3fn swizzled scale factors.

    Raises:
        NotImplementedError: pre-SM100 / missing CuteDSL runtime, or stochastic_rounding=True.
        ValueError: bad dtype/shape/divisibility/amax, or unsupported hadamard_dimension/scaling_type.
    """
    if torch.cuda.is_available() and not cutedsl_nvfp4_kernels_available():
        raise NotImplementedError(
            f"cutedsl_rht_quantize_row_col requires {CUTEDSL_NVFP4_REQUIREMENTS} "
            f"({cutedsl_nvfp4_unavailable_reason()})."
        )
    if stochastic_rounding:
        raise NotImplementedError(
            "stochastic rounding is not implemented in the cutedsl kernel"
        )
    if hadamard_dimension != 16:
        raise ValueError(f"hadamard_dimension must be 16, got {hadamard_dimension}")
    if scaling_type != _DEFAULT_SCALING_TYPE:
        raise ValueError(
            f"scaling_type={scaling_type!r} is not supported; only ScalingType.TensorWise."
        )

    from ._cutedsl_kernels_impl import _cutedsl_rht_quantize_row_col_impl

    return _cutedsl_rht_quantize_row_col_impl(
        A, col_global_amax, row_global_amax, tuple(sign_vector)
    )


@cutedsl_rht_quantize_row_col.register_fake
def _(
    A,
    col_global_amax,
    row_global_amax,
    sign_vector,
    stochastic_rounding=False,
    hadamard_dimension=16,
    scaling_type=_DEFAULT_SCALING_TYPE,
):
    M, N = A.shape
    col_fp4 = A.new_empty((N, M // 2), dtype=torch.uint8)
    col_sf = A.new_empty((N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn)
    row_fp4 = A.new_empty((M, N // 2), dtype=torch.uint8)
    row_sf = A.new_empty((M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn)
    return col_fp4, col_sf, row_fp4, row_sf
