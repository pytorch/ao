# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL 2D NVFP4 E2M1 weight quantization (no RHT), SM100+.

Drop-in for ``triton_weight_quantize_2d``: produces both the rowwise weight quantize (for the
forward GEMM) and the colwise = quantize of ``W.T`` (for the dgrad GEMM). Weights are NOT
Hadamard-rotated, so this uses the no-MMA (``apply_rht=False``) variant of the fused kernel: with
no tensor-core matmul, the columnwise warps read the transposed bf16 tile straight from SMEM (a
plain ``A.t()`` transpose-quantize) and the rowwise warps read it in the row grain (plain
``NVFP4(A)``). The warp layout is balanced col=8/row=8 for the symmetric 2D-quantize work (the RHT
kernel's 4-col/8-row split was sized for a cheap TMEM col-epilogue and starves this path).

Unlike the Triton 2D weight kernel — which shares one scale across each 16x16 block — this emits
canonical NVFP4 1x16 block scales (one scale per 16 contiguous elements): finer / slightly more
accurate, with the identical output layout, and ~3.8-4.0x faster than the Triton kernel.
"""

from typing import Tuple

import torch

from .hadamard_cutedsl_utils import raise_if_cutedsl_nvfp4_unavailable


@torch.library.custom_op("torchao::cutedsl_weight_quantize_2d", mutates_args=())
def cutedsl_weight_quantize_2d(
    A: torch.Tensor,
    global_amax: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D NVFP4 E2M1 weight quantization without RHT (CuteDSL, SM100+).

    Args:
        A:           (M, N) bfloat16, row-major. M == out_features must be divisible by 256,
                     N == in_features by 128 (the fused kernel's tiling constraints, stricter
                     than the Triton kernel's M % 128).
        global_amax: scalar float32 ``A.float().abs().max()`` (caller may all-reduce for TP).

    Returns:
        4-tuple matching ``triton_weight_quantize_2d``:
          - (M, N//2) uint8: rowwise FP4 codes (W).
          - (M//128, N//64, 32, 16) float8_e4m3fn: rowwise swizzled scale factors.
          - (N, M//2) uint8: colwise FP4 codes (W.T).
          - (N//128, M//64, 32, 16) float8_e4m3fn: colwise swizzled scale factors.

    Raises:
        NotImplementedError: pre-SM100 / missing CuteDSL runtime.
        ValueError: bad dtype/shape, or out_features not divisible by 256.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_weight_quantize_2d")
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    M, N = A.shape
    if M % 256 != 0:
        raise ValueError(
            f"cutedsl_weight_quantize_2d requires out_features (dim 0) divisible by 256, got {M}"
        )

    from ._cutedsl_kernels_impl import (
        DEFAULT_SIGN_VECTOR,
        _cutedsl_rht_quantize_row_col_impl,
    )

    # apply_rht=False -> no-MMA path: col reads transposed A from SMEM -> NVFP4(A.t()) = W.T, row
    # path = NVFP4(A) = W. Both global amaxes are max|A| (max|A.t()| == max|A|). sign_vector unused.
    col_fp4, col_sf, row_fp4, row_sf = _cutedsl_rht_quantize_row_col_impl(
        A, global_amax, global_amax, DEFAULT_SIGN_VECTOR, apply_rht=False
    )
    # impl returns (col=W.T, col_sf, row=W, row_sf); reorder to triton's (W, W_sf, W.T, W.T_sf).
    return row_fp4, row_sf, col_fp4, col_sf


@cutedsl_weight_quantize_2d.register_fake
def _(A, global_amax):
    M, N = A.shape
    codes = A.new_empty((M, N // 2), dtype=torch.uint8)
    sf = A.new_empty((M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn)
    t_codes = A.new_empty((N, M // 2), dtype=torch.uint8)
    t_sf = A.new_empty((N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn)
    return codes, sf, t_codes, t_sf
