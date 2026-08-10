# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL per-expert 2D NVFP4 E2M1 weight quantization (no RHT), SM100+.

Quantizes stacked ``(E, M, N)`` expert weights, producing rowwise (W) and
colwise (W.T) FP4 codes and swizzled scale factors per expert.
"""

from typing import Tuple

import torch

from .hadamard_cutedsl_utils import raise_if_cutedsl_nvfp4_unavailable


@torch.library.custom_op("torchao::cutedsl_group_weight_quantize_2d", mutates_args=())
def cutedsl_group_weight_quantize_2d(
    A: torch.Tensor,
    global_amax: torch.Tensor,
    num_tensors: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-expert 2D NVFP4 E2M1 weight quantization without RHT (CuteDSL, SM100+).

    Args:
        A:           (E, M, N) bfloat16, contiguous. M == out_features and
                     N == in_features must both be divisible by 128 (M % 256 shapes
                     use the faster 256-row supertile).
        global_amax: (E,) float32 per-expert ``A[e].float().abs().max()`` (caller may
                     all-reduce for tensor parallelism).
        num_tensors: Number of experts; must equal E.

    Returns:
        4-tuple:
          - (E, M, N//2) uint8: rowwise FP4 codes (W).
          - (E, M//128, N//64, 32, 16) float8_e4m3fn: rowwise swizzled scale factors.
          - (E, N, M//2) uint8: colwise FP4 codes (W.T).
          - (E, N//128, M//64, 32, 16) float8_e4m3fn: colwise swizzled scale factors.

    Raises:
        NotImplementedError: pre-SM100 / missing CuteDSL runtime.
        ValueError: bad dtype/shape/expert count.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_weight_quantize_2d")
    if A.ndim != 3:
        raise ValueError("A must be 3-D (E, M, N)")
    if A.shape[0] != num_tensors:
        raise ValueError(f"Expected {num_tensors} experts, got {A.shape[0]}")

    from ._cutedsl_kernels_impl import _cutedsl_group_weight_quantize_2d_impl

    return _cutedsl_group_weight_quantize_2d_impl(A, global_amax)


@cutedsl_group_weight_quantize_2d.register_fake
def _(A, global_amax, num_tensors):
    E, M, N = A.shape
    codes = A.new_empty((E, M, N // 2), dtype=torch.uint8)
    sf = A.new_empty((E, M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn)
    t_codes = A.new_empty((E, N, M // 2), dtype=torch.uint8)
    t_sf = A.new_empty((E, N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn)
    return codes, sf, t_codes, t_sf
