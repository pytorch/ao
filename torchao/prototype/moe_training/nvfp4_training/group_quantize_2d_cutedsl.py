# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL dense-expert grouped 2D NVFP4 E2M1 weight quantization (no RHT), SM100+.

Drop-in for ``triton_group_weight_quantize_2d``: for each expert it produces the rowwise weight
quantize (for the forward grouped GEMM) and the colwise = quantize of ``W.T`` (for the dgrad
grouped GEMM). One kernel launch covers the whole ``(E, M, N)`` stack -- experts are equal-sized
and contiguous, so the stack is handed to the fused kernel as its ``(E*M, N)`` view and only the
columnwise stores carry a per-expert offset.

Weights are NOT Hadamard-rotated, so this uses the no-MMA (``apply_rht=False``) variant with the
balanced 8-col/8-row warp split, same as the non-grouped ``cutedsl_weight_quantize_2d``.
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
    """Per-expert 2D (16x16) NVFP4 E2M1 weight quantization without RHT (CuteDSL, SM100+).

    Args:
        A: Dense ``(E, M, N)`` BF16 weights, contiguous. Each expert is a contiguous 2D matrix.
            M == out_features and N == in_features must both be divisible by 128 (the fused
            kernel's tiling constraints, the same shapes the Triton kernel accepts).
        global_amax: ``(E,)`` float32 per-expert absolute maxima. The caller computes
            ``A[e].float().abs().max()`` (and optionally all-reduces for tensor parallelism).
        num_tensors: Number of experts; must equal ``E``.

    Returns:
        A 4-tuple matching ``triton_group_weight_quantize_2d``:
          - ``(E, M, N//2)`` uint8 rowwise FP4 codes.
          - ``(E, M//128, N//64, 32, 16)`` float8_e4m3fn rowwise swizzled scales.
          - ``(E, N, M//2)`` uint8 colwise FP4 codes (rowwise W.T).
          - ``(E, N//128, M//64, 32, 16)`` float8_e4m3fn colwise swizzled scales.

    Raises:
        NotImplementedError: pre-SM100 / missing CuteDSL runtime.
        ValueError: bad dtype/shape/storage, or out_features not divisible by 128.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_weight_quantize_2d")
    if A.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16, got {A.dtype}")
    if A.ndim != 3:
        raise ValueError("Tensor A must be 3-D")
    if not A.is_contiguous():
        raise ValueError("A must be contiguous")

    E, M, N = A.shape
    if E != num_tensors:
        raise ValueError(f"Expected {num_tensors} experts, got {E}")
    if global_amax.shape != (E,):
        raise ValueError(f"global_amax must have shape ({E},)")
    if global_amax.dtype != torch.float32:
        raise ValueError(f"Expected float32 global_amax, got {global_amax.dtype}")
    if not global_amax.is_cuda or global_amax.device != A.device:
        raise ValueError("global_amax must be on the same device as A")
    if not global_amax.is_contiguous():
        raise ValueError("global_amax must be contiguous")
    if M % 128 != 0:
        raise ValueError(
            f"cutedsl_group_weight_quantize_2d requires out_features (dim 1) divisible "
            f"by 128, got {M}"
        )
    if N % 128 != 0:
        raise ValueError(f"Expected N divisible by 128, got N={N}")

    from ._cutedsl_kernels_impl import _cutedsl_group_weight_quantize_2d_impl

    return _cutedsl_group_weight_quantize_2d_impl(A, global_amax)


@cutedsl_group_weight_quantize_2d.register_fake
def _(A, global_amax, num_tensors):
    E, M, N = A.shape
    qa = A.new_empty((E, M, N // 2), dtype=torch.uint8)
    sfa = A.new_empty((E, M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn)
    qa_t = A.new_empty((E, N, M // 2), dtype=torch.uint8)
    sfa_t = A.new_empty((E, N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn)
    return qa, sfa, qa_t, sfa_t
