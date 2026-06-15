# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL RHT global-amax kernel for NVFP4 training (SM100+).

In one tensor-core pass over ``A`` computes the two global amaxes NVFP4 two-level
scaling needs: ``max|RHT(A.t())|`` for the columnwise output and ``max|A|`` for the
rowwise output. The columnwise amax is taken over the post-RHT data, not the plain
amax: RHT can raise the per-block max, and a too-small global scale saturates the
E4M3 block scales.
"""

from typing import List, Tuple

import torch

from .hadamard_cutedsl_utils import (
    CUTEDSL_NVFP4_REQUIREMENTS,
    cutedsl_nvfp4_kernels_available,
    cutedsl_nvfp4_unavailable_reason,
)


@torch.library.custom_op("torchao::cutedsl_rht_amax", mutates_args=())
def cutedsl_rht_amax(
    A: torch.Tensor,
    sign_vector: List[int],
    hadamard_dimension: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the post-RHT columnwise amax and the plain rowwise amax of ``A``.

    Args:
        A: (M, N) bfloat16 tensor, row-major. M must be divisible by 256, N by 128.
        sign_vector: Sign vector for the RHT as a list of ints.
        hadamard_dimension: Dimension of the Hadamard matrix (only 16 supported).

    Returns:
        Tuple ``(col_amax, row_amax)`` of scalar float32 tensors:
          - col_amax: ``max(abs(RHT(A.t())))``.
          - row_amax: ``max(abs(A))``.

    Raises:
        NotImplementedError: If the CuteDSL runtime / Blackwell hardware is unavailable.
        ValueError: If ``hadamard_dimension`` is not 16.
    """
    if torch.cuda.is_available() and not cutedsl_nvfp4_kernels_available():
        raise NotImplementedError(
            f"cutedsl_rht_amax requires {CUTEDSL_NVFP4_REQUIREMENTS} "
            f"({cutedsl_nvfp4_unavailable_reason()})."
        )
    if hadamard_dimension != 16:
        raise ValueError(f"hadamard_dimension must be 16, got {hadamard_dimension}")

    from ._cutedsl_kernels_impl import _cutedsl_rht_amax_impl

    col_amax, row_amax = _cutedsl_rht_amax_impl(A, tuple(sign_vector))
    # Return scalars (shape ()) for the per-tensor NVFP4 scale.
    return col_amax.reshape(()), row_amax.reshape(())


@cutedsl_rht_amax.register_fake
def _(A, sign_vector, hadamard_dimension=16):
    col_amax = A.new_empty((), dtype=torch.float32)
    row_amax = A.new_empty((), dtype=torch.float32)
    return col_amax, row_amax
