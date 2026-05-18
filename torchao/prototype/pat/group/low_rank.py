# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import Tensor

from ..distributed_utils import _is_dtensor
from ..utils import use_deterministic_algorithms
from .grouper import Grouper
from .packed import PackedGrouperMixin


class SVDGrouper(Grouper):
    """Apply SVD to regularize the singular values of a parameter tensor."""

    # Set False to allow non-deterministic SVD (e.g. cuSOLVER's gesvdj) for
    # speed; safe when only the reconstructed weight is consumed downstream.
    deterministic: bool = True

    def __init__(self, p: Tensor, pack_dim: Optional[int] = None):
        # SVD on a sharded DTensor would silently decompose only the local
        # shard; require the caller to materialize the full tensor first.
        if _is_dtensor(p):
            raise TypeError(
                f"{type(self).__name__} does not support DTensor inputs; "
                "materialize the full tensor before constructing the grouper."
            )

        self._p = p
        self.orig_shape = p.shape

        # Reshape input to 2D, then regularize its singular values
        p = p.data.squeeze()
        if pack_dim is None and p.dim() > 2:
            p = p.flatten(start_dim=1)

        # torch.linalg.svd does not support half-precision inputs
        if self._p.dtype in (torch.bfloat16, torch.float16):
            p = p.to(torch.float32)

        with use_deterministic_algorithms(self.deterministic):
            (self.U, self.p, self.Vh) = torch.linalg.svd(p, full_matrices=False)

        self.in_dims = 0

    @torch.no_grad()
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reconstruct as (U * S) @ Vh, broadcasting S along U's column axis.
        # This avoids materializing a [k, k] diagonal matrix and collapses the
        # U @ diag(S) gemm into a cheap elementwise scale. copy_ also handles
        # any dtype conversion, so the .to(...) cast can be skipped.
        self._p.copy_(((self.U * self.p) @ self.Vh).view(self.orig_shape))


class PackedSVDGrouper(PackedGrouperMixin, SVDGrouper):
    """Wrapper around SVDGrouper to handle packed tensors."""

    def __init__(self, p: Tensor, npack: int, pack_dim: int = 0):
        super().__init__(p, npack, pack_dim)

        if pack_dim == 1:
            p = p.t()

        super(PackedGrouperMixin, self).__init__(
            p.view(npack, -1, self.embed_dim), pack_dim=pack_dim
        )

    @torch.no_grad()
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Broadcast S along U's column axis to avoid building a [npack, k, k]
        # diagonal matrix.
        self._p.copy_((self.U * self.p.unsqueeze(-2)) @ self.Vh)
