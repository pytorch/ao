# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import Tensor

from ..utils import use_deterministic_algorithms
from .grouper import Grouper
from .packed import PackedGrouperMixin


class SVDGrouper(Grouper):
    """Apply SVD to regularize the singular values of a parameter tensor."""

    def __init__(self, p: Tensor, pack_dim: Optional[int] = None):
        self._p = p
        self.orig_shape = p.shape

        # Reshape input to 2D, then regularize its singular values
        p = p.data.squeeze()
        if pack_dim is None and p.dim() > 2:
            p = p.flatten(start_dim=1)

        # torch.linalg.svd does not support half-precision inputs
        if self._p.dtype in (torch.bfloat16, torch.float16):
            p = p.to(torch.float32)

        with use_deterministic_algorithms():
            (self.U, self.p, self.Vh) = torch.linalg.svd(p, full_matrices=False)

        self.in_dims = 0

    @torch.no_grad()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._p.copy_(
            torch.linalg.multi_dot([self.U, torch.diag(self.p), self.Vh])
            .view(self.orig_shape)
            .to(self._p.dtype)
        )


class QKSVDGrouper(PackedGrouperMixin, SVDGrouper):
    def __init__(self, p: Tensor, pack_dim: int = 0):
        super().__init__(p, 3, pack_dim)

        self.qk_dim = self.embed_dim * 2
        super(PackedGrouperMixin, self).__init__(
            p[: self.qk_dim] if pack_dim == 0 else p[:, : self.qk_dim],
            pack_dim=pack_dim,
        )

    @torch.no_grad()
    def __exit__(self, exc_type, exc_val, exc_tb):
        p = torch.linalg.multi_dot([self.U, torch.diag(self.p), self.Vh])
        if self.pack_dim == 0:
            self._p[: self.qk_dim].copy_(p)
        else:
            self._p[:, : self.qk_dim].copy_(p)


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
        self._p.copy_(self.U @ torch.diag_embed(self.p) @ self.Vh)
