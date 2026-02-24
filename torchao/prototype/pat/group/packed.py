# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


class PackedGrouperMixin:
    def __init__(self, p: Tensor, npack: int, pack_dim: int = 0):
        assert p.dim() == 2, f"Expected 2D tensor, got {p.dim()=}"
        assert pack_dim < p.dim(), f"Invalid {pack_dim=} for {p.shape=}"

        if pack_dim == 0:
            embed_dim = p.size(1)
            expect_shape = (embed_dim * npack, embed_dim)
        else:
            embed_dim = p.size(0)
            expect_shape = (embed_dim, embed_dim * npack)
        assert p.shape == torch.Size(expect_shape), (
            f"Expected {expect_shape=}, got {p.shape=}"
        )
        self.embed_dim = embed_dim
        self.pack_dim = pack_dim
