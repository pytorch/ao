# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from torch import Tensor

from .dim import Dim0Grouper, Dim1Grouper
from .packed import PackedGrouperMixin


class QKGrouper(PackedGrouperMixin, Dim1Grouper):
    """Grouper applied only to query and key weights. Assumes that query, key,
    value weights are packed along `qk_pack_dim` dimension.

    Args:
        p (Tensor): The packed query, key, value weights.
        qk_pack_dim (int): Dimension along which query and key are packed.
        qk_reg_index (int, optional): 0 for query, 1 for key. Default: 0.
    """

    def __init__(
        self,
        p: Tensor,
        qk_pack_dim: int = 0,
        qk_reg_index: int = 0,
    ):
        super().__init__(p, 3, qk_pack_dim)

        if qk_reg_index == 0:  # query
            start, end = 0, self.embed_dim
        else:  # key
            start, end = self.embed_dim, self.embed_dim * 2

        super(PackedGrouperMixin, self).__init__(
            p[start:end] if qk_pack_dim == 0 else p[:, start:end]
        )


class AttentionHeadGrouperDim0(Dim0Grouper):
    """Grouper for attention heads.

    Args:
        p (Tensor): Input tensor with packed attention heads.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, p: Tensor, num_heads: int):
        super().__init__(p)

        self.num_heads = num_heads
        self.head_dim = p.size(0) // num_heads

    def __enter__(self):
        self.p.data = self.p.data.view(self.num_heads, -1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.p.data = self.p.data.view(self.num_heads * self.head_dim, -1)
        super().__exit__(exc_type, exc_val, exc_tb)


class AttentionHeadGrouperDim1(Dim1Grouper):
    def __init__(self, p: Tensor, num_heads: int):
        self._orig_p = p
        self.head_dim = p.size(1) // num_heads
        p = p.view(-1, num_heads, self.head_dim).transpose(1, 2).contiguous()
        super().__init__(p)

        self.num_heads = num_heads

    def __enter__(self):
        self.p.data = self.p.data.view(-1, self.num_heads)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.p.data = self.p.data.view(-1, self.head_dim, self.num_heads)
        self.p = self.p.data.transpose(1, 2).contiguous().view(self._orig_p.shape)
        self._orig_p.data.copy_(self.p.data)
        super().__exit__(exc_type, exc_val, exc_tb)
