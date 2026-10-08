# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor

from .dim import Dim0Grouper, Dim1Grouper


def _resolve_heads(
    packed_dim: int, num_heads: int | None, head_dim: int | None
) -> tuple[int, int]:
    if (num_heads is None) == (head_dim is None):
        raise ValueError("specify exactly one of num_heads or head_dim")
    name, value = (
        ("head_dim", head_dim) if head_dim is not None else ("num_heads", num_heads)
    )
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if packed_dim % value:
        raise ValueError(
            f"packed dimension {packed_dim} not divisible by {name} {value}"
        )
    if head_dim is not None:
        return packed_dim // value, value
    return value, packed_dim // value


class AttentionHeadGrouperDim0(Dim0Grouper):
    """Grouper for attention heads packed along dimension 0.

    Specify exactly one of ``num_heads`` or ``head_dim``. A fixed head width
    allows tensors with different head counts to share a pruning config.
    """

    def __init__(
        self, p: Tensor, num_heads: int | None = None, head_dim: int | None = None
    ):
        super().__init__(p)
        self.num_heads, self.head_dim = _resolve_heads(p.size(0), num_heads, head_dim)

    def __enter__(self):
        self.p = self.p.view(self.num_heads, -1)
        return self


class AttentionHeadGrouperDim1(Dim1Grouper):
    """Grouper for attention heads packed along dimension 1.

    Specify exactly one of ``num_heads`` or ``head_dim``.
    """

    def __init__(
        self, p: Tensor, num_heads: int | None = None, head_dim: int | None = None
    ):
        self._orig_p = p
        self.num_heads, self.head_dim = _resolve_heads(p.size(1), num_heads, head_dim)
        p = p.view(-1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        super().__init__(p)

    def __enter__(self):
        self.p = self.p.view(-1, self.num_heads)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        data = self.p.view(-1, self.head_dim, self.num_heads)
        data = data.transpose(1, 2).contiguous().view(self._orig_p.shape)
        self._orig_p.data.copy_(data)
        super().__exit__(exc_type, exc_val, exc_tb)
