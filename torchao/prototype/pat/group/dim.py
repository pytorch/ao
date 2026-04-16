# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor

from .grouper import Grouper


class DimGrouperMixin:
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __enter__(self):
        if self.p.dim() > 2:
            self._write_back = not self.p.is_contiguous()
            self.p = self.p.flatten(start_dim=self.start_dim, end_dim=self.end_dim)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.p.dim() > 2 and self._write_back:
            self._param.copy_(self.p.view_as(self._param))
        return super().__exit__(exc_type, exc_val, exc_tb)


class Dim0Grouper(DimGrouperMixin, Grouper):
    def __init__(self, p: Tensor, start_dim: int = 1, end_dim: int = -1):
        super().__init__(start_dim, end_dim)
        super(DimGrouperMixin, self).__init__(p, in_dims=0)


class Dim1Grouper(DimGrouperMixin, Grouper):
    def __init__(self, p: Tensor, start_dim: int = 1, end_dim: int = -1):
        super().__init__(start_dim, end_dim)
        super(DimGrouperMixin, self).__init__(p, in_dims=1)
