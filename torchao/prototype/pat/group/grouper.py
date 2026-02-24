# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from typing import Optional, Union

from torch import Tensor


class Grouper(ABC):
    def __init__(self, p: Tensor, in_dims: Optional[Union[int, tuple]] = None) -> None:
        self.p = p
        self.orig_shape = p.shape
        self.in_dims = in_dims

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.p.data = self.p.data.view(self.orig_shape)

    def group_size(self):
        return self.p.numel() // self.p.size(self.in_dims)

    def n_groups(self):
        return self.p.numel() // self.group_size()


class ElemGrouper(Grouper):
    def group_size(self):
        return 1


class LayerGrouper(Grouper):
    def group_size(self):
        return self.p.numel()
