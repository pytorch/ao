# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor

from .grouper import Grouper


class ConvFilterGrouper(Grouper):
    def __init__(self, p: Tensor):
        assert p.dim() == 4, "ConvFilterGrouper only supports 4D tensors"
        super().__init__(p, in_dims=0)

    def __enter__(self):
        self.p.data = self.p.data.view(self.orig_shape[0] * self.orig_shape[1], -1)
        return self
