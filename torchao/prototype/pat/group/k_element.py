# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from torch import Tensor

from .dim import DimGrouperMixin
from .grouper import Grouper


class KElementGrouper(DimGrouperMixin, Grouper):
    """Groups each row of a parameter tensor into chunks of k elements.

    For a 2D tensor (M, N), produces ceil(N/k) groups per row,
    each of size k (last group may be smaller if N % k != 0).
    """

    def __init__(self, p: Tensor, k: int, start_dim: int = 1, end_dim: int = -1):
        assert k > 0, "k must be positive"
        super().__init__(start_dim, end_dim)
        super(DimGrouperMixin, self).__init__(p, in_dims=0)
        self.k = k

    def __enter__(self):
        super().__enter__()
        M, N = self.p.shape
        remainder = N % self.k
        if remainder == 0:
            self.p = self.p.view(-1, self.k)
        else:
            pad_size = self.k - remainder
            self._pad_size = pad_size
            self._unpadded_shape = self.p.shape
            self.p = F.pad(self.p, (0, pad_size)).view(-1, self.k)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_pad_size"):
            M = self._unpadded_shape[0]
            N_padded = self._unpadded_shape[1] + self._pad_size
            unpadded = self.p.reshape(M, N_padded)[:, : self._unpadded_shape[1]]
            self._param.data.copy_(unpadded.contiguous().view(self.orig_shape))
            del self._pad_size, self._unpadded_shape
        super().__exit__(exc_type, exc_val, exc_tb)

    def group_size(self):
        return self.k
