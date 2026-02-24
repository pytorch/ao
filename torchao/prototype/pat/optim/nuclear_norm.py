# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
from torch import Tensor

from .proxmap import ProxMap


class ProxNuclearNorm(ProxMap):
    @staticmethod
    def tau(p: Tensor) -> float:
        return 1.0

    def apply_(self, p: Tensor, gamma: Union[Tensor, float]) -> Tensor:
        super().apply_(p, gamma)
        thresh = self.threshold(p, gamma)
        zero_mask = p.le(thresh)
        p.sub_(torch.where(zero_mask, p, thresh))
        return zero_mask.sum()
