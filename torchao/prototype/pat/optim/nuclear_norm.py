# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

import torch
from torch import Tensor

from .proxmap import ProxMap


class ProxNuclearNorm(ProxMap):
    def tau(self, p: Tensor) -> float:
        return 1.0

    def _get_norm(self, p: Tensor) -> Tensor:
        return p

    def apply_(
        self,
        p: Tensor,
        gamma: Union[Tensor, float],
        tau_reweight: Union[Tensor, float] = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        thresh = self.threshold(p, gamma, tau_reweight)
        zero_mask = p.le(thresh)
        p.sub_(torch.where(zero_mask, p, thresh))
        return zero_mask.sum(), self._get_norm(p)
