# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

from torch import Tensor

from .proxmap import ProxMap


class ProxLasso(ProxMap):
    def tau(self, p: Tensor) -> float:
        return 1.0

    def _get_norm(self, p):
        return p.abs()

    def apply_(
        self,
        p: Tensor,
        gamma: Union[Tensor, float],
        tau_reweight: Union[Tensor, float] = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        mult = (1 - self.threshold(p, gamma, tau_reweight) / self._get_norm(p)).clamp(
            min=0
        )
        p.mul_(mult)
        return mult.eq(0).sum(), self._get_norm(p)
