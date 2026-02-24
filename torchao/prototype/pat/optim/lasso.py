# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

from torch import Tensor

from .proxmap import ProxMap


class ProxLasso(ProxMap):
    @staticmethod
    def tau(p: Tensor) -> float:
        return 1.0

    def apply_(self, p: Tensor, gamma: Union[Tensor, float]) -> Tensor:
        super().apply_(p, gamma)
        mult = (1 - self.threshold(p, gamma) / p.abs()).clamp(min=0)
        p.mul_(mult)
        return mult.eq(0).sum()
