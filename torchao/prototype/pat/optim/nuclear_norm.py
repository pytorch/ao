# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

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
    ) -> tuple[Tensor, Tensor]:
        thresh = self.threshold(p, gamma, tau_reweight)
        # Soft-threshold non-negative singular values to zero: equivalent to
        # relu(p - thresh) for p >= 0. Avoids a torch.where intermediate and
        # composes under torch.vmap (relu_ has a batching rule; clamp_ does
        # not at the time of writing).
        p.sub_(thresh).relu_()
        return p.eq(0).sum(), self._get_norm(p)
