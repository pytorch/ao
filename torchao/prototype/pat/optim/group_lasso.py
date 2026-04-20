# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple, Union

import torch
from torch import Tensor

from .proxmap import ProxMap


class ProxGroupLasso(ProxMap):
    def tau(self, p: Tensor) -> float:
        """Assumes that p is a group within the full tensor"""
        return math.sqrt(p.numel())

    def _get_norm(self, p):
        return torch.linalg.vector_norm(p)

    def apply_(
        self,
        p: Tensor,
        gamma: Union[Tensor, float],
        tau_reweight: Union[Tensor, float] = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        p_norm = self._get_norm(p)
        mult = torch.maximum(
            1 - self.threshold(p, gamma, tau_reweight) / p_norm,
            torch.zeros_like(p_norm),
        )
        p.mul_(mult)
        group_norm = self._get_norm(p).div_(self.tau(p))
        return mult.eq(0).sum(), group_norm


class ProxGroupLassoVectorized(ProxGroupLasso):
    def __init__(self, reg_lambda: float, reduce_dim: int) -> None:
        assert 0 <= reduce_dim < 2, (
            f"Expected reduce_dim to be 0 or 1 but got {reduce_dim}"
        )
        super().__init__(reg_lambda)
        self.reduce_dim = reduce_dim

    def tau(self, p: Tensor) -> float:
        return math.sqrt(p.size(self.reduce_dim))

    def _get_norm(self, p):
        return torch.linalg.vector_norm(p, dim=self.reduce_dim, keepdim=True)

    def apply_(
        self,
        p: Tensor,
        gamma: Union[Tensor, float],
        tau_reweight: Union[Tensor, float] = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        p_norm_vec = self._get_norm(p)
        mult = torch.maximum(
            1 - self.threshold(p, gamma, tau_reweight) / p_norm_vec,
            torch.zeros_like(p_norm_vec),
        )
        p.mul_(mult)
        group_norm = self._get_norm(p).squeeze().div_(self.tau(p))
        return mult.eq(0).sum(), group_norm
