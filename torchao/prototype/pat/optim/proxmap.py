# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Tuple, Union

from torch import Tensor


class ProxMap(ABC):
    """Abstract base class that defines the proximal mapping interface"""

    def __init__(self, reg_lambda: float) -> None:
        self.reg_lambda = reg_lambda

    @abstractmethod
    def _get_norm(self, p: Tensor) -> Tensor:
        """Return group-level norm of p"""

    @abstractmethod
    def tau(self, p: Tensor) -> Union[float, Tensor]:
        """Return group-level regularization strength"""

    def threshold(
        self,
        p: Tensor,
        gamma: Union[Tensor, float],
        tau_reweight: Union[Tensor, float] = 1.0,
    ) -> Union[Tensor, float]:
        """Return pruning threshold"""
        return self.reg_lambda * self.tau(p) * tau_reweight * gamma

    @abstractmethod
    def apply_(
        self,
        p: Tensor,
        gamma: Union[Tensor, float],
        tau_reweight: Union[Tensor, float] = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """Apply proximal mapping to p in-place and return number of zero
        elements and group-level norm of p.

        Arguments:
            p (Tensor): full or group-level tensor to be pruned.
            gamma (float): typically the cumulative sum over step sizes.

        Returns:
            zero_elts (Tensor): number of zero elements in p after pruning.
            norm (Tensor): norm of p after pruning divided by self.tau(p).
        """
