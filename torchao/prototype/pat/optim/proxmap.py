# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import Tensor


class ProxMap(ABC):
    """Abstract base class that defines the proximal mapping interface"""

    def __init__(self, reg_lambda: float) -> None:
        self.reg_lambda = reg_lambda

    @staticmethod
    @abstractmethod
    def tau(p: Tensor) -> float:
        """Return group-level regularization strength"""

    def threshold(self, p: Tensor, gamma: Union[Tensor, float]) -> Union[Tensor, float]:
        """Return pruning threshold"""
        return self.reg_lambda * self.tau(p) * gamma

    def apply_(self, p: Tensor, gamma: Union[Tensor, float]) -> Tensor:
        """Provide interface for pruning (modify p in-place):
            pruner.apply_(p, q, step_count)
        Inputs:
            p (Tensor): full or group-level tensor to be pruned
            gamma (float): typically the cumulative sum over step sizes
        """
        if isinstance(gamma, float) and gamma == 0:
            return torch.zeros(1, dtype=torch.long, device=p.device)
