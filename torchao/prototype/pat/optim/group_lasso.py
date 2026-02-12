# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Union

import torch
from torch import Tensor
from torch.distributed.tensor.experimental import local_map
from torch.distributed.tensor.placement_types import Replicate

from ..distributed_utils import is_dtensor
from .proxmap import ProxMap


class ProxGroupLasso(ProxMap):
    @staticmethod
    def tau(p: Tensor) -> float:
        """Assumes that p is a group within the full tensor"""
        return math.sqrt(p.numel())

    def _get_norm(self, p):
        return torch.linalg.vector_norm(p)

    def apply_(self, p: Tensor, gamma: Union[Tensor, float]) -> Tensor:
        super().apply_(p, gamma)
        p_norm = self._get_norm(p)
        mult = torch.maximum(
            1 - self.threshold(p, gamma) / p_norm, torch.zeros_like(p_norm)
        )
        p.mul_(mult)
        return mult.eq(0).sum()


class ProxGroupLassoReduce(ProxGroupLasso):
    @staticmethod
    def partial_norm(p):
        return p.square().sum()

    def _get_norm(self, p):
        assert is_dtensor(p), f"Expected DTensor input but got {type(p)}"
        partial_norm = local_map(
            self.partial_norm,
            out_placements=(Replicate() for _ in p.placements),
            device_mesh=p.device_mesh,
        )(p)
        if partial_norm.dim() > 0:
            partial_norm = partial_norm.sum()
        return partial_norm.sqrt()
