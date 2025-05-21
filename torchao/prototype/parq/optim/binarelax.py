# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import Tensor

from ..utils import channel_bucketize
from .proxmap import ProxMap


class ProxBinaryRelax(ProxMap):
    """Prox-map of Binary Relax, Q may not be evenly spaced."""

    def __init__(self, anneal_start: int, anneal_end: int) -> None:
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end

    @torch.no_grad()
    def apply_(
        self,
        p: Tensor,
        q: Tensor,
        Q: Tensor,
        step_count: int,
        dim: Optional[int] = None,
    ) -> None:
        if step_count < self.anneal_start:
            return

        if q is None:
            # hard quantization to the nearest point in Q
            Q_mid = (Q[..., :-1] + Q[..., 1:]) / 2
            if dim is None:
                q = Q[torch.bucketize(p, Q_mid)]
            else:
                q = Q.gather(1, channel_bucketize(p, Q_mid))

        if step_count >= self.anneal_end:
            p.copy_(q)
        else:
            # linear annealing of relaxation coefficient
            theta = (step_count - self.anneal_start) / (
                self.anneal_end - self.anneal_start
            )
            p.mul_(1 - theta).add_(q, alpha=theta)
