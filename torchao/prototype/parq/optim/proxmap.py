# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from ..utils import channel_bucketize


# Create an abstract class to provide proximal-mapping interface
class ProxMap(ABC):
    @abstractmethod
    def apply_(self, p: Tensor, q: Tensor, Q: Tensor, step_count: int) -> None:
        """Provide interface for proximal mapping (modify p in-place):
            prox_map.apply_(p, q, Q, step_count)
        Inputs:
            p (Tensor): tensor to be quantized
            q (Tensor): None or hard quantized tensor of same size as p
            Q (Tensor): set of target quantization values
            step_count: trigger iteration-dependent mapping if needed
        """


class ProxHardQuant(ProxMap):
    """Prox-map of hard quantization, Q may not be evenly spaced."""

    @torch.no_grad()
    def apply_(
        self,
        p: Tensor,
        q: Tensor,
        Q: Tensor,
        step_count: int,
        dim: Optional[int] = None,
    ) -> None:
        if q is None:
            # quantize to the nearest point in Q
            Q_mid = (Q[..., :-1] + Q[..., 1:]) / 2
            if dim is None:
                q = Q[torch.bucketize(p, Q_mid)]
            else:
                q = Q.gather(1, channel_bucketize(p, Q_mid))
        p.copy_(q)
