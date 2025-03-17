# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from torch import Tensor

from .quantizer import Quantizer


class UnifQuantizer(Quantizer):
    """Uniform quantizer, range determined by multiples of |p|.mean()"""

    def __init__(self, center: bool = False) -> None:
        super().__init__(center)

    def quantize(
        self, p: Tensor, b: int, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        """Instantiation of Quantizer.quantize() method"""
        assert b >= 1
        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)

        # set range of quantization: min( b * |q|.mean(), |q|.max())
        q_abs = q.abs()
        if dim is not None:
            q_max = torch.minimum(
                b * q_abs.mean(dim=dim, keepdim=True),  # pyre-ignore[6,9]
                torch.max(q_abs, dim=dim, keepdim=True)[0],  # pyre-ignore[6]
            )
        else:
            q_max = torch.minimum(b * q_abs.mean(), torch.max(q_abs))  # pyre-ignore[6]

        # clamp to quantization range
        q.copy_(torch.minimum(torch.maximum(q, -q_max), q_max))

        # compute scale from [-2^{b-1}+0.5, 2^{b-1}-0.5] to [-q_max, q_max]
        s = q_max / (2 ** (b - 1) - 0.5)

        # scale by 1/s -> shift -0.5 -> round -> shift +0.5 -> scale by s
        # where shift ensures rounding to integers 2^{b-1}, ..., 2^{b-1}-1
        q.div_(s).sub_(0.5).round_().add_(0.5).mul_(s)

        # set of all target quantization values
        Q = s * (
            torch.arange(-(2 ** (b - 1)) + 0.5, 2 ** (b - 1), step=1, device=q.device)
        )

        # return quantized tensor and set of possible quantization values
        if self.center:
            q += mean
            Q += mean
        return q, Q
