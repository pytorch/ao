# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional, Tuple

import torch
from torch.ao.quantization import (
    FakeQuantizeBase,
    ObserverBase,
)


class SimpleFakeQuantize(FakeQuantizeBase):
    """
    Thin wrapper module around an Observer instance and a fake quant op.

    Args:
        observer: Observer module instance
        fake_quant_op: function with args (input, scale, zp, qmin, qmax),
            defaulting to `torch.fake_quantize_per_tensor_affine`

    TODO: This should capture most of the functionality in the existing
    toq.FakeQuantize class, but is much simpler. In the future we should
    consider deprecating toq.FakeQuantize in favor of this class
    """

    def __init__(
        self,
        observer: ObserverBase,
        fake_quant_op: Optional[Callable] = None,
    ):
        super().__init__()
        self.observer = observer
        if fake_quant_op is not None:
            self.fake_quant_op = fake_quant_op
        else:
            self.fake_quant_op = torch.fake_quantize_per_tensor_affine

    def forward(self, x):
        if self.observer_enabled[0] == 1:
            self.observer(x.detach())
            scale, zp = self.calculate_qparams()
        if self.fake_quant_enabled[0] == 1:
            qmin = self.observer.quant_min
            qmax = self.observer.quant_max
            x = self.fake_quant_op(
                x, scale, zp, self.observer.quant_min, self.observer.quant_max,
            )
        return x

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.observer.calculate_qparams()

    @property
    def scale(self) -> torch.Tensor:
        return self.observer.scale

    @property
    def zero_point(self) -> torch.Tensor:
        return self.observer.zero_point
