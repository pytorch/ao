# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor


class Quantizer(ABC):
    """Abstract base class that defines the quantization interface"""

    def __init__(self, center: bool = False) -> None:
        self.center = center

    @abstractmethod
    def get_quant_size(self, b: int) -> int:
        """Given number of bits b, return total number of quantization values"""

    @abstractmethod
    def quantize(self, p: Tensor, b: int) -> tuple[Tensor, Tensor]:
        """Provide interface for quantization:
            q, Q = Quantizer.quantize(p)
        Inputs:
            p (Tensor): tensor to be quantized
        Outputs:
            q (Tensor): quantized tensor of same size as p
            Q (Tensor): set of 2^b quantization values
        Instantiation should not modify p, leaving update to ProxMap.
        """

    @staticmethod
    def remove_mean(p: Tensor, dim: Optional[int] = None) -> tuple[Tensor, Tensor]:
        """Center parameters in a Tensor, called if self.center == True.
        Note that this is different from direct asymmetric quantization,
        and may lead to (hopefully only) slightly different performance.
        """
        mean = p.mean(dim=dim, keepdim=dim is not None)
        q = p - mean
        return q, mean
