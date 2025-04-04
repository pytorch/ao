# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
from torch import Tensor

from .quantizer import Quantizer


def get_q_max(
    q: Tensor, b: int, dim: Optional[int] = None, scale_method: str = "mean"
) -> Tensor:
    if scale_method == "mean":
        # set range of quantization: min(b * |q|.mean(), |q|.max())
        q_abs = q.abs()
        if dim is not None:
            q_max = torch.minimum(
                b * q_abs.mean(dim=dim, keepdim=True),  # pyre-ignore[6,9]
                torch.max(q_abs, dim=dim, keepdim=True).values,  # pyre-ignore[6]
            )
        else:
            q_max = torch.minimum(b * q_abs.mean(), torch.max(q_abs))  # pyre-ignore[6]
    elif scale_method == "max":
        q_max = (
            q.abs().max(dim=dim, keepdim=True).values
            if dim is not None
            else q.abs().max()
        )
    else:
        raise NotImplementedError(f"Invalid {scale_method=}, choices=('mean','max')")
    return q_max


class UnifQuantizer(Quantizer):
    """Uniform and symmetric quantizer"""

    def __init__(
        self,
        center: bool = False,
        scale_method: str = "mean",
        int_shift: float = 0.5,
        zero_point: float = 0.5,
    ):
        """Set quantization function parameters.

        Args:
            center: whether to subtract p.mean() prior to quantization
            scale_method: compute scale based 'mean', multiples of |p|.mean(),
                or 'max', |p|.max() (default: 'mean')
            int_shift: float value to shift the lower bound of integer range by:
                -2^{b - 1} + int_shift (default: 0.5). Using 0.5 results in 2^b
                values. E.g., [-1.5, -0.5, 0.5, 1.5] for b=2.
            zero_point: float value to shift p by after scale and round.
        """
        assert scale_method in ("max", "mean"), f"Invalid {scale_method=}"
        super().__init__(center=center)

        self.scale_method = scale_method
        self.int_shift = int_shift
        self.zero_point = zero_point

    def get_quant_size(self, b: int) -> int:
        """Levels in [-2^{b-1} + self.int_shift, 2^{b-1} - self.int_shift].

        Note that range_absmax = 2^{b-1} - self.int_shift on both ends of the
        boundary and the interval is closed."""
        return math.floor(2**b - 2 * self.int_shift) + 1

    def quantize(
        self, p: Tensor, b: int, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        """Instantiation of Quantizer.quantize() method"""
        assert b != 0, "Please use TernaryUnifQuantizer instead"

        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)
        q_max = get_q_max(q, b, dim=dim, scale_method=self.scale_method)
        q_max.clamp_(min=torch.finfo(q.dtype).tiny)

        # clamp to quantization range
        q.copy_(torch.minimum(torch.maximum(q, -q_max), q_max))

        # scale from [-2^{b-1}+int_shift, 2^{b-1}-int_shift] to [-q_max, q_max]
        range_absmax = 2 ** (b - 1) - self.int_shift
        s = q_max / range_absmax

        # scale by 1/s -> shift -zero_point -> round -> shift +zero_point ->
        # scale by s, where shift ensures rounding to integers
        q.div_(s).sub_(self.zero_point).round_().add_(self.zero_point).mul_(s)

        # set of all target quantization values
        Q = torch.arange(
            -range_absmax, range_absmax + 1e-5, dtype=p.dtype, device=p.device
        )
        if dim is not None:
            Q = Q.unsqueeze(0).mul(s)  # broadcasted multiply requires copy
        else:
            Q.mul_(s)

        # return quantized tensor and set of possible quantization values
        if self.center:
            q += mean
            Q += mean
        return q, Q


class MaxUnifQuantizer(UnifQuantizer):
    def __init__(
        self,
        center: bool = False,
        scale_method: str = "max",
        int_shift: float = 1.0,
        zero_point: float = 0.0,
    ):
        """Set quantization function with int_shift=1.0.

        The final quantization range includes 2^b - 1 quantized values. E.g.,
        [-1, 0, 1] for b=2. The quantization scale is determined by |p|.max()
        by default and zero point is 0.0.
        """
        super().__init__(
            center=center,
            scale_method=scale_method,
            int_shift=int_shift,
            zero_point=zero_point,
        )


class AsymUnifQuantizer(Quantizer):
    def get_quant_size(self, b: int) -> int:
        """Equivalent to int_max - int_min + 1, where int_min = -2^{b-1} and
        int_max = 2^{b-1} - 1."""
        return 2**b

    def quantize(
        self, p: Tensor, b: int, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        assert b != 0, "Please use TernaryUnifQuantizer instead"

        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)

        if dim is not None:
            q_min = q.min(dim=dim, keepdim=True).values
            q_max = q.max(dim=dim, keepdim=True).values
        else:
            q_min = q.min()
            q_max = q.max()

        int_min = -(2 ** (b - 1))
        int_max = 2 ** (b - 1) - 1
        s = (q_max - q_min) / (int_max - int_min)
        s.clamp_(min=torch.finfo(q.dtype).tiny)

        zero_point = q_min.div_(s).round_()
        q.div_(s).round_().sub_(zero_point).add_(zero_point).mul_(s)

        Q = torch.arange(int_min, int_max + 1, dtype=p.dtype, device=p.device)
        if dim is not None:
            Q = Q.unsqueeze(0).mul(s)  # broadcasted multiply requires copy
        else:
            Q.mul_(s)

        # return quantized tensor and set of possible quantization values
        if self.center:
            q += mean
            Q += mean
        return q, Q


class TernaryUnifQuantizer(Quantizer):
    """Uniform quantizer for ternary bit case. Quantization range is [-1, 1]."""

    def get_quant_size(self, b: int) -> int:
        return 3

    def quantize(
        self, p: Tensor, b: int, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        assert b == 0, f"Unexpected {b=} for ternary case"

        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)

        q_max = get_q_max(q, b, dim=dim, scale_method="max")
        q_max.clamp_(min=torch.finfo(q.dtype).tiny)
        s = q_max / 1.5
        q.div_(s).round_().clamp_(min=-1, max=1).mul_(s)

        Q = torch.tensor([-1, 0, 1], dtype=p.dtype, device=p.device)
        if dim is not None:
            Q = Q.unsqueeze(0).mul(s)
        else:
            Q.mul_(s)

        if self.center:
            q += mean
            Q += mean
        return q, Q
