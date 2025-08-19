# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F

from torchao.quantization.observer import AffineQuantizedMinMaxObserver, PerAxis
from torchao.quantization.quant_primitives import MappingType


class SmoothQuantStep(str, Enum):
    PREPARE = "prepare"
    CONVERT = "convert"
    PREPARE_FOR_LOADING = "prepare_for_loading"


class SmoothQuantObserver(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        alpha: Optional[float] = 0.5,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
    ):
        """
        A custom observer for SmoothQuant

        Args:
            weight: The weight tensor to be observed.
            alpha: The alpha value to determine smoothing factor, normally between 0 and 1.
                   Fall back to conventional quantization if alpha is None.
            quant_min: The minimum quantized value
            quant_max: The maximum quantized value
            eps: The minimum scale to avoid dividing by zero.
        """
        super().__init__()
        assert weight.ndim == 2
        self.weight = weight
        self.device = self.weight.device
        self.alpha = alpha
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps or torch.finfo(torch.float32).eps
        # act.shape = [mb, ic] (reshape if needed), wei.shape = [oc, ic]
        # *_ic_obs are used to determine smoothing_factor
        # wei_oc_obs is used to find qparams for quantization
        self.act_ic_obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC, torch.int8, PerAxis(-1), eps=self.eps
        )
        self.wei_ic_obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC, torch.int8, PerAxis(-1), eps=self.eps
        )
        self.wei_oc_obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.int8,
            PerAxis(0),
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            eps=self.eps,
        )
        self.wei_ic_obs(self.weight)

    @torch.no_grad()
    def forward(self, input: torch.Tensor):
        self.act_ic_obs(input.to("cpu"))
        return input

    def calculate_qparams(self):
        # Step 1: Get min/max per input channel (IC) from observers
        wei_min_per_ic = self.wei_ic_obs.min_val
        wei_max_per_ic = self.wei_ic_obs.max_val
        act_min_per_ic = self.act_ic_obs.min_val
        act_max_per_ic = self.act_ic_obs.max_val
        x_abs_max_per_ic = (
            torch.max(torch.abs(act_min_per_ic), torch.abs(act_max_per_ic)) + self.eps
        )
        w_abs_max_per_ic = (
            torch.max(torch.abs(wei_min_per_ic), torch.abs(wei_max_per_ic)) + self.eps
        )

        # Step 2: Calculate smoothing factor
        if self.alpha is None:
            # fall back to conventional quantization if alpha is None
            smoothing_factor = torch.ones_like(x_abs_max_per_ic)
        else:
            smoothing_factor = torch.pow(x_abs_max_per_ic, self.alpha) / torch.pow(
                w_abs_max_per_ic.to(x_abs_max_per_ic.device), 1 - self.alpha
            )

        # Step 3: Calculate activation scales for static quantization
        act_scales = None

        # Step 4: Update weight and find scales
        self.wei_oc_obs(self.weight * smoothing_factor.to(self.device))
        wei_scales, _ = self.wei_oc_obs.calculate_qparams()

        return (
            smoothing_factor.to(self.device),
            act_scales,
            wei_scales.to(self.device),
        )


class SmoothQuantObservedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        obs: SmoothQuantObserver,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.obs = obs

    def forward(self, input: torch.Tensor):
        input = self.obs(input)
        return F.linear(input, self.weight, self.bias)

    @classmethod
    def from_float(cls, float_linear: torch.nn.Linear, obs: SmoothQuantObserver):
        observed_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            obs,
            float_linear.bias is not None,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear
