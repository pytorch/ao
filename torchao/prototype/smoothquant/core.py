# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
import torch.nn.functional as F

from torchao.quantization.quantize_.common import (
    _choose_quant_func_and_quantize_tensor,
)


class SmoothQuantObserver(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        alpha: Optional[float] = 0.5,
    ):
        """
        A custom observer for smoothing factor, main concept of SmoothQuant.

        Args:
            weight: The weight tensor to be observed.
            alpha: The alpha value to determine smoothing factor, normally between 0 and 1.
        """
        super().__init__()
        assert weight.ndim == 2
        self.weight = weight
        self.alpha = alpha
        self.inputs = []
        self.device = weight.device

    @torch.no_grad()
    def forward(self, input: torch.Tensor):
        self.inputs.append(input.to("cpu"))
        return input

    def calculate_qparams(self, weight_quant_kwargs=None):
        assert self.inputs and len(self.inputs) > 0, (
            "calibrate observer first by running model on exemplar data"
        )
        inputs = [inp.to(self.device) for inp in self.inputs]
        acc = torch.cat(inputs, dim=0)
        # Reshape if needed: [batch, seq, features] -> [batch*seq, features]
        example_input_for_quantization = acc
        if acc.ndim > 2:
            acc = acc.view(-1, acc.shape[-1])

        # Calculate per-channel max values
        x_abs_max = torch.max(torch.abs(acc), dim=0)[0]
        w_abs_max = torch.max(torch.abs(self.weight), dim=0)[0]

        # Calculate smoothing factor
        if self.alpha is None:
            smoothing_factor = torch.ones_like(x_abs_max)
        else:
            eps = torch.finfo(torch.float32).eps
            smoothing_factor = torch.pow(x_abs_max + eps, self.alpha) / torch.pow(
                w_abs_max + eps, 1 - self.alpha
            )

        if weight_quant_kwargs is not None:
            quant_smooth_activation = _choose_quant_func_and_quantize_tensor(
                example_input_for_quantization / smoothing_factor, weight_quant_kwargs
            )
            return (
                smoothing_factor,
                quant_smooth_activation.scale,
                quant_smooth_activation.zero_point,
            )
        else:
            return smoothing_factor, None, None


class SmoothQuantObservedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        obs: SmoothQuantObserver,
        has_bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features, out_features, bias=has_bias, device=device, dtype=dtype
        )
        self.obs = obs

    def forward(self, input: torch.Tensor):
        input = self.obs(input)
        return F.linear(input, self.weight, self.bias)

    @classmethod
    def from_float(cls, float_linear: torch.nn.Linear, obs: SmoothQuantObserver):
        with torch.device("meta"):
            observed_linear = cls(
                float_linear.in_features,
                float_linear.out_features,
                obs,
                has_bias=float_linear.bias is not None,
                device=float_linear.weight.device,
                dtype=float_linear.weight.dtype,
            )
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear
