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
            return smoothing_factor, quant_smooth_activation.scale
        else:
            return smoothing_factor, None


class RunningAbsMaxSmoothQuantObserver(torch.nn.Module):
    """Memory-efficient SmoothQuant observer using running per-channel absmax.

    Unlike ``SmoothQuantObserver`` which stores every calibration input in a
    list and concatenates them at convert time, this observer maintains a
    *running* per-channel absolute-maximum that is updated incrementally
    during each ``forward`` call.  Only a single example input is kept for
    activation-scale computation during ``calculate_qparams``.

    This reduces calibration memory from **O(N x features)** (where N is the
    total number of calibration samples) to **O(features)**, which prevents
    RAM spikes and OOM kills when calibrating on large datasets.

    Args:
        weight: The weight tensor to be observed (must be 2-D).
        alpha: Smoothing factor exponent, normally between 0 and 1.
               ``None`` disables smoothing (factor = 1).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        alpha: Optional[float] = 0.5,
    ):
        super().__init__()
        assert weight.ndim == 2
        self.weight = weight
        self.alpha = alpha
        self.device = weight.device
        self.x_abs_max: Optional[torch.Tensor] = None
        self.calibration_count: int = 0
        self._example_input: Optional[torch.Tensor] = None

    @torch.no_grad()
    def forward(self, input: torch.Tensor):
        flat = input.view(-1, input.shape[-1]) if input.ndim > 2 else input
        batch_abs_max = torch.max(torch.abs(flat), dim=0)[0].to("cpu")

        if self.x_abs_max is None:
            self.x_abs_max = batch_abs_max
        else:
            self.x_abs_max = torch.max(self.x_abs_max, batch_abs_max)

        self.calibration_count += 1
        if self._example_input is None:
            self._example_input = input.to("cpu")

        return input

    def calculate_qparams(self, weight_quant_kwargs=None):
        assert self.x_abs_max is not None and self.calibration_count > 0, (
            "calibrate observer first by running model on exemplar data"
        )
        x_abs_max = self.x_abs_max.to(self.device)
        w_abs_max = torch.max(torch.abs(self.weight), dim=0)[0]

        if self.alpha is None:
            smoothing_factor = torch.ones_like(x_abs_max)
        else:
            eps = torch.finfo(torch.float32).eps
            smoothing_factor = torch.pow(x_abs_max + eps, self.alpha) / torch.pow(
                w_abs_max + eps, 1 - self.alpha
            )

        if weight_quant_kwargs is not None:
            example_input = self._example_input.to(self.device)
            quant_smooth_activation = _choose_quant_func_and_quantize_tensor(
                example_input / smoothing_factor, weight_quant_kwargs
            )
            return smoothing_factor, quant_smooth_activation.scale
        else:
            return smoothing_factor, None


class SmoothQuantObservedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        obs: "SmoothQuantObserver | RunningAbsMaxSmoothQuantObserver",
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
    def from_float(
        cls,
        float_linear: torch.nn.Linear,
        obs: "SmoothQuantObserver | RunningAbsMaxSmoothQuantObserver",
    ):
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
