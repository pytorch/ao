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


class RunningAbsMaxSmoothQuantObserver(torch.nn.Module):
    """Memory-efficient SmoothQuant observer using running per-channel absmax.

    Unlike ``SmoothQuantObserver`` which stores every calibration input in a
    list and concatenates them at convert time, this observer maintains a
    *running* per-channel absolute-maximum that is updated incrementally
    during each ``forward`` call.

    This reduces calibration memory from **O(N x features)** (where N is the
    total number of calibration samples) to **O(features)**, which prevents
    RAM spikes and OOM kills when calibrating on large datasets.

    **Two-Pass Calibration:**

    To compute activation quantization scales accurately, this observer uses
    a two-pass approach:

    1. **First pass**: Run calibration data through the model. The observer
       collects running per-channel abs-max values to compute the smoothing
       factor.

    2. Call ``compute_smoothing_factor()`` to finalize the smoothing factor
       and switch to second-pass mode.

    3. **Second pass**: Run the same calibration data through the model again.
       The observer now collects running min/max statistics on the *smoothed*
       activations (input / smoothing_factor).

    4. Call ``calculate_qparams()`` to get the smoothing factor and activation
       scale.

    If ``compute_smoothing_factor()`` is not called, ``calculate_qparams()``
    will use the running min/max statistics collected during the first pass
    (less accurate than two-pass but still functional).

    Args:
        weight: The weight tensor to be observed (must be 2-D).
        alpha: Smoothing factor exponent, normally between 0 and 1.
               ``None`` disables smoothing (factor = 1).

    Example::

        # First pass: collect x_abs_max
        observer = RunningAbsMaxSmoothQuantObserver(weight, alpha=0.5)
        for batch in calibration_data:
            model(batch)

        # Compute smoothing factor and prepare for second pass
        observer.compute_smoothing_factor()

        # Second pass: collect smoothed activation stats
        for batch in calibration_data:
            model(batch)

        # Get qparams
        smoothing_factor, act_scale = observer.calculate_qparams(weight_quant_kwargs)
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

        # First pass state: collect x_abs_max for smoothing factor
        self.x_abs_max: Optional[torch.Tensor] = None
        self.calibration_count: int = 0

        # Running min/max for first pass (fallback when no second pass)
        self._first_pass_min: Optional[torch.Tensor] = None
        self._first_pass_max: Optional[torch.Tensor] = None

        # Second pass state: collect smoothed activation stats
        self._smoothing_factor: Optional[torch.Tensor] = None
        self._in_second_pass: bool = False
        self._smooth_input_min: Optional[torch.Tensor] = None
        self._smooth_input_max: Optional[torch.Tensor] = None
        self._second_pass_count: int = 0

    @torch.no_grad()
    def forward(self, input: torch.Tensor):
        flat = input.view(-1, input.shape[-1]) if input.ndim > 2 else input

        if not self._in_second_pass:
            # First pass: collect x_abs_max for smoothing factor computation
            batch_abs_max = torch.max(torch.abs(flat), dim=0)[0].to("cpu")

            if self.x_abs_max is None:
                self.x_abs_max = batch_abs_max
            else:
                self.x_abs_max = torch.max(self.x_abs_max, batch_abs_max)

            # Also track running global min/max of raw activations (for fallback)
            # Note: these are GLOBAL min/max across all samples and channels
            batch_min = torch.min(flat).to("cpu")
            batch_max = torch.max(flat).to("cpu")

            if self._first_pass_min is None:
                self._first_pass_min = batch_min
                self._first_pass_max = batch_max
            else:
                self._first_pass_min = torch.min(self._first_pass_min, batch_min)
                self._first_pass_max = torch.max(self._first_pass_max, batch_max)

            self.calibration_count += 1
        else:
            # Second pass: collect min/max of smoothed activations
            assert self._smoothing_factor is not None
            smooth_factor = self._smoothing_factor.to(input.device)
            smoothed = flat / smooth_factor

            batch_min = torch.min(smoothed).to("cpu")
            batch_max = torch.max(smoothed).to("cpu")

            if self._smooth_input_min is None:
                self._smooth_input_min = batch_min
                self._smooth_input_max = batch_max
            else:
                self._smooth_input_min = torch.min(self._smooth_input_min, batch_min)
                self._smooth_input_max = torch.max(self._smooth_input_max, batch_max)

            self._second_pass_count += 1

        return input

    def compute_smoothing_factor(self) -> torch.Tensor:
        """Compute smoothing factor and prepare for second calibration pass.

        Call this method after the first pass through calibration data and
        before the second pass.

        Returns:
            The computed smoothing factor tensor.

        Raises:
            AssertionError: If called before any calibration data was observed.
        """
        assert self.x_abs_max is not None and self.calibration_count > 0, (
            "calibrate observer first by running model on exemplar data"
        )

        x_abs_max = self.x_abs_max.to(self.device)
        w_abs_max = torch.max(torch.abs(self.weight), dim=0)[0]

        if self.alpha is None:
            self._smoothing_factor = torch.ones_like(x_abs_max)
        else:
            eps = torch.finfo(torch.float32).eps
            self._smoothing_factor = torch.pow(x_abs_max + eps, self.alpha) / torch.pow(
                w_abs_max + eps, 1 - self.alpha
            )

        self._in_second_pass = True
        self._smooth_input_min = None
        self._smooth_input_max = None
        self._second_pass_count = 0

        return self._smoothing_factor

    def calculate_qparams(self, weight_quant_kwargs=None):
        """Calculate quantization parameters.

        Args:
            weight_quant_kwargs: Optional kwargs for weight quantization.
                If provided and second pass was completed, activation scale
                and zero_point are computed from the collected smoothed
                activation statistics.

        Returns:
            A tuple of (smoothing_factor, activation_scale, activation_zero_point).
            activation_scale and activation_zero_point are None if
            weight_quant_kwargs is None or if the second pass was not completed.
        """
        assert self.x_abs_max is not None and self.calibration_count > 0, (
            "calibrate observer first by running model on exemplar data"
        )

        # Compute or retrieve smoothing factor
        if self._smoothing_factor is not None:
            smoothing_factor = self._smoothing_factor.to(self.device)
        else:
            x_abs_max = self.x_abs_max.to(self.device)
            w_abs_max = torch.max(torch.abs(self.weight), dim=0)[0]

            if self.alpha is None:
                smoothing_factor = torch.ones_like(x_abs_max)
            else:
                eps = torch.finfo(torch.float32).eps
                smoothing_factor = torch.pow(x_abs_max + eps, self.alpha) / torch.pow(
                    w_abs_max + eps, 1 - self.alpha
                )

        if weight_quant_kwargs is None:
            return smoothing_factor, None, None

        # Compute activation scale from second pass statistics
        if self._in_second_pass and self._second_pass_count > 0:
            assert self._smooth_input_min is not None
            assert self._smooth_input_max is not None

            smooth_min = self._smooth_input_min.to(self.device)
            smooth_max = self._smooth_input_max.to(self.device)

            # Determine quantization parameters from weight_quant_kwargs
            qmin = weight_quant_kwargs.get("quant_min", -128)
            qmax = weight_quant_kwargs.get("quant_max", 127)
            is_symmetric = weight_quant_kwargs.get(
                "is_symmetric",
                weight_quant_kwargs.get("qscheme", None)
                in (torch.per_tensor_symmetric, torch.per_channel_symmetric),
            )

            if is_symmetric:
                # Symmetric quantization: scale based on max abs value
                abs_max = torch.max(torch.abs(smooth_min), torch.abs(smooth_max))
                scale = abs_max / ((qmax - qmin) / 2)
                scale = torch.clamp(scale, min=torch.finfo(torch.float32).eps)
                zero_point = torch.zeros(1, dtype=torch.int64, device=self.device)
            else:
                # Affine quantization: scale based on range
                scale = (smooth_max - smooth_min) / (qmax - qmin)
                scale = torch.clamp(scale, min=torch.finfo(torch.float32).eps)
                zero_point = torch.round(qmin - smooth_min / scale).to(torch.int64)

            return smoothing_factor, scale, zero_point
        else:
            # Second pass not completed, return None for activation scale
            return smoothing_factor, None, None


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
