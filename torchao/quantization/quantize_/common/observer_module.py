# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.quantization.observer import AffineQuantizedMinMaxObserver


# TODO: Build enum backend for more observers like AWQObserver, PerChannelHistogramObserver
# User should be able to handle `act_obs` directly for easy customization
class ObservedTensor(torch.nn.Linear):
    """
    A linear module with an observer for static quantization.

    This module wraps a linear layer and adds an observer that collects
    statistics during calibration. After calibration, use `quantize_`
    with subclass to convert to a quantized module.

    Example usage:
        # Step 1: PREPARE - Insert observers into the model
        model = torch.nn.Sequential(torch.nn.Linear(64, 128))
        quantize_(model, Int8StaticActivationInt8WeightConfig(step="prepare", granularity=PerRow()))

        # Step 2: CALIBRATE - Run calibration data to collect activation statistics
        for _ in range(10):
            calibration_input = torch.randn(32, 64)
            model(calibration_input)

        # Step 3: CONVERT - Convert observers to quantized layers
        quantize_(model, Int8StaticActivationInt8WeightConfig(step="convert"))

        # Step 4: INFERENCE - Use the quantized model
        output = model(torch.randn(32, 64))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_obs: "AffineQuantizedMinMaxObserver",
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass that observes activations and performs linear computation"""
        self.act_obs(input)
        output = torch.nn.functional.linear(input, self.weight, self.bias)
        return output

    @classmethod
    def from_float(
        cls,
        float_linear: torch.nn.Linear,
        act_obs: "AffineQuantizedMinMaxObserver",
    ) -> "ObservedTensor":
        """Create an observed linear from a float linear module."""
        observed_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            act_obs,
            bias=float_linear.bias is not None,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear
