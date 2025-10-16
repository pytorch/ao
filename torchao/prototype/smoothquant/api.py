# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import types
from dataclasses import dataclass
from typing import Optional

import torch

from torchao.core.config import AOBaseConfig
from torchao.quantization.linear_activation_scale import (
    to_weight_tensor_with_linear_activation_scale_metadata,
)
from torchao.quantization.quant_api import (
    _QUANTIZE_CONFIG_HANDLER,
    Int8StaticActivationInt8WeightConfig,
    _linear_extra_repr,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import DummyModule

from .core import (
    SmoothQuantObservedLinear,
    SmoothQuantObserver,
    SmoothQuantStep,
)


@dataclass
class SmoothQuantConfig(AOBaseConfig):
    """
    Configuration for SmoothQuant quantization when passed into quantize_()

    Args:
        base_config: Base quantization configuration that SmoothQuant is applied on top of
        step (SmoothQuantStep): The step for SmoothQuant process
            PREPARE: insert SmoothQuant Observers to linear layers
            CONVERT: convert the observed linear modules to quantized modules
            PREPARE_FOR_LOADING: convert the floating point model to a dummy smoothquant quantized model, so we can
            load the quantized weights through copy_ later
        alpha: The alpha value to determine smoothing factor. Factor = 1 if alpha is None, which means
            Fall back to conventional quantization if None
    """

    base_config: AOBaseConfig
    step: SmoothQuantStep
    alpha: Optional[float] = 0.5

    def __post_init__(self):
        self.step = self.step.lower() if isinstance(self.step, str) else self.step.value
        all_step_values = [s.value for s in SmoothQuantStep]
        if self.step not in all_step_values:
            raise ValueError(f"{self.step} is not one of {all_step_values}")


@register_quantize_module_handler(SmoothQuantConfig)
def _smooth_quant_transform(
    module: torch.nn.Module,
    config: SmoothQuantConfig,
) -> torch.nn.Module:
    step = config.step
    base_config = config.base_config

    if step == SmoothQuantStep.PREPARE:
        observer = SmoothQuantObserver(
            weight=module.weight,
            alpha=config.alpha,
        )
        return SmoothQuantObservedLinear.from_float(module, observer)

    if step == SmoothQuantStep.PREPARE_FOR_LOADING:
        # loading from pre-quantized checkpoint
        observer = SmoothQuantObserver(
            weight=module.weight,
            alpha=config.alpha,
        )
        observed_linear = SmoothQuantObservedLinear.from_float(module, observer)
        example_input = torch.randn(
            (1, module.weight.shape[1]),
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        observed_linear(example_input)

    elif step == SmoothQuantStep.CONVERT:
        if not isinstance(module, SmoothQuantObservedLinear):
            print(
                f"convert: module is not SmoothQuantObservedLinear, skipping: {type(module)}"
            )
            return module
        observed_linear = module
    else:
        raise ValueError(f"Unexpected step: {step}")

    # Compute smoothed weight parameters
    act_quant_min, act_quant_max = None, None
    if isinstance(base_config, Int8StaticActivationInt8WeightConfig):
        act_quant_min, act_quant_max = -127, 127
    smoothing_factor, act_scale = observed_linear.obs.calculate_qparams(
        act_quant_min, act_quant_max
    )
    weight = observed_linear.weight * smoothing_factor

    # Create new linear layer
    with torch.device("meta"):
        linear = torch.nn.Linear(
            observed_linear.in_features,
            observed_linear.out_features,
            observed_linear.bias is not None,
            device=observed_linear.weight.device,
            dtype=observed_linear.weight.dtype,
        )
    linear.bias = observed_linear.bias

    # Quantize weights
    if isinstance(base_config, Int8StaticActivationInt8WeightConfig):
        base_config = Int8StaticActivationInt8WeightConfig(act_scale)
    base_config_handler = _QUANTIZE_CONFIG_HANDLER[type(base_config)]
    dummy_mod = DummyModule(weight)
    quant_mod = base_config_handler(dummy_mod, base_config)
    qw = quant_mod.weight

    # Add smoothing factor metadata
    qw = to_weight_tensor_with_linear_activation_scale_metadata(
        qw, smoothing_factor.to(qw.dtype)
    )
    linear.weight = torch.nn.Parameter(qw, requires_grad=False)
    linear.extra_repr = types.MethodType(_linear_extra_repr, linear)

    return linear
