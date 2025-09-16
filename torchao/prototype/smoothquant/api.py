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
from torchao.quantization.quant_api import (
    _linear_extra_repr,
)
from torchao.quantization.quantize_.common import SupportsActivationPreScaling
from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
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
    observed_linear = None

    if step == SmoothQuantStep.PREPARE:
        observer = SmoothQuantObserver(
            weight=module.weight,
            alpha=config.alpha,
        )
        return SmoothQuantObservedLinear.from_float(module, observer)

    elif step == SmoothQuantStep.PREPARE_FOR_LOADING:
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
    smoothing_factor = observed_linear.obs.calculate_qparams()
    smoothing_factor = torch.clamp(smoothing_factor, min=1e-6)

    base_config_handler = _QUANTIZE_CONFIG_HANDLER[type(config.base_config)]
    dummy_mod = DummyModule(observed_linear.weight * smoothing_factor)
    quant_mod = base_config_handler(dummy_mod, config.base_config)
    qw = quant_mod.weight
    assert isinstance(qw, SupportsActivationPreScaling), (
        "weight must support activation scaling through implementing `SupportsActivationPreScaling`"
    )
    # since we want to do `act` / `smoothing_factor` during runtime for speed, we'll save the
    # reciprocal of the `smoothing_factor`
    qw.act_pre_scale = (1.0 / smoothing_factor).to(qw.dtype)

    with torch.device("meta"):
        linear = torch.nn.Linear(
            observed_linear.in_features,
            observed_linear.out_features,
            observed_linear.bias is not None,
            device=observed_linear.weight.device,
            dtype=observed_linear.weight.dtype,
        )
    linear.bias = observed_linear.bias

    linear.weight = torch.nn.Parameter(qw, requires_grad=False)
    linear.extra_repr = types.MethodType(_linear_extra_repr, linear)

    return linear
