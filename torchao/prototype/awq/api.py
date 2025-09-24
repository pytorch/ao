# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
import types
from dataclasses import dataclass

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
    AWQObservedLinear,
    AWQObserver,
    AWQStep,
)

logger = logging.getLogger(__name__)


@dataclass
class AWQConfig(AOBaseConfig):
    """
    Configuration for quantizing linear layers when passed into quantize_()

    Args:
        base_config (AOBaseConfig): The quantization config that we can apply awq on top of, e.g. 8da4w, int4 weight only
        step (AWQStep): specifies the step for AWQ, one of PREPARE, CONVERT and PREPARE_FOR_LOADING indicating the step of AWQ process
            PREPARE: insert AWQ Observers to linear
            CONVERT: convert the observed linear modules to linear modules with awq quantized weights
            PREPARE_FOR_LOADING: convert the floating point model to a dummy awq quantized model, so we can
            load the quantized weights through copy_ later
            can use the corresponding string "prepare", "convert", "prepare_for_loading" for simplicity
        scale_search_space_size (int): the number of scales to search for
    """

    base_config: AOBaseConfig
    step: AWQStep
    scale_search_space_size: int = 20

    def __post_init__(self):
        self.step = self.step.lower()
        all_step_values = [s.value for s in AWQStep]
        if self.step not in all_step_values:
            raise ValueError(f"{self.step} is not one of {all_step_values}")


@register_quantize_module_handler(AWQConfig)
def _awq_transform(
    module: torch.nn.Module,
    config: AWQConfig,
) -> torch.nn.Module:
    step = config.step
    scale_search_space_size = config.scale_search_space_size
    observed_linear = None
    base_config = config.base_config

    if step == AWQStep.PREPARE:
        observer = AWQObserver(
            module.weight,
            module.bias,
            base_config,
            scale_search_space_size,
        )
        return AWQObservedLinear.from_float(module, observer)
    elif step == AWQStep.PREPARE_FOR_LOADING:
        # loading from pre-quantized checkpoint
        observer = AWQObserver(
            module.weight,
            module.bias,
            base_config,
            scale_search_space_size,
        )
        observed_linear = AWQObservedLinear.from_float(module, observer)
        example_input = torch.randn(
            (1, module.weight.shape[1]),
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        observed_linear(example_input)
    else:
        assert step == AWQStep.CONVERT, f"Unexpected step: {step}"
        if not isinstance(module, AWQObservedLinear):
            logger.info(
                f"convert: module is not AWQObservedLinear, skipping: {type(module)}"
            )
            return module
        observed_linear = module

    assert observed_linear is not None
    equalization_scale = observed_linear.act_obs.calculate_qparams()

    base_config_handler = _QUANTIZE_CONFIG_HANDLER[type(config.base_config)]
    dummy_mod = DummyModule(observed_linear.weight * equalization_scale)
    quant_mod = base_config_handler(dummy_mod, config.base_config)
    qw = quant_mod.weight
    assert isinstance(qw, SupportsActivationPreScaling), (
        "weight must support activation scaling through implementing `SupportsActivationPreScaling`"
    )
    # since we want to do `act` * `act_pre_scale` during runtime for speed, we'll save the
    # reciprocal of the `equalization_scale`
    qw.act_pre_scale = 1.0 / equalization_scale

    linear = torch.nn.Linear(
        observed_linear.in_features,
        observed_linear.out_features,
        observed_linear.bias != None,
        device=observed_linear.weight.device,
        dtype=observed_linear.weight.dtype,
    )
    linear.weight = torch.nn.Parameter(qw, requires_grad=False)
    linear.extra_repr = types.MethodType(_linear_extra_repr, linear)
    linear.bias = observed_linear.bias
    return linear
