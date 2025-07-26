# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import types
from dataclasses import dataclass
from typing import List, Optional

import torch

import torchao
from torchao.core.config import AOBaseConfig
from torchao.quantization import to_weight_tensor_with_linear_activation_scale_metadata
from torchao.quantization.quant_api import (
    _linear_extra_repr,
)
from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
    register_quantize_module_handler,
)
from torchao.utils import DummyModule

from .core import (
    AWQObservedLinear,
    AWQObserver,
)


@dataclass
class AWQConfig(AOBaseConfig):
    """
    Configuration for quantizing linear layers when passed into quantize_()

    Args:
        base_config (AOBaseConfig): The quantization config that we can apply awq on top of, e.g. 8da4w, int4 weight only
        step (str): a string of "prepare", "convert" or "load" indicating the step of AWQ process
            prepare: insert AWQ Observers to linear
            convert: convert the observed linear modules to linear modules with awq quantized weights
            load: convert the floating point model to a dummy awq quantized model
        example_input_shape (Optional[List[int]])): This is used for load step to initialize a random example input
        scale_search_space_size (int): the number of scales to search for
        set_inductor_config: if True, adjusts `torchinductor` settings to recommended values.
    """

    base_config: AOBaseConfig
    step: str
    example_input_shape: Optional[List[int]] = None
    scale_search_space_size: int = 20
    set_inductor_config: bool = True

    def __post_init__(self):
        OPTIONS = ["prepare", "convert", "load"]
        assert self.step in OPTIONS, f"Only {OPTIONS} are supported, got {self.step}"


@register_quantize_module_handler(AWQConfig)
def _awq_transform(
    module: torch.nn.Module,
    config: AWQConfig,
) -> torch.nn.Module:
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    step = config.step
    scale_search_space_size = config.scale_search_space_size
    observed_linear = None
    base_config = config.base_config

    if step == "prepare":
        observer = AWQObserver(
            module.weight,
            module.bias,
            base_config,
            scale_search_space_size,
        )
        return AWQObservedLinear.from_float(module, observer)
    elif step == "load":
        # loading from pre-quantized checkpoint
        observer = AWQObserver(
            module.weight,
            module.bias,
            base_config,
            scale_search_space_size,
        )
        observed_linear = AWQObservedLinear.from_float(module, observer)
        assert config.example_input_shape is not None, (
            "When step is load, we expect example_input_shape to be specified as well"
        )
        example_input = torch.randn(
            config.example_input_shape,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        observed_linear(example_input)
    else:
        if not isinstance(module, AWQObservedLinear):
            print(f"convert: module is not AWQObservedLinear, skipping: {type(module)}")
            return module
        observed_linear = module

    assert observed_linear is not None
    equalization_scale = observed_linear.act_obs.calculate_qparams()

    base_config_handler = _QUANTIZE_CONFIG_HANDLER[type(config.base_config)]
    dummy_mod = DummyModule(observed_linear.weight * equalization_scale)
    quant_mod = base_config_handler(dummy_mod, config.base_config)
    qw = quant_mod.weight
    qw = to_weight_tensor_with_linear_activation_scale_metadata(qw, equalization_scale)

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
