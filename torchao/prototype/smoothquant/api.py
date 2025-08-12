# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import types
from dataclasses import dataclass
from typing import Optional

import torch

import torchao
from torchao.core.config import AOBaseConfig
from torchao.dtypes import to_affine_quantized_intx, to_affine_quantized_intx_static
from torchao.quantization.linear_activation_quantized_tensor import (
    to_linear_activation_quantized,
)
from torchao.quantization.linear_activation_scale import (
    to_weight_tensor_with_linear_activation_scale_metadata,
)
from torchao.quantization.quant_api import (
    _linear_extra_repr,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.quantization.utils import _get_per_token_block_size

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
        step (SmoothQuantStep): The step for SmoothQuant process
            PREPARE: insert SmoothQuant Observers to linear layers
            CONVERT: convert the observed linear modules to quantized modules
        alpha: The alpha value to determine smoothing factor. Factor = 1 if alpha is None, which means
            Fall back to conventional quantization if None
        quant_mode: dynamic or static quantization of activation
        smoothing_factor: The smoothing factor for the layer. Acquired from the layer's observer if None.
        act_scales: The activation scales for the layer. Acquired from the layer's observer if None.
        wei_scales: The weight scales for the layer. Acquired from the layer's observer if None.
        set_inductor_config: if True, adjusts `torchinductor` settings to recommended values.
    """

    step: SmoothQuantStep
    alpha: Optional[float] = 0.5
    quant_mode: str = "dynamic"
    smoothing_factor: Optional[torch.Tensor] = None
    act_scales: Optional[torch.Tensor] = None
    wei_scales: Optional[torch.Tensor] = None
    set_inductor_config = True

    def __post_init__(self):
        self.step = self.step.lower() if isinstance(self.step, str) else self.step.value
        all_step_values = [s.value for s in SmoothQuantStep]
        if self.step not in all_step_values:
            raise ValueError(f"{self.step} is not one of {all_step_values}")
        assert self.quant_mode in ["static", "dynamic"]


class _ActQuantizer:
    def __init__(self, target_dtype, quant_min=-127):
        self.target_dtype = target_dtype
        self.quant_min = quant_min

    def dynamic_quantize(self, input):
        return to_affine_quantized_intx(
            input,
            MappingType.SYMMETRIC,
            _get_per_token_block_size(input),
            self.target_dtype,
            self.quant_min,
        )

    def static_quantize(self, input, scale, zero_point):
        # Use tensor-wise quantization for static mode
        # This matches the expected behavior for SmoothQuant static quantization
        return to_affine_quantized_intx_static(
            input,
            scale,
            zero_point,
            (1,) + (1,) * (input.ndim - 1),
            self.target_dtype,
            self.quant_min,
        )


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
            quant_mode=config.quant_mode,
            quant_min=-127,
            quant_max=127,
            eps=torch.finfo(torch.float32).eps,
        )
        return SmoothQuantObservedLinear.from_float(module, observer)

    elif step == SmoothQuantStep.CONVERT:
        if not isinstance(module, SmoothQuantObservedLinear):
            print(
                f"convert: module is not SmoothQuantObservedLinear, skipping: {type(module)}"
            )
            return module
        observed_linear = module
    else:
        raise ValueError(f"Unexpected step: {step}")

    # Convert observed linear to quantized linear
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    # Get quantization parameters
    if all(x is not None for x in (config.smoothing_factor, config.wei_scales)):
        smoothing_factor, act_scales, wei_scales = (
            config.smoothing_factor,
            config.act_scales,
            config.wei_scales,
        )
        weight = observed_linear.weight * smoothing_factor
    else:
        smoothing_factor, act_scales, wei_scales = (
            observed_linear.obs.calculate_qparams()
        )
        weight = observed_linear.obs.weight * smoothing_factor

    # Create new linear layer
    linear = torch.nn.Linear(
        observed_linear.in_features,
        observed_linear.out_features,
        observed_linear.bias is not None,
        device=observed_linear.weight.device,
        dtype=observed_linear.weight.dtype,
    )
    linear.bias = observed_linear.bias

    # Quantize weights
    target_dtype = torch.int8
    weight = weight.to(observed_linear.weight.dtype)
    block_size = (1, weight.size(1))
    wei_zero_points = torch.zeros_like(wei_scales, dtype=torch.int64)

    qw = to_affine_quantized_intx_static(
        weight, wei_scales, wei_zero_points, block_size, target_dtype
    )

    # Apply activation quantization
    qw = to_linear_activation_quantized(
        qw, _ActQuantizer(target_dtype).dynamic_quantize
    )

    # Add smoothing factor metadata
    qw = to_weight_tensor_with_linear_activation_scale_metadata(
        qw, smoothing_factor.to(qw.dtype)
    )
    linear.weight = torch.nn.Parameter(qw, requires_grad=False)
    linear.extra_repr = types.MethodType(_linear_extra_repr, linear)

    return linear
