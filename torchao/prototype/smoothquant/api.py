# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import types
from dataclasses import dataclass
from typing import Dict, Optional

import torch

import torchao
from torchao.core.config import AOBaseConfig
from torchao.dtypes import to_affine_quantized_intx, to_affine_quantized_intx_static
from torchao.prototype.smoothquant.core import (
    SmoothQuantObservedLinear,
    SmoothQuantObserver,
)
from torchao.quantization import quantize_
from torchao.quantization.linear_activation_quantized_tensor import (
    to_linear_activation_quantized,
)
from torchao.quantization.linear_activation_scale import (
    to_weight_tensor_with_linear_activation_scale_metadata,
)
from torchao.quantization.quant_api import (
    _linear_extra_repr,
    _replace_with_custom_fn_if_matches_filter,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.quantization.utils import _get_per_token_block_size
from torchao.quantization.weight_tensor_linear_activation_quantization import (
    to_weight_tensor_with_linear_activation_quantization_metadata,
)


def insert_smooth_quant_observer_(
    model: torch.nn.Module, alpha: Optional[float] = 0.5, quant_mode: str = "dynamic"
):
    """
    Inserts SmoothQuantObserver into Linear layers of a given model.

    Args:
        model: The model to be modified (in place). Ensure model is on the desired device for calibration
        alpha: The alpha value to determine smoothing factor. Factor = 1 if alpha is None, which means
               falling back to conventional quantization.
        quant_mode: dynamic or static quantization of activation
    """
    _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)

    quant_min, quant_max = -127, 127
    eps = torch.finfo(torch.float32).eps

    def replace_with_observer(layer):
        # creates observer and replaces linear layers with observed linear layers
        observer = SmoothQuantObserver(
            layer.weight,
            alpha,
            quant_mode,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
        )
        return SmoothQuantObservedLinear.from_float(layer, observer)

    _replace_with_custom_fn_if_matches_filter(model, replace_with_observer, _is_linear)


def save_smooth_quant_recipe(
    model: torch.nn.Module, save_path: str
) -> Dict[str, torch.Tensor]:
    """
    Save smoothing_factors, act_scales, and wei_scales for each SmoothQuantObservedLinear layer in the model.
    """
    result = {}

    def recurse(module: torch.nn.Module, name: str = ""):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Apply the analysis function to this layer
            if isinstance(child, SmoothQuantObservedLinear):
                smoothing_factor, act_scales, wei_scales = child.obs.calculate_qparams()
                result[full_name + ".smoothing_factor"] = smoothing_factor
                result[full_name + ".act_scales"] = act_scales
                result[full_name + ".wei_scales"] = wei_scales

            # Recurse into child modules
            recurse(child, full_name)

    recurse(model)

    torch.save(result, save_path)


def load_smooth_quant_recipe(
    model: torch.nn.Module, recipe_path: str, device=None
) -> torch.nn.Module:
    recipe = torch.load(recipe_path, weights_only=True)

    def recurse(module: torch.nn.Module, name: str = ""):
        if isinstance(module, SmoothQuantObservedLinear):
            smoothing_factor = recipe.get(name + ".smoothing_factor", None)
            act_scales = recipe.get(name + ".act_scales", None)
            wei_scales = recipe.get(name + ".wei_scales", None)
            if device is not None:
                module.to(device=device)
            # act_scales is None for dynamic quantization
            if any(x is None for x in (smoothing_factor, wei_scales)):
                return module
            is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
            wrapper = torch.nn.Sequential(module)
            quantize_(
                wrapper,
                SmoothQuantConfig(smoothing_factor, act_scales, wei_scales),
                is_observed_linear,
            )
            return wrapper[0]

        mod_new = module

        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            setattr(mod_new, child_name, recurse(child, full_name))
        return mod_new

    recurse(model)


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
        return to_affine_quantized_intx_static(
            input,
            scale,
            zero_point,
            list(input.shape),
            self.target_dtype,
            self.quant_min,
        )


@dataclass
class SmoothQuantConfig(AOBaseConfig):
    """
    Configuration for quantizing linear layers when passed into quantize_()

    Args:
        smoothing_factor: The smoothing factor for the layer. Acquired from the layer's observer if None.
        act_scales: The activation scales for the layer. Acquired from the layer's observer if None.
        wei_scales: The weight scales for the layer. Acquired from the layer's observer if None.
        set_inductor_config: if True, adjusts `torchinductor` settings to recommended values.
    """

    smoothing_factor: Optional[torch.Tensor] = None
    act_scales: Optional[torch.Tensor] = None
    wei_scales: Optional[torch.Tensor] = None
    set_inductor_config: bool = True


@register_quantize_module_handler(SmoothQuantConfig)
def _smooth_quant_transform(
    module: torch.nn.Module,
    config: SmoothQuantConfig,
):
    smoothing_factor = config.smoothing_factor
    act_scales = config.act_scales
    wei_scales = config.wei_scales
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()
    observed_linear = module

    linear = torch.nn.Linear(
        observed_linear.in_features,
        observed_linear.out_features,
        observed_linear.bias is not None,
        device=observed_linear.weight.device,
        dtype=observed_linear.weight.dtype,
    )
    linear.bias = observed_linear.bias

    target_dtype = torch.int8
    # act_scales is None for dynamic quantization thus not checked
    if any(x is None for x in (smoothing_factor, wei_scales)):
        factor, x_scale, w_scales = observed_linear.obs.calculate_qparams()
        weight = observed_linear.obs.weight * factor
    else:
        factor, x_scale, w_scales = smoothing_factor, act_scales, wei_scales
        weight = observed_linear.weight * factor
    weight = weight.to(observed_linear.weight.dtype)
    block_size = (1, weight.size(1))
    wei_zero_points = torch.zeros_like(w_scales, dtype=torch.int64)
    qw = to_affine_quantized_intx_static(
        weight,
        w_scales,
        wei_zero_points,
        block_size,
        target_dtype,
    )

    if x_scale is None:
        # dynamic quant
        qw = to_linear_activation_quantized(
            qw, _ActQuantizer(target_dtype).dynamic_quantize
        )
    else:
        # static quant
        x_zero_point = torch.zeros_like(x_scale, dtype=torch.int64)
        qw = to_weight_tensor_with_linear_activation_quantization_metadata(
            qw, _ActQuantizer(target_dtype).static_quantize, x_scale, x_zero_point
        )

    qw = to_weight_tensor_with_linear_activation_scale_metadata(qw, factor.to(qw.dtype))
    linear.weight = torch.nn.Parameter(qw, requires_grad=False)
    linear.extra_repr = types.MethodType(_linear_extra_repr, module)
    return linear
