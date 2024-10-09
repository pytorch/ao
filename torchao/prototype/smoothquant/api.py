import torch
from torchao.quantization.quant_primitives import (
     _DTYPE_TO_QVALUE_BOUNDS,
)
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.dtypes import to_affine_quantized_intx_static
from torchao.prototype.smoothquant.core import(
    SmoothQuantObserver, 
    SmoothQuantObservedLinear, 
    SmoothQuantLayoutType,
) 
from typing import Dict, Optional


def insert_smooth_quant_observer(
        model: torch.nn.Module,
        alpha: float = 0.5,
        quant_mode: str = "static",
        n_calib_examples: int = 20):
    """
    Inserts SmoothQuantObserver into Linear layers of a given model.

    Args:
        model: The model to be modified (in place). Ensure model is on the desired device for calibration
        mapping_type: symmetric or asymmetric quantization of weight
        n_calib_examples: Number of examples used for calibration
    """
    _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)

    quant_min = _DTYPE_TO_QVALUE_BOUNDS[torch.int8][0]
    quant_max = _DTYPE_TO_QVALUE_BOUNDS[torch.int8][1]
    eps = torch.finfo(torch.float32).eps

    def replace_with_observer(layer):
        # creates observer and replaces linear layers with observed linear layers
        observer = SmoothQuantObserver(
            layer.weight,
            alpha,
            quant_mode,
            n_calib_examples,
            quant_min=quant_min,
            quant_max = quant_max,
            eps = eps)
        return SmoothQuantObservedLinear.from_float(layer, observer)

    _replace_with_custom_fn_if_matches_filter(model, replace_with_observer, _is_linear)


def _observed_linear_subclass_inserter(constructor):
    """
    Replaces unquantized observed linear instances with quantized linear instances.

    Args:
        constructor: the function which applies quantization to the observed linear layer
    """
    def insert_subclass(observed_linear):
        # creates the new linear layer using constructor
        linear = torch.nn.Linear(
            observed_linear.in_features,
            observed_linear.out_features,
            observed_linear.bias is not None,
            device=observed_linear.weight.device,
            dtype=observed_linear.weight.dtype
        )
        linear.weight = torch.nn.Parameter(constructor(observed_linear), requires_grad=False)
        linear.bias = observed_linear.bias
        return linear

    return insert_subclass


def save_smooth_quant_recipe(model: torch.nn.Module, save_path: str) -> Dict[str, torch.Tensor]:
    """
    Save smoothing_factors, act_scales, and wei_scales for each SmoothQuantObservedLinear layer in the model.
    """
    result = {}

    def recurse(module: torch.nn.Module, name: str = ''):
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

def load_smooth_quant_recipe(model: torch.nn.Module, recipe_path: str, device=None) -> torch.nn.Module:
    recipe = torch.load(recipe_path, weights_only=True)
    
    def recurse(module: torch.nn.Module, name: str = ''):
        if isinstance(module, SmoothQuantObservedLinear):
            smoothing_factor = recipe.get(name + ".smoothing_factor", None)
            act_scales = recipe.get(name + ".act_scales", None)
            wei_scales = recipe.get(name + ".wei_scales", None)
            if device is not None:
                module.to(device=device)
            # act_scales is None for dynamic quantization
            if any(x is None for x in (smoothing_factor, wei_scales)):
                return module
            return smooth_quant(smoothing_factor, act_scales, wei_scales)(module)

        mod_new = module

        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            setattr(mod_new, child_name, recurse(child, full_name))
        return mod_new
    
    recurse(model)
    

def smooth_quant(
        smoothing_factor: Optional[torch.Tensor] = None,
        act_scales: Optional[torch.Tensor] = None,
        wei_scales: Optional[torch.Tensor] = None
    ):
    """
    Quantizes linear layers when passed into quantize_()

    Args:
        smoothing_factor: The smoothing factor for the layer. Acquired from the layer's observer if None.
        act_scales: The activation scales for the layer. Acquired from the layer's observer if None.
        wei_scales: The weight scales for the layer. Acquired from the layer's observer if None.
    """

    def quantize_weight(observed_linear):
        target_dtype = torch.int8
        quant_min = _DTYPE_TO_QVALUE_BOUNDS[target_dtype][0]
        quant_max = _DTYPE_TO_QVALUE_BOUNDS[target_dtype][1]
        nonlocal smoothing_factor, act_scales, wei_scales
        # act_scales is None for dynamic quantization
        if any(x is None for x in (smoothing_factor, wei_scales)):
            factor, s_act, s_wei = observed_linear.obs.calculate_qparams()
            weight = observed_linear.obs.weight * factor
        else:
            factor, s_act, s_wei = smoothing_factor, act_scales, wei_scales
            weight = observed_linear.weight * factor
        weight_t = weight.t().contiguous()
        block_size = (weight_t.size(0), 1)
        inv_smoothing_factor = 1 / factor
        layout_type = SmoothQuantLayoutType(
            inv_smoothing_factor, s_act, s_wei
        )
        wei_zero_points = torch.zeros_like(s_wei, dtype=torch.int64)
        return to_affine_quantized_intx_static(
            weight_t,
            s_wei,
            wei_zero_points,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            layout_type=layout_type
        )
    
    return _observed_linear_subclass_inserter(quantize_weight)


