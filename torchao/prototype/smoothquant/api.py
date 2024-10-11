import torch
from torchao.quantization.quant_primitives import (
     _DTYPE_TO_QVALUE_BOUNDS,
)
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.dtypes import to_affine_quantized_intx, to_affine_quantized_intx_static
from torchao.quantization.linear_activation_scale_quantized import (
    to_linear_scale_activation_quantized,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.utils import _get_per_token_block_size
from torchao.prototype.smoothquant.core import(
    SmoothQuantObserver,
    SmoothQuantObservedLinear,
)
from typing import Dict, Optional
from torch._dynamo import is_compiling as dynamo_is_compiling


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


# StaticQuantizeAct and DynamicQuantizeAct are defined as classes to allow for easy serialization and deserialization
class StaticQuantizeAct:
    def __init__(self, x_scale, target_dtype, quant_min=-127):
        super().__init__()
        self.x_scale = x_scale
        self.x_zp = torch.zeros_like(x_scale, dtype=torch.int64)
        self.target_dtype = target_dtype
        self.quant_min = quant_min

    def __call__(self, input):
        x_zp = torch.zeros([1], dtype=torch.int64)
        qx = to_affine_quantized_intx_static(
            input, self.x_scale, x_zp, list(input.shape), self.target_dtype, self.quant_min
        )
        if dynamo_is_compiling() or "FakeTensor" in input.__repr__():
            return qx.tensor_impl.int_data.to(qx.dtype)
        return qx


class DynamicQuantizeAct:
    def __init__(self, target_dtype, quant_min=-127):
        self.target_dtype = target_dtype
        self.quant_min = quant_min

    def __call__(self, input):
        block_size = _get_per_token_block_size(input)
        qx = to_affine_quantized_intx(
            input, MappingType.SYMMETRIC, block_size, self.target_dtype, self.quant_min
        )
        if dynamo_is_compiling() or "FakeTensor" in input.__repr__():
            return qx.tensor_impl.int_data.to(qx.dtype)
        return qx


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
            quant_min,
            quant_max,
        )

        is_dynamic = x_scale is None
        if is_dynamic:
            input_quant_func = DynamicQuantizeAct(target_dtype)
        else:
            input_quant_func = StaticQuantizeAct(x_scale, target_dtype)

        return to_linear_scale_activation_quantized(qw, factor, input_quant_func)

    return _observed_linear_subclass_inserter(quantize_weight)
