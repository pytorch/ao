import torch
import torch.nn.functional as F
from torchao.dtypes.affine_quantized_tensor import AWQ_INT4_LayoutType, AWQLayoutType
from torchao.prototype.awq.core import AWQObserver, _awq_quant
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from torchao.dtypes import to_affine_quantized
from torchao.dtypes.uintx.Uintx import to_uintx
from typing import Optional, Tuple


class ObservedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, act_obs: torch.nn.Module, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs

    def forward(self, input: torch.Tensor):
        output = F.linear(input, self.weight, self.bias)
        self.act_obs(input, output)
        return output

    @classmethod
    def from_float(cls, float_linear, act_obs):
        observed_linear = cls(float_linear.in_features, float_linear.out_features, act_obs, False, device=float_linear.weight.device, dtype=float_linear.weight.dtype)
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear

class AWQ_int4(torch.nn.Module):
    def __init__(
        self, 
        int_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eq_scales: torch.Tensor,
        original_shape: Tuple,
        scales: torch.Tensor, 
        zeros: torch.Tensor,
        qdtype,
        device=None):

        super().__init__()
        self.weight = to_uintx(int_weight.to(torch.uint8), qdtype)
        self.bias = bias
        self.scales = scales
        self.zeros = zeros
        self.eq_scales = eq_scales
        self.original_shape = original_shape

    def forward(self, input: torch.Tensor):
        dq = (self.weight.get_plain() - self.zeros) * self.scales
        return torch.nn.functional.linear(input / self.eq_scales, dq.view(self.original_shape), self.bias)
    
def _replace_with_custom_fn_if_matches_filter(
    model,
    replacement_fn,
    filter_fn,
    cur_fqn="",
    device=None,
) -> None:
    """
    Recursively replaces each child module in `model` with the result of `replacement_fn(child)`
    if `filter_fn(child)` returns `True`.

    Args:
        model (torch.nn.Module): The model containing modules to be replaced.
        replacement_fn (Callable[[torch.nn.Module], torch.nn.Module]): The function to replace matching modules.
        filter_fn (Callable[[torch.nn.Module], bool]): The filter function to determine which modules to replace.
        cur_fqn (str, optional): The current fully qualified name of the module being processed. Defaults to "".
        device (device, optional): Device to move the model to before applying `filter_fn`. Defaults to None.

    Returns:
        None
    """
    # print(model)
    if filter_fn(model, cur_fqn[:-1]):
        if device is not None:
            model.to(device=device)  # move to device before quantization
        # print("replacing ", model)
        model = replacement_fn(model)
        return model
    else:
        for name, child in model.named_children():
            if "attn" in name:
                continue
            new_child = _replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, f"{cur_fqn}{name}.", device
            )
            if new_child is not child:
                setattr(model, name, new_child)
        if device is not None:
            model.to(device=device)  # move parent module to device
        return model
    
def insert_awq_observer(model, quant_dtype, group_size, input_dtype, device):
    assert quant_dtype in ["int4", "int8"]
    _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)
    if quant_dtype == "int4":
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        eps = 1e-6
        preserve_zero = False
        zero_point_dtype = torch.bfloat16
        zero_point_domain = ZeroPointDomain.FLOAT

    elif quant_dtype == "int8":
        mapping_type = MappingType.SYMMETRIC
        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int64
        zero_point_domain = ZeroPointDomain.INT
        preserve_zero = True
        block_size = (1, -1)
        quant_min = None
        quant_max = None

    def replace_with_observer(layer):
        observer = AWQObserver(
            layer.weight,
            layer.bias, 
            block_size, 
            input_dtype, 
            mapping_type,
            target_dtype, 
            device,
            preserve_zero = preserve_zero,
            zero_point_domain = zero_point_domain,
            zero_point_dtype = zero_point_dtype,
            quant_min=quant_min,
            quant_max = quant_max,
            eps = eps)
        return ObservedLinear.from_float(layer, observer)
    _replace_with_custom_fn_if_matches_filter(model, replace_with_observer, _is_linear)

# variant of _get_linear_subclass_inserter that works with observed linear class
def _observed_linear_subclass_inserter(constructor):
    def insert_subclass(observed_linear):
        linear = torch.nn.Linear(observed_linear.in_features, observed_linear.out_features, observed_linear.bias!=None, device=observed_linear.weight.device, dtype=observed_linear.weight.dtype)
        linear.weight = torch.nn.Parameter(constructor(observed_linear), requires_grad=False)
        linear.bias = observed_linear.bias
        return linear

    return insert_subclass

def awq_quant(quant_dtype = "int4", group_size = 128):
    
    def weight_quant_func(observed_linear):
        assert quant_dtype in ["int4", "int8"]
        mapping_type = MappingType.ASYMMETRIC
        # weight quantization
        equalization_scale = observed_linear.act_obs.calculate_qparams()
        if quant_dtype == "int4":
            mapping_type = MappingType.ASYMMETRIC
            block_size = (1, group_size)
            target_dtype = torch.int32
            quant_min = 0
            quant_max = 15
            eps = 1e-6
            preserve_zero = False
            zero_point_dtype = torch.bfloat16
            zero_point_domain = ZeroPointDomain.FLOAT
            layout_type = AWQ_INT4_LayoutType(equalization_scale)
        elif quant_dtype == "int8":
            mapping_type = MappingType.SYMMETRIC
            target_dtype = torch.int8
            eps = torch.finfo(torch.float32).eps
            zero_point_dtype = torch.int64
            zero_point_domain = ZeroPointDomain.INT
            preserve_zero = True
            block_size = (1, observed_linear.weight.shape[1])
            layout_type = AWQLayoutType(equalization_scale)
            quant_min = None
            quant_max = None
        
        return to_affine_quantized(
            observed_linear.weight,
            mapping_type, block_size, 
            target_dtype, quant_min, 
            quant_max, eps, 
            zero_point_dtype=zero_point_dtype,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
            layout_type=layout_type)
    
    return _observed_linear_subclass_inserter(weight_quant_func)


