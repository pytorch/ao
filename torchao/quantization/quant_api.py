# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Quantization APIs

Generally these APIs can be applied directly to any model
with Linear modules to obtain quantized linear ops. The intended
usage involves applying torch.compile to the model afterwards
both because primitives were designed based on the fusions that
come along with it and because that is how we access the intended quantized
and mixed GEMM kernels
"""
from functools import partial
import torch
import torchao
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Union, Dict, Optional

from torchao.dtypes import PlainLayoutType
from torchao.utils import (
    TORCH_VERSION_AFTER_2_4,
    unwrap_tensor_subclass,
)
from .subclass import (
    QuantizedLinearWeightBase,
)

from .linear_activation_quantized_tensor import (
    LinearActivationQuantizedTensor,
    to_linear_activation_quantized,
)

from .quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from .weight_only import WeightOnlyInt8QuantLinear
from .unified import Quantizer, TwoStepQuantizer
from .GPTQ import (
    Int4WeightOnlyGPTQQuantizer,
    Int4WeightOnlyQuantizer,
)
from .utils import _get_per_token_block_size
import logging
from .autoquant import autoquant, AutoQuantizableLinearWeight


__all__ = [
    "swap_conv2d_1x1_to_linear",
    "Quantizer",
    "TwoStepQuantizer",
    "Int4WeightOnlyGPTQQuantizer",
    "Int4WeightOnlyQuantizer",
    "autoquant",
    "_get_subclass_inserter",
    "quantize_",
    "int8_dynamic_activation_int4_weight",
    "int8_dynamic_activation_int8_weight",
    "int8_dynamic_activation_int8_semi_sparse_weight",
    "int4_weight_only",
    "int8_weight_only",
]

from .GPTQ import (
    Int8DynActInt4WeightQuantizer,
    Int8DynActInt4WeightGPTQQuantizer,

)
__all__ += [
    "Int8DynActInt4WeightQuantizer",
    "Int8DynActInt4WeightGPTQQuantizer",
]

### TO BE DEPRECATED START
from .subclass import (
    Int4WeightOnlyQuantizedLinearWeight,
    Int8DynamicallyQuantizedLinearWeight,
    Int8WeightOnlyQuantizedLinearWeight,
)

def _in_features_greater_than_16(mod, *args):
    return hasattr(mod, "in_features") and mod.in_features > 16

def change_linear_weights_to_int8_dqtensors(model, filter_fn=None, **kwargs):
    """
    Converts all linear weight tensors to the `Int8DynamicallyQuantizedLinearWeight`
    Tensor subclass, effectively applying the same form of quantization
    as apply_dynamic_quant while not modifying the linear modules.
    """
    if TORCH_VERSION_AFTER_2_4:
        raise ImportError("This API is deprecated for pytorch 2.4+, please checkout quantization/README.md for most up to date APIs")

    if filter_fn is None:
        filter_fn = lambda *args: _is_linear(*args) and _in_features_greater_than_16(
            *args
        )

    _replace_with_custom_fn_if_matches_filter(
        model, _get_subclass_inserter(Int8DynamicallyQuantizedLinearWeight, enable_parametrization=False, **kwargs), filter_fn
    )


def change_linear_weights_to_int8_woqtensors(model, filter_fn=None, **kwargs):
    """
    Converts all linear weight tensors to the
    `Int8WeightOnlyQuantizedLinearWeight` tensor subclass,
    effectively applying the same form of quantization
    as apply_weight_only_int8_quant while not modifying the linear modules.
    """
    if TORCH_VERSION_AFTER_2_4:
        raise ImportError("This API is deprecated for pytorch 2.4+, please checkout quantization/README.md for most up to date APIs")

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(Int8WeightOnlyQuantizedLinearWeight, enable_parametrization=False, **kwargs),
        _is_linear if filter_fn is None else filter_fn,
    )

def change_linear_weights_to_int4_woqtensors(model, groupsize=128, inner_k_tiles=8, filter_fn=None):
    """
    Converts all linear weight tensors to the
    `Int4WeightOnlyQuantizedLinearWeight` tensor subclass,
    effectively applying the same form of quantization
    as apply_dynamic_quant while not modifying the linear modules.
    Args:
        `groupsize`: parameter for quantization, controls the granularity of quantization, smaller
         size is more fine grained, choices are [256, 128, 64, 32]
        `inner_k_tiles`: parameter for int4 mm kernel, choices are [8, 4, 2]
    """
    if TORCH_VERSION_AFTER_2_4:
        raise ImportError("This API is deprecated for pytorch 2.4+, please checkout quantization/README.md for most up to date APIs")

    if filter_fn is None:
        filter_fn = _is_linear

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(Int4WeightOnlyQuantizedLinearWeight, enable_parametrization=False, groupsize=groupsize, inner_k_tiles=inner_k_tiles),
        filter_fn,
    )

### TO BE DEPRECATED END



def _replace_with_custom_fn_if_matches_filter(
    model,
    replacement_fn,
    filter_fn,
    cur_fqn="",
) -> None:
    """
    Recursively replaces each child module in `model` with the result of `replacement_fn(child)`
    if `filter_fn(child)` returns `True`.

    Args:
        model (torch.nn.Module): The model containing modules to be replaced.
        replacement_fn (Callable[[torch.nn.Module], torch.nn.Module]): The function to replace matching modules.
        filter_fn (Callable[[torch.nn.Module], bool]): The filter function to determine which modules to replace.
        cur_fqn (str, optional): The current fully qualified name of the module being processed. Defaults to "".

    Returns:
        None
    """
    if filter_fn(model, cur_fqn[:-1]):
        model = replacement_fn(model)
        return model
    else:
        for name, child in model.named_children():
            new_child = _replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, f"{cur_fqn}{name}."
            )
            if new_child is not child:
                setattr(model, name, new_child)
        return model


def _is_linear(mod, *args):
    # avoid circular dep
    from torchao.dtypes import AffineQuantizedTensor

    # adding weight tensor subclass isinstance check to make sure the weight is only quantized once
    # when it is shared by multiple linear modules
    return (
        isinstance(mod, torch.nn.Linear)
        and hasattr(mod, "weight")
        and not isinstance(mod.weight, QuantizedLinearWeightBase)
        and not isinstance(mod.weight, AutoQuantizableLinearWeight)
        and not isinstance(mod.weight, AffineQuantizedTensor)
        and not isinstance(mod.weight, LinearActivationQuantizedTensor)
    )

import torch.nn.utils.parametrize as parametrize

def _get_subclass_inserter(cls, enable_parametrization=False, **kwargs):
    """
    Returns a function which inserts the given subclass into all linear modules
    in the model. The inserted module will have its weight set to the result of
    `cls(mod.weight, **kwargs)`. If parametrization is enabled then this will be done using
    torch.nn.utils.parametrize instead of directly setting the attribute on the module.

    Args:
        cls (torch.Tensor): The class to insert as a child module.
        kwargs (Any): Any additional arguments for the constructor.
    """
    constructor = kwargs.pop("constructor", "subclass_constructor")
    from_float = kwargs.pop("method", "from_float")
    def insert_subclass(lin):
        if enable_parametrization:
            lin.weight = torch.nn.Parameter(cls.from_float(lin.weight, **kwargs), requires_grad=False)
            _, args = lin.weight.__tensor_flatten__()
            parametrize.register_parametrization(lin, "weight", getattr(cls, constructor)(*args))
        else:
            lin.weight = torch.nn.Parameter(
                # cls.from_float(...)
                getattr(cls, from_float)(lin.weight, **kwargs), requires_grad=False
            )
        return lin

    return insert_subclass


def swap_conv2d_1x1_to_linear(model, filter_fn=None):
    """
    Changes all conv2d 1x1 modules to equivalent linear modules so that they can then be quantized.
    """

    class PermuteSandwich(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, *args):
            return self.mod(args[0].permute(0, 2, 3, 1)).permute(-0, 3, 1, 2)

    def replace_conv2d_1x1(conv):
        assert conv.kernel_size == (1, 1)
        lin = torch.nn.Linear(
            conv.in_channels, conv.out_channels, bias=(conv.bias is None)
        )
        lin.weight = torch.nn.Parameter(conv.weight.squeeze(-1, -2))
        lin.bias = conv.bias
        return PermuteSandwich(lin)

    if filter_fn is None:
        filter_fn = lambda mod, *args: isinstance(
            mod, torch.nn.Conv2d
        ) and mod.kernel_size == (1, 1)

    _replace_with_custom_fn_if_matches_filter(
        model, replace_conv2d_1x1, filter_fn=filter_fn
    )


def _get_linear_subclass_inserter(constructor):
    def insert_subclass(lin):
        lin.weight = torch.nn.Parameter(constructor(lin.weight), requires_grad=False)
        return lin

    return insert_subclass

def quantize_(model: torch.nn.Module, apply_tensor_subclass: Callable[[torch.nn.Module], torch.nn.Module], filter_fn: Optional[Callable[[torch.nn.Module, str], bool]]=None, set_inductor_config: bool=True):
    """Convert the weight of linear modules in the model with `apply_tensor_subclass`, model is modified inplace

    Args:
        model (torch.nn.Module): input model
        apply_tensor_subclass (Callable[[torch.nn.Module], torch.nn.Module]): function that applies tensor subclass conversion to the weight of a module and return the module (e.g. convert the weight tensor of linear to affine quantized tensor)
        filter_fn (Optional[Callable[[torch.nn.Module, str], bool]]): function that takes a nn.Module instance and fully qualified name of the module, returns True if we want to run `apply_tensor_subclass` on
        the weight of the module
        set_inductor_config (bool, optional): Whether to automatically use recommended inductor config settings (defaults to True)

    Example::

        import torch
        import torch.nn as nn
        from torchao import quantize_

        # 1. quantize with some predefined `apply_tensor_subclass` method that corresponds to
        # optimized execution paths or kernels (e.g. int4 tinygemm kernel)
        # also customizable with arguments
        # currently options are
        # int8_dynamic_activation_int4_weight (for executorch)
        # int8_dynamic_activation_int8_weight (optimized with int8 mm op and torch.compile)
        # int4_weight_only (optimized with int4 tinygemm kernel and torch.compile)
        # int8_weight_only (optimized with int8 mm op and torch.compile
        from torchao.quantization.quant_api import int4_weight_only

        m = nn.Sequential(nn.Linear(32, 1024), nn.Linear(1024, 32))
        quantize_(m, int4_weight_only(group_size=32))

        # 2. write your own new apply_tensor_subclass
        # You can also add your own apply_tensor_subclass by manually calling tensor subclass constructor
        # on weight

        from torchao.dtypes import to_affine_quantized

        # weight only uint4 asymmetric groupwise quantization
        groupsize = 32
        apply_weight_quant = lambda x: to_affine_quantized(
          x, "asymmetric", (1, groupsize), torch.int32, 0, 15, 1e-6,
          zero_point_dtype=torch.bfloat16, preserve_zero=False, zero_point_domain="float")

        def apply_weight_quant_to_linear(linear):
            linear.weight = torch.nn.Parameter(apply_weight_quant(linear.weight), requires_grad=False)
            return linear

        # apply to modules under block0 submodule
        def filter_fn(module: nn.Module, fqn: str) -> bool:
            return isinstance(module, nn.Linear)

        m = nn.Sequential(nn.Linear(32, 1024), nn.Linear(1024, 32))
        quantize_(m, apply_weight_quant_to_linear, filter_fn)

    """
    if set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    _replace_with_custom_fn_if_matches_filter(
        model,
        apply_tensor_subclass,
        _is_linear if filter_fn is None else filter_fn,
    )


def int8_dynamic_activation_int4_weight(group_size=32):
    """Applies int8 dynamic per token asymmetric activation quantization and int4 per group weight symmetric quantization to linear
    This is used to produce a model for executorch backend, but currently executorch did not
    support lowering for the quantized model from this flow yet

    Args:
        `group_size`: parameter for quantization, controls the granularity of quantization, smaller
         size is more fine grained
    """
    def apply_int8_dynamic_activation_int4_weight_quant(weight):
        if weight.shape[-1] % group_size != 0:
            return weight

        # avoid circular dep
        from torchao.dtypes import to_affine_quantized

        # weight settings
        mapping_type = MappingType.SYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        quant_min = -8
        quant_max = 7

        # input settings
        input_mapping_type = MappingType.ASYMMETRIC
        input_target_dtype = torch.int8
        input_quant_func = lambda x: to_affine_quantized(x, input_mapping_type, _get_per_token_block_size(x), input_target_dtype)

        weight = to_affine_quantized(weight, mapping_type, block_size, target_dtype, quant_min, quant_max, eps)
        weight = to_linear_activation_quantized(weight, input_quant_func)
        return weight

    return _get_linear_subclass_inserter(apply_int8_dynamic_activation_int4_weight_quant)


def int4_weight_only(group_size=128, inner_k_tiles=8):
    """
    Applies uint4 weight-only asymmetric per-group quantization to linear layers, using
    "tensor_core_tiled" layout for speedup with tinygemm kernel

    Note:
        This is targeting `tinygemm` int4mm kernel (`torch.ops.aten._weight_int4pack_mm`), the main difference
        of quantization algorithm compared to the more traditional type of integer quantization is the following:
        1). zero_point is in floating point domain instead of integer domain (`zero_point_domain`=`ZeroPointDomain.FLOAT`)
        2). floating point zero does not have to be exactly representable (`preserve_zero`=False in `choose_qparams_affine`)
        please follow the relevant code in `choose_qparams_affine`, `quantize_affine` and `dequantize_affine`
        to learn about how the quantization parameters are chosen and how the Tensor is quantized/dequantized for tinygemm

    Args:
        `group_size`: parameter for quantization, controls the granularity of quantization, smaller
         size is more fine grained, choices are [256, 128, 64, 32]
        `inner_k_tiles`: parameter for int4 mm kernel, choices are [8, 4, 2]
    """
    def apply_int4_weight_only_quant(weight):
        if weight.shape[-1] % group_size != 0:
            return weight

        # avoid circular dep
        from torchao.dtypes import to_affine_quantized
        from torchao.dtypes import TensorCoreTiledLayoutType

        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        eps = 1e-6
        preserve_zero = False
        zero_point_dtype = torch.bfloat16
        zero_point_domain = ZeroPointDomain.FLOAT
        layout_type = TensorCoreTiledLayoutType(inner_k_tiles=inner_k_tiles)
        return to_affine_quantized(weight, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, zero_point_dtype=zero_point_dtype, preserve_zero=preserve_zero, zero_point_domain=zero_point_domain, layout_type=layout_type)

    return _get_linear_subclass_inserter(apply_int4_weight_only_quant)


def int8_weight_only():
    """
    Applies int8 weight-only symmetric per-channel quantization to linear layers.
    """
    def apply_int8wo_quant(weight):
        # avoid circular dep
        from torchao.dtypes import to_affine_quantized

        mapping_type = MappingType.SYMMETRIC
        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int64
        block_size = (1, weight.shape[1])
        return to_affine_quantized(weight, mapping_type, block_size, target_dtype, eps=eps, zero_point_dtype=zero_point_dtype)

    return _get_linear_subclass_inserter(apply_int8wo_quant)


def int8_dynamic_activation_int8_weight(layout_type=PlainLayoutType()):
    """
    Applies int8 dynamic symmetric per-token activation and int8 per-channel weight
    quantization to linear layers
    """
    def apply_int8_dynamic_activation_int8_weight_quant(weight):
        in_features = weight.shape[1]
        # int8 dynamic quantization only has benefit when in_feature > 16
        if in_features <= 16:
            return weight

        # avoid circular dep
        from torchao.dtypes import to_affine_quantized
        # weight settings
        mapping_type = MappingType.SYMMETRIC
        def get_weight_block_size(x):
            return (1, x.shape[1])
        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int64

        # input settings
        input_mapping_type = MappingType.SYMMETRIC
        input_target_dtype = torch.int8
        input_eps = 1e-5
        input_quant_min = -127
        input_quant_max = 127
        input_quant_func = lambda x: to_affine_quantized(x, input_mapping_type, _get_per_token_block_size(x), input_target_dtype, eps=input_eps, quant_min=input_quant_min, quant_max=input_quant_max, scale_dtype=torch.float32 if x.dtype == torch.float16 else None)

        block_size = get_weight_block_size(weight)
        weight = to_affine_quantized(weight, mapping_type, block_size, target_dtype, eps=eps, zero_point_dtype=zero_point_dtype, layout_type=layout_type)
        weight = to_linear_activation_quantized(weight, input_quant_func)
        return weight

    return _get_linear_subclass_inserter(apply_int8_dynamic_activation_int8_weight_quant)


def int8_dynamic_activation_int8_semi_sparse_weight():
    """
    Applies int8 dnynamic symmetric per-token activation and int8 per-channel weight
    quantization + 2:4 sparsity to linear layers.
    """
    from torchao.dtypes import SemiSparseLayoutType
    return int8_dynamic_activation_int8_weight(layout_type=SemiSparseLayoutType())
