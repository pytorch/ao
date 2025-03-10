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

import logging
import types
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

import torchao
from torchao.core.config import AOBaseConfig
from torchao.dtypes import (
    AffineQuantizedTensor,
    CutlassInt4PackedLayout,
    Float8Layout,
    Int4CPULayout,
    MarlinQQQLayout,
    MarlinSparseLayout,
    PlainLayout,
    SemiSparseLayout,
    TensorCoreTiledLayout,
    UintxLayout,
    to_affine_quantized_floatx,
    to_affine_quantized_floatx_static,
    to_affine_quantized_intx,
    to_marlinqqq_quantized_intx,
)
from torchao.dtypes.utils import Layout
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.inference import Float8MMConfig
from torchao.quantization.linear_activation_weight_observed_tensor import (
    LinearActivationWeightObservedTensor,
)
from torchao.quantization.observer import AffineQuantizedObserverBase, get_block_size
from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
    register_quantize_module_handler,
)
from torchao.quantization.weight_tensor_linear_activation_quantization import (
    to_weight_tensor_with_linear_activation_quantization_metadata,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_4,
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
    is_MI300,
    is_sm_at_least_89,
    is_sm_at_least_90,
)

from .autoquant import AutoQuantizableLinearWeight, autoquant
from .GPTQ import (
    Int4WeightOnlyGPTQQuantizer,
    Int4WeightOnlyQuantizer,
    Int8DynActInt4WeightGPTQQuantizer,
    Int8DynActInt4WeightQuantizer,
)
from .granularity import (
    PerRow,
    PerTensor,
)
from .linear_activation_quantized_tensor import (
    LinearActivationQuantizedTensor,
    to_linear_activation_quantized,
)
from .qat import (
    intx_quantization_aware_training,
)
from .quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from .subclass import (
    Int4WeightOnlyQuantizedLinearWeight,
    Int8DynamicallyQuantizedLinearWeight,
    Int8WeightOnlyQuantizedLinearWeight,
    QuantizedLinearWeightBase,
)
from .unified import Quantizer, TwoStepQuantizer
from .utils import _get_per_token_block_size

logger = logging.getLogger(__name__)

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
    "intx_quantization_aware_training",
    "float8_weight_only",
    "uintx_weight_only",
    "fpx_weight_only",
    "gemlite_uintx_weight_only",
    "float8_dynamic_activation_float8_weight",
    "float8_static_activation_float8_weight",
    "Int8DynActInt4WeightQuantizer",
    "Int8DynActInt4WeightGPTQQuantizer",
]

LAYOUT_TO_ZERO_POINT_DOMAIN = {
    TensorCoreTiledLayout: [ZeroPointDomain.FLOAT],
    MarlinSparseLayout: [ZeroPointDomain.INT],
    Int4CPULayout: [ZeroPointDomain.FLOAT],
}

LAYOUT_TO_PRESERVE_ZEROS = {
    TensorCoreTiledLayout: False,
    MarlinSparseLayout: True,
    Int4CPULayout: False,
}


######
# TO BE DEPRECATED START
######
def _in_features_greater_than_16(mod, *args):
    return hasattr(mod, "in_features") and mod.in_features > 16


def change_linear_weights_to_int8_dqtensors(model, filter_fn=None, **kwargs):
    """
    Converts all linear weight tensors to the `Int8DynamicallyQuantizedLinearWeight`
    Tensor subclass, effectively applying the same form of quantization
    as apply_dynamic_quant while not modifying the linear modules.
    """
    if TORCH_VERSION_AT_LEAST_2_4:
        raise ImportError(
            "This API is deprecated for pytorch 2.4+, please checkout quantization/README.md for most up to date APIs"
        )

    if filter_fn is None:
        filter_fn = lambda *args: _is_linear(*args) and _in_features_greater_than_16(
            *args
        )

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(
            Int8DynamicallyQuantizedLinearWeight, enable_parametrization=False, **kwargs
        ),
        filter_fn,
    )


def change_linear_weights_to_int8_woqtensors(model, filter_fn=None, **kwargs):
    """
    Converts all linear weight tensors to the
    `Int8WeightOnlyQuantizedLinearWeight` tensor subclass,
    effectively applying the same form of quantization
    as apply_weight_only_int8_quant while not modifying the linear modules.
    """
    if TORCH_VERSION_AT_LEAST_2_4:
        raise ImportError(
            "This API is deprecated for pytorch 2.4+, please checkout quantization/README.md for most up to date APIs"
        )

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(
            Int8WeightOnlyQuantizedLinearWeight, enable_parametrization=False, **kwargs
        ),
        _is_linear if filter_fn is None else filter_fn,
    )


def change_linear_weights_to_int4_woqtensors(
    model, groupsize=128, inner_k_tiles=8, filter_fn=None
):
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
    if TORCH_VERSION_AT_LEAST_2_4:
        raise ImportError(
            "This API is deprecated for pytorch 2.4+, please checkout quantization/README.md for most up to date APIs"
        )

    if filter_fn is None:
        filter_fn = _is_linear

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(
            Int4WeightOnlyQuantizedLinearWeight,
            enable_parametrization=False,
            groupsize=groupsize,
            inner_k_tiles=inner_k_tiles,
        ),
        filter_fn,
    )


########
# TO BE DEPRECATED END
########


def _replace_with_custom_fn_if_matches_filter(
    model,
    replacement_fn,
    filter_fn,
    cur_fqn="",
    device=None,
    extra_args: Optional[Tuple[Any, ...]] = (),
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
        extra_args (Tuple[Any, ...], optional): optional extra args to pass to `replacement_fn`.

    Returns:
        None
    """
    if isinstance(model, Float8Linear):
        with torch.device("meta"):
            new_module = nn.Linear(model.in_features, model.out_features)
        new_module.weight = model.weight
        new_module.bias = model.bias
        model = new_module
    if filter_fn(model, cur_fqn[:-1]):
        if device is not None:
            model.to(device=device)  # move to device before quantization
        model = replacement_fn(model, *extra_args)
        return model
    else:
        named_children_list = list(model.named_children())
        for name, child in named_children_list:
            new_child = _replace_with_custom_fn_if_matches_filter(
                child,
                replacement_fn,
                filter_fn,
                f"{cur_fqn}{name}.",
                device,
                extra_args,
            )
            if new_child is not child:
                setattr(model, name, new_child)
        if device is not None:
            model.to(device=device)  # move parent module to device
        return model


def _is_linear(mod, *args):
    # avoid circular dependencies
    from torchao.quantization.qat.affine_fake_quantized_tensor import (
        AffineFakeQuantizedTensor,
    )

    # adding weight tensor subclass isinstance check to make sure the weight is only quantized once
    # when it is shared by multiple linear modules
    return (
        isinstance(mod, torch.nn.Linear)
        and hasattr(mod, "weight")
        and not isinstance(mod.weight, QuantizedLinearWeightBase)
        and not isinstance(mod.weight, AutoQuantizableLinearWeight)
        and not isinstance(mod.weight, AffineQuantizedTensor)
        and not isinstance(mod.weight, LinearActivationQuantizedTensor)
        and not isinstance(mod.weight, AffineFakeQuantizedTensor)
        and not isinstance(mod, nn.modules.linear.NonDynamicallyQuantizableLinear)
    )


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
            lin.weight = torch.nn.Parameter(
                cls.from_float(lin.weight, **kwargs), requires_grad=False
            )
            _, args = lin.weight.__tensor_flatten__()
            parametrize.register_parametrization(
                lin, "weight", getattr(cls, constructor)(*args)
            )
        else:
            lin.weight = torch.nn.Parameter(
                # cls.from_float(...)
                getattr(cls, from_float)(lin.weight, **kwargs),
                requires_grad=False,
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


def insert_observers_(
    model: nn.Module,
    input_observer: Optional[AffineQuantizedObserverBase],
    weight_observer: Optional[AffineQuantizedObserverBase],
    *,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
):
    """
    Converts the weight of a linear module to a LinearActivationWeightObservedTensor.

    This function wraps the weight of the given linear module with a LinearActivationWeightObservedTensor,
    which enables observation of both input and weight tensors during forward passes.
    The wrapped weight is then re-wrapped as a nn.Parameter to maintain compatibility
    with PyTorch's module system.

    Example::

    ```
        import torch
        import torch.nn as nn
        from torchao.quantization.linear_observer_tensor import insert_observers_
        from torchao.quantization.observer import (
            AffineQuantizedMinMaxObserver,
            PerTensor,
            MappingType
        )

        # Create observers
        input_observer = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.float8_e4m3fn,
            granularity_type=PerTensor(),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
            zero_point_domain=ZeroPointDomain.NONE,
        )

        # Create a linear module
        linear_module = nn.Linear(10, 20)

        # Convert the linear module's weight to an observed tensor
        insert_observers_(linear_module, input_observer, weight_observer=None)

        # The linear_module can now be used as usual, with observers calculating statistics
        output = linear_module(torch.randn(10, 10))

        # Get the scale and zero point of the input observer
        scale, zero_point = linear_module.weight.input_observer.calculate_qparams()
    ```

    Args:
        model (nn.Module): The nn.Module to convert.
        input_observer (Optional[AffineQuantizedObserverBase]): Observer for input tensor.
        weight_observer (Optional[AffineQuantizedObserverBase]): Observer for weight tensor.
        filter_fn (Optional[Callable[[torch.nn.Module, str], bool]]): Filter function to select which modules to convert.
            If not provided, all linear modules will be converted. This function should take a module and its fully qualified name.

    Returns:
        nn.Linear: The modified linear module with its weight wrapped in a LinearActivationWeightObservedTensor.
    """

    def convert_to_linear_observer(linear_module: nn.Linear):
        # Wrap the weight with LinearActivationWeightObservedTensor and then with nn.Parameter
        linear_module.weight = nn.Parameter(
            LinearActivationWeightObservedTensor.from_float(
                linear_module.weight,
                input_observer=input_observer,
                weight_observer=weight_observer,
            ),
            requires_grad=linear_module.weight.requires_grad,
        )
        return linear_module

    _replace_with_custom_fn_if_matches_filter(
        model,
        convert_to_linear_observer,
        _is_linear if filter_fn is None else filter_fn,
    )


def _quantization_type(weight: torch.Tensor):
    if isinstance(weight, AffineQuantizedTensor):
        return f"{weight.__class__.__name__}({weight._quantization_type()})"

    if isinstance(weight, LinearActivationQuantizedTensor):
        return f"{weight.__class__.__name__}(activation={weight.input_quant_func}, weight={_quantization_type(weight.original_weight_tensor)})"

    if type(weight) is torch.Tensor:
        return "not quantized"

    return "not recognized"


def _linear_extra_repr(self):
    return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight={_quantization_type(self.weight)}"


def _get_linear_subclass_inserter(
    constructor, *, allow_requires_grad=False, propagate_bias=False, **kwargs
):
    """Helper function to apply the constructor that quantizes the weight Tensor (with additional kwargs)
    to the weight of linear module
    """

    def insert_subclass(lin):
        requires_grad = allow_requires_grad and lin.weight.requires_grad
        if propagate_bias == True:
            kwargs["bias"] = lin.bias
        lin.weight = torch.nn.Parameter(
            constructor(lin.weight, **kwargs), requires_grad=requires_grad
        )
        lin.extra_repr = types.MethodType(_linear_extra_repr, lin)
        return lin

    return insert_subclass


def quantize_(
    model: torch.nn.Module,
    config: AOBaseConfig,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
    set_inductor_config: Optional[bool] = None,
    device: Optional[torch.types.Device] = None,
):
    """Convert the weight of linear modules in the model with `config`, model is modified inplace

    Args:
        model (torch.nn.Module): input model
        config (AOBaseConfig): a workflow configuration object.
        filter_fn (Optional[Callable[[torch.nn.Module, str], bool]]): function that takes a nn.Module instance and fully qualified name of the module, returns True if we want to run `config` on
        the weight of the module
        set_inductor_config (bool, optional): Whether to automatically use recommended inductor config settings (defaults to None)
        device (device, optional): Device to move module to before applying `filter_fn`. This can be set to `"cuda"` to speed up quantization. The final model will be on the specified `device`.
            Defaults to None (do not change device).

    Example::

        import torch
        import torch.nn as nn
        from torchao import quantize_

        # quantize with some predefined `config` method that corresponds to
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

    """
    if set_inductor_config != None:
        warnings.warn(
            """The `set_inductor_config` argument to `quantize_` will be removed in a future release. This functionality is being migrated to individual workflows. Please see https://github.com/pytorch/ao/issues/1715 for more details."""
        )
    else:  # None
        # for now, default to True to not change existing behavior when the
        # argument is not specified
        set_inductor_config = True

    if set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    if isinstance(config, AOBaseConfig):
        handler = _QUANTIZE_CONFIG_HANDLER[type(config)]
        # for each linear in the model, apply the transform if filtering passes
        _replace_with_custom_fn_if_matches_filter(
            model,
            handler,
            _is_linear if filter_fn is None else filter_fn,
            device=device,
            extra_args=(config,),
        )

    else:
        raise AssertionError(
            """Passing a generic Callable to `quantize_` is no longer recommended and will be deprecated at a later release. Please see https://github.com/pytorch/ao/issues/1690 for instructions on how to pass in workflow configuration instead."""
        )


def _int8_asymm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
    """This is defined here instead of local function to support serialization"""
    mapping_type = MappingType.ASYMMETRIC
    target_dtype = torch.int8
    if TORCH_VERSION_AT_LEAST_2_6:
        return to_affine_quantized_intx(
            x,
            mapping_type,
            _get_per_token_block_size(x),
            target_dtype,
            scale_dtype=torch.float64,
            zero_point_dtype=torch.int64,
        )
    else:
        return to_affine_quantized_intx(
            x, mapping_type, _get_per_token_block_size(x), target_dtype
        )


def _int8_symm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
    mapping_type = MappingType.SYMMETRIC
    target_dtype = torch.int8
    eps = 1e-5
    quant_min = -127
    quant_max = 127

    return to_affine_quantized_intx(
        x,
        mapping_type,
        _get_per_token_block_size(x),
        target_dtype,
        eps=eps,
        quant_min=quant_min,
        quant_max=quant_max,
        scale_dtype=torch.float32,
    )


@dataclass
class Int8DynamicActivationInt4WeightConfig(AOBaseConfig):
    """Configuration for applying int8 dynamic per token asymmetric activation quantization and int4 per group weight symmetric quantization to linear
    This is used to produce a model for executorch backend, but currently executorch did not
    support lowering for the quantized model from this flow yet

    Args:
        `group_size`: parameter for quantization, controls the granularity of quantization, smaller
         size is more fine grained
        `layout`: layout type for quantized weight tensor, only supports `MarlinQQQLayout()` and `CutlassInt4PackedLayout()` for now
        `mapping_type`: quantization type for weight, controls the weight quantization is symmetric or asymmetric
        `act_mapping_type`: quantization type for activation, controls the activation quantization is symmetric or asymmetric
    """

    group_size: int = 32
    layout: Layout = PlainLayout()
    mapping_type: MappingType = MappingType.SYMMETRIC
    act_mapping_type: MappingType = MappingType.ASYMMETRIC


# for BC
int8_dynamic_activation_int4_weight = Int8DynamicActivationInt4WeightConfig


@register_quantize_module_handler(Int8DynamicActivationInt4WeightConfig)
def _int8_dynamic_activation_int4_weight_transform(
    module: torch.nn.Module, config: Int8DynamicActivationInt4WeightConfig
):
    group_size = config.group_size
    layout = config.layout
    mapping_type = config.mapping_type
    act_mapping_type = config.act_mapping_type

    weight = module.weight

    if group_size is None or group_size == -1:
        group_size = weight.shape[-1]
    if weight.shape[-1] % group_size != 0:
        return module

    # weight settings
    block_size = (1, group_size)
    target_dtype = torch.int8
    eps = torch.finfo(torch.float32).eps
    quant_min = -8
    quant_max = 7

    # input settings
    if act_mapping_type == MappingType.ASYMMETRIC:
        input_quant_func = _int8_asymm_per_token_quant
    elif act_mapping_type == MappingType.SYMMETRIC:
        if isinstance(layout, MarlinQQQLayout):
            input_quant_func = _int8_symm_per_token_quant
        elif isinstance(layout, CutlassInt4PackedLayout):
            input_quant_func = _int8_symm_per_token_reduced_range_quant_cutlass
        else:
            input_quant_func = _int8_symm_per_token_quant
    else:
        assert False, f"Unsupported activation mapping type: {act_mapping_type}"

    if isinstance(layout, MarlinQQQLayout):
        weight = to_marlinqqq_quantized_intx(
            weight, block_size, quant_min, quant_max, _layout=layout
        )
    else:
        weight = to_affine_quantized_intx(
            weight,
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
            _layout=layout,
        )
    weight = to_linear_activation_quantized(weight, input_quant_func)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class Int4DynamicActivationInt4WeightConfig(AOBaseConfig):
    """Applies int4 dynamic per token symmetric activation quantization and int4 per row weight symmetric quantization to linear

    Args:
        `layout`: layout type for quantized weight tensor, only supports `MarlinQQQLayout()` and `CutlassInt4PackedLayout()` for now
        `mapping_type`: quantization type for weight, controls the weight quantization is symmetric or asymmetric
        `act_mapping_type`: quantization type for activation, controls the activation quantization is symmetric or asymmetric
    """

    layout: Layout = CutlassInt4PackedLayout()
    mapping_type: MappingType = MappingType.SYMMETRIC
    act_mapping_type: MappingType = MappingType.SYMMETRIC


# for bc
int4_dynamic_activation_int4_weight = Int4DynamicActivationInt4WeightConfig


@register_quantize_module_handler(Int4DynamicActivationInt4WeightConfig)
def _int4_dynamic_activation_int4_weight_transform(
    module: torch.nn.Module, config: Int4DynamicActivationInt4WeightConfig
) -> torch.nn.Module:
    weight = module.weight
    layout = config.layout
    mapping_type = config.mapping_type
    act_mapping_type = config.act_mapping_type

    if not isinstance(layout, CutlassInt4PackedLayout):
        raise NotImplementedError(
            f"Only CutlassInt4PackedLayout layout is supported. Received {layout}."
        )
    if mapping_type != MappingType.SYMMETRIC:
        raise NotImplementedError("Only mapping_type=SYMMETRIC is supported.")
    if act_mapping_type != MappingType.SYMMETRIC:
        raise NotImplementedError("Only act_mapping_type=SYMMETRIC is supported.")

    weight = to_affine_quantized_intx(
        weight,
        mapping_type=mapping_type,
        block_size=(1, weight.shape[1]),
        target_dtype=torch.int8,
        quant_min=-8,
        quant_max=7,
        eps=torch.finfo(torch.float32).eps,
        zero_point_domain=ZeroPointDomain.NONE,
        _layout=layout,
    )
    weight = to_linear_activation_quantized(
        weight,
        _int4_symm_per_token_quant_cutlass,
    )
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class GemliteUIntXWeightOnlyConfig(AOBaseConfig):
    """
    applies weight only 4 or 8 bit integer quantization and utilizes the gemlite triton kernel and its associated weight packing format.
    This only works for fp16 models. 8 bit quantization is symmetric, 4 bit quantization is asymmetric.

    Args:
        `group_size`: parameter for quantization, controls the granularity of quantization, smaller
         size is more fine grained
        `bit_width`: bit width of the quantized weight.
        `packing_bitwidth`: bit width of the packed weight, should be 8 or 32. Can have performance impacts depending on hardware.
        `contiguous`: if set, the weight will be packed as specified. Leaving it as None lets gemlite determine the best choice.
    """

    group_size: Optional[int] = 64
    bit_width: int = 4
    packing_bitwidth: int = 32
    contiguous: Optional[bool] = None


# for BC
gemlite_uintx_weight_only = GemliteUIntXWeightOnlyConfig


@register_quantize_module_handler(GemliteUIntXWeightOnlyConfig)
def _gemlite_uintx_weight_only_transform(
    module: torch.nn.Module, config: GemliteUIntXWeightOnlyConfig
):
    group_size = config.group_size
    bit_width = config.bit_width
    packing_bitwidth = config.packing_bitwidth
    contiguous = config.contiguous

    weight = module.weight

    from torchao.dtypes.uintx.gemlite_layout import get_gemlite_aqt_kwargs

    use_hqq = True if bit_width == 4 else False
    new_weight = to_affine_quantized_intx(
        weight,
        **get_gemlite_aqt_kwargs(
            weight, group_size, bit_width, packing_bitwidth, contiguous, use_hqq
        ),
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class Int4WeightOnlyConfig(AOBaseConfig):
    """
    Configuration for applying uint4 weight-only asymmetric per-group quantization to linear layers, using
    "tensor_core_tiled" layout for speedup with tinygemm kernel

    Note:
        This is targeting `tinygemm` int4mm kernel (`torch.ops.aten._weight_int4pack_mm`
        and `torch.ops.aten._weight_int4pack_mm_for_cpu`), the main difference
        of quantization algorithm compared to the more traditional type of integer quantization is the following:
        1). zero_point is in floating point domain instead of integer domain (`zero_point_domain`=`ZeroPointDomain.FLOAT`)
        2). floating point zero does not have to be exactly representable (`preserve_zero`=False in `choose_qparams_affine`)
        please follow the relevant code in `choose_qparams_affine`, `quantize_affine` and `dequantize_affine`
        to learn about how the quantization parameters are chosen and how the Tensor is quantized/dequantized for tinygemm

    Args:
        `group_size`: parameter for quantization, controls the granularity of quantization, smaller
         size is more fine grained, choices are [256, 128, 64, 32]
        `layout`: layout type for quantized tensor, default is `TensorCoreTiledLayout(inner_k_tiles=8)`
        `use_hqq`: whether to use hqq or default quantization mode, default is False
        `zero_point_domain`: data type of zeros points, choices are [ZeroPointDomain.FLOAT, ZeroPointDomain.INT, ZeroPointDomain.NONE]
    """

    group_size: int = 128
    layout: Optional[TensorCoreTiledLayout] = TensorCoreTiledLayout(inner_k_tiles=8)
    use_hqq: bool = False
    zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.NONE


# for BC
# TODO maybe change other callsites
int4_weight_only = Int4WeightOnlyConfig


@register_quantize_module_handler(Int4WeightOnlyConfig)
def _int4_weight_only_transform(
    module: torch.nn.Module, config: Int4WeightOnlyConfig
) -> torch.nn.Module:
    # TODO(future PR): perhaps move this logic to a different file, to keep the API
    # file clean of implementation details

    # for now, make these local variables to allow the rest of the function
    # to be a direct copy-paste
    weight = module.weight
    group_size = config.group_size
    layout = config.layout
    use_hqq = config.use_hqq
    zero_point_domain = config.zero_point_domain

    if weight.shape[-1] % group_size != 0:
        logger.info(
            f"Skipping quantizing weight with int4 weight only quantization because the shape of weight {weight.shape} is not compatible with group_size {group_size}"
        )
        return module

    mapping_type = MappingType.ASYMMETRIC
    block_size = (1, group_size)
    target_dtype = torch.int32
    quant_min = 0
    quant_max = 15
    eps = 1e-6
    preserve_zero = LAYOUT_TO_PRESERVE_ZEROS[type(layout)]
    zero_point_dtype = (
        weight.dtype if isinstance(layout, Int4CPULayout) else torch.bfloat16
    )

    # nonlocal zero_point_domain
    assert (
        type(layout) in LAYOUT_TO_ZERO_POINT_DOMAIN.keys()
    ), f"Only support layout: {LAYOUT_TO_ZERO_POINT_DOMAIN.keys()}"
    if zero_point_domain == ZeroPointDomain.NONE:
        # the first value is the default one
        zero_point_domain = LAYOUT_TO_ZERO_POINT_DOMAIN[type(layout)][0]
    else:
        assert (
            zero_point_domain in LAYOUT_TO_ZERO_POINT_DOMAIN[type(layout)]
        ), f"Layout only support {LAYOUT_TO_ZERO_POINT_DOMAIN[layout]}"

    # Sparse Marlin only supports symmetric quantization.
    # NOTE: If we start having lots of layouts that require different configurations,
    # we should consider moving this logic somewhere else.
    if isinstance(layout, MarlinSparseLayout):
        mapping_type = MappingType.SYMMETRIC
        assert (
            group_size == 128 or group_size == weight.shape[-1]
        ), f"MarlinSparseLayout only supports 128 group size or per channel quantization, got {group_size}"

    new_weight = to_affine_quantized_intx(
        weight,
        mapping_type,
        block_size,
        target_dtype,
        quant_min,
        quant_max,
        eps,
        zero_point_dtype=zero_point_dtype,
        preserve_zero=preserve_zero,
        zero_point_domain=zero_point_domain,
        _layout=layout,
        use_hqq=use_hqq,
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class Int8WeightOnlyConfig(AOBaseConfig):
    """
    Configuration for applying int8 weight-only symmetric per-channel quantization to linear layers.
    """

    group_size: Optional[int] = None


# for BC
int8_weight_only = Int8WeightOnlyConfig


@register_quantize_module_handler(Int8WeightOnlyConfig)
def _int8_weight_only_transform(module: torch.nn.Module, config: Int8WeightOnlyConfig):
    group_size = config.group_size
    weight = module.weight

    mapping_type = MappingType.SYMMETRIC
    target_dtype = torch.int8
    eps = torch.finfo(torch.float32).eps
    zero_point_dtype = torch.int64
    if group_size is None:
        group_size = weight.shape[1]
    block_size = (1, group_size)
    new_weight = to_affine_quantized_intx(
        weight,
        mapping_type,
        block_size,
        target_dtype,
        eps=eps,
        zero_point_dtype=zero_point_dtype,
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


def _int8_symm_per_token_reduced_range_quant(x: torch.Tensor) -> torch.Tensor:
    mapping_type = MappingType.SYMMETRIC
    target_dtype = torch.int8
    eps = 1e-5
    quant_min = -127
    quant_max = 127
    return to_affine_quantized_intx(
        x,
        mapping_type,
        _get_per_token_block_size(x),
        target_dtype,
        eps=eps,
        quant_min=quant_min,
        quant_max=quant_max,
        scale_dtype=torch.float32 if x.dtype == torch.float16 else None,
    )


def _int8_symm_per_token_reduced_range_quant_noop_decode(
    x: torch.Tensor,
) -> torch.Tensor:
    mapping_type = MappingType.SYMMETRIC
    target_dtype = torch.int8
    eps = 1e-5
    quant_min = -127
    quant_max = 127
    if x.shape[1] == 1:
        return x
    else:
        return to_affine_quantized_intx(
            x,
            mapping_type,
            _get_per_token_block_size(x),
            target_dtype,
            eps=eps,
            quant_min=quant_min,
            quant_max=quant_max,
            scale_dtype=torch.float32 if x.dtype == torch.float16 else None,
        )


def _int8_symm_per_token_reduced_range_quant_cutlass(
    x: torch.Tensor,
) -> torch.Tensor:
    mapping_type = MappingType.SYMMETRIC
    target_dtype = torch.int8
    eps = 1e-5
    quant_min = -127
    quant_max = 127
    return to_affine_quantized_intx(
        x,
        mapping_type,
        _get_per_token_block_size(x),
        target_dtype,
        eps=eps,
        zero_point_domain=ZeroPointDomain.NONE,
        quant_min=quant_min,
        quant_max=quant_max,
        scale_dtype=torch.float16 if x.dtype == torch.float16 else None,
    )


def _int4_symm_per_token_quant_cutlass(x: torch.Tensor) -> torch.Tensor:
    return to_affine_quantized_intx(
        x,
        mapping_type=MappingType.SYMMETRIC,
        block_size=_get_per_token_block_size(x),
        target_dtype=torch.int8,
        quant_min=-8,
        quant_max=7,
        eps=1e-5,
        zero_point_domain=ZeroPointDomain.NONE,
        _layout=CutlassInt4PackedLayout(),
    )


@dataclass
class Int8DynamicActivationInt8WeightConfig(AOBaseConfig):
    """
    Configuration for applying int8 dynamic symmetric per-token activation and int8 per-channel weight
    quantization to linear layers
    """

    layout: Optional[Layout] = PlainLayout()
    act_mapping_type: Optional[MappingType] = MappingType.SYMMETRIC
    weight_only_decode: bool = False


# for BC
int8_dynamic_activation_int8_weight = Int8DynamicActivationInt8WeightConfig


@register_quantize_module_handler(Int8DynamicActivationInt8WeightConfig)
def _int8_dynamic_activation_int8_weight_transform(
    module: torch.nn.Module, config: Int8DynamicActivationInt8WeightConfig
) -> torch.nn.Module:
    layout = config.layout
    act_mapping_type = config.act_mapping_type
    weight_only_decode = config.weight_only_decode

    weight = module.weight

    in_features = weight.shape[1]
    # int8 dynamic quantization only has benefit when in_feature > 16
    if in_features <= 16:
        logger.info(
            f"Skipping applying int8_dynamic_activation_int8_weight to weight of shape {weight.shape}"
            f" because `in_feature` is <= 16: {in_features}"
        )
        return module

    # weight settings
    mapping_type = MappingType.SYMMETRIC
    weight_zero_point_domain = ZeroPointDomain.NONE

    def get_weight_block_size(x):
        return (1, x.shape[1])

    target_dtype = torch.int8
    eps = torch.finfo(torch.float32).eps
    zero_point_dtype = torch.int64

    if weight_only_decode:
        input_quant_func = _int8_symm_per_token_reduced_range_quant_noop_decode
    else:
        # input settings
        if act_mapping_type == MappingType.SYMMETRIC:
            input_quant_func = _int8_symm_per_token_reduced_range_quant
        else:
            input_quant_func = _int8_asymm_per_token_quant

    block_size = get_weight_block_size(weight)
    weight = to_affine_quantized_intx(
        weight,
        mapping_type,
        block_size,
        target_dtype,
        eps=eps,
        zero_point_dtype=zero_point_dtype,
        _layout=layout,
        zero_point_domain=weight_zero_point_domain,
    )
    weight = to_linear_activation_quantized(weight, input_quant_func)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


def int8_dynamic_activation_int8_semi_sparse_weight():
    """
    Applies int8 dnynamic symmetric per-token activation and int8 per-channel weight
    quantization + 2:4 sparsity to linear layers.
    """
    warnings.warn("""int8_dyanmic_activation_int8_semi_sparse_weight() will be deprecated at a later release. Please use the layout kwarg in int8_dynamic_activation_int8_weight instead.

    from torchao.dtypes import SemiSparseLayout
    int8_dynamic_activation_int8_weight(layout=SemiSparseLayout()""")

    return int8_dynamic_activation_int8_weight(layout=SemiSparseLayout())


@dataclass
class Float8WeightOnlyConfig(AOBaseConfig):
    """
    Configuration for applying float8 weight-only symmetric per-channel quantization to linear layers.

    Args:
        weight_dtype (torch.dtype): The target data type for weight quantization. Default is torch.float8_e4m3fn.

    Note:
        The actual matmul will be computed in original precision of the weight tensor.
    """

    weight_dtype: torch.dtype = torch.float8_e4m3fn


# for BC
float8_weight_only = Float8WeightOnlyConfig


@register_quantize_module_handler(Float8WeightOnlyConfig)
def _float8_weight_only_transform(
    module: torch.nn.Module, config: Float8WeightOnlyConfig
) -> torch.nn.Module:
    from torchao.dtypes import to_affine_quantized_floatx

    weight = module.weight
    block_size = (1, weight.shape[1])
    new_weight = to_affine_quantized_floatx(
        input_float=weight,
        block_size=block_size,
        target_dtype=config.weight_dtype,
        scale_dtype=None,
        _layout=Float8Layout(mm_config=None),
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


_fp8_granularities = Union[PerTensor, PerRow]


# Validate and process granularity input
def _normalize_granularity(
    granularity: Optional[
        Union[_fp8_granularities, Tuple[_fp8_granularities, _fp8_granularities]]
    ],
) -> Tuple[_fp8_granularities, _fp8_granularities]:
    processed_granularity = None
    if granularity is None:
        processed_granularity = (PerTensor(), PerTensor())
    elif isinstance(granularity, (PerTensor, PerRow)):
        processed_granularity = (granularity, granularity)
    elif isinstance(granularity, tuple) and len(granularity) == 2:
        if not (
            isinstance(granularity[0], (PerTensor, PerRow))
            and isinstance(granularity[1], (PerTensor, PerRow))
        ):
            raise ValueError(
                f"Invalid granularity types: {granularity}, only PerTensor or PerRow are supported."
            )
        if not isinstance(granularity[0], type(granularity[1])):
            raise ValueError(
                f"Different granularities for activation and weight are not supported: {granularity}, only PerTensor or PerRow are supported."
            )
        processed_granularity = granularity
    else:
        raise ValueError(
            f"Invalid granularity specification: {granularity}, only PerTensor or PerRow are supported."
        )
    # Validate granularity with supported Hardware
    for _granularity in processed_granularity:
        if isinstance(_granularity, PerTensor):
            assert (
                is_sm_at_least_89() or is_MI300()
            ), "PerTensor quantization only works for CUDA>=8.9 and MI300+"
        elif isinstance(_granularity, PerRow):
            assert (
                is_sm_at_least_90() or is_MI300()
            ), "PerRow quantization only works for CUDA>=9.0 and MI300+"
        else:
            raise ValueError(f"Invalid granularity type: {_granularity}")

    return processed_granularity


def _input_activation_quant_func_fp8(
    x: torch.Tensor,
    activation_granularity: _fp8_granularities,
    activation_dtype: torch.dtype,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
):
    """This function is used to quantize the input activation tensor for an aqt_float variant. If scale
    is not provided it will be dynamically calculate the scales otherwise it will use the provided scale.
    """
    assert (
        zero_point is None
    ), "Zero point is not supported for dynamic FP8 quantization"
    if isinstance(activation_granularity, PerRow):
        assert (
            x.dtype == torch.bfloat16
        ), "PerRow quantization only works for bfloat16 precision input activation"

    block_size = get_block_size(x.shape, activation_granularity)
    if scale is None:
        activation = to_affine_quantized_floatx(
            input_float=x,
            block_size=block_size,
            target_dtype=activation_dtype,
            scale_dtype=torch.float32,
            _layout=Float8Layout(mm_config=None),  # Config is stored on weight
        )
    else:
        assert isinstance(
            activation_granularity, PerTensor
        ), "Static quantization only supports PerTensor granularity"
        activation = to_affine_quantized_floatx_static(
            input_float=x,
            block_size=block_size,
            scale=scale,
            target_dtype=activation_dtype,
            _layout=Float8Layout(mm_config=None),  # Config is stored on weight
        )
    return activation


def _fp8_mm_compat(weight: torch.Tensor) -> bool:
    """
    Check if a weight tensor meets float8 quantization requirements.

    Args:
        weight (torch.Tensor): The weight tensor to check

    Returns:
        bool: True if the tensor can be quantized to float8, False otherwise
    """
    assert (
        weight.dim() == 2
    ), f"float8 quantization only works for 2-D tensors, got {weight.dim()}D tensor"

    out_dim, in_dim = weight.shape
    is_compatible = (in_dim % 16 == 0) and (out_dim % 16 == 0)

    if not is_compatible:
        logger.info(
            f"Skipping float8 quantization: weight shape {weight.shape} is not compatible with _scaled_mm. "
            f"Both input dimension ({in_dim}) and output dimension ({out_dim}) must be multiples of 16. "
        )

    return is_compatible


@dataclass
class Float8DynamicActivationFloat8WeightConfig(AOBaseConfig):
    """
    Configuration for applying float8 dynamic symmetric quantization to both activations and weights of linear layers.

    Args:
        activation_dtype (torch.dtype): The target data type for activation quantization. Default is torch.float8_e4m3fn.
        weight_dtype (torch.dtype): The target data type for weight quantization. Default is torch.float8_e4m3fn.
        granularity:
            The granularity for quantization. Can be either a single granularity (applied to both
            activations and weights) or a tuple of two granularities (one for activations, one for weights).
            If None, defaults to PerTensor for both. Currently both quantizations need to be the same type. And
            only PerTensor and PerRow are supported.
        mm_config (Float8MMConfig): Configuration for the matrix multiplication. Default uses fast accumulation.

    """

    activation_dtype: torch.dtype = torch.float8_e4m3fn
    weight_dtype: torch.dtype = torch.float8_e4m3fn
    granularity: Optional[
        Union[_fp8_granularities, Tuple[_fp8_granularities, _fp8_granularities]]
    ] = None
    mm_config: Optional[Float8MMConfig] = None

    def __post_init__(self):
        if self.mm_config is None:
            self.mm_config = Float8MMConfig(use_fast_accum=True)


# for bc
float8_dynamic_activation_float8_weight = Float8DynamicActivationFloat8WeightConfig


@register_quantize_module_handler(Float8DynamicActivationFloat8WeightConfig)
def _float8_dynamic_activation_float8_weight_transform(
    module: torch.nn.Module, config: Float8DynamicActivationFloat8WeightConfig
):
    assert (
        is_sm_at_least_89() or is_MI300()
    ), "Float8 dynamic activation quantization is only supported on CUDA>=8.9 and MI300+"

    activation_dtype = config.activation_dtype
    weight_dtype = config.weight_dtype
    granularity = config.granularity
    mm_config = config.mm_config
    weight = module.weight

    activation_granularity, weight_granularity = _normalize_granularity(granularity)

    if not _fp8_mm_compat(weight):
        # TODO(future PR): this should really throw an exception instead of silently
        # not doing what the user asked
        return module
    if isinstance(weight_granularity, PerRow):
        assert (
            weight.dtype == torch.bfloat16
        ), "PerRow quantization only works for bfloat16 precision input weight"

    block_size = get_block_size(weight.shape, weight_granularity)
    quantized_weight = to_affine_quantized_floatx(
        input_float=weight,
        block_size=block_size,
        target_dtype=weight_dtype,
        scale_dtype=torch.float32,
        _layout=Float8Layout(mm_config=mm_config),
    )

    input_quant_func = _input_activation_quant_func_fp8
    input_quant_kwargs = {
        "activation_granularity": activation_granularity,
        "activation_dtype": activation_dtype,
    }

    quantized_weight = to_linear_activation_quantized(
        quantized_weight, input_quant_func, quant_kwargs=input_quant_kwargs
    )

    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class Float8StaticActivationFloat8WeightConfig(AOBaseConfig):
    """
    Configuration for applying float8 static symmetric quantization to

    Args:
        scale (torch.Tensor): The scale tensor for activation quantization.
        activation_dtype (torch.dtype): The target data type for activation quantization. Default is torch.float8_e4m
        weight_dtype (torch.dtype): The target data type for weight quantization. Default is torch.float8_e4m
        mm_config (Float8MMConfig): Configuration for the matrix multiplication. Default uses fast accumulation.
    """

    scale: torch.Tensor
    activation_dtype: torch.dtype = torch.float8_e4m3fn
    weight_dtype: torch.dtype = torch.float8_e4m3fn
    granularity: Optional[
        Union[_fp8_granularities, Tuple[_fp8_granularities, _fp8_granularities]]
    ] = None
    mm_config: Optional[Float8MMConfig] = None

    def __post_init__(self):
        if self.mm_config is None:
            self.mm_config = Float8MMConfig(use_fast_accum=True)


# for bc
float8_static_activation_float8_weight = Float8StaticActivationFloat8WeightConfig


@register_quantize_module_handler(Float8StaticActivationFloat8WeightConfig)
def _float8_static_activation_float8_weight_transform(
    module: torch.nn.Module, config: Float8StaticActivationFloat8WeightConfig
):
    assert (
        is_sm_at_least_89() or is_MI300()
    ), "Float8 static activation quantization is only supported on CUDA 8.9 and above"

    scale = config.scale
    activation_dtype = config.activation_dtype
    weight_dtype = config.weight_dtype
    granularity = config.granularity
    mm_config = config.mm_config

    weight = module.weight
    activation_granularity, weight_granularity = _normalize_granularity(granularity)
    assert isinstance(
        activation_granularity, PerTensor
    ), "Static quantization only supports PerTensor granularity"

    if not _fp8_mm_compat(weight):
        # TODO(future PR): this should really throw an exception instead of silently
        # not doing what the user asked
        return module
    block_size = get_block_size(weight.shape, weight_granularity)
    quantized_weight = to_affine_quantized_floatx(
        input_float=weight,
        block_size=block_size,
        target_dtype=weight_dtype,
        scale_dtype=torch.float32,
        _layout=Float8Layout(mm_config=mm_config),
    )

    input_quant_func = _input_activation_quant_func_fp8
    input_quant_kwargs = {
        "activation_granularity": activation_granularity,
        "activation_dtype": activation_dtype,
    }

    quantized_weight = to_weight_tensor_with_linear_activation_quantization_metadata(
        quantized_weight,
        input_quant_func,
        scale=scale,
        zero_point=None,
        quant_kwargs=input_quant_kwargs,
    )

    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class UIntXWeightOnlyConfig(AOBaseConfig):
    """
    Configuration for applying uintx weight-only asymmetric per-group quantization to linear layers, using uintx quantization where
    x is the number of bits specified by `dtype`

    Args:
        `dtype`: torch.uint1 to torch.uint7 sub byte dtypes
        `group_size`: parameter for quantization, controls the granularity of quantization, smaller
         size is more fine grained, defaults to 64
        `pack_dim`: the dimension we use for packing, defaults to -1
        `use_hqq`: whether to use hqq algorithm or the default algorithm to quantize the weight
    """

    dtype: torch.dtype
    group_size: int = 64
    pack_dim: int = -1
    use_hqq: bool = False


# for BC
uintx_weight_only = UIntXWeightOnlyConfig


@register_quantize_module_handler(UIntXWeightOnlyConfig)
def _uintx_weight_only_transform(
    module: torch.nn.Module, config: UIntXWeightOnlyConfig
):
    dtype = config.dtype
    group_size = config.group_size
    pack_dim = config.pack_dim
    use_hqq = config.use_hqq

    weight = module.weight

    from torchao.quantization.quant_primitives import _DTYPE_TO_QVALUE_BOUNDS

    SUPPORTED_DTYPES = {
        torch.uint1,
        torch.uint2,
        torch.uint3,
        torch.uint4,
        torch.uint5,
        torch.uint6,
        torch.uint7,
        torch.uint8,
    }
    assert dtype in SUPPORTED_DTYPES, f"Unsupported dtype for hqq: {dtype}"

    mapping_type = MappingType.ASYMMETRIC
    block_size = (1, group_size)

    if use_hqq:
        if dtype == torch.uint4:
            logger.warn(
                "Recommended to use `int4_weight_only(group_size, use_hqq=True)` for the best performance"
            )
        quant_min, quant_max = _DTYPE_TO_QVALUE_BOUNDS[dtype]
        dtype = torch.uint8
        eps = None
        zero_point_dtype = None
        zero_point_domain = ZeroPointDomain.FLOAT
        preserve_zero = False
        _layout = PlainLayout()
    else:
        quant_min, quant_max = None, None
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int32
        zero_point_domain = ZeroPointDomain.INT
        preserve_zero = True
        _layout = UintxLayout(dtype=dtype, pack_dim=pack_dim)

    new_weight = to_affine_quantized_intx(
        weight,
        mapping_type,
        block_size,
        dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        eps=eps,
        zero_point_dtype=zero_point_dtype,
        zero_point_domain=zero_point_domain,
        preserve_zero=preserve_zero,
        _layout=_layout,
        use_hqq=use_hqq,
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class FPXWeightOnlyConfig(AOBaseConfig):
    """Sub-byte floating point dtypes defined by `ebits`: exponent bits and `mbits`: mantissa bits
    e.g. fp6_e3_m2, fp6_e2_m3, ...
    The packing format and kernels are from the fp6-llm paper: https://arxiv.org/abs/2401.14112
    github repo: https://github.com/usyd-fsalab/fp6_llm, now renamed to quant-llm
    For more details for packing please see: :class:`~torchao.dtypes.fpx.FpxTensorCoreAQTTensorImpl`

    This is experimental, will be merged with `to_affine_quantized_floatx`
    in the future
    """

    ebits: int
    mbits: int


# for BC
fpx_weight_only = FPXWeightOnlyConfig


@register_quantize_module_handler(FPXWeightOnlyConfig)
def _fpx_weight_only_transform(
    module: torch.nn.Module, config: FPXWeightOnlyConfig
) -> torch.nn.Module:
    ebits = config.ebits
    mbits = config.mbits
    weight = module.weight

    from torchao.dtypes import to_affine_quantized_fpx
    from torchao.dtypes.floatx import FloatxTensorCoreLayout

    assert weight.dim() == 2, f"floatx only works for 2-d Tensor, got: {weight.dim()}"
    out_dim, in_dim = weight.shape
    if (in_dim % 64 != 0) or (out_dim % 256 != 0):
        logger.info(
            f"Skipping floatx quantization float{ebits + mbits + 1}_{ebits}_{mbits} because "
            f"the shape is not compatible with the kernel: in_dim={in_dim}, out_dim={out_dim} "
            "expected in_dim % 64 == 0 and out_dim % 256 == 0"
        )
        return module

    _layout = FloatxTensorCoreLayout(ebits, mbits)
    new_weight = to_affine_quantized_fpx(weight, _layout)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


if TORCH_VERSION_AT_LEAST_2_5:
    torch.serialization.add_safe_globals(
        [
            _int8_asymm_per_token_quant,
            _int8_symm_per_token_reduced_range_quant,
            _int8_symm_per_token_reduced_range_quant_cutlass,
            _int4_symm_per_token_quant_cutlass,
            _input_activation_quant_func_fp8,
        ]
    )
