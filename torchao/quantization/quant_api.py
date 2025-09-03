# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and affiliates.
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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

import torchao
from torchao.core.config import AOBaseConfig
from torchao.dtypes import (
    AffineQuantizedTensor,
    CutlassInt4PackedLayout,
    CutlassSemiSparseLayout,
    Float8Layout,
    Int4CPULayout,
    Int4XPULayout,
    Int8DynamicActInt4WeightCPULayout,
    MarlinQQQLayout,
    MarlinSparseLayout,
    PackedLinearInt8DynamicActivationIntxWeightLayout,
    PlainLayout,
    QDQLayout,
    SemiSparseLayout,
    TensorCoreTiledLayout,
    UintxLayout,
    Int4NPULayout,
    to_affine_quantized_floatx,
    to_affine_quantized_floatx_static,
    to_affine_quantized_intx,
    to_fbgemm_fp8,
    to_marlinqqq_quantized_intx,
)
from torchao.dtypes.uintx.packed_linear_int8_dynamic_activation_intx_weight_layout import (
    Target,
    make_packed_linear_int8_dynamic_activation_intx_weight_tensor,
)
from torchao.dtypes.utils import Layout
from torchao.float8.config import e4m3_dtype, e5m2_dtype
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.inference import (
    Float8MMConfig,
    FP8Granularity,
    _check_hardware_support,
    _normalize_granularity,
)
from torchao.quantization.linear_activation_weight_observed_tensor import (
    LinearActivationWeightObservedTensor,
)
from torchao.quantization.observer import AffineQuantizedObserverBase, get_block_size
from torchao.quantization.quantize_.common import (
    KernelPreference,
    PackingFormat,
)
from torchao.quantization.quantize_.workflows import (
    Float8Tensor,
    Int4MarlinSparseTensor,
    Int4OpaqueTensor,
    Int4PreshuffledTensor,
    Int4Tensor,
    Int4TilePackedTo4dTensor,
    IntxOpaqueTensor,
    IntxUnpackedToInt8Tensor,
    QuantizeTensorToFloat8Kwargs,
)
from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
    register_quantize_module_handler,
)
from torchao.quantization.weight_tensor_linear_activation_quantization import (
    to_weight_tensor_with_linear_activation_quantization_metadata,
)
from torchao.utils import (
    _is_fbgemm_genai_gpu_available,
    is_MI300,
    is_sm_at_least_89,
    is_sm_at_least_90,
)

from .autoquant import AutoQuantizableLinearWeight, autoquant
from .GPTQ import (
    Int4WeightOnlyGPTQQuantizer,
)
from .granularity import (
    Granularity,
    PerAxis,
    PerGroup,
    PerRow,
    PerTensor,
)
from .linear_activation_quantized_tensor import (
    LinearActivationQuantizedTensor,
    to_linear_activation_quantized,
)
from .linear_quant_modules import (
    Int4WeightOnlyQuantizer,
    Int8DynActInt4WeightQuantizer,
)
from .qat import (
    intx_quantization_aware_training,
)
from .quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
    ZeroPointDomain,
    quantize_affine,
)
from .subclass import (
    QuantizedLinearWeightBase,
)
from .unified import Quantizer, TwoStepQuantizer
from .utils import _get_per_token_block_size

logger = logging.getLogger(__name__)

# TODO: revisit this list?
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
    "Float8DynamicActivationFloat8SemiSparseWeightConfig",
    "ModuleFqnToConfig",
    "FbgemmConfig",
]

LAYOUT_TO_ZERO_POINT_DOMAIN = {
    TensorCoreTiledLayout: [ZeroPointDomain.FLOAT],
    MarlinSparseLayout: [ZeroPointDomain.INT],
    Int4CPULayout: [ZeroPointDomain.FLOAT],
    Int4XPULayout: [ZeroPointDomain.FLOAT, ZeroPointDomain.INT],
    Int4NPULayout: [ZeroPointDomain.FLOAT],
}

LAYOUT_TO_PRESERVE_ZEROS = {
    TensorCoreTiledLayout: False,
    MarlinSparseLayout: True,
    Int4CPULayout: False,
    Int4XPULayout: False,
    Int4NPULayout: False,
}


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
            if new_child is not child and new_child is not None:
                setattr(model, name, new_child)
        if device is not None:
            model.to(device=device)  # move parent module to device
        return model


def _replace_with_custom_fn_if_matches_filter_with_name(
    model,
    replacement_fn,
    filter_fn,
    cur_fqn="",
    device=None,
    extra_args: Optional[Tuple[Any, ...]] = (),
) -> None:
    """
    A variant of _replace_with_custom_fn_if_matches_filter where replacement_fn takes module name as well
        ...
        replacement_fn (Callable[[torch.nn.Module, str], torch.nn.Module]): The function to replace matching modules.
        ...

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
        model = replacement_fn(model, cur_fqn[:-1], *extra_args)
        return model
    else:
        named_children_list = list(model.named_children())
        for name, child in named_children_list:
            new_child = _replace_with_custom_fn_if_matches_filter_with_name(
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
        _AffineFakeQuantizedTensor,
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
        and not isinstance(mod.weight, _AffineFakeQuantizedTensor)
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

    if hasattr(weight, "_quantization_type"):
        return f"{weight.__class__.__name__}({weight._quantization_type()})"

    if type(weight) is torch.Tensor or isinstance(weight, torch.nn.Parameter):
        return f"Tensor: {type(weight)}"

    return f"not recognized: {type(weight)}"


def _linear_extra_repr(self):
    return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight={_quantization_type(self.weight)}"


def _embedding_extra_repr(self):
    return f"num_embeddings={self.weight.shape[0]}, embedding_dim={self.weight.shape[1]}, weight={_quantization_type(self.weight)}"


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
    device: Optional[torch.types.Device] = None,
):
    """Convert the weight of linear modules in the model with `config`, model is modified inplace

    Args:
        model (torch.nn.Module): input model
        config (AOBaseConfig): a workflow configuration object.
        filter_fn (Optional[Callable[[torch.nn.Module, str], bool]]): function that takes a nn.Module instance and fully qualified name of the module, returns True if we want to run `config` on
        the weight of the module
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
    torch._C._log_api_usage_once("torchao.quantization.quantize_")

    filter_fn = _is_linear if filter_fn is None else filter_fn

    if isinstance(config, ModuleFqnToConfig):
        _replace_with_custom_fn_if_matches_filter_with_name(
            model,
            _module_fqn_to_config_handler,
            filter_fn,
            device=device,
            extra_args=(config,),
        )
        return

    if isinstance(config, AOBaseConfig):
        handler = _QUANTIZE_CONFIG_HANDLER[type(config)]
        # for each linear in the model, apply the transform if filtering passes
        _replace_with_custom_fn_if_matches_filter(
            model,
            handler,
            filter_fn,
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
    scale_dtype = torch.float32
    eps = torch.finfo(torch.float32).eps
    zero_point_dtype = torch.int8
    return to_affine_quantized_intx(
        x,
        mapping_type,
        _get_per_token_block_size(x),
        target_dtype,
        eps=eps,
        scale_dtype=scale_dtype,
        zero_point_dtype=zero_point_dtype,
    )


def _uint8_asymm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
    mapping_type = MappingType.ASYMMETRIC
    target_dtype = torch.uint8
    scale_dtype = torch.float32
    eps = torch.finfo(torch.float32).eps
    zero_point_dtype = torch.int32
    quant_min = 0
    quant_max = 255
    out = to_affine_quantized_intx(
        x,
        mapping_type,
        _get_per_token_block_size(x),
        target_dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        eps=eps,
        scale_dtype=scale_dtype,
        zero_point_dtype=zero_point_dtype,
    )
    return out


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
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values.
    """

    group_size: int = 32
    layout: Layout = PlainLayout()
    mapping_type: MappingType = MappingType.SYMMETRIC
    act_mapping_type: MappingType = MappingType.ASYMMETRIC
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.Int8DynamicActivationInt4WeightConfig"
        )


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
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    weight = module.weight

    if group_size is None or group_size == -1:
        group_size = weight.shape[-1]
    if weight.shape[-1] % group_size != 0:
        return module

    # weight settings
    block_size = (1, group_size)
    target_dtype = torch.int8
    quant_min = -8
    quant_max = 7

    # input settings
    if act_mapping_type == MappingType.ASYMMETRIC:
        if isinstance(layout, Int8DynamicActInt4WeightCPULayout):
            input_quant_func = _uint8_asymm_per_token_quant
        else:
            input_quant_func = _int8_asymm_per_token_quant
    elif act_mapping_type == MappingType.SYMMETRIC:
        if isinstance(layout, MarlinQQQLayout):
            input_quant_func = _int8_symm_per_token_quant
        elif isinstance(layout, CutlassInt4PackedLayout):
            input_quant_func = _int8_symm_cutlass_quant
        else:
            input_quant_func = _int8_symm_per_token_quant
    else:
        assert False, f"Unsupported activation mapping type: {act_mapping_type}"

    if isinstance(layout, MarlinQQQLayout):
        weight = to_marlinqqq_quantized_intx(
            weight, block_size, quant_min, quant_max, _layout=layout
        )
    elif isinstance(layout, CutlassInt4PackedLayout):
        weight = _int4_symm_cutlass_quant(weight)
    elif isinstance(layout, Int8DynamicActInt4WeightCPULayout):
        weight = to_affine_quantized_intx(
            weight,
            mapping_type,
            block_size,
            target_dtype=torch.uint8,
            quant_min=0,
            quant_max=15,
            _layout=layout,
        )
    else:
        weight = to_affine_quantized_intx(
            weight,
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            _layout=layout,
        )
    weight = to_linear_activation_quantized(weight, input_quant_func)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class Int8DynamicActivationIntxWeightConfig(AOBaseConfig):
    """
    Configuration for dynamically quantizing activations to torch.int8 and weights to torch.intx, with 1 <= x <= 8.
    More specifically, activations are dynamically quantized to 8-bits at a per-token granularity with scales/zeros.
    Weights are quantized with scales/zeros in a groupwise or channelwise manner using the number of bits specified by weight_dtype.

    This layout is identical to Int8DynamicActivationInt4WeightConfig when weight_dtype is torch.int4 and other args
    are the same.  However, this layout is more general and supports other weight dtypes.

    args:
        weight_dtype: The dtype to use for weight quantization.  Must be torch.intx, where 1 <= x <= 8.
        weight_granularity: The granularity to use for weight quantization.  Must be PerGroup or PerAxis(axis=0).
        weight_mapping_type: The type of mapping to use for the weight quantization.
            Must be one of MappingType.ASYMMETRIC or MappingType.SYMMETRIC.  MappingType.SYMMETRIC requires ZeroPointDomain.NONE
        weight_scale_dtype: The dtype to use for the weight scale.
        act_mapping_type: The type of mapping to use for the activation quantization.
            Must be one of MappingType.ASYMMETRIC or MappingType.SYMMETRIC.
        layout: The layout to use for the packed weight tensor:
            - PackedLinearInt8DynamicActivationIntxWeightLayout: this layout is optimized for CPU performance.
            - QDQLayout: this layout represents the quantization with Q/DQ quant primitives, and is intended for
                export applications like ExecuTorch.
    """

    weight_dtype: torch.dtype = torch.int8
    weight_granularity: Granularity = PerGroup(32)
    weight_mapping_type: MappingType = MappingType.SYMMETRIC
    # TODO: add weight_scale_dtype to Int8DynamicActivationInt4WeightConfig
    weight_scale_dtype: Optional[torch.dtype] = None
    act_mapping_type: MappingType = MappingType.ASYMMETRIC
    layout: Layout = QDQLayout()
    packing_format: PackingFormat = PackingFormat.UNPACKED_TO_INT8

    # Used with PackingFormat.OPAQUE
    compute_target: Optional[str] = None
    version: int = 1

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.Int8DynamicActivationIntxWeightConfig"
        )
        assert self.weight_dtype in [getattr(torch, f"int{b}") for b in range(1, 9)], (
            f"weight_dtype must be torch.intx, where 1 <= x <= 8, but got {self.weight_dtype}"
        )
        assert isinstance(self.weight_granularity, (PerAxis, PerGroup)), (
            f"weight_granularity must be PerAxis or PerGroup, but got {self.weight_granularity}"
        )
        if isinstance(self.weight_granularity, PerAxis):
            assert self.weight_granularity.axis == 0, (
                f"axis must be 0, but got {self.weight_granularity.axis}"
            )
        assert self.weight_mapping_type in [
            MappingType.ASYMMETRIC,
            MappingType.SYMMETRIC,
            MappingType.SYMMETRIC_NO_CLIPPING_ERR,
        ], (
            f"weight_mapping_type must be MappingType.ASYMMETRIC or MappingType.SYMMETRIC or MappingType.SYMMETRIC_NO_CLIPPING_ERR, but got {self.weight_mapping_type}"
        )
        assert self.act_mapping_type in [
            MappingType.ASYMMETRIC,
            MappingType.SYMMETRIC,
        ], (
            f"act_mapping_type must be MappingType.ASYMMETRIC or MappingType.SYMMETRIC, but got {self.act_mapping_type}"
        )
        assert isinstance(
            self.layout, (PackedLinearInt8DynamicActivationIntxWeightLayout, QDQLayout)
        ), (
            f"layout must be PackedLinearInt8DynamicActivationIntxWeightLayout or QDQLayout, but got {self.layout}"
        )

        if isinstance(self.layout, PackedLinearInt8DynamicActivationIntxWeightLayout):
            if self.layout.target in [Target.AUTO, Target.KLEIDIAI, Target.ATEN]:
                if (self.weight_scale_dtype) is None or (
                    self.weight_scale_dtype != torch.bfloat16
                ):
                    logging.warning(
                        f"When using layout PackedLinearInt8DynamicActivationIntxWeightLayout with target {self.layout.target}, "
                        f"the weight scale may be cast to bfloat16 by the kernel, but weight_scale_dtype is set to {self.weight_scale_dtype}. "
                        "Explicitly set weight_scale_dtype to torch.bfloat16 to suppress this warning. "
                        "If you need weight_scale_dtype = torch.float32, use target=Target.UNIVERSAL instead."
                    )


def _int8_dynamic_activation_intx_weight_quantize_tensor(weight, bias, config):
    weight_dtype = config.weight_dtype
    weight_granularity = config.weight_granularity
    weight_mapping_type = config.weight_mapping_type
    weight_scale_dtype = config.weight_scale_dtype
    act_mapping_type = config.act_mapping_type
    layout = config.layout
    packing_format = config.packing_format
    compute_target = config.compute_target

    assert weight.dim() == 2, (
        f"Int8DynamicActivationIntxWeightConfig only works for 2-d Tensor, got: {weight.dim()}"
    )
    if isinstance(weight_granularity, PerGroup):
        group_size = weight_granularity.group_size
    elif isinstance(weight_granularity, PerAxis):
        assert weight_granularity.axis == 0, (
            f"axis must be 0 with PerAxis, but got {weight_granularity.axis}"
        )
        group_size = weight.shape[-1]
    else:
        raise ValueError(
            f"weight_granularity must be PerGroup or PerAxis, got {weight_granularity}"
        )

    block_size = (1, group_size)

    if config.version == 2:
        assert act_mapping_type == MappingType.ASYMMETRIC
        assert packing_format in [
            PackingFormat.UNPACKED_TO_INT8,
            PackingFormat.OPAQUE,
        ], f"Unsupported packing format: {packing_format}"
        new_weight = IntxUnpackedToInt8Tensor.from_hp(
            weight,
            block_size,
            weight_dtype,
            mapping_type=weight_mapping_type,
            apply_int8_act_asym_per_token_quant=True,
        )
        if weight_scale_dtype is not None and weight_scale_dtype != weight.dtype:
            _adjust_scale_dtype_in_intx_unpacked_tensor(
                new_weight, weight, weight_scale_dtype
            )

        new_bias = bias

        # Create packed tensor
        if packing_format == PackingFormat.OPAQUE:
            assert compute_target is not None, (
                "Must specify a compute target for PackingFormat.OPAQUE"
            )
            new_weight = IntxOpaqueTensor.from_intx_unpacked_to_int8_tensor(
                new_weight, bias=new_bias, compute_target=compute_target
            )
            new_bias = None  # bias is packed with weights

        return new_weight, new_bias

    # Version 1
    quant_min, quant_max = _DTYPE_TO_QVALUE_BOUNDS[weight_dtype]

    # We quantize with QDQLayout, and then construct the packed weight tensor later
    # set preserve_zero based on weight mapping type
    preserve_zero = weight_mapping_type in [
        MappingType.SYMMETRIC,
        MappingType.SYMMETRIC_NO_CLIPPING_ERR,
    ]

    weight = to_affine_quantized_intx(
        input_float=weight,
        mapping_type=weight_mapping_type,
        block_size=(1, group_size),
        target_dtype=torch.int8,
        quant_min=quant_min,
        quant_max=quant_max,
        scale_dtype=weight_scale_dtype,
        zero_point_dtype=torch.int8,
        preserve_zero=preserve_zero,
        zero_point_domain=ZeroPointDomain.INT,
        _layout=QDQLayout(),
    )
    if isinstance(layout, QDQLayout):
        # TODO: _int8_asymm_per_token_quant uses scale_dtype=torch.float64, zero_point_dtype=torch.int64,
        # which is not great for export with QDQLayout.  It is also not consistent with _int8_symm_per_token_quant,
        # which uses scale_dtype=torch.float32, zero_point_dtype=torch.int32.
        # Maybe introduce new fp32/int32 versions of _int8_asymm_per_token_quant?
        if act_mapping_type == MappingType.ASYMMETRIC:
            activation_quant_func = _int8_asymm_per_token_quant
        elif act_mapping_type == MappingType.SYMMETRIC:
            activation_quant_func = _int8_symm_per_token_quant
        else:
            assert False, f"Unsupported activation mapping type: {act_mapping_type}"
        weight = to_linear_activation_quantized(weight, activation_quant_func)
    elif isinstance(layout, PackedLinearInt8DynamicActivationIntxWeightLayout):
        # PackedLinearInt8DynamicActivationIntxWeightLayout has dynamic activation quantization
        # fused with the kernel and it should not be applied separately
        assert act_mapping_type == MappingType.ASYMMETRIC, (
            "PackedLinearInt8DynamicActivationIntxWeightLayout requires act_mapping_type=MappingType.ASYMMETRIC"
        )
        data, scale, zero_point = weight.tensor_impl.get_plain()
        groups_per_row = weight.shape[-1] // group_size
        scale = scale.reshape(-1, groups_per_row)

        assert zero_point is not None
        zero_point = zero_point.reshape(-1, groups_per_row)
        has_weight_zeros = (zero_point != 0).any()
        weight = make_packed_linear_int8_dynamic_activation_intx_weight_tensor(
            data,
            scale,
            zero_point if has_weight_zeros else None,
            bias,
            weight_dtype,
            layout.target,
            validate_inputs=False,
        )
        # bias is packed with weights if present
        bias = None

    return weight, bias


@register_quantize_module_handler(Int8DynamicActivationIntxWeightConfig)
def _int8_dynamic_activation_intx_weight_transform(
    module: torch.nn.Module, config: Int8DynamicActivationIntxWeightConfig
) -> torch.nn.Module:
    new_weight, new_bias = _int8_dynamic_activation_intx_weight_quantize_tensor(
        module.weight, module.bias, config
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    if new_bias is None:
        module.bias = None
    if isinstance(module, nn.Linear):
        module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class Int4DynamicActivationInt4WeightConfig(AOBaseConfig):
    """Applies int4 dynamic per token symmetric activation quantization and int4 per row weight symmetric quantization to linear

    Args:
        `layout`: layout type for quantized weight tensor, only supports `MarlinQQQLayout()` and `CutlassInt4PackedLayout()` for now
        `mapping_type`: quantization type for weight, controls the weight quantization is symmetric or asymmetric
        `act_mapping_type`: quantization type for activation, controls the activation quantization is symmetric or asymmetric
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values.
    """

    layout: Layout = CutlassInt4PackedLayout()
    mapping_type: MappingType = MappingType.SYMMETRIC
    act_mapping_type: MappingType = MappingType.SYMMETRIC
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.Int4DynamicActivationInt4WeightConfig"
        )


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
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    if not isinstance(layout, CutlassInt4PackedLayout):
        raise NotImplementedError(
            f"Only CutlassInt4PackedLayout layout is supported. Received {layout}."
        )
    if mapping_type != MappingType.SYMMETRIC:
        raise NotImplementedError("Only mapping_type=SYMMETRIC is supported.")
    if act_mapping_type != MappingType.SYMMETRIC:
        raise NotImplementedError("Only act_mapping_type=SYMMETRIC is supported.")

    weight = _int4_symm_cutlass_quant(weight)
    weight = to_linear_activation_quantized(
        weight,
        _int4_symm_cutlass_quant,
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
        `mode`: if set to "dynamic", activations are quantized at runtime; default is "weight_only" (weight-only quantization).
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values.
    """

    group_size: Optional[int] = 128
    bit_width: int = 4
    packing_bitwidth: Optional[int] = None
    mode: Optional[str] = "weight_only"
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.GemliteUIntXWeightOnlyConfig"
        )


# for BC
gemlite_uintx_weight_only = GemliteUIntXWeightOnlyConfig


@register_quantize_module_handler(GemliteUIntXWeightOnlyConfig)
def _gemlite_uintx_weight_only_transform(
    module: torch.nn.Module, config: GemliteUIntXWeightOnlyConfig
):
    group_size = config.group_size
    bit_width = config.bit_width
    packing_bitwidth = config.packing_bitwidth
    mode = config.mode
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    weight = module.weight

    from torchao.dtypes.uintx.gemlite_layout import get_gemlite_aqt_kwargs

    use_hqq = True if bit_width == 4 else False
    new_weight = to_affine_quantized_intx(
        weight,
        **get_gemlite_aqt_kwargs(
            weight,
            group_size=group_size,
            bit_width=bit_width,
            packing_bitwidth=packing_bitwidth,
            mode=mode,
            use_hqq=use_hqq,
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
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values.
        `preserve_zero`: whether to preserve zero, default is None. Will be set to True if zero_point_domain is ZeroPointDomain.INT
        `packing_format`: the packing format for int4 tensor, available from version 2 and above
    """

    group_size: int = 128
    layout: Optional[TensorCoreTiledLayout] = TensorCoreTiledLayout(inner_k_tiles=8)
    use_hqq: bool = False
    zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.NONE
    set_inductor_config: bool = True
    preserve_zero: Optional[bool] = None
    # only used in version >= 2
    packing_format: PackingFormat = PackingFormat.PLAIN
    version: int = 1

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.Int4WeightOnlyConfig")


# for BC
# TODO maybe change other callsites
int4_weight_only = Int4WeightOnlyConfig


def _int4_weight_only_quantize_tensor(weight, config):
    # TODO(future PR): perhaps move this logic to a different file, to keep the API
    # file clean of implementation details

    # for now, make these local variables to allow the rest of the function
    # to be a direct copy-paste
    group_size = config.group_size
    layout = config.layout
    use_hqq = config.use_hqq
    zero_point_domain = config.zero_point_domain
    packing_format = config.packing_format

    if weight.shape[-1] % group_size != 0:
        logger.info(
            f"Skipping quantizing weight with int4 weight only quantization because the shape of weight {weight.shape} is not compatible with group_size {group_size}"
        )
        return weight

    block_size = tuple([1 for _ in range(weight.ndim - 1)] + [group_size])

    if config.version == 2:
        block_size = list(block_size)
        if packing_format == PackingFormat.PRESHUFFLED:
            new_weight = Int4PreshuffledTensor.from_hp(
                weight,
                block_size,
                activation_dtype=torch.bfloat16,
            )
            return new_weight
        elif packing_format == PackingFormat.PLAIN:
            new_weight = Int4Tensor.from_hp(
                weight,
                block_size,
            )
            return new_weight
        elif packing_format == PackingFormat.MARLIN_SPARSE:
            new_weight = Int4MarlinSparseTensor.from_hp(
                weight,
                block_size,
            )
            return new_weight
        elif packing_format == PackingFormat.OPAQUE:
            new_weight = Int4OpaqueTensor.from_hp(
                weight,
                block_size,
            )
            return new_weight
        elif packing_format == PackingFormat.TILE_PACKED_TO_4D:
            new_weight = Int4TilePackedTo4dTensor.from_hp(
                weight,
                block_size,
            )
            return new_weight
        else:
            raise ValueError(f"Unsupported packing format: {packing_format}")

    assert config.version == 1

    mapping_type = MappingType.ASYMMETRIC
    target_dtype = torch.int32
    quant_min = 0
    quant_max = 15
    eps = 1e-6
    zero_point_dtype = (
        weight.dtype if ( isinstance(layout, Int4CPULayout) or isinstance(layout, Int4NPULayout) ) else torch.bfloat16
    )

    # nonlocal zero_point_domain
    assert type(layout) in LAYOUT_TO_ZERO_POINT_DOMAIN.keys(), (
        f"Only support layout: {LAYOUT_TO_ZERO_POINT_DOMAIN.keys()}"
    )
    if zero_point_domain == ZeroPointDomain.NONE:
        # the first value is the default one
        zero_point_domain = LAYOUT_TO_ZERO_POINT_DOMAIN[type(layout)][0]
    else:
        assert zero_point_domain in LAYOUT_TO_ZERO_POINT_DOMAIN[type(layout)], (
            f"Layout only support {LAYOUT_TO_ZERO_POINT_DOMAIN[layout]}"
        )

    if zero_point_domain == ZeroPointDomain.INT and isinstance(layout, Int4XPULayout):
        zero_point_dtype = torch.int32

    preserve_zero = (
        config.preserve_zero
        if config.preserve_zero is not None
        else LAYOUT_TO_PRESERVE_ZEROS[type(layout)]
    )
    # Sparse Marlin only supports symmetric quantization.
    # NOTE: If we start having lots of layouts that require different configurations,
    # we should consider moving this logic somewhere else.
    if isinstance(layout, MarlinSparseLayout):
        mapping_type = MappingType.SYMMETRIC
        assert group_size == 128 or group_size == weight.shape[-1], (
            f"MarlinSparseLayout only supports 128 group size or per channel quantization, got {group_size}"
        )

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
    return new_weight


@register_quantize_module_handler(Int4WeightOnlyConfig)
def _int4_weight_only_transform(
    module: torch.nn.Module, config: Int4WeightOnlyConfig
) -> torch.nn.Module:
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, "weight"), (
        "applying int8 weight only quant requires module to have weight attribute"
        + " but {module} does not have one"
    )
    new_weight = _int4_weight_only_quantize_tensor(module.weight, config)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class Float8DynamicActivationInt4WeightConfig(AOBaseConfig):
    """Configuration for apply float8 dynamic per row quantization and int4
    per group weight quantization to linear
    (only group_size 128 is supported right now since underlying kernel used only supports 128
    and above and no benefits of making it bigger)

    Args:
        `packing_format`: how the weight is packed, only preshuffled is supported
    """

    packing_format: PackingFormat = "preshuffled"


@register_quantize_module_handler(Float8DynamicActivationInt4WeightConfig)
def _float8_dynamic_activation_int4_weight_transform(
    module: torch.nn.Module, config: Float8DynamicActivationInt4WeightConfig
) -> torch.nn.Module:
    assert hasattr(module, "weight"), (
        "applying int8 weight only quant requires module to have weight attribute"
        + " but {module} does not have one"
    )
    packing_format = config.packing_format

    assert packing_format == "preshuffled", (
        f"only preshuffled packing_format supported right now, got: {packing_format}"
    )
    weight = module.weight
    group_size = 128
    block_size = tuple([1 for _ in range(weight.ndim - 1)] + [group_size])
    new_weight = Int4PreshuffledTensor.from_hp(
        module.weight,
        block_size,
        activation_dtype=torch.float8_e4m3fn,
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class Int8WeightOnlyConfig(AOBaseConfig):
    """
    Configuration for applying int8 weight-only symmetric per-channel quantization to linear layers.

    Args:
        group_size: Optional[int] = None - Controls the granularity of quantization. If None, applies per-channel quantization.
            Otherwise, applies per-group quantization with the specified group size.
        set_inductor_config: bool = True - If True, adjusts `torchinductor` settings to recommended values
            for better performance with this quantization scheme.
    """

    group_size: Optional[int] = None
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.Int8WeightOnlyConfig")


# for BC
int8_weight_only = Int8WeightOnlyConfig


def _int8_weight_only_quantize_tensor(weight, config):
    mapping_type = MappingType.SYMMETRIC
    target_dtype = torch.int8
    eps = torch.finfo(torch.float32).eps
    zero_point_dtype = torch.int64
    group_size = config.group_size
    if group_size is None:
        group_size = weight.shape[-1]
    block_size = tuple([1 for x in range(weight.dim() - 1)] + [group_size])
    new_weight = to_affine_quantized_intx(
        weight,
        mapping_type,
        block_size,
        target_dtype,
        eps=eps,
        zero_point_dtype=zero_point_dtype,
    )
    return new_weight


@register_quantize_module_handler(Int8WeightOnlyConfig)
def _int8_weight_only_transform(module: torch.nn.Module, config: Int8WeightOnlyConfig):
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, "weight"), (
        "applying int8 weight only quant requires module to have weight attribute"
        + " but {module} does not have one"
    )
    new_weight = _int8_weight_only_quantize_tensor(module.weight, config)
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


def _int8_symm_cutlass_quant(x: torch.Tensor) -> torch.Tensor:
    return to_affine_quantized_intx(
        x,
        mapping_type=MappingType.SYMMETRIC,
        block_size=_get_per_token_block_size(x),
        target_dtype=torch.int8,
        scale_dtype=torch.float32,
        eps=torch.finfo(torch.float32).eps,
        zero_point_domain=ZeroPointDomain.NONE,
    )


def _int4_symm_cutlass_quant(x: torch.Tensor) -> torch.Tensor:
    return to_affine_quantized_intx(
        x,
        mapping_type=MappingType.SYMMETRIC,
        block_size=_get_per_token_block_size(x),
        target_dtype=torch.int8,
        quant_min=-8,
        quant_max=7,
        scale_dtype=torch.float32,
        eps=torch.finfo(torch.float32).eps,
        zero_point_domain=ZeroPointDomain.NONE,
        _layout=CutlassInt4PackedLayout(),
    )


def _float8_cutlass_quant(
    x: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    return to_affine_quantized_floatx(
        x,
        block_size=_get_per_token_block_size(x),
        scale_dtype=torch.float32,
        target_dtype=target_dtype,
        _layout=Float8Layout(mm_config=None),
    )


def _float8_cutlass_quant_sparse(
    x: torch.Tensor,
    target_dtype: torch.dtype,
) -> (torch.Tensor, torch.Tensor):
    return to_affine_quantized_floatx(
        x,
        block_size=_get_per_token_block_size(x),
        scale_dtype=torch.float32,
        target_dtype=target_dtype,
        _layout=CutlassSemiSparseLayout(),
    )


@dataclass
class Int8DynamicActivationInt8WeightConfig(AOBaseConfig):
    """
    Configuration for applying int8 dynamic symmetric per-token activation and int8 per-channel weight
    quantization to linear layers.

    Args:
        layout: Optional[Layout] = PlainLayout() - Tensor layout for the quantized weights. Controls how the
            quantized data is stored and accessed.
        act_mapping_type: Optional[MappingType] = MappingType.SYMMETRIC - Mapping type for activation quantization.
            SYMMETRIC uses symmetric quantization around zero.
        weight_only_decode: bool = False - If True, only quantizes weights during forward pass and keeps activations
            in original precision during decode operations.
        set_inductor_config: bool = True - If True, adjusts `torchinductor` settings to recommended values
            for better performance with this quantization scheme.
    """

    layout: Optional[Layout] = PlainLayout()
    act_mapping_type: Optional[MappingType] = MappingType.SYMMETRIC
    weight_only_decode: bool = False
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.Int8DynamicActivationInt8WeightConfig"
        )


# for BC
int8_dynamic_activation_int8_weight = Int8DynamicActivationInt8WeightConfig


def _int8_dynamic_activation_int8_weight_quantize_tensor(weight, config):
    layout = config.layout
    act_mapping_type = config.act_mapping_type
    weight_only_decode = config.weight_only_decode

    in_features = weight.shape[-1]
    # int8 dynamic quantization only has benefit when in_feature > 16
    if in_features <= 16:
        logger.info(
            f"Skipping applying int8_dynamic_activation_int8_weight to weight of shape {weight.shape}"
            f" because `in_feature` is <= 16: {in_features}"
        )
        return weight

    # weight settings
    mapping_type = MappingType.SYMMETRIC
    weight_zero_point_domain = ZeroPointDomain.NONE

    def get_weight_block_size(x):
        return tuple([1 for _ in range(x.dim() - 1)] + [x.shape[-1]])

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
    new_weight = to_affine_quantized_intx(
        weight,
        mapping_type,
        block_size,
        target_dtype,
        eps=eps,
        zero_point_dtype=zero_point_dtype,
        _layout=layout,
        zero_point_domain=weight_zero_point_domain,
    )
    new_weight = to_linear_activation_quantized(new_weight, input_quant_func)
    return new_weight


@register_quantize_module_handler(Int8DynamicActivationInt8WeightConfig)
def _int8_dynamic_activation_int8_weight_transform(
    module: torch.nn.Module, config: Int8DynamicActivationInt8WeightConfig
) -> torch.nn.Module:
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, "weight"), (
        "applying int8 dynamic activation int8 weight quant requires module to have weight attribute"
        + "but {module} does not have one"
    )
    new_weight = _int8_dynamic_activation_int8_weight_quantize_tensor(
        module.weight, config
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


def int8_dynamic_activation_int8_semi_sparse_weight():
    """
    Applies int8 dnynamic symmetric per-token activation and int8 per-channel weight
    quantization + 2:4 sparsity to linear layers.
    """
    warnings.warn(
        """int8_dyanmic_activation_int8_semi_sparse_weight() will be deprecated at a later release. Please use the layout kwarg in int8_dynamic_activation_int8_weight instead.

    from torchao.dtypes import SemiSparseLayout
    int8_dynamic_activation_int8_weight(layout=SemiSparseLayout()"""
    )

    return int8_dynamic_activation_int8_weight(layout=SemiSparseLayout())


@dataclass
class Float8WeightOnlyConfig(AOBaseConfig):
    """
    Configuration for applying float8 weight-only symmetric per-channel quantization to linear layers.

    Args:
        weight_dtype (torch.dtype): The target data type for weight quantization. Default is torch.float8_e4m3fn.
        set_inductor_config (bool): if True, adjusts `torchinductor` settings to recommended values.
        version (int): the version of the config, version 1 is using AffineQuantizedTensor that we plan to deprecate/split, version 2 is using Float8Tensor (default)

    Note:
        The actual matmul will be computed in original precision of the weight tensor.
    """

    weight_dtype: torch.dtype = e4m3_dtype
    set_inductor_config: bool = True
    version: int = 2

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.Float8WeightOnlyConfig")


# for BC
float8_weight_only = Float8WeightOnlyConfig


def _float8_weight_only_quant_tensor(weight, config):
    if config.version == 1:
        warnings.warn(
            "version 1 of Float8WeightOnlyConfig is deprecated and will no longer be supported in a future release, please use version 2, see https://github.com/pytorch/ao/issues/2649 for more details"
        )
        from torchao.dtypes import to_affine_quantized_floatx

        block_size = tuple([1 for _ in range(weight.dim() - 1)] + [weight.shape[-1]])
        new_weight = to_affine_quantized_floatx(
            input_float=weight,
            block_size=block_size,
            target_dtype=config.weight_dtype,
            scale_dtype=None,
            _layout=Float8Layout(mm_config=None),
        )
    else:
        assert config.version == 2, f"Unexpected version: {config.version}"
        weight_dtype = config.weight_dtype
        new_weight = Float8Tensor.from_hp(
            weight, float8_dtype=weight_dtype, granularity=PerRow()
        )
    return new_weight


@register_quantize_module_handler(Float8WeightOnlyConfig)
def _float8_weight_only_transform(
    module: torch.nn.Module, config: Float8WeightOnlyConfig
) -> torch.nn.Module:
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, "weight"), (
        "applying int8 weight only quant requires module to have weight attribute"
        + " but {module} does not have one"
    )
    new_weight = _float8_weight_only_quant_tensor(module.weight, config)

    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


def _input_activation_quant_func_fp8(
    x: torch.Tensor,
    activation_granularity: FP8Granularity,
    activation_dtype: torch.dtype,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
):
    """This function is used to quantize the input activation tensor for an aqt_float variant. If scale
    is not provided it will be dynamically calculate the scales otherwise it will use the provided scale.
    """
    assert zero_point is None, (
        "Zero point is not supported for dynamic FP8 quantization"
    )
    if isinstance(activation_granularity, PerRow):
        assert x.dtype == torch.bfloat16, (
            "PerRow quantization only works for bfloat16 precision input activation"
        )

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
        assert isinstance(activation_granularity, PerTensor), (
            "Static quantization only supports PerTensor granularity"
        )
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
    assert weight.dim() in [
        2,
        3,
    ], f"float8 quantization only works for 2/3-D tensors, got {weight.dim()}D tensor"

    out_dim, in_dim = weight.shape[-2:]
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
        granularity (Optional[Union[FP8Granularity, List[FP8Granularity]]]):
            The granularity for quantization. Can be either a single granularity (applied to both
            activations and weights) or a tuple of two granularities (one for activations, one for weights).
            If None, defaults to PerTensor for both. Currently both quantizations need to be the same type. And
            only PerTensor and PerRow are supported.
        mm_config (Float8MMConfig): Configuration for the matrix multiplication. Default uses fast accumulation.
        activation_value_lb (Optional[float]): the lower bound for activation value for calculating scale
        activation_value_ub (Optional[float]): the upper bound for activation value for calculating scale
        kernel_preference (KernelPreference): kernel preference for ops like matmul, grouped matmul etc. by defalut (KernelPreference.AUTO) it will be chosen for user based on hardware or other information, this only needs to be set in weight
        set_inductor_config (bool): if True, adjusts `torchinductor` settings to recommended values.
        version (int): the version of the config, version 1 is using AffineQuantizedTensor that we plan to deprecate/split, version 2 is using Float8Tensor (default)

    """

    activation_dtype: torch.dtype = e4m3_dtype
    weight_dtype: torch.dtype = e4m3_dtype
    granularity: Optional[Union[FP8Granularity, List[FP8Granularity]]] = None
    mm_config: Optional[Float8MMConfig] = None
    activation_value_lb: Optional[float] = None
    activation_value_ub: Optional[float] = None
    kernel_preference: KernelPreference = KernelPreference.AUTO
    set_inductor_config: bool = True
    version: int = 2

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.Float8DynamicActivationFloat8WeightConfig"
        )
        if self.mm_config is None:
            self.mm_config = Float8MMConfig(use_fast_accum=True)
        activation_granularity, weight_granularity = _normalize_granularity(
            self.granularity
        )
        self.granularity = [activation_granularity, weight_granularity]


# for bc
float8_dynamic_activation_float8_weight = Float8DynamicActivationFloat8WeightConfig


def _float8_dynamic_activation_float8_weight_quantize_tensor(weight, config):
    activation_dtype = config.activation_dtype
    weight_dtype = config.weight_dtype
    granularity = config.granularity
    mm_config = config.mm_config
    activation_value_lb = config.activation_value_lb
    activation_value_ub = config.activation_value_ub
    kernel_preference = config.kernel_preference

    # Ensure works on device
    _check_hardware_support(granularity)
    activation_granularity, weight_granularity = granularity

    if not _fp8_mm_compat(weight):
        # TODO(future PR): this should really throw an exception instead of silently
        # not doing what the user asked
        return weight

    if isinstance(weight_granularity, PerRow):
        assert weight.dtype == torch.bfloat16, (
            "PerRow quantization only works for bfloat16 precision input weight"
        )

    if config.version == 1:
        warnings.warn(
            "version 1 of Float8DynamicActivationFloat8WeightConfig is deprecated and will no longer be supported in a future release, please use version 2, see https://github.com/pytorch/ao/issues/2649 for more details"
        )

        block_size = get_block_size(weight.shape[-2:], weight_granularity)
        if weight.dim() == 3:
            block_size = tuple([1] + list(block_size))
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
    else:
        assert config.version == 2, f"Unexpected version: {config.version}"
        act_quant_kwargs = QuantizeTensorToFloat8Kwargs(
            activation_dtype,
            activation_granularity,
            hp_value_lb=activation_value_lb,
            hp_value_ub=activation_value_ub,
            kernel_preference=kernel_preference,
        )

        quantized_weight = Float8Tensor.from_hp(
            weight,
            float8_dtype=weight_dtype,
            granularity=weight_granularity,
            mm_config=mm_config,
            kernel_preference=kernel_preference,
            act_quant_kwargs=act_quant_kwargs,
        )

    return quantized_weight


@register_quantize_module_handler(Float8DynamicActivationFloat8WeightConfig)
def _float8_dynamic_activation_float8_weight_transform(
    module: torch.nn.Module, config: Float8DynamicActivationFloat8WeightConfig
):
    assert is_sm_at_least_89() or is_MI300(), (
        "Float8 dynamic activation quantization is only supported on CUDA>=8.9 and MI300+"
    )
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, "weight"), (
        "applying float8 dynamic activation quant requires module to have weight attribute"
        + f"but {module} does not have one"
    )
    quantized_weight = _float8_dynamic_activation_float8_weight_quantize_tensor(
        module.weight, config
    )
    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class Float8DynamicActivationFloat8SemiSparseWeightConfig(AOBaseConfig):
    """
    Applies float8 dynamic quantization to activations and float8 quantization followed by compression to sparse semi-structured tensor to weights of linear layers.

    Args:
        `layout`: layout type for quantized weight tensor, only supports `CutlassSemiSparseLayout` at the moment.
        `activation_dtype`: data type for quantized activation tensor.
        `weight_dtype`: data type for quantized weight tensor.
    """

    layout: Layout = CutlassSemiSparseLayout()
    activation_dtype: torch.dtype = e5m2_dtype
    weight_dtype: torch.dtype = e4m3_dtype

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.Float8DynamicActivationFloat8SemiSparseWeightConfig"
        )


@register_quantize_module_handler(Float8DynamicActivationFloat8SemiSparseWeightConfig)
def _float8_dynamic_activation_float8_semi_sparse_weight_transform(
    module: torch.nn.Module, config: Float8DynamicActivationFloat8SemiSparseWeightConfig
):
    assert is_sm_at_least_90(), "Float8 quantization is only supported on CUDA>=9.0"

    weight = module.weight
    weight_dtype = config.weight_dtype
    activation_dtype = config.activation_dtype
    layout = config.layout

    if not isinstance(layout, CutlassSemiSparseLayout):
        raise NotImplementedError(
            f"Only CutlassSemiSparseLayout layout is supported. Received {layout}."
        )

    weight = _float8_cutlass_quant_sparse(weight, weight_dtype)
    weight = to_linear_activation_quantized(
        weight,
        _float8_cutlass_quant,
        quant_kwargs={"target_dtype": activation_dtype},
    )

    module.weight = torch.nn.Parameter(weight, requires_grad=False)
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
        set_inductor_config (bool): if True, adjusts `torchinductor` settings to recommended values.
    """

    scale: torch.Tensor
    activation_dtype: torch.dtype = e4m3_dtype
    weight_dtype: torch.dtype = e4m3_dtype
    granularity: Optional[
        Union[FP8Granularity, Tuple[FP8Granularity, FP8Granularity]]
    ] = None
    mm_config: Optional[Float8MMConfig] = Float8MMConfig(use_fast_accum=True)
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.Float8StaticActivationFloat8WeightConfig"
        )


# for bc
float8_static_activation_float8_weight = Float8StaticActivationFloat8WeightConfig


@register_quantize_module_handler(Float8StaticActivationFloat8WeightConfig)
def _float8_static_activation_float8_weight_transform(
    module: torch.nn.Module, config: Float8StaticActivationFloat8WeightConfig
):
    assert is_sm_at_least_89() or is_MI300(), (
        "Float8 static activation quantization is only supported on CUDA 8.9 and above"
    )

    scale = config.scale
    activation_dtype = config.activation_dtype
    weight_dtype = config.weight_dtype
    granularity = config.granularity
    mm_config = config.mm_config
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    weight = module.weight
    activation_granularity, weight_granularity = _normalize_granularity(granularity)
    assert isinstance(activation_granularity, PerTensor), (
        "Static quantization only supports PerTensor granularity"
    )

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
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values.
    """

    dtype: torch.dtype
    group_size: int = 64
    pack_dim: int = -1
    use_hqq: bool = False
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.UIntXWeightOnlyConfig")


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
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

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
            logger.warning(
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


def _adjust_scale_dtype_in_intx_unpacked_tensor(
    intx_unpacked_tensor: IntxUnpackedToInt8Tensor,
    hp_tensor: torch.Tensor,
    scale_dtype: torch.dtype,
) -> None:
    """
    Adjusts the scale_dtype on IntxUnpackedToInt8Tensor.
    Updating the scale dtype requires updating the qdata because qdata is calculated after the scale.
    This is used in IntxWeightOnlyConfig and Int8DynamicActivationIntxWeightConfig to make
    version=2 and version=1 numerically equivalent when the scale_dtype differs from the input dtype
    """
    assert isinstance(intx_unpacked_tensor, IntxUnpackedToInt8Tensor)
    intx_unpacked_tensor.scale = intx_unpacked_tensor.scale.to(scale_dtype)
    qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[intx_unpacked_tensor.target_dtype]
    intx_unpacked_tensor.qdata = quantize_affine(
        hp_tensor,
        intx_unpacked_tensor.block_size,
        intx_unpacked_tensor.scale,
        intx_unpacked_tensor.zero_point,
        output_dtype=torch.int8,
        quant_min=qmin,
        quant_max=qmax,
    )


@dataclass
class IntxWeightOnlyConfig(AOBaseConfig):
    """
    Configuration for quantizing weights to torch.intx, with 1 <= x <= 8.
    Weights are quantized with scales/zeros in a groupwise or channelwise
    manner using the number of bits specified by weight_dtype.
    args:
        weight_dtype: The dtype to use for weight quantization.  Must be torch.intx, where 1 <= x <= 8.
        granularity: The granularity to use for weight quantization.  Must be PerGroup or PerAxis(0).
        mapping_type: The type of mapping to use for the weight quantization.
            Must be one of MappingType.ASYMMETRIC or MappingType.SYMMETRIC.
        scale_dtype: The dtype to use for the weight scale.
        layout: The layout to use for the packed weight tensor:
            - QDQLayout: this layout is designed for export to ExecuTorch.this layout represents the quantization with Q/DQ quant primitives,
                and is intended for export applications like ExecuTorch.
    """

    weight_dtype: torch.dtype = torch.int8
    granularity: Granularity = PerAxis(0)
    mapping_type: MappingType = MappingType.SYMMETRIC
    scale_dtype: Optional[torch.dtype] = None
    layout: Layout = QDQLayout()
    packing_format: PackingFormat = PackingFormat.UNPACKED_TO_INT8
    version: int = 1

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.IntxWeightOnlyConfig")
        assert self.weight_dtype in [getattr(torch, f"int{b}") for b in range(1, 9)], (
            f"weight_dtype must be torch.intx, where 1 <= x <= 8, but got {self.weight_dtype}"
        )
        assert isinstance(self.granularity, (PerAxis, PerGroup)), (
            f"granularity must be PerAxis or PerGroup, but got {self.granularity}"
        )
        if isinstance(self.granularity, PerAxis):
            assert self.granularity.axis == 0, (
                f"axis must be 0 with PerAxis, but got {self.granularity.axis}"
            )
        assert self.mapping_type in [
            MappingType.ASYMMETRIC,
            MappingType.SYMMETRIC,
        ], (
            f"mapping_type must be MappingType.ASYMMETRIC or MappingType.SYMMETRIC, but got {self.mapping_type}"
        )


def _intx_weight_only_quantize_tensor(weight, config):
    weight_dtype = config.weight_dtype
    granularity = config.granularity
    mapping_type = config.mapping_type
    scale_dtype = config.scale_dtype
    layout = config.layout
    packing_format = config.packing_format

    assert weight.dim() == 2, (
        f"IntxWeightOnlyConfig only works for 2-d Tensor, got: {weight.dim()}"
    )
    if isinstance(granularity, PerGroup):
        group_size = granularity.group_size
    elif isinstance(granularity, PerAxis):
        assert granularity.axis == 0, (
            f"axis must be 0 with PerAxis, but got {granularity.axis}"
        )
        group_size = weight.shape[-1]
    else:
        raise ValueError(f"granularity must be PerGroup or PerAxis, got {granularity}")

    block_size = (1, group_size)

    if config.version == 2:
        if config.packing_format == PackingFormat.UNPACKED_TO_INT8:
            new_weight = IntxUnpackedToInt8Tensor.from_hp(
                weight,
                block_size,
                weight_dtype,
                mapping_type=mapping_type,
            )
            if scale_dtype is not None and scale_dtype != weight.dtype:
                _adjust_scale_dtype_in_intx_unpacked_tensor(
                    new_weight, weight, scale_dtype
                )

            return new_weight
        else:
            raise ValueError(f"Unsupported packing format: {packing_format}")

    # Version 1
    quant_min, quant_max = _DTYPE_TO_QVALUE_BOUNDS[weight_dtype]
    weight = to_affine_quantized_intx(
        input_float=weight,
        mapping_type=mapping_type,
        block_size=block_size,
        target_dtype=torch.int8,
        quant_min=quant_min,
        quant_max=quant_max,
        scale_dtype=scale_dtype,
        zero_point_dtype=torch.int8,
        preserve_zero=(mapping_type == MappingType.SYMMETRIC),
        zero_point_domain=ZeroPointDomain.INT,
        _layout=layout,
    )
    return weight


@register_quantize_module_handler(IntxWeightOnlyConfig)
def _intx_weight_only_transform(
    module: torch.nn.Module, config: IntxWeightOnlyConfig
) -> torch.nn.Module:
    assert hasattr(module, "weight"), (
        "applying intx weight only quant requires module to have weight attribute"
        + " but {module} does not have one"
    )
    new_weight = _intx_weight_only_quantize_tensor(module.weight, config)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)

    if isinstance(module, nn.Linear):
        module.extra_repr = types.MethodType(_linear_extra_repr, module)
    elif isinstance(module, nn.Embedding):
        module.extra_repr = types.MethodType(_embedding_extra_repr, module)

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
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.FPXWeightOnlyConfig")


# for BC
fpx_weight_only = FPXWeightOnlyConfig


@register_quantize_module_handler(FPXWeightOnlyConfig)
def _fpx_weight_only_transform(
    module: torch.nn.Module, config: FPXWeightOnlyConfig
) -> torch.nn.Module:
    ebits = config.ebits
    mbits = config.mbits
    weight = module.weight
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

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


@dataclass
class FbgemmConfig(AOBaseConfig):
    """Quantization Config for fbgemm-genai kernels
    Args:
       input_dtype (torch.dtype): input dtype of the kernel
       weight_dtype (torch.dtype): weight dtype of the kernel
       output_dtype (torch.dtype): output dtype of the kernel
       group_size (int): The group size for weight
       preshuffle (bool): whether preshuffle the weights or not
    """

    input_dtype: torch.dtype
    weight_dtype: torch.dtype
    output_dtype: torch.dtype
    block_size: Optional[List[int]] = None
    activation_scale_ub: float = 1200.0
    preshuffle: bool = False


@register_quantize_module_handler(FbgemmConfig)
def _(module: torch.nn.Module, config: FbgemmConfig) -> torch.nn.Module:
    if not _is_fbgemm_genai_gpu_available():
        raise ImportError("Requires fbgemm-gpu-genai >= 1.2.0")

    _SUPPORTED_DTYPES = {
        (torch.bfloat16, torch.int4, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16),
    }

    if (
        (config.input_dtype == torch.bfloat16)
        and (config.weight_dtype == torch.int4)
        and (config.output_dtype == torch.bfloat16)
    ):
        if config.preshuffle:
            weight = Int4PreshuffledTensor.from_hp(
                module.weight,
                config.block_size,
                activation_dtype=torch.bfloat16,
            )
        else:
            weight = Int4Tensor.from_hp(
                module.weight,
                config.block_size,
            )
        module.weight = torch.nn.Parameter(weight, requires_grad=False)
        module.extra_repr = types.MethodType(_linear_extra_repr, module)
        return module
    if (
        (config.input_dtype == e4m3_dtype)
        and (config.weight_dtype == torch.int4)
        and (config.output_dtype == torch.bfloat16)
    ):
        if config.preshuffle:
            weight = Int4PreshuffledTensor.from_hp(
                module.weight,
                config.block_size,
                activation_dtype=torch.float8_e4m3fn,
            )
            module.weight = torch.nn.Parameter(weight, requires_grad=False)
            module.extra_repr = types.MethodType(_linear_extra_repr, module)
            return module
    elif (
        (config.input_dtype == e4m3_dtype)
        and (config.weight_dtype == e4m3_dtype)
        and (config.output_dtype == torch.bfloat16)
    ):
        weight = to_fbgemm_fp8(
            module.weight,
            config.activation_scale_ub,
        )
        module.weight = torch.nn.Parameter(weight, requires_grad=False)
        module.extra_repr = types.MethodType(_linear_extra_repr, module)
        return module
    else:
        raise NotImplementedError(
            f"{config} is not supported. supported input, weight, output kernel dtypes are: {_SUPPORTED_DTYPES}"
        )


@dataclass
class ModuleFqnToConfig(AOBaseConfig):
    """Per module configurations for torchao quantize_ API

    Args:
        `module_fqn_to_config`: Dict[str, Optional[AOBaseConfig]]: a dictionary from
         the fully qualified name of module to the AOBaseConfig that we want to apply to the module.
         Also has a special key: "_default", if "_default" is present in the dictionary,
         the config for "_default" will be applied to all the remaining modules that does not have
         per module configuration specified.
    """

    module_fqn_to_config: Dict[str, Optional[AOBaseConfig]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.ModuleFqnToConfig")


def _module_fqn_to_config_handler(
    module: torch.nn.Module, module_fqn: str, config: ModuleFqnToConfig
):
    c = None
    if module_fqn in config.module_fqn_to_config:
        # Maybe: we can add module type specific config in the future, in needed
        c = config.module_fqn_to_config[module_fqn]
    else:
        # fallback to use default if no module specific config is provided
        c = config.module_fqn_to_config.get("_default", None)

    if c is not None:
        handler = _QUANTIZE_CONFIG_HANDLER[type(c)]
        return handler(module, c)

    return module


torch.serialization.add_safe_globals(
    [
        _int8_asymm_per_token_quant,
        _int8_symm_per_token_reduced_range_quant,
        _input_activation_quant_func_fp8,
        _int4_symm_cutlass_quant,
        _int8_symm_cutlass_quant,
        _float8_cutlass_quant,
        _float8_cutlass_quant_sparse,
        Target,
    ]
)
