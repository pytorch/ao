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

import inspect
import logging
import re
import types
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union
from typing import OrderedDict as OrderedDictType

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

import torchao
from torchao.core.config import AOBaseConfig
from torchao.dtypes import (
    AffineQuantizedTensor,
    CutlassSemiSparseLayout,
    Float8Layout,
    Int4CPULayout,
    Int4XPULayout,
    PlainLayout,
    SemiSparseLayout,
    TensorCoreTiledLayout,
    to_affine_quantized_floatx,
    to_affine_quantized_floatx_static,
    to_affine_quantized_intx,
)
from torchao.dtypes.uintx.packed_linear_int8_dynamic_activation_intx_weight_layout import (
    Target,
)
from torchao.dtypes.utils import Layout
from torchao.float8.config import e4m3_dtype
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.inference import (
    Float8MMConfig,
    FP8Granularity,
    _check_hardware_support,
    _granularity_is_a_1_128_w_128_128,
    _normalize_granularity,
)
    _normalize_granularity,
)

# for BC, make sure to keep the `noqa: F401` comments to prevent
# ruff from removing "unused imports"
from torchao.prototype.quantization.quant_api import (
    Float8StaticActivationFloat8WeightConfig,  # noqa: F401
    GemliteUIntXWeightOnlyConfig,  # noqa: F401
)
from torchao.quantization.linear_activation_weight_observed_tensor import (
)
from torchao.quantization.observer import AffineQuantizedObserverBase
from torchao.quantization.quantize_.common import (
    KernelPreference,
)
from torchao.quantization.quantize_.workflows import (
    Float8PackingFormat,
    Float8Tensor,
    Int4ChooseQParamsAlgorithm,
    Int4PackingFormat,
    Int4PlainInt32Tensor,
    Int4PreshuffledTensor,
    Int4Tensor,
    Int4TilePackedTo4dTensor,
    Int8Tensor,
    IntxChooseQParamsAlgorithm,
    IntxOpaqueTensor,
    IntxPackingFormat,
    IntxUnpackedToInt8Tensor,
    QuantizeTensorToFloat8Kwargs,
    QuantizeTensorToInt8Kwargs,
    Sparse2x4CUTLASSFloat8Tensor,
)
from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
    register_quantize_module_handler,
)
from torchao.quantization.utils import (
    _fp8_mm_compat,
    _linear_extra_repr,
    _quantization_type,
    get_block_size,
)
from torchao.utils import (
    is_MI300,
    is_sm_at_least_89,
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
from .unified import Quantizer, TwoStepQuantizer
from .utils import _get_per_token_block_size

logger = logging.getLogger(__name__)

# TODO: revisit this list?
__all__ = [
    "swap_conv2d_1x1_to_linear",
    "Quantizer",
    "TwoStepQuantizer",
    "Int4WeightOnlyQuantizer",
    "autoquant",  # noqa: F822
    "_get_subclass_inserter",
    "quantize_",
    "intx_quantization_aware_training",
    "Int8DynActInt4WeightQuantizer",
    "ModuleFqnToConfig",
]

# Lazy imports to avoid CUDA initialization at import time
_lazy_imports = {
    "autoquant": ".autoquant",
}


def __getattr__(name):
    if name in _lazy_imports:
        import importlib

        module_path = _lazy_imports[name]
        module = importlib.import_module(module_path, __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


LAYOUT_TO_ZERO_POINT_DOMAIN = {
    TensorCoreTiledLayout: [ZeroPointDomain.FLOAT],
    Int4CPULayout: [ZeroPointDomain.FLOAT],
    Int4XPULayout: [ZeroPointDomain.FLOAT, ZeroPointDomain.INT],
}

LAYOUT_TO_PRESERVE_ZEROS = {
    TensorCoreTiledLayout: False,
    Int4CPULayout: False,
    Int4XPULayout: False,
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


def _is_linear(mod, *args):
    # avoid circular dependencies
    from torchao.quantization.qat.affine_fake_quantized_tensor import (
        _AffineFakeQuantizedTensor,
    )

    from .autoquant import AutoQuantizableLinearWeight

    # adding weight tensor subclass isinstance check to make sure the weight is only quantized once
    # when it is shared by multiple linear modules
    return (
        isinstance(mod, torch.nn.Linear)
        and hasattr(mod, "weight")
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
        from torchao.quantization import PerTensor
        from torchao.quantization.linear_observer_tensor import insert_observers_
        from torchao.quantization.observer import (
            AffineQuantizedMinMaxObserver,
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


def _embedding_extra_repr(self):
    return f"num_embeddings={self.weight.shape[0]}, embedding_dim={self.weight.shape[1]}, weight={_quantization_type(self.weight)}"


def _module_extra_repr(self, original_extra_repr, parameter_name):
    module_torchao_extra_repr = []

    original_extra_repr_str = original_extra_repr()
    if len(original_extra_repr_str) > 0:
        module_torchao_extra_repr.append(original_extra_repr_str)

    module_torchao_extra_repr.append(
        f"{parameter_name}={_quantization_type(getattr(self, parameter_name))}"
    )
    return ", ".join(module_torchao_extra_repr)


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
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = _is_linear,
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
        # Int8DynamicActivationInt8WeightConfig (optimized with int8 mm op and torch.compile)
        # Int4WeightOnlyConfig (optimized with int4 tinygemm kernel and torch.compile)
        # Int8WeightOnlyConfig (optimized with int8 mm op and torch.compile
        from torchao.quantization.quant_api import Int4WeightOnlyConfig

        m = nn.Sequential(nn.Linear(32, 1024), nn.Linear(1024, 32))
        quantize_(m, Int4WeightOnlyConfig(group_size=32))

    """
    torch._C._log_api_usage_once("torchao.quantization.quantize_")

    if isinstance(config, FqnToConfig):
        if filter_fn is not None:
            raise ValueError(
                "Custom filter_fn and FqnToConfig were both specified. Only filter_fn=None is supported when FqnToConfig is specified."
            )
        named_modules = dict(model.named_modules())
        for module_fqn, module in named_modules.items():
            if (
                fqn_matches_fqn_config(module_fqn, config)
                or _module_param_matches_fqn_config(module, module_fqn, config)
                or ("_default" in config.fqn_to_config and _is_linear(module))
            ):
                replacement = _fqn_to_config_handler(module, module_fqn, config)
                if device is not None:
                    replacement = replacement.to(device=device)
                # handle module swap
                if replacement is not module and module_fqn != "":
                    child_name = module_fqn.split(".")[-1]
                    parent_fqn = module_fqn.removesuffix(child_name).removesuffix(".")
                    parent_module = named_modules[parent_fqn]
                    setattr(parent_module, child_name, replacement)
    elif isinstance(config, AOBaseConfig):
        filter_fn = _is_linear if filter_fn is None else filter_fn
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
class Int8DynamicActivationIntxWeightConfig(AOBaseConfig):
    """
    Configuration for dynamically quantizing activations to torch.int8 and weights to torch.intx, with 1 <= x <= 8.
    More specifically, activations are dynamically quantized to 8-bits at a per-token granularity with scales/zeros.
    Weights are quantized with scales/zeros in a groupwise or channelwise manner using the number of bits specified by weight_dtype.

    args:
        `weight_dtype`: The dtype to use for weight quantization.  Must be torch.intx, where 1 <= x <= 8.
       ` weight_granularity`: The granularity to use for weight quantization.  Must be PerGroup or PerAxis(axis=0).
        `weight_mapping_type`: The type of mapping to use for the weight quantization.
            Must be one of MappingType.ASYMMETRIC or MappingType.SYMMETRIC.  MappingType.SYMMETRIC requires ZeroPointDomain.NONE
        `weight_scale_dtype`: The dtype to use for the weight scale.
        `act_mapping_type`: The type of mapping to use for the activation quantization.
            Must be one of MappingType.ASYMMETRIC or MappingType.SYMMETRIC.
        `intx_packing_format`: The format to use for the packed weight tensor (version 2 only).
            - unpacked_to_int8: this format is the default and is intended for export applications like ExecuTorch.
            - opaque_torchao_auto: this format is optimized for CPU performance.
        `intx_choose_qparams_algorithm`: The algorithm to use for choosing the quantization parameters.
        `version`: version of the config to use, only subset of above args are valid based on version, see note for more details.

    Example:

    .. literalinclude:: ../../examples/inference/int8_dynamic_activation_intx_weight.py
       :language: python
    """

    weight_dtype: torch.dtype = torch.int8
    weight_granularity: Granularity = PerGroup(32)
    weight_mapping_type: MappingType = MappingType.SYMMETRIC
    weight_scale_dtype: Optional[torch.dtype] = None
    act_mapping_type: MappingType = MappingType.ASYMMETRIC
    intx_packing_format: IntxPackingFormat = IntxPackingFormat.UNPACKED_TO_INT8
    intx_choose_qparams_algorithm: IntxChooseQParamsAlgorithm = (
        IntxChooseQParamsAlgorithm.AFFINE
    )

    version: int = 2

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


def _int8_dynamic_activation_intx_weight_quantize_tensor(
    weight,
    bias,
    config,
    *,
    custom_scale: Optional[torch.Tensor] = None,
    custom_zero_point: Optional[torch.Tensor] = None,
):
    weight_dtype = config.weight_dtype
    weight_granularity = config.weight_granularity
    weight_mapping_type = config.weight_mapping_type
    weight_scale_dtype = config.weight_scale_dtype
    act_mapping_type = config.act_mapping_type
    intx_packing_format = config.intx_packing_format
    intx_choose_qparams_algorithm = config.intx_choose_qparams_algorithm

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

    assert config.version == 2
    assert act_mapping_type == MappingType.ASYMMETRIC
    opaque_formats = [
        IntxPackingFormat.OPAQUE_ATEN_KLEIDIAI,
        IntxPackingFormat.OPAQUE_TORCHAO_AUTO,
        IntxPackingFormat.OPAQUE_TORCHAO_KLEIDIAI,
        IntxPackingFormat.OPAQUE_TORCHAO_LOWBIT,
    ]
    assert (
        intx_packing_format == IntxPackingFormat.UNPACKED_TO_INT8
        or intx_packing_format in opaque_formats
    ), f"Unsupported packing format: {intx_packing_format}"
    if custom_zero_point is not None and custom_zero_point.dtype == torch.int32:
        custom_zero_point = custom_zero_point.to(torch.int8)
    new_weight = IntxUnpackedToInt8Tensor.from_hp(
        weight,
        block_size,
        weight_dtype,
        mapping_type=weight_mapping_type,
        activation_quantization="int8_asym_per_token",
        intx_choose_qparams_algorithm=intx_choose_qparams_algorithm,
        custom_scale=custom_scale,
        custom_zero_point=custom_zero_point,
    )
    if weight_scale_dtype is not None and weight_scale_dtype != weight.dtype:
        _adjust_scale_dtype_in_intx_unpacked_tensor(
            new_weight, weight, weight_scale_dtype
        )

    new_bias = bias

    # Create packed tensor
    if intx_packing_format in opaque_formats:
        new_weight = IntxOpaqueTensor.from_intx_unpacked_to_int8_tensor(
            new_weight, bias=new_bias, intx_packing_format=intx_packing_format
        )
        new_bias = None  # bias is packed with weights

    return new_weight, new_bias


@register_quantize_module_handler(Int8DynamicActivationIntxWeightConfig)
def _int8_dynamic_activation_intx_weight_transform(
    module: torch.nn.Module,
    config: Int8DynamicActivationIntxWeightConfig,
    *,
    parameter_name: str = "weight",
    custom_scale: Optional[torch.Tensor] = None,
    custom_zero_point: Optional[torch.Tensor] = None,
) -> torch.nn.Module:
    new_weight, new_bias = _int8_dynamic_activation_intx_weight_quantize_tensor(
        getattr(module, parameter_name),
        module.bias,
        config,
        custom_scale=custom_scale,
        custom_zero_point=custom_zero_point,
    )
    setattr(
        module,
        parameter_name,
        torch.nn.Parameter(new_weight, requires_grad=False),
    )
    if new_bias is None:
        module.bias = None
    module.extra_repr = types.MethodType(
        partial(
            _module_extra_repr,
            original_extra_repr=module.extra_repr,
            parameter_name=parameter_name,
        ),
        module,
    )
    return module


@dataclass
class Int4WeightOnlyConfig(AOBaseConfig):
    """
    Configuration for int4 weight only quantization, only groupwise quantization is supported
    right now, and we support version 1 and version 2, that are implemented differently although with
    same support. In version 2, different target are mainly distinguished by `packing_format` arg, and in version 1, mainly by `layout`.

    Args:
        `group_size`: parameter for quantization, controls the granularity of quantization, smaller
         size is more fine grained, choices are [256, 128, 64, 32], used in both version 1 and 2
        `int4_packing_format`: the packing format for int4 tensor, used in version 2 only
         `int4_choose_qparams_algorithm`: variants of choose qparams algorithm to use for int4,
         currently support TINYGEMM ("tinygemm") and HQQ ("hqq"), used in version 2 only
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values. used in both version 1 and 2
        `version`: version of the config to use, default is 2

    Example:

    .. literalinclude:: ../../examples/inference/int4_weight_only.py
       :language: python
    """

    group_size: int = 128
    set_inductor_config: bool = True
    # only used in version >= 2
    int4_packing_format: Int4PackingFormat = Int4PackingFormat.PLAIN
    int4_choose_qparams_algorithm: Int4ChooseQParamsAlgorithm = (
        Int4ChooseQParamsAlgorithm.TINYGEMM
    )
    version: int = 2

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.Int4WeightOnlyConfig")


def _int4_weight_only_quantize_tensor(weight, config):
    # TODO(future PR): perhaps move this logic to a different file, to keep the API
    # file clean of implementation details

    # for now, make these local variables to allow the rest of the function
    # to be a direct copy-paste
    group_size = config.group_size
    int4_choose_qparams_algorithm = config.int4_choose_qparams_algorithm
    int4_packing_format = config.int4_packing_format

    if weight.shape[-1] % group_size != 0:
        logger.info(
            f"Skipping quantizing weight with int4 weight only quantization because the shape of weight {weight.shape} is not compatible with group_size {group_size}"
        )
        return weight

    block_size = tuple([1 for _ in range(weight.ndim - 1)] + [group_size])

    assert config.version == 2
    block_size = list(block_size)

    if int4_choose_qparams_algorithm == Int4ChooseQParamsAlgorithm.HQQ:
        assert int4_packing_format == Int4PackingFormat.TILE_PACKED_TO_4D, (
            f"Int4ChooseQParamsAlgorithm.HQQ is not supported by packing format {int4_packing_format}, "
            f"it's only supported by Int4PackingFormat.TILE_PACKED_TO_4D currently"
        )

    if int4_packing_format == Int4PackingFormat.PRESHUFFLED:
        new_weight = Int4PreshuffledTensor.from_hp(
            weight,
            block_size,
            activation_dtype=torch.bfloat16,
        )
        return new_weight
    elif int4_packing_format == Int4PackingFormat.PLAIN:
        new_weight = Int4Tensor.from_hp(
            weight,
            block_size,
        )
        return new_weight
    elif int4_packing_format == Int4PackingFormat.PLAIN_INT32:
        new_weight = Int4PlainInt32Tensor.from_hp(
            weight,
            block_size,
        )
        return new_weight
    elif int4_packing_format == Int4PackingFormat.TILE_PACKED_TO_4D:
        new_weight = Int4TilePackedTo4dTensor.from_hp(
            weight,
            block_size,
            int4_choose_qparams_algorithm=int4_choose_qparams_algorithm,
        )
        return new_weight
    else:
        raise ValueError(f"Unsupported int4 packing format: {int4_packing_format}")


@register_quantize_module_handler(Int4WeightOnlyConfig)
def _int4_weight_only_transform(
    module: torch.nn.Module,
    config: Int4WeightOnlyConfig,
    *,
    parameter_name: str = "weight",
) -> torch.nn.Module:
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, parameter_name), (
        f"applying int4 weight only quant requires module to have {parameter_name} attribute"
        + f" but {module} does not have one"
    )
    new_weight = _int4_weight_only_quantize_tensor(
        getattr(module, parameter_name), config
    )
    setattr(
        module,
        parameter_name,
        torch.nn.Parameter(new_weight, requires_grad=False),
    )
    module.extra_repr = types.MethodType(
        partial(
            _module_extra_repr,
            original_extra_repr=module.extra_repr,
            parameter_name=parameter_name,
        ),
        module,
    )
    return module


@dataclass
class Float8DynamicActivationInt4WeightConfig(AOBaseConfig):
    """Configuration for apply float8 dynamic per row quantization and int4
    per group weight quantization to linear
    (only group_size 128 is supported right now since underlying kernel used only supports 128
    and above and no benefits of making it bigger)

    Args:
        `int4_packing_format`: how the weight is packed, only preshuffled is supported

    Example:

    .. literalinclude:: ../../examples/inference/float8_dynamic_activation_int4_weight.py
       :language: python
    """

    int4_packing_format: Int4PackingFormat = "preshuffled"


@register_quantize_module_handler(Float8DynamicActivationInt4WeightConfig)
def _float8_dynamic_activation_int4_weight_transform(
    module: torch.nn.Module,
    config: Float8DynamicActivationInt4WeightConfig,
    *,
    parameter_name: str = "weight",
) -> torch.nn.Module:
    assert hasattr(module, parameter_name), (
        f"applying float8 dynamic activation int4 weight quant requires module to have {parameter_name} attribute"
        + f" but {module} does not have one"
    )
    int4_packing_format = config.int4_packing_format

    assert int4_packing_format == "preshuffled", (
        f"only preshuffled int4_packing_format supported right now, got: {int4_packing_format}"
    )
    weight = getattr(module, parameter_name)
    group_size = 128
    block_size = tuple([1 for _ in range(weight.ndim - 1)] + [group_size])
    new_weight = Int4PreshuffledTensor.from_hp(
        weight,
        block_size,
        activation_dtype=torch.float8_e4m3fn,
    )
    setattr(
        module,
        parameter_name,
        torch.nn.Parameter(new_weight, requires_grad=False),
    )
    module.extra_repr = types.MethodType(
        partial(
            _module_extra_repr,
            original_extra_repr=module.extra_repr,
            parameter_name=parameter_name,
        ),
        module,
    )
    return module


@dataclass
class Int8WeightOnlyConfig(AOBaseConfig):
    """
    Configuration for applying int8 weight-only symmetric per-channel quantization to linear layers.

    Args:
        group_size (version 1) - Controls the granularity of quantization.
        If None, applies per-channel quantization. Otherwise, applies per-group quantization with the specified group size.
        granularity (version 2) - Quantization granularity.
            PerRow() for per-channel quantization, PerTensor() for per-tensor quantization.
        set_inductor_config: bool = True - If True, adjusts `torchinductor` settings to recommended values
            for better performance with this quantization scheme.

    Example:

    .. literalinclude:: ../../examples/inference/int8_weight_only.py
       :language: python
    """

    group_size: Optional[int] = None
    granularity: Optional[Granularity] = PerRow()
    set_inductor_config: bool = True
    version: int = 1

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.Int8WeightOnlyConfig")
        if self.version == 2:
            assert self.group_size is None, (
                f"Only support version 2 with group_size=None, got {self.group_size}"
            )


def _int8_weight_only_quantize_tensor(weight, config):
    if config.version == 1:
        warnings.warn(
            "Config Deprecation: version 1 of Int8WeightOnlyConfig is deprecated and will no longer be supported in a future release, please use version 2, see https://github.com/pytorch/ao/issues/2752 for more details"
        )
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
    else:
        assert config.version == 2, f"Unexpected version: {config.version}"
        new_weight = Int8Tensor.from_hp(weight, granularity=config.granularity)
    return new_weight


@register_quantize_module_handler(Int8WeightOnlyConfig)
def _int8_weight_only_transform(
    module: torch.nn.Module,
    config: Int8WeightOnlyConfig,
    *,
    parameter_name: str = "weight",
):
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, parameter_name), (
        "applying int8 weight only quant requires module to have {parameter_name} attribute"
        + " but {module} does not have one"
    )
    quantized_tensor = _int8_weight_only_quantize_tensor(
        getattr(module, parameter_name), config
    )
    setattr(
        module,
        parameter_name,
        torch.nn.Parameter(quantized_tensor, requires_grad=False),
    )
    module.extra_repr = types.MethodType(
        partial(
            _module_extra_repr,
            original_extra_repr=module.extra_repr,
            parameter_name=parameter_name,
        ),
        module,
    )
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
        version (int): the version of the config, version 1 is using AffineQuantizedTensor that we plan to deprecate/split, version 2 is using Int8Tensor

    Example:

    .. literalinclude:: ../../examples/inference/int8_dynamic_activation_int8_weight.py
       :language: python
    """

    layout: Optional[Layout] = PlainLayout()
    act_mapping_type: Optional[MappingType] = MappingType.SYMMETRIC
    weight_only_decode: bool = False
    granularity: Granularity = PerRow()
    set_inductor_config: bool = True
    version: int = 1

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.Int8DynamicActivationInt8WeightConfig"
        )


def _int8_dynamic_activation_int8_weight_quantize_tensor(weight, config):
    if config.version == 1:
        layout = config.layout
        act_mapping_type = config.act_mapping_type
        weight_only_decode = config.weight_only_decode

        in_features = weight.shape[-1]
        # int8 dynamic quantization only has benefit when in_feature > 16
        if in_features <= 16:
            logger.info(
                f"Skipping applying Int8DynamicActivationInt8WeightConfig to weight of shape {weight.shape}"
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
        quantized_weight = to_linear_activation_quantized(new_weight, input_quant_func)
    else:
        assert config.granularity in {PerRow(), PerTensor()}, (
            "Only PerRow and PerTensor are supported"
        )
        weight_granularity = config.granularity
        act_granularity = config.granularity

        assert config.act_mapping_type == MappingType.SYMMETRIC, (
            "asymmetric dynamic quant not supported currently"
        )
        assert config.version == 2, f"Unexpected version: {config.version}"

        # TODO: Symmentric/Asymmetric choice for weight quantization
        # https://github.com/pytorch/ao/pull/3241#discussion_r2551515539
        quantized_weight = Int8Tensor.from_hp(
            weight,
            granularity=weight_granularity,
            act_quant_kwargs=QuantizeTensorToInt8Kwargs(
                granularity=act_granularity,
                mapping_type=config.act_mapping_type,
            ),
        )

    return quantized_weight


@register_quantize_module_handler(Int8DynamicActivationInt8WeightConfig)
def _int8_dynamic_activation_int8_weight_transform(
    module: torch.nn.Module,
    config: Int8DynamicActivationInt8WeightConfig,
    *,
    parameter_name="weight",
) -> torch.nn.Module:
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, parameter_name), (
        f"applying int8 dynamic activation int8 weight quant requires module to have {parameter_name} attribute"
        + f" but {module} does not have one"
    )
    new_weight = _int8_dynamic_activation_int8_weight_quantize_tensor(
        getattr(module, parameter_name), config
    )
    setattr(
        module,
        parameter_name,
        torch.nn.Parameter(new_weight, requires_grad=False),
    )
    module.extra_repr = types.MethodType(
        partial(
            _module_extra_repr,
            original_extra_repr=module.extra_repr,
            parameter_name=parameter_name,
        ),
        module,
    )
    return module


@dataclass
class Int8StaticActivationInt8WeightConfig(AOBaseConfig):
    """
    Configuration for applying int8 static symmetric quantization to both activation and weight

    Args:
        act_quant_scale (torch.Tensor): The scale tensor for activation quantization.
        granularity (Granularity): The granularity of quantization. PerRow() and PerTensor() are supported currently
        act_mapping_type (MappingType): The mapping type for activation quantization. only SYMMETRIC is supported currently
        set_inductor_config (bool): if True, adjusts `torchinductor` settings to recommended values.
        version (int): the version of the config
    """

    act_quant_scale: Optional[torch.Tensor] = None
    granularity: Granularity = PerRow()
    act_mapping_type: Optional[MappingType] = MappingType.SYMMETRIC
    set_inductor_config: bool = True
    version: int = 1

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.Int8StaticActivationInt8WeightConfig"
        )

        # Validate activation granularity for static quantization
        if isinstance(self.granularity, PerRow) and self.granularity.dim != -1:
            raise ValueError(
                f"Int8StaticActivationInt8WeightConfig only supports PerRow(dim=-1) "
                f"for activation quantization, got PerRow(dim={self.granularity.dim}). "
                f"Per-feature activation quantization is not supported due to slicing limitations."
            )

    def get_act_quant_kwargs(self) -> QuantizeTensorToInt8Kwargs:
        """Get the activation quantization kwargs for static quantization.

        Returns:
            QuantizeTensorToInt8Kwargs with the configured granularity and mapping type.
        """
        return QuantizeTensorToInt8Kwargs(
            granularity=self.granularity,
            mapping_type=self.act_mapping_type,
        )


@register_quantize_module_handler(Int8StaticActivationInt8WeightConfig)
def _int8_static_activation_int8_weight_transform(
    module: torch.nn.Module,
    config: Int8StaticActivationInt8WeightConfig,
    *,
    parameter_name="weight",
):
    assert config.granularity in {PerRow(), PerTensor()}, (
        "Only PerRow and PerTensor is supported currently"
    )
    assert config.act_mapping_type == MappingType.SYMMETRIC, (
        "asymmetric static quant not supported currently"
    )
    assert hasattr(module, parameter_name), (
        f"Expected module to have attribute `{parameter_name}` but not found"
    )

    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    activation_granularity = config.granularity
    weight_granularity = config.granularity

    quantized_tensor = Int8Tensor.from_hp(
        getattr(module, parameter_name),
        granularity=weight_granularity,
        act_quant_kwargs=QuantizeTensorToInt8Kwargs(
            granularity=activation_granularity,
            mapping_type=config.act_mapping_type,
        ),
        act_quant_scale=config.act_quant_scale.detach(),
    )

    setattr(
        module,
        parameter_name,
        torch.nn.Parameter(quantized_tensor, requires_grad=False),
    )
    module.extra_repr = types.MethodType(
        partial(
            _module_extra_repr,
            original_extra_repr=module.extra_repr,
            parameter_name=parameter_name,
        ),
        module,
    )
    return module


def int8_dynamic_activation_int8_semi_sparse_weight():
    """
    Applies int8 dnynamic symmetric per-token activation and int8 per-channel weight
    quantization + 2:4 sparsity to linear layers.
    """
    warnings.warn(
        """int8_dyanmic_activation_int8_semi_sparse_weight() will be deprecated at a later release. Please use the layout kwarg in Int8DynamicActivationInt8WeightConfig instead.

    from torchao.dtypes import SemiSparseLayout
    Int8DynamicActivationInt8WeightConfig(layout=SemiSparseLayout()"""
    )

    return Int8DynamicActivationInt8WeightConfig(layout=SemiSparseLayout())


@dataclass
class Float8WeightOnlyConfig(AOBaseConfig):
    """
    Configuration for applying float8 weight-only symmetric per-channel quantization to linear layers.

    Args:
        weight_dtype (torch.dtype): The target data type for weight quantization. Default is torch.float8_e4m3fn.
        set_inductor_config (bool): if True, adjusts `torchinductor` settings to recommended values.
        version (int): the version of the config, version 1 is deprecated, version 2 is using Float8Tensor (default)

    Note:
        The actual matmul will be computed in original precision of the weight tensor.

    Example:

    .. literalinclude:: ../../examples/inference/float8_weight_only.py
       :language: python
    """

    weight_dtype: torch.dtype = e4m3_dtype
    set_inductor_config: bool = True
    version: int = 2

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.Float8WeightOnlyConfig")


def _float8_weight_only_quant_tensor(weight, config):
    assert config.version == 2, f"Unexpected version: {config.version}"
    weight_dtype = config.weight_dtype
    new_weight = Float8Tensor.from_hp(
        weight, float8_dtype=weight_dtype, granularity=PerRow()
    )
    return new_weight


@register_quantize_module_handler(Float8WeightOnlyConfig)
def _float8_weight_only_transform(
    module: torch.nn.Module,
    config: Float8WeightOnlyConfig,
    *,
    parameter_name: str = "weight",
) -> torch.nn.Module:
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, parameter_name), (
        "applying float8 weight only quant requires module to have {parameter_name} attribute"
        + " but {module} does not have one"
    )

    if isinstance(module, Float8Linear):
        module = _unwrap_float8_linear(module)

    quantized_tensor = _float8_weight_only_quant_tensor(
        getattr(module, parameter_name), config
    )

    setattr(
        module,
        parameter_name,
        torch.nn.Parameter(quantized_tensor, requires_grad=False),
    )
    module.extra_repr = types.MethodType(
        partial(
            _module_extra_repr,
            original_extra_repr=module.extra_repr,
            parameter_name=parameter_name,
        ),
        module,
    )
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
        version (int): the version of the config, version 1 is deprecated, version 2 is using Float8Tensor (default)

    Example:

    .. literalinclude:: ../../examples/inference/float8_dynamic_activation_float8_weight.py
       :language: python
    """

    activation_dtype: torch.dtype = e4m3_dtype
    weight_dtype: torch.dtype = e4m3_dtype
    granularity: Optional[Union[FP8Granularity, List[FP8Granularity]]] = None
    packing_format: Optional[Float8PackingFormat] = Float8PackingFormat.PLAIN
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
        activation_granularity, weight_granularity = _normalize_granularity(
            self.granularity
        )
        self.granularity = [activation_granularity, weight_granularity]

        default_use_fast_accum = True
        if _granularity_is_a_1_128_w_128_128(self.granularity):
            assert self.kernel_preference in (
                KernelPreference.AUTO,
                KernelPreference.TORCH,
            ), "unimplemented"
            assert self.version >= 2, "unimplemented"
            default_use_fast_accum = False
        if torch.xpu.is_available():
            # XPU does not support fast_accum for now
            default_use_fast_accum = False

        if self.mm_config is None:
            self.mm_config = Float8MMConfig(use_fast_accum=default_use_fast_accum)


def _float8_dynamic_activation_float8_weight_quantize_tensor(weight, config):
    activation_dtype = config.activation_dtype
    weight_dtype = config.weight_dtype
    granularity = config.granularity
    mm_config = config.mm_config
    activation_value_lb = config.activation_value_lb
    activation_value_ub = config.activation_value_ub
    kernel_preference = config.kernel_preference
    packing_format = config.packing_format

    # Ensure works on device
    _check_hardware_support(granularity)
    activation_granularity, weight_granularity = granularity

    # Note: right now we assume it's weights of conv2d and conv3d purely based
    # on the dimension of weight, currently there is no conflict with linear 2d
    # and moe weights 3d
    # if we need to support conv1d, which also has 3d weight, we may have to
    # pass around the module as well to distinguish between conv1d and 3d moe weight
    if weight.dim() in [4, 5]:
        # weights for conv2d or 3d
        assert isinstance(activation_granularity, PerTensor) and isinstance(
            weight_granularity, PerTensor
        ), "4D/5D tensor only supports per tensor activation and weight quantization"

        # conv3d weight dim: (C_out, C_in, K1, K2, K3)
        # conv2d weight dim: (C_out, C_in, K1, K2)
        # skip quantization when either C_out or C_in
        # is not a multiple of 16
        if weight.shape[0] % 16 != 0 or weight.shape[1] % 16 != 0:
            return weight
    elif not _fp8_mm_compat(weight):
        # TODO(future PR): this should really throw an exception instead of silently
        # not doing what the user asked
        return weight
    assert config.version == 2, f"Unexpected version: {config.version}"
    if packing_format == Float8PackingFormat.PLAIN and isinstance(
        weight_granularity, PerRow
    ):
        assert weight.dtype == torch.bfloat16, (
            "PerRow quantization only works for bfloat16 precision input weight"
        )
    act_quant_kwargs = QuantizeTensorToFloat8Kwargs(
        activation_dtype,
        activation_granularity,
        hp_value_lb=activation_value_lb,
        hp_value_ub=activation_value_ub,
        kernel_preference=kernel_preference,
    )
    if packing_format == Float8PackingFormat.PLAIN:
        quantized_weight = Float8Tensor.from_hp(
            weight,
            float8_dtype=weight_dtype,
            granularity=weight_granularity,
            mm_config=mm_config,
            kernel_preference=kernel_preference,
            act_quant_kwargs=act_quant_kwargs,
        )
        return quantized_weight
    elif packing_format == Float8PackingFormat.SPARSE_CUTLASS:
        assert isinstance(weight_granularity, PerRow), (
            "Sparse packing format only supports per-row quantization"
        )
        quantized_weight = Sparse2x4CUTLASSFloat8Tensor.from_hp(
            weight,
            float8_dtype=weight_dtype,
            granularity=weight_granularity,
            act_quant_kwargs=act_quant_kwargs,
        )
        return quantized_weight


@register_quantize_module_handler(Float8DynamicActivationFloat8WeightConfig)
def _float8_dynamic_activation_float8_weight_transform(
    module: torch.nn.Module,
    config: Float8DynamicActivationFloat8WeightConfig,
    *,
    parameter_name: str = "weight",
):
    if torch.cuda.is_available():
        assert is_sm_at_least_89() or is_MI300(), (
            "Float8 dynamic activation quantization is only supported on CUDA>=8.9 and MI300+"
        )
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, parameter_name), (
        f"applying float8 dynamic activation quant requires module to have parameter {parameter_name} attribute"
        + f" but {module} does not have one"
    )
    if isinstance(module, Float8Linear):
        module = _unwrap_float8_linear(module)
    quantized_tensor = _float8_dynamic_activation_float8_weight_quantize_tensor(
        getattr(module, parameter_name), config
    )
    setattr(
        module,
        parameter_name,
        torch.nn.Parameter(quantized_tensor, requires_grad=False),
    )
    module.extra_repr = types.MethodType(
        partial(
            _module_extra_repr,
            original_extra_repr=module.extra_repr,
            parameter_name=parameter_name,
        ),
        module,
    )
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
        `weight_dtype`: The dtype to use for weight quantization.  Must be torch.intx, where 1 <= x <= 8.
        `granularity`: The granularity to use for weight quantization.  Must be PerGroup or PerAxis(0).
        `mapping_type`: The type of mapping to use for the weight quantization.
            Must be one of MappingType.ASYMMETRIC or MappingType.SYMMETRIC.
        `scale_dtype`: The dtype to use for the weight scale.
        `intx_packing_format`: The format to use for the packed weight tensor (version 2 only).
        `intx_choose_qparams_algorithm`: The algorithm to use for choosing the quantization parameters.
        `version`: version of the config to use, only subset of above args are valid based on version, see note for more details.

    Example:

    .. literalinclude:: ../../examples/inference/intx_weight_only.py
       :language: python
    """

    weight_dtype: torch.dtype = torch.int8
    granularity: Granularity = PerAxis(0)
    mapping_type: MappingType = MappingType.SYMMETRIC
    scale_dtype: Optional[torch.dtype] = None
    intx_packing_format: IntxPackingFormat = IntxPackingFormat.UNPACKED_TO_INT8
    intx_choose_qparams_algorithm: IntxChooseQParamsAlgorithm = (
        IntxChooseQParamsAlgorithm.AFFINE
    )
    version: int = 2

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
            MappingType.SYMMETRIC_NO_CLIPPING_ERR,
        ], (
            f"mapping_type must be MappingType.ASYMMETRIC, MappingType.SYMMETRIC, or MappingType.SYMMETRIC_NO_CLIPPING_ERR, but got {self.mapping_type}"
        )


def _intx_weight_only_quantize_tensor(
    weight,
    config,
    *,
    custom_scale: Optional[torch.Tensor] = None,
    custom_zero_point: Optional[torch.Tensor] = None,
):
    weight_dtype = config.weight_dtype
    granularity = config.granularity
    mapping_type = config.mapping_type
    scale_dtype = config.scale_dtype
    intx_packing_format = config.intx_packing_format
    intx_choose_qparams_algorithm = config.intx_choose_qparams_algorithm

    if weight.dim() == 2:
        input_dim = -1
    elif weight.dim() == 4:
        # conv2d: N, C_in, H, W
        input_dim = 1
    else:
        raise ValueError(
            f"IntxWeightOnlyConfig only works for 2-d and 4-d Tensors, got: {weight.dim()}"
        )

    if isinstance(granularity, PerGroup):
        group_size = granularity.group_size
    elif isinstance(granularity, PerAxis):
        assert granularity.axis == 0, (
            f"axis must be 0 with PerAxis, but got {granularity.axis}"
        )
        group_size = weight.shape[input_dim]
    else:
        raise ValueError(f"granularity must be PerGroup or PerAxis, got {granularity}")

    if weight.dim() == 2:
        block_size = (1, group_size)
    else:
        # conv2d: N, C_in, H, W
        assert weight.dim() == 4
        block_size = (1, group_size, 1, 1)

    assert config.version == 2
    if config.intx_packing_format == IntxPackingFormat.UNPACKED_TO_INT8:
        if custom_zero_point is not None and custom_zero_point.dtype == torch.int32:
            custom_zero_point = custom_zero_point.to(torch.int8)
        new_weight = IntxUnpackedToInt8Tensor.from_hp(
            weight,
            block_size,
            weight_dtype,
            mapping_type=mapping_type,
            custom_scale=custom_scale,
            custom_zero_point=custom_zero_point,
            intx_choose_qparams_algorithm=intx_choose_qparams_algorithm,
        )
        if scale_dtype is not None and scale_dtype != weight.dtype:
            _adjust_scale_dtype_in_intx_unpacked_tensor(new_weight, weight, scale_dtype)

        return new_weight
    else:
        raise ValueError(f"Unsupported packing format: {intx_packing_format}")


@register_quantize_module_handler(IntxWeightOnlyConfig)
def _intx_weight_only_transform(
    module: torch.nn.Module,
    config: IntxWeightOnlyConfig,
    *,
    custom_scale: Optional[torch.Tensor] = None,
    custom_zero_point: Optional[torch.Tensor] = None,
) -> torch.nn.Module:
    assert hasattr(module, "weight"), (
        "applying intx weight only quant requires module to have weight attribute"
        + " but {module} does not have one"
    )
    new_weight = _intx_weight_only_quantize_tensor(
        module.weight,
        config,
        custom_scale=custom_scale,
        custom_zero_point=custom_zero_point,
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)

    if isinstance(module, nn.Linear):
        module.extra_repr = types.MethodType(_linear_extra_repr, module)
    elif isinstance(module, nn.Embedding):
        module.extra_repr = types.MethodType(_embedding_extra_repr, module)

    return module


@dataclass
class FqnToConfig(AOBaseConfig):
    """Configuration class for applying different quantization configs to modules or parameters based on their fully qualified names (FQNs).

    Args:
        `fqn_to_config`: typing.OrderedDict[str, Optional[AOBaseConfig]]: an
         ordered dictionary from
             (1). fully qualified name (fqn) of module or parameter
             (2). regex of fully qualified name (in python `re` module regex format), should
                  start with prefix "re:" or
             (3). "_default"
         to the config that we want to apply to the module/param or None

         Config key ordered by precedence:
           * fully qualified parameter name, e.g. `language.layers.0.q_proj.weight`
           * fully qualified module name, e.g. `language.layers.0.q_proj`
           * regex for parameter names, must start with `re:`, e.g. `re:language\.layers\..+\.q_proj.weight`.
             The first regex that matches will be applied.
           * regex for module names, must start with `re:`, e.g. `re:language\.layers\..+\.q_proj`,
             whichever regex fully matches the module fqn first will be applied
             (order of keys for dictionary are kept consistent since we are using OrderedDict)
           * "_default", fallback if no match for all previous keys
             (Note, when using `_default`, the config is applied to all modules, to apply
              it to only a subset of modules, e.g. with some types, it's better to filter
              the modules that we don't want to quantize before hand and configure them to
              None, e.g. `{"re:.+norm.+": None, "_default": linear_config}`) "_default" is not supported when filter_fn is not specified.
        `module_fqn_to_config`: typing.OrderedDict[str, Optional[AOBaseConfig]]: To maintain BC with ModuleFqnToConfig, to be deprecated later
        `version`: int: Version of config to use.

    Note:
        - The order of patterns in the OrderedDict may matter as only the first matching pattern is applied
        - "_default" is ignored for parameter replacement.
    """

    fqn_to_config: OrderedDictType[str, Optional[AOBaseConfig]] = field(
        default_factory=OrderedDict
    )
    # to maintain BC, we keep the previous module_fqn_to_config field
    module_fqn_to_config: OrderedDictType[str, Optional[AOBaseConfig]] = field(
        default_factory=OrderedDict
    )
    version: int = 1

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.FqnToConfig")

        if (
            len(self.fqn_to_config) > 0
            and len(self.module_fqn_to_config) > 0
            and self.fqn_to_config != self.module_fqn_to_config
        ):
            raise ValueError(
                "`fqn_to_config` and `module_fqn_to_config` are both specified and are not equal!"
            )

        # This code handles BC compatibility with `ModuleFqnToConfig`. It ensures that `self.module_fqn_to_config` and `self.fqn_to_config` share the same object.
        if len(self.module_fqn_to_config) > 0 and len(self.fqn_to_config) == 0:
            self.fqn_to_config = self.module_fqn_to_config

        if len(self.fqn_to_config) > 0 and len(self.module_fqn_to_config) == 0:
            self.module_fqn_to_config = self.fqn_to_config

        # TODO we plan to deprecate `_default later, so raise a warning if we find it passed in`
        if "_default" in self.fqn_to_config:
            warnings.warn(
                "Config Deprecation: _default is deprecated and will no longer be supported in a future release. Please see https://github.com/pytorch/ao/issues/3229 for more details."
            )

    def __str__(self):
        return "\n".join(
            [
                "FqnToConfig({",
                *(
                    f"  '{key}':\n    {value},"
                    for key, value in self.fqn_to_config.items()
                ),
                "})",
            ]
        )


# maintain BC
ModuleFqnToConfig = FqnToConfig


def _handler_supports_fqn_quantization(
    handler: Callable[[torch.nn.Module, AOBaseConfig], torch.nn.Module],
) -> bool:
    """
    Returns True if the handler function has a "parameter_name" kwarg in its type signature, False otherwise.
    """
    return inspect.signature(handler).parameters.get("parameter_name", None) is not None


def _fqn_to_config_handler(
    module: torch.nn.Module,
    fqn: str,
    config: FqnToConfig,
):
    """This function expects a module that either is specified in FqnToConfig or has a parameter that is specified in FqnToConfig.

    Args:
        module (torch.nn.Module): The module to be processed.
        fqn (str): The fully qualified name of the module containing the parameters.
        config (FqnToConfig): Configuration object containing regex patterns / fqn mapped
            to quantization configurations.

    Returns:
        torch.nn.Module: The modified module with quantized parameters.

    Raises:
        NotImplementedError: If the quantization configuration is not yet supported for parameter quantization.
    """
    parameter_config_found = False
    top_level_params = []
    for i, (parameter_name, param) in enumerate(list(module.named_parameters())):
        if parameter_name in dir(module):
            parameter_fqn = (
                f"{fqn}.{parameter_name}" if len(fqn) > 0 else parameter_name
            )
            top_level_params.append((i, parameter_name, param, parameter_fqn))

    # First we see if any parameter fqn matches with FqnToConfig, if so, we apply the appropriate transform
    for i, parameter_name, param, parameter_fqn in list(top_level_params):
        if parameter_fqn in config.fqn_to_config:
            parameter_config_found = True
            c = config.fqn_to_config[parameter_fqn]
            # if None, remove from subsequent regex check
            if c is None:
                top_level_params.pop(i)
            else:
                handler = _QUANTIZE_CONFIG_HANDLER[type(c)]
                if _handler_supports_fqn_quantization(handler):
                    # may be more than one param specified, so don't return prematurely
                    module = handler(module, c, parameter_name=parameter_name)
                else:
                    raise NotImplementedError(
                        f"{type(c)} does not yet support parameter quantization! Please see https://github.com/pytorch/ao/issues/3252 for more details"
                    )

    # then we see if we match module_fqn exactly
    if not parameter_config_found and fqn in config.fqn_to_config:
        c = config.fqn_to_config[fqn]
        if c is not None:
            handler = _QUANTIZE_CONFIG_HANDLER[type(c)]
            return handler(module, c)
        else:
            return module

    # Next try to match parameters on regex patterns
    for i, parameter_name, param, parameter_fqn in top_level_params:
        for pattern in config.fqn_to_config:
            if pattern.startswith("re:") and re.fullmatch(pattern[3:], parameter_fqn):
                parameter_config_found = True
                c = config.fqn_to_config[pattern]
                if c is not None:
                    handler = _QUANTIZE_CONFIG_HANDLER[type(c)]
                    if _handler_supports_fqn_quantization(handler):
                        # may be more than one param specified, so don't return prematurely
                        module = handler(module, c, parameter_name=parameter_name)
                    else:
                        raise NotImplementedError(
                            f"{type(c)} does not yet support parameter quantization! Please see https://github.com/pytorch/ao/issues/3252 for more details"
                        )

    # try to match regex on module fqn
    if not parameter_config_found:
        for pattern in config.fqn_to_config:
            # we'll apply the config for first fully matched pattern
            if pattern.startswith("re:") and re.fullmatch(pattern[3:], fqn):
                c = config.fqn_to_config[pattern]
                if c is not None:
                    handler = _QUANTIZE_CONFIG_HANDLER[type(c)]
                    return handler(module, c)
                else:
                    return module

    # If no module_fqn or parameter_fqn matches, then we apply _default
    if not parameter_config_found:
        c = config.fqn_to_config.get("_default", None)
        if c is not None:
            handler = _QUANTIZE_CONFIG_HANDLER[type(c)]
            # safe to return here as at most only one module will match
            return handler(module, c)

    return module


def fqn_matches_fqn_config(
    fqn: str,
    config: FqnToConfig,
):
    """Check if a given fqn matches the exact fqn or regex pattern specified in FqnToConfig.

    Args:
        fqn (str): The fully qualified name of the module.
        config (FqnToConfig): Configuration object containing regex patterns or raw FQNs for quantization.

    Returns:
        bool: True if the fqn is specified in FqnToConfig. False otherwise.
    """
    if fqn in config.fqn_to_config:
        assert not fqn.startswith("re:"), (
            f"Error: Exact match but regex {fqn} specified."
        )
        return True
    else:
        for maybe_module_or_param_fqn_pattern in config.fqn_to_config:
            if maybe_module_or_param_fqn_pattern.startswith("re:") and re.fullmatch(
                maybe_module_or_param_fqn_pattern[3:], fqn
            ):
                return True
    return False


def _module_param_matches_fqn_config(
    module: nn.Module,
    fqn: str,
    config: FqnToConfig,
):
    """Check if a given module contains top-level parameters that match the exact fqn or regex pattern specified in FqnToConfig.

    Args:
        module (nn.Module): The module to be checked.
        fqn (str): The fully qualified name of the module.
        config (FqnToConfig): Configuration object containing regex patterns or raw FQNs for quantization.

    Returns:
        bool: True if the module contains top-level parameters that match the fqn or regex pattern specified in FqnTo
    """
    for name, param in module.named_parameters():
        if name in dir(module):
            parameter_fqn = f"{fqn}.{name}" if len(fqn) > 0 else name
            if fqn_matches_fqn_config(parameter_fqn, config):
                return True

    return False


def _unwrap_float8_linear(module: Float8Linear) -> nn.Linear:
    """
    Unwrap a torchao Float8Linear by returning a nn.Linear with the same weights and bias.

    Torchao inference quantization techniques are generally only applicable to nn.Linear
    layers, so this helper is useful for unwrapping models trained with torchao float8 training,
    which replaces nn.Linear layers with Float8Linear layers.
    """
    with torch.device("meta"):
        new_module = nn.Linear(module.in_features, module.out_features)
    new_module.weight = module.weight
    new_module.bias = module.bias
    return new_module


torch.serialization.add_safe_globals(
    [
        _int8_asymm_per_token_quant,
        _int8_symm_per_token_reduced_range_quant,
        _input_activation_quant_func_fp8,
        _float8_cutlass_quant,
        _float8_cutlass_quant_sparse,
        Target,
    ]
)
