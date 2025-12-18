# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Quantization workflow APIs moved from `torch/quantization/quant_api.py`
to prototype.
"""

import logging
import types
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

import torchao
from torchao.core.config import AOBaseConfig
from torchao.dtypes import (
    CutlassInt4PackedLayout,
    Float8Layout,
    Int8DynamicActInt4WeightCPULayout,
    MarlinQQQLayout,
    PlainLayout,
    UintxLayout,
    to_affine_quantized_floatx,
    to_affine_quantized_intx,
    to_marlinqqq_quantized_intx,
)
from torchao.dtypes.utils import Layout
from torchao.float8.config import e4m3_dtype
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.inference import (
    Float8MMConfig,
    FP8Granularity,
    _normalize_granularity,
)
from torchao.quantization.granularity import (
    PerTensor,
)
from torchao.quantization.linear_activation_quantized_tensor import (
    to_linear_activation_quantized,
)
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.quantization.utils import (
    _fp8_mm_compat,
    _linear_extra_repr,
    get_block_size,
)
from torchao.quantization.weight_tensor_linear_activation_quantization import (
    to_weight_tensor_with_linear_activation_quantization_metadata,
)
from torchao.utils import (
    is_MI300,
    is_sm_at_least_89,
)

logger = logging.getLogger(__name__)


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
        warnings.warn(
            "`Int8DynamicActivationInt4WeightConfig` will be deleted in a future release of torchao. Please see https://github.com/pytorch/ao/issues/2752 for more details."
        )


@register_quantize_module_handler(Int8DynamicActivationInt4WeightConfig)
def _int8_dynamic_activation_int4_weight_transform(
    module: torch.nn.Module,
    config: Int8DynamicActivationInt4WeightConfig,
    *,
    custom_scale: Optional[torch.Tensor] = None,
    custom_zero_point: Optional[torch.Tensor] = None,
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

    # avoid circular import
    from torchao.quantization.quant_api import (
        _int4_symm_cutlass_quant,
        _int8_asymm_per_token_quant,
        _int8_symm_cutlass_quant,
        _int8_symm_per_token_quant,
        _uint8_asymm_per_token_quant,
    )

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
            custom_scale=custom_scale,
            custom_zero_point=custom_zero_point,
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
            custom_scale=custom_scale,
            custom_zero_point=custom_zero_point,
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
        warnings.warn(
            "`Int4DynamicActivationInt4WeightConfig` will be deleted in a future release of torchao. Please see https://github.com/pytorch/ao/issues/2752 for more details."
        )


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

    # avoid circular import
    from torchao.quantization.quant_api import _int4_symm_cutlass_quant

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
        warnings.warn(
            "`GemliteUIntXWeightOnlyConfig` will be deleted in a future release of torchao. Please see https://github.com/pytorch/ao/issues/2752 for more details."
        )


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

    from torchao.prototype.dtypes.uintx.gemlite_layout import get_gemlite_aqt_kwargs

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
        warnings.warn(
            "`Float8StaticActivationFloat8WeightConfig` will be deleted in a future release of torchao. Please see https://github.com/pytorch/ao/issues/2752 for more details."
        )


@register_quantize_module_handler(Float8StaticActivationFloat8WeightConfig)
def _float8_static_activation_float8_weight_transform(
    module: torch.nn.Module, config: Float8StaticActivationFloat8WeightConfig
):
    assert is_sm_at_least_89() or is_MI300(), (
        "Float8 static activation quantization is only supported on CUDA 8.9 and above"
    )

    if isinstance(module, Float8Linear):
        # avoid circular import
        from torchao.quantization.quant_api import _unwrap_float8_linear

        module = _unwrap_float8_linear(module)

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

    # prevent circular import
    from torchao.quantization.quant_api import _input_activation_quant_func_fp8

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
        warnings.warn(
            "`UIntXWeightOnlyConfig` will be deleted in a future release of torchao. Please see https://github.com/pytorch/ao/issues/2752 for more details."
        )


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
                "Recommended to use `Int4WeightOnlyConfig(group_size, use_hqq=True, version=1)` for the best performance"
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
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.FPXWeightOnlyConfig")
        warnings.warn(
            "`FPXWeightOnlyConfig` will be deleted in a future release of torchao. Please see https://github.com/pytorch/ao/issues/2752 for more details."
        )


@register_quantize_module_handler(FPXWeightOnlyConfig)
def _fpx_weight_only_transform(
    module: torch.nn.Module, config: FPXWeightOnlyConfig
) -> torch.nn.Module:
    ebits = config.ebits
    mbits = config.mbits
    weight = module.weight
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    if isinstance(module, Float8Linear):
        # avoid circular import
        from torchao.quantization.quant_api import _unwrap_float8_linear

        module = _unwrap_float8_linear(module)

    from torchao.dtypes import to_affine_quantized_fpx
    from torchao.prototype.dtypes.floatx import FloatxTensorCoreLayout

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
