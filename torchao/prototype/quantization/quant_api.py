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
from enum import Enum
from typing import Optional

import torch

import torchao
from torchao.core.config import AOBaseConfig
from torchao.dtypes import (
    CutlassInt4PackedLayout,
    Int8DynamicActInt4WeightCPULayout,
    PlainLayout,
    UintxLayout,
    to_affine_quantized_intx,
)
from torchao.dtypes.utils import Layout
from torchao.float8.config import e4m3_dtype
from torchao.float8.inference import (
    Float8MMConfig,
)
from torchao.quantization.granularity import Granularity, PerTensor
from torchao.quantization.linear_activation_quantized_tensor import (
    to_linear_activation_quantized,
)
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization.quantize_.common import (
    KernelPreference,
    QuantizeTensorKwargs,
)
from torchao.quantization.quantize_.workflows import (
    QuantizeTensorToFloat8Kwargs,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.quantization.utils import (
    _linear_extra_repr,
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
        `layout`: layout type for quantized weight tensor, only supports `CutlassInt4PackedLayout()` for now
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
        if isinstance(layout, CutlassInt4PackedLayout):
            input_quant_func = _int8_symm_cutlass_quant
        else:
            input_quant_func = _int8_symm_per_token_quant
    else:
        assert False, f"Unsupported activation mapping type: {act_mapping_type}"

    if isinstance(layout, CutlassInt4PackedLayout):
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
        `layout`: layout type for quantized weight tensor, only supports `CutlassInt4PackedLayout()` for now
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
    Configuration for applying float8 static symmetric quantization to both activation and weight.

    This supports two workflows:

    1. Observer based flow (recommended):
        - Use step="prepare" to insert observers
        - Calibrate with representative data
        - Use step="convert" to convert to quantized model

    2. Direct quantization with known scale:
        - Provide act_quant_scale directly (step is not required)

    Args:
        step (Optional[Float8StaticStep]): Specifies the step for the observer-based quantization process.
            PREPARE: insert observers to linear
            CONVERT: convert the observed linear modules to linear modules with quantized weights
            Can use the corresponding string "prepare", "convert" for simplicity
            If not provided, act_quant_scale must be provided for direct quantization.
        act_quant_scale (Optional[torch.Tensor]): The scale tensor for activation quantization.
            Required when step is not provided.
        activation_dtype (torch.dtype): The target data type for activation quantization. Default is torch.float8_e4m3fn
        weight_dtype (torch.dtype): The target data type for weight quantization. Default is torch.float8_e4m3fn
        granularity (Granularity): The granularity of quantization. Only PerTensor() is supported for static activation quantization because the scale must be fixed at calibration time and work for any batch size at inference.
        mm_config (Float8MMConfig): Configuration for the matrix multiplication. Default uses fast accumulation.
        kernel_preference (KernelPreference): Kernel preference for quantization and matmul operations.
        set_inductor_config (bool): if True, adjusts `torchinductor` settings to recommended values.

    Example (Observer flow):
        # Step 1: Prepare model by inserting observers
        quantize_(model, Float8StaticActivationFloat8WeightConfig(step="prepare"))

        # Step 2: Calibrate with representative data
        for batch in calibration_data:
            model(batch)

        # Step 3: Convert observed model to quantized model
        quantize_(model, Float8StaticActivationFloat8WeightConfig(step="convert"))

    Example (Direct quantization):
        config = Float8StaticActivationFloat8WeightConfig(
            act_quant_scale=my_scale,
            granularity=PerTensor(),
        )
        quantize_(model, config)
    """

    step: Optional["Float8StaticStep"] = None
    act_quant_scale: Optional[torch.Tensor] = None
    activation_dtype: torch.dtype = e4m3_dtype
    weight_dtype: torch.dtype = e4m3_dtype
    granularity: Optional[Granularity] = None
    mm_config: Optional[Float8MMConfig] = None
    kernel_preference: KernelPreference = KernelPreference.AUTO
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.Float8StaticActivationFloat8WeightConfig"
        )
        if self.mm_config is None:
            self.mm_config = Float8MMConfig(use_fast_accum=True)

        # Allow string step values for convenience
        if self.step is not None:
            if isinstance(self.step, str):
                self.step = self.step.lower()
            all_step_values = [s.value for s in Float8StaticStep]
            if self.step not in all_step_values and self.step not in list(
                Float8StaticStep
            ):
                raise ValueError(f"{self.step} is not one of {all_step_values}")

    def get_act_quant_kwargs(self) -> QuantizeTensorKwargs:
        """Return the activation quantization kwargs.

        This method is required by the IsStaticQuantizationConfig protocol.
        """
        granularity = self.granularity if self.granularity is not None else PerTensor()
        return QuantizeTensorToFloat8Kwargs(
            float8_dtype=self.activation_dtype,
            granularity=granularity,
            mm_config=self.mm_config,
            kernel_preference=self.kernel_preference,
        )


# Float8 static quantization step enum
class Float8StaticStep(str, Enum):
    PREPARE = "prepare"
    CONVERT = "convert"


@register_quantize_module_handler(Float8StaticActivationFloat8WeightConfig)
def _float8_static_activation_float8_weight_transform(
    module: torch.nn.Module,
    config: Float8StaticActivationFloat8WeightConfig,
) -> torch.nn.Module:
    """
    Transform handler for Float8StaticActivationFloat8WeightConfig.

    Behavior depends on the step:
    - PREPARE: Insert observer into linear module
    - CONVERT: Convert observed linear to quantized linear
    """
    from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
        PrototypeFloat8Tensor,
    )
    from torchao.quantization.observer import (
        AffineQuantizedMinMaxObserver,
        ObservedLinear,
    )

    step = config.step
    granularity = config.granularity if config.granularity is not None else PerTensor()

    if step == Float8StaticStep.PREPARE or step == "prepare":
        # Create observer and wrap linear
        observer = AffineQuantizedMinMaxObserver(
            mapping_type=MappingType.SYMMETRIC,
            target_dtype=config.activation_dtype,
            granularity=granularity,
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.float32,
        )
        return ObservedLinear.from_float(module, observer)

    elif step == Float8StaticStep.CONVERT or step == "convert":
        if not isinstance(module, ObservedLinear):
            logger.info(
                f"convert: module is not ObservedLinear, skipping: {type(module)}"
            )
            return module

        # Extract activation scale from observer
        # Scale needs to be 2D for 2D activation tensors
        act_quant_scale, _ = module.act_obs.calculate_qparams()
        if act_quant_scale.ndim == 0:
            # TODO: add keep_dim arg for `choose_qparams_affine_with_min_max`
            # to avoid this workaround
            act_quant_scale = act_quant_scale.view(1, 1)

        if config.set_inductor_config:
            torchao.quantization.utils.recommended_inductor_config_setter()

        activation_dtype = config.activation_dtype
        weight_dtype = config.weight_dtype

        # Create quantized weight tensor
        quantized_tensor = PrototypeFloat8Tensor.from_hp(
            module.weight,
            float8_dtype=weight_dtype,
            granularity=granularity,
            mm_config=config.mm_config,
            kernel_preference=config.kernel_preference,
            act_quant_kwargs=QuantizeTensorToFloat8Kwargs(
                float8_dtype=activation_dtype,
                granularity=granularity,
                mm_config=config.mm_config,
                kernel_preference=config.kernel_preference,
            ),
            act_quant_scale=act_quant_scale.detach(),
        )

        # Create new Linear module with quantized weight
        linear = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        linear.weight = torch.nn.Parameter(quantized_tensor, requires_grad=False)
        linear.bias = module.bias
        return linear

    elif step is None:
        # Direct quantization path - use provided act_quant_scale
        if config.act_quant_scale is None:
            raise ValueError(
                "When step is None, act_quant_scale must be provided for direct quantization. "
                "Alternatively, use step='prepare' followed by step='convert' for observer-based flow."
            )

        if config.set_inductor_config:
            torchao.quantization.utils.recommended_inductor_config_setter()

        activation_dtype = config.activation_dtype
        weight_dtype = config.weight_dtype
        act_quant_scale = config.act_quant_scale

        # Create quantized weight tensor
        quantized_tensor = PrototypeFloat8Tensor.from_hp(
            module.weight,
            float8_dtype=weight_dtype,
            granularity=granularity,
            mm_config=config.mm_config,
            kernel_preference=config.kernel_preference,
            act_quant_kwargs=QuantizeTensorToFloat8Kwargs(
                float8_dtype=activation_dtype,
                granularity=granularity,
                mm_config=config.mm_config,
                kernel_preference=config.kernel_preference,
            ),
            act_quant_scale=act_quant_scale.detach(),
        )

        module.weight = torch.nn.Parameter(quantized_tensor, requires_grad=False)
        return module

    else:
        raise ValueError(
            f"Unexpected step: {step}. Expected one of {[s.value for s in Float8StaticStep]} or None."
        )


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
