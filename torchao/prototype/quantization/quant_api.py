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
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

import torchao
from torchao.core.config import AOBaseConfig
from torchao.float8.config import e4m3_dtype
from torchao.float8.inference import (
    Float8MMConfig,
)
from torchao.quantization.granularity import Granularity, PerTensor
from torchao.quantization.quant_primitives import (
    MappingType,
)
from torchao.quantization.quantize_.common import (
    KernelPreference,
    QuantizeTensorKwargs,
)
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep
from torchao.quantization.quantize_.workflows import (
    QuantizeTensorToFloat8Kwargs,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.quantization.utils import (
    _linear_extra_repr,
    _module_extra_repr,
)

logger = logging.getLogger(__name__)


@dataclass
class UIntxWeightOnlyConfig(AOBaseConfig):
    """Weight-only uintx quantization using bit-packed format with gemlite (https://github.com/dropbox/gemlite)
       Triton kernels.

    Supports 4-bit (asymmetric, grouped) and 8-bit (symmetric, per-channel) quantization.
    Uses gemlite library for efficient Triton-based GEMM.

    Args:
        group_size: quantization group size. Use None for per-channel (required for 8-bit).
            Valid values: 32, 64, 128, 256, 512, 1024, None. Default: 128.
        bit_width: quantization bit width, 4 or 8. Default: 4.
        packing_bitwidth: bit width for packing, 8/16/32/None (auto). Default: None.
        set_inductor_config: if True, set recommended torchinductor config. Default: True.

    Example:

    .. literalinclude:: ../../examples/inference/uintx_weight_only.py
       :language: python
    """

    group_size: Optional[int] = 128
    bit_width: int = 4
    packing_bitwidth: Optional[int] = None
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.UIntxWeightOnlyConfig")
        if self.bit_width not in [4, 8]:
            raise ValueError(f"bit_width must be 4 or 8, got {self.bit_width}")
        valid_group_sizes = [32, 64, 128, 256, 512, 1024, None]
        if self.group_size not in valid_group_sizes:
            raise ValueError(
                f"group_size must be one of {valid_group_sizes}, got {self.group_size}"
            )
        if self.bit_width == 8 and self.group_size is not None:
            raise ValueError("group_size must be None for bit_width=8")
        if self.packing_bitwidth not in [8, 16, 32, None]:
            raise ValueError(
                f"packing_bitwidth must be 8, 16, 32, or None, got {self.packing_bitwidth}"
            )


@register_quantize_module_handler(UIntxWeightOnlyConfig)
def _uintx_weight_only_transform(
    module: torch.nn.Module,
    config: UIntxWeightOnlyConfig,
    *,
    parameter_name: str = "weight",
) -> torch.nn.Module:
    from torchao.prototype.quantization.uintx.uintx_bit_packed_tensor import (
        UIntxBitPackedTensor,
    )

    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, parameter_name), (
        f"applying uintx weight only quant requires module to have {parameter_name} attribute"
        + f" but {module} does not have one"
    )
    weight = getattr(module, parameter_name)
    quantized_weight = UIntxBitPackedTensor.from_hp(
        weight,
        bit_width=config.bit_width,
        group_size=config.group_size,
        packing_bitwidth=config.packing_bitwidth,
    )
    setattr(
        module,
        parameter_name,
        torch.nn.Parameter(quantized_weight, requires_grad=False),
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
class Int8DynamicActivationUIntxWeightConfig(AOBaseConfig):
    """Dynamic activation + uintx weight quantization using gemlite (https://github.com/dropbox/gemlite)
       Triton kernels.

    Activations are quantized dynamically at runtime (int8). Weights use bit-packed
    uintx format. Supports 4-bit and 8-bit weight quantization.

    Args:
        group_size: quantization group size. Use None for per-channel (required for 8-bit).
            Valid values: 32, 64, 128, 256, 512, 1024, None. Default: 128.
        bit_width: weight quantization bit width, 4 or 8. Default: 4.
        packing_bitwidth: bit width for packing, 8/16/32/None (auto). Default: None.
        set_inductor_config: if True, set recommended torchinductor config. Default: True.

    Example:

    .. literalinclude:: ../../examples/inference/int8_dynamic_activation_uintx_weight.py
       :language: python
    """

    group_size: Optional[int] = 128
    bit_width: int = 4
    packing_bitwidth: Optional[int] = None
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.Int8DynamicActivationUIntxWeightConfig"
        )
        if self.bit_width not in [4, 8]:
            raise ValueError(f"bit_width must be 4 or 8, got {self.bit_width}")
        valid_group_sizes = [32, 64, 128, 256, 512, 1024, None]
        if self.group_size not in valid_group_sizes:
            raise ValueError(
                f"group_size must be one of {valid_group_sizes}, got {self.group_size}"
            )
        if self.bit_width == 8 and self.group_size is not None:
            raise ValueError("group_size must be None for bit_width=8")
        if self.packing_bitwidth not in [8, 16, 32, None]:
            raise ValueError(
                f"packing_bitwidth must be 8, 16, 32, or None, got {self.packing_bitwidth}"
            )


@register_quantize_module_handler(Int8DynamicActivationUIntxWeightConfig)
def _int8_dynamic_activation_uintx_weight_transform(
    module: torch.nn.Module,
    config: Int8DynamicActivationUIntxWeightConfig,
    *,
    parameter_name: str = "weight",
) -> torch.nn.Module:
    from torchao.prototype.quantization.uintx.uintx_bit_packed_tensor import (
        UIntxBitPackedTensor,
    )

    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, parameter_name), (
        f"applying int8 dynamic activation uintx weight quant requires module to have {parameter_name} attribute"
        + f" but {module} does not have one"
    )
    weight = getattr(module, parameter_name)
    quantized_weight = UIntxBitPackedTensor.from_hp(
        weight,
        bit_width=config.bit_width,
        group_size=config.group_size,
        packing_bitwidth=config.packing_bitwidth,
        mode="dynamic",
    )
    setattr(
        module,
        parameter_name,
        torch.nn.Parameter(quantized_weight, requires_grad=False),
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
        step (Optional[QuantizationStep]): Specifies the step for the observer-based quantization process.
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
        quantize_and_dequantize_output (bool): If True, the output of the linear layer will also be statically
            quantized to float8 and then dequantized back to the original dtype. This is useful for simulating
            the quantization error of the output while keeping the output as a regular tensor for downstream ops.
            This requires an additional output observer during the prepare step. Default is False.

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

    step: Optional["QuantizationStep"] = None
    act_quant_scale: Optional[torch.Tensor] = None
    activation_dtype: torch.dtype = e4m3_dtype
    weight_dtype: torch.dtype = e4m3_dtype
    granularity: Optional[Granularity] = None
    mm_config: Optional[Float8MMConfig] = None
    kernel_preference: KernelPreference = KernelPreference.AUTO
    set_inductor_config: bool = True
    quantize_and_dequantize_output: bool = False
    output_act_quant_scale: Optional[torch.Tensor] = None

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
            all_step_values = [s.value for s in QuantizationStep]
            if self.step not in all_step_values and self.step not in list(
                QuantizationStep
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


class Float8ObservedLinear(torch.nn.Linear):
    """
    A linear module with an observer for float8 static quantization.

    This module wraps a linear layer and adds an AffineQuantizedMinMaxObserver
    that collects statistics during calibration. After calibration, use
    `quantize_` with `Float8StaticActivationFloat8WeightConfig(step="convert")`
    to convert to a quantized module.

    Optionally, an output observer can be provided to collect statistics for
    output quantization when `quantize_and_dequantize_output=True` is used.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_act_obs: "AffineQuantizedMinMaxObserver",  # noqa: F821
        bias: bool = True,
        device=None,
        dtype=None,
        output_act_obs: Optional["AffineQuantizedMinMaxObserver"] = None,  # noqa: F821
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.input_act_obs = input_act_obs
        self.output_act_obs = output_act_obs

    def forward(self, input: Tensor) -> Tensor:
        self.input_act_obs(input)
        output = F.linear(input, self.weight, self.bias)
        if self.output_act_obs is not None:
            self.output_act_obs(output)
        return output

    @classmethod
    def from_float(
        cls,
        float_linear: torch.nn.Linear,
        input_act_obs: "AffineQuantizedMinMaxObserver",  # noqa: F821
        output_act_obs: Optional["AffineQuantizedMinMaxObserver"] = None,  # noqa: F821
    ) -> "Float8ObservedLinear":
        """Create an observed linear from a float linear module."""
        observed_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            input_act_obs,
            bias=float_linear.bias is not None,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
            output_act_obs=output_act_obs,
        )
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear


class Float8ObservedSoftmax(torch.nn.Softmax):
    """
    A softmax module with an observer for float8 static quantization.

    This module wraps a softmax layer and adds an AffineQuantizedMinMaxObserver
    that collects statistics on the output during calibration. After calibration,
    use `quantize_` with `Float8StaticActivationFloat8WeightConfig(step="convert")`
    to convert to a quantized softmax module that applies quantize-and-dequantize
    to simulate quantization error.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        output_act_obs: Optional["AffineQuantizedMinMaxObserver"] = None,  # noqa: F821
    ):
        super().__init__(dim=dim)
        self.output_act_obs = output_act_obs

    def forward(self, input: Tensor) -> Tensor:
        output = F.softmax(input, self.dim, _stacklevel=5)
        if self.output_act_obs is not None:
            self.output_act_obs(output)
        return output

    @classmethod
    def from_float(
        cls,
        float_softmax: torch.nn.Softmax,
        output_act_obs: "AffineQuantizedMinMaxObserver",  # noqa: F821
    ) -> "Float8ObservedSoftmax":
        """Create an observed softmax from a float softmax module."""
        observed_softmax = cls(
            dim=float_softmax.dim,
            output_act_obs=output_act_obs,
        )
        return observed_softmax


class Float8QuantizedSoftmax(torch.nn.Module):
    """
    A softmax module that applies quantize-and-dequantize to its output.

    This module computes softmax and then quantizes the output to float8,
    immediately dequantizing it back to the original dtype. This simulates
    the quantization error while returning a regular tensor.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        output_act_quant_scale: Optional[torch.Tensor] = None,
        output_act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
    ):
        super().__init__()
        self._dim = dim
        # Register scale as a buffer so it moves with the module
        if output_act_quant_scale is not None:
            self.register_buffer("output_act_quant_scale", output_act_quant_scale)
        else:
            self.output_act_quant_scale = None
        self.output_act_quant_kwargs = output_act_quant_kwargs

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    def forward(self, input: Tensor) -> Tensor:
        from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
            _choose_quant_func_and_quantize_tensor,
        )

        output = F.softmax(input, self._dim, _stacklevel=5)

        # Apply quantize-and-dequantize if configured
        if self.output_act_quant_kwargs is not None:
            quantized_output = _choose_quant_func_and_quantize_tensor(
                output,
                self.output_act_quant_kwargs,
                act_quant_scale=self.output_act_quant_scale,
            )
            output = quantized_output.dequantize()

        return output

    @classmethod
    def from_observed(
        cls,
        observed_softmax: "Float8ObservedSoftmax",
        output_act_quant_scale: torch.Tensor,
        output_act_quant_kwargs: QuantizeTensorToFloat8Kwargs,
    ) -> "Float8QuantizedSoftmax":
        """Create a quantized softmax from an observed softmax module."""
        return cls(
            dim=observed_softmax.dim,
            output_act_quant_scale=output_act_quant_scale,
            output_act_quant_kwargs=output_act_quant_kwargs,
        )


@register_quantize_module_handler(Float8StaticActivationFloat8WeightConfig)
def _float8_static_activation_float8_weight_transform(
    module: torch.nn.Module,
    config: Float8StaticActivationFloat8WeightConfig,
) -> torch.nn.Module:
    """
    Transform handler for Float8StaticActivationFloat8WeightConfig.

    Behavior depends on the step:
    - PREPARE: Insert observer into linear or softmax module
    - CONVERT: Convert observed modules to quantized modules

    Supported modules:
    - torch.nn.Linear: Static activation quantization with float8 weights
    - torch.nn.Softmax: Output quantization simulation (quantize-and-dequantize)
    """
    from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
        PrototypeFloat8Tensor,
    )
    from torchao.quantization.observer import AffineQuantizedMinMaxObserver

    step = config.step
    granularity = config.granularity if config.granularity is not None else PerTensor()

    if step == QuantizationStep.PREPARE or step == "prepare":
        # Handle Softmax modules
        if isinstance(module, torch.nn.Softmax):
            output_observer = AffineQuantizedMinMaxObserver(
                mapping_type=MappingType.SYMMETRIC,
                target_dtype=config.activation_dtype,
                granularity=granularity,
                eps=torch.finfo(torch.float32).eps,
                scale_dtype=torch.float32,
                zero_point_dtype=torch.float32,
                keepdim=True,
            )
            return Float8ObservedSoftmax.from_float(module, output_observer)

        # Handle Linear modules
        # Create input observer and wrap linear
        input_observer = AffineQuantizedMinMaxObserver(
            mapping_type=MappingType.SYMMETRIC,
            target_dtype=config.activation_dtype,
            granularity=granularity,
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.float32,
            keepdim=True,
        )
        # Create output observer if quantize_and_dequantize_output is True
        output_observer = None
        if config.quantize_and_dequantize_output:
            output_observer = AffineQuantizedMinMaxObserver(
                mapping_type=MappingType.SYMMETRIC,
                target_dtype=config.activation_dtype,
                granularity=granularity,
                eps=torch.finfo(torch.float32).eps,
                scale_dtype=torch.float32,
                zero_point_dtype=torch.float32,
                keepdim=True,
            )
        return Float8ObservedLinear.from_float(module, input_observer, output_observer)

    elif step == QuantizationStep.CONVERT or step == "convert":
        # Handle observed Softmax modules
        if isinstance(module, Float8ObservedSoftmax):
            if module.output_act_obs is None:
                logger.warning(
                    "Float8ObservedSoftmax has no output observer, returning as-is"
                )
                return module

            # Extract output scale from observer
            output_act_quant_scale, _ = module.output_act_obs.calculate_qparams()

            output_act_quant_kwargs = QuantizeTensorToFloat8Kwargs(
                float8_dtype=config.activation_dtype,
                granularity=granularity,
                mm_config=config.mm_config,
                kernel_preference=config.kernel_preference,
            )

            return Float8QuantizedSoftmax.from_observed(
                module,
                output_act_quant_scale=output_act_quant_scale.detach(),
                output_act_quant_kwargs=output_act_quant_kwargs,
            )

        # Handle observed Linear modules
        if not isinstance(module, Float8ObservedLinear):
            logger.info(
                f"convert: module is not Float8ObservedLinear or Float8ObservedSoftmax, skipping: {type(module)}"
            )
            return module

        # Extract activation scale from observer
        act_quant_scale, _ = module.input_act_obs.calculate_qparams()

        if config.set_inductor_config:
            torchao.quantization.utils.recommended_inductor_config_setter()

        activation_dtype = config.activation_dtype
        weight_dtype = config.weight_dtype

        # Extract output activation scale from observer if available
        output_act_quant_scale = None
        output_act_quant_kwargs = None
        if module.output_act_obs is not None:
            output_act_quant_scale, _ = module.output_act_obs.calculate_qparams()
            output_act_quant_kwargs = QuantizeTensorToFloat8Kwargs(
                float8_dtype=activation_dtype,
                granularity=granularity,
                mm_config=config.mm_config,
                kernel_preference=config.kernel_preference,
            )

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
            output_act_quant_scale=output_act_quant_scale.detach()
            if output_act_quant_scale is not None
            else None,
            output_act_quant_kwargs=output_act_quant_kwargs,
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
        linear.extra_repr = types.MethodType(_linear_extra_repr, linear)
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

        # Handle output quantization kwargs if output_act_quant_scale is provided
        output_act_quant_scale = config.output_act_quant_scale
        output_act_quant_kwargs = None
        if output_act_quant_scale is not None:
            output_act_quant_kwargs = QuantizeTensorToFloat8Kwargs(
                float8_dtype=activation_dtype,
                granularity=granularity,
                mm_config=config.mm_config,
                kernel_preference=config.kernel_preference,
            )

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
            output_act_quant_scale=output_act_quant_scale.detach()
            if output_act_quant_scale is not None
            else None,
            output_act_quant_kwargs=output_act_quant_kwargs,
        )

        module.weight = torch.nn.Parameter(quantized_tensor, requires_grad=False)
        module.extra_repr = types.MethodType(_linear_extra_repr, module)
        return module

    else:
        raise ValueError(
            f"Unexpected step: {step}. Expected one of {[s.value for s in QuantizationStep]} or None."
        )
