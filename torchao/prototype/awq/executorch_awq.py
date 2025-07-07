# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
AWQ implementation with QDQLayout support and dynamic activation quantization for ExecuTorch.

This module extends the existing AWQ implementation to support:
1. QDQLayout (Quantize-Dequantize Layout) for ExecuTorch compatibility
2. 8-bit dynamic activation quantization for scales
3. Improved quantization workflow for ExecuTorch deployment
"""

import types
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

import torch
import torch.nn.functional as F

import torchao
from torchao.core.config import AOBaseConfig
from torchao.dtypes import to_affine_quantized_intx
from torchao.dtypes.uintx.q_dq_layout import QDQLayout
from torchao.quantization.quant_primitives import _DTYPE_TO_BIT_WIDTH
from torchao.quantization import to_weight_tensor_with_linear_activation_scale_metadata
from torchao.quantization.granularity import PerGroup
from torchao.quantization.observer import AffineQuantizedObserverBase
from torchao.quantization.quant_api import (
    _linear_extra_repr,
    _replace_with_custom_fn_if_matches_filter,
)
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.quantization.quant_api import _int8_asymm_per_token_quant
from torchao.quantization.quant_primitives import (
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
)

from torchao.quantization.linear_activation_quantized_tensor import (
    LinearActivationQuantizedTensor,
    to_linear_activation_quantized,
)


class AWQObserverQDQ(AffineQuantizedObserverBase):
    """
    AWQ Observer with QDQLayout support and dynamic activation quantization.

    This observer extends the base AWQ implementation to support:
    - QDQLayout for ExecuTorch compatibility
    - 8-bit dynamic activation quantization for scales
    - Improved calibration workflow
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        quantization_granularity: PerGroup,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        n_validation_examples: int,
        validation_sequence_len: int,
        scale_search_space_size: int = 20,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: Optional[bool] = True,
        zero_point_domain=ZeroPointDomain.INT,
    ):
        """
        Args:
            weight: The weight tensor to be observed
            bias: The bias tensor to be observed
            quantization_granularity: Granularity for weight quantization
            mapping_type: Quantization mapping type
            target_dtype: Target dtype for quantized weights
            n_validation_examples: Number of calibration examples
            validation_sequence_len: Sequence length for calibration
            scale_search_space_size: Number of scale options to search
            quant_min: Minimum quantization value
            quant_max: Maximum quantization value
            eps: Minimum scale value
            scale_dtype: Scale tensor dtype
            zero_point_dtype: Zero point tensor dtype
            preserve_zero: Whether to preserve zero exactly
            zero_point_domain: Domain for zero point values
        """
        super().__init__(
            mapping_type,
            target_dtype,
            quantization_granularity,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            scale_dtype=scale_dtype,
            zero_point_dtype=zero_point_dtype,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
        )

        self.quantization_granularity = quantization_granularity
        self.weight = weight
        self.bias = bias
        self.n_validation_examples = n_validation_examples
        self.validation_sequence_len = validation_sequence_len
        self.scale_search_space_size = scale_search_space_size

        # Calibration state
        self.calibration_token_count = 0
        self.inputs = []
        self.outputs = []
        self.device = self.weight.device
        self.average = torch.zeros((1, weight.shape[1]), device=self.device)

        if self.bias is not None:
            self.bias = self.bias.to(self.device)

    @torch.no_grad()
    def forward(self, input: torch.Tensor, output: torch.Tensor):
        """Collect calibration data during forward pass."""
        if len(self.inputs) < self.n_validation_examples:
            self.inputs.append(input.to("cpu"))
            self.outputs.append(output.to("cpu"))

        # Handle different input shapes
        if len(input.shape) == 2:  # [batch, hidden]
            self.calibration_token_count += input.shape[0]
            self.average += input.abs().sum(dim=0)
        else:  # [batch, seq_len, hidden]
            self.calibration_token_count += (
                input.shape[-2] * input.shape[0]
            )  # batch * seq_len
            # Sum over batch and sequence dimensions to get [hidden_size]
            self.average += input.abs().sum(dim=(0, -2))

    def calculate_qparams(self):
        """
        Calculate optimal quantization parameters using AWQ with optional dynamic activation quantization.

        Returns:
            Optimal activation scales for AWQ quantization
        """
        assert self.outputs, "Must calibrate observer first by running model on data"

        # Normalize average activation magnitudes
        self.average /= self.calibration_token_count

        # Move calibration data to device
        for i in range(self.n_validation_examples):
            self.inputs[i] = self.inputs[i].to(self.device)
            self.outputs[i] = self.outputs[i].to(self.device)

        best_loss = float("inf")
        best_scales = None

        # Search over scale options
        for i in range(self.scale_search_space_size):
            ratio = i * 1.0 / self.scale_search_space_size
            scales = self.average.pow(ratio).to(self.weight.dtype)
            scales = scales / (scales.max() * scales.min()).sqrt()

            # Create quantized weight with QDQLayout
            layout = QDQLayout()
            tensor_dtype = torch.int8

            quantized_weight = to_affine_quantized_intx(
                self.weight * scales,
                self.mapping_type,
                (1, self.quantization_granularity.group_size),
                tensor_dtype,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                eps=self.eps,
                scale_dtype=self.scale_dtype,
                zero_point_dtype=self.zero_point_dtype,
                preserve_zero=self.preserve_zero,
                zero_point_domain=self.zero_point_domain,
                _layout=layout,
            )

            # Evaluate quantization loss
            total_loss = 0
            for j in range(self.n_validation_examples):
                quantized_output = F.linear(
                    self.inputs[j] / scales, quantized_weight, self.bias
                )
                loss = (self.outputs[j] - quantized_output).pow(2).mean().item()
                total_loss += loss

            # Update best scales if this is better
            if total_loss < best_loss:
                best_scales = scales
                best_loss = total_loss

        # Move calibration data back to CPU to save memory
        for i in range(self.n_validation_examples):
            self.inputs[i] = self.inputs[i].to("cpu")
            self.outputs[i] = self.outputs[i].to("cpu")

        return best_scales.detach()


class AWQObservedLinearQDQ(torch.nn.Linear):
    """
    AWQ Observed Linear layer with QDQLayout support.

    This layer captures activations during calibration and applies AWQ
    quantization with QDQLayout for ExecuTorch compatibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_obs: AWQObserverQDQ,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs

    def forward(self, input: torch.Tensor):
        """Forward pass with activation observation."""
        output = F.linear(input, self.weight, self.bias)
        self.act_obs(input, output)
        return output

    @classmethod
    def from_float(cls, float_linear: torch.nn.Linear, act_obs: AWQObserverQDQ):
        """Create AWQObservedLinearQDQ from a regular Linear layer."""
        observed_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            act_obs,
            float_linear.bias is not None,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear


def _awq_int8_dynamic_activation_intx_weight_quant(input_tensor, awq_scale):
    """
    AWQ-aware activation quantization function that applies AWQ scaling before standard int8 quantization.

    Args:
        input_tensor: Input tensor to quantize
        awq_scale: AWQ scaling factor to apply before quantization

    Returns:
        Quantized input tensor
    """
    # Step 1: Apply AWQ scaling
    scaled_input = input_tensor / awq_scale
    # Step 2: Apply standard int8 dynamic activation quantization
    return _int8_asymm_per_token_quant(scaled_input)


def insert_awq_observer_qdq_(
    model: torch.nn.Module,
    n_validation_examples: int,
    validation_sequence_len: int,
    quant_dtype: torch.dtype = torch.int4,
    scale_search_space_size: int = 20,
    group_size: int = 128,
):
    """
    Insert AWQ observers with QDQLayout support into Linear layers.

    Args:
        model: Model to modify in-place
        n_validation_examples: Number of calibration examples
        validation_sequence_len: Sequence length for calibration
        quant_dtype: Target quantization dtype
        scale_search_space_size: Number of scale options to search
        group_size: Group size for quantization granularity
    """
    _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)

    assert quant_dtype != torch.uint8, (
        "Invalid quant_dtype. Please use torch.int1 .. torch.int8"
    )

    # Quantization configuration
    mapping_type = MappingType.ASYMMETRIC
    quantization_granularity = PerGroup(group_size)
    quant_min = 0
    quant_max = (
        255 if quant_dtype == torch.int8 else 2 ** _DTYPE_TO_BIT_WIDTH[quant_dtype] - 1
    )
    eps = torch.finfo(torch.float32).eps
    preserve_zero = True
    zero_point_dtype = torch.int64
    zero_point_domain = ZeroPointDomain.INT

    def replace_with_observer(layer):
        """Replace Linear layer with AWQObservedLinearQDQ."""
        observer = AWQObserverQDQ(
            layer.weight,
            layer.bias,
            quantization_granularity,
            mapping_type,
            quant_dtype,
            n_validation_examples,
            validation_sequence_len,
            scale_search_space_size,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
            zero_point_dtype=zero_point_dtype,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
        )
        return AWQObservedLinearQDQ.from_float(layer, observer)

    _replace_with_custom_fn_if_matches_filter(model, replace_with_observer, _is_linear)


def _is_awq_observed_linear_qdq(mod, *args):
    """Filter function to identify AWQObservedLinearQDQ modules for quantization."""
    return isinstance(mod, AWQObservedLinearQDQ)


@dataclass
class AWQQDQConfig(AOBaseConfig):
    """
    Configuration for AWQ quantization with QDQLayout support.

    Args:
        quant_dtype: Target quantization dtype
        group_size: Group size for quantization granularity
        set_inductor_config: Whether to set recommended inductor settings
    """

    quant_dtype: torch.dtype = torch.int4
    group_size: int = 64
    set_inductor_config: bool = True


@register_quantize_module_handler(AWQQDQConfig)
def _awq_qdq_transform(
    module: torch.nn.Module,
    config: AWQQDQConfig,
) -> torch.nn.Module:
    """
    Transform AWQObservedLinearQDQ to quantized Linear with QDQLayout.

    Args:
        module: AWQObservedLinearQDQ module to transform
        config: AWQ QDQ configuration

    Returns:
        Quantized Linear module with QDQLayout weights
    """
    # Only transform AWQObservedLinearQDQ modules
    if not isinstance(module, AWQObservedLinearQDQ):
        return module

    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    observed_linear = module
    quant_dtype = config.quant_dtype
    group_size = config.group_size

    assert quant_dtype != torch.uint8, (
        "Invalid quant_dtype. Please use torch.int1 .. torch.int8"
    )

    # Get optimal activation scales from AWQ calibration
    equalization_scale = observed_linear.act_obs.calculate_qparams()

    # Configure quantization parameters to match working Int8DynamicActivationIntxWeightConfig
    mapping_type = MappingType.SYMMETRIC
    block_size = (1, group_size)

    # Use int8 target_dtype and bounds for weight quantization to match QDQLayout expectations
    target_dtype = torch.int8
    quant_min = _DTYPE_TO_QVALUE_BOUNDS[quant_dtype][0]
    quant_max = _DTYPE_TO_QVALUE_BOUNDS[quant_dtype][1]
    eps = torch.finfo(torch.float32).eps
    preserve_zero = True
    zero_point_dtype = torch.int64  # Standard for int8
    zero_point_domain = ZeroPointDomain.INT
    scale_dtype = torch.float32  # Match working implementation
    layout = QDQLayout()

    # Apply AWQ scaling to weight BEFORE quantization (this is the key)
    awq_scaled_weight = observed_linear.weight * equalization_scale

    # Create quantized weight with QDQLayout using AWQ-scaled weight
    qw = to_affine_quantized_intx(
        awq_scaled_weight,  # Apply AWQ scaling before quantization
        mapping_type,
        block_size,
        target_dtype,
        quant_min,
        quant_max,
        eps,
        scale_dtype=scale_dtype,
        zero_point_dtype=zero_point_dtype,
        preserve_zero=preserve_zero,
        zero_point_domain=zero_point_domain,
        _layout=layout,
    )

    activation_quant_func = _int8_asymm_per_token_quant
    # First wrap up the AffineQuantized weight Tensor into LinearActivationQuantizedTensor
    # This way before going to quantized weight tensor we go through activation
    # dynamic quant
    qw = to_linear_activation_quantized(qw, activation_quant_func)
    # Second wrap it up in WeightTensorWithLinearActivationScaleMetadata to scale activation
    # before they are fed to LinearActivationQuantizedTensor
    qw = to_weight_tensor_with_linear_activation_scale_metadata(qw, equalization_scale)

    linear = torch.nn.Linear(
        observed_linear.in_features,
        observed_linear.out_features,
        observed_linear.bias != None,
        device=observed_linear.weight.device,
        dtype=observed_linear.weight.dtype,
    )
    linear.weight = torch.nn.Parameter(qw, requires_grad=False)
    linear.extra_repr = types.MethodType(_linear_extra_repr, module)
    linear.bias = observed_linear.bias
    return linear
