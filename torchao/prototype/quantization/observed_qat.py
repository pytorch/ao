# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from torchao.core.config import AOBaseConfig
from torchao.quantization.granularity import PerAxis, PerGroup, PerTensor
from torchao.quantization.observer import AffineQuantizedMinMaxObserver
from torchao.quantization.qat.fake_quantize_config import (
    IntxFakeQuantizeConfig,
)
from torchao.quantization.qat.linear import FakeQuantizedLinear
from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
    _get_reduction_params,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.quantization.utils import get_block_size


def _validate_activation_config(config: IntxFakeQuantizeConfig) -> None:
    if config.zero_point_domain != ZeroPointDomain.INT:
        raise ValueError("Calibration requires an integer zero-point domain")
    if (
        config.is_dynamic
        or config.range_learning
        or not isinstance(config.granularity, (PerTensor, PerAxis, PerGroup))
    ):
        raise ValueError(
            "Calibration is only supported for static per-tensor, per-axis, "
            "or per-group activation quantization"
        )


class IntxMinMaxObserver(AffineQuantizedMinMaxObserver):
    """Min-max observer with a reset operation for repeated QAT calibration."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.numel() == 0:
            return input
        observed_input = input.detach()
        if self.scale_dtype is not None:
            observed_input = observed_input.to(dtype=self.scale_dtype)
        if isinstance(self.granularity, PerTensor):
            min_val, max_val = torch.aminmax(observed_input)
            if not hasattr(self, "min_val") or not hasattr(self, "max_val"):
                self.min_val = min_val
                self.max_val = max_val
            else:
                self.min_val.copy_(torch.minimum(self.min_val, min_val))
                self.max_val.copy_(torch.maximum(self.max_val, max_val))
            return input
        block_size = get_block_size(observed_input.shape, self.granularity)
        shape_for_reduction, reduction_dims = _get_reduction_params(
            block_size, observed_input.size()
        )
        reduced_input = observed_input.reshape(shape_for_reduction)
        min_val = torch.amin(reduced_input, dim=reduction_dims, keepdim=self.keepdim)
        max_val = torch.amax(reduced_input, dim=reduction_dims, keepdim=self.keepdim)
        if not hasattr(self, "min_val") or not hasattr(self, "max_val"):
            self.min_val = min_val
            self.max_val = max_val
        else:
            if (
                self.min_val.shape != min_val.shape
                or self.max_val.shape != max_val.shape
            ):
                raise ValueError(
                    "Calibration range shape changed between calibration inputs"
                )
            self.min_val.copy_(torch.minimum(self.min_val, min_val))
            self.max_val.copy_(torch.maximum(self.max_val, max_val))
        return input

    def reset_min_max(self) -> None:
        """Remove temporary ranges before a new calibration cycle."""
        for name in ("min_val", "max_val"):
            if hasattr(self, name):
                delattr(self, name)

    def get_running_min_max(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return copies of the collected calibration range."""
        if not hasattr(self, "min_val") or not hasattr(self, "max_val"):
            raise ValueError("No calibration data was collected")
        return self.min_val.clone(), self.max_val.clone()

    def set_running_min_max(
        self,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
    ) -> None:
        """Replace the collected calibration range."""
        if not hasattr(self, "min_val") or not hasattr(self, "max_val"):
            raise ValueError("No calibration data was collected")
        if min_val.shape != self.min_val.shape or max_val.shape != self.max_val.shape:
            raise ValueError("Calibration range shapes must match collected ranges")
        requires_conversion = (
            min_val.device != self.min_val.device
            or max_val.device != self.max_val.device
            or min_val.dtype != self.min_val.dtype
            or max_val.dtype != self.max_val.dtype
        )
        min_val = min_val.detach()
        max_val = max_val.detach()
        if (
            not torch.isfinite(min_val).all().item()
            or not torch.isfinite(max_val).all().item()
        ):
            raise ValueError("Calibration ranges must be finite")
        min_val = min_val.to(self.min_val)
        max_val = max_val.to(self.max_val)
        if (
            not torch.isfinite(min_val).all().item()
            or not torch.isfinite(max_val).all().item()
        ):
            raise ValueError("Calibration range conversion produced non-finite values")
        if torch.any(min_val > max_val).item():
            raise ValueError("Calibration minimum must not exceed the maximum")
        if requires_conversion:
            warnings.warn(
                "Converting calibration ranges to match the collected ranges",
                stacklevel=2,
            )
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class IntxObservedLinear(FakeQuantizedLinear):
    """Persistent QAT linear with repeatable static activation calibration."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_config: Optional[IntxFakeQuantizeConfig] = None,
        weight_config: Optional[IntxFakeQuantizeConfig] = None,
        *args,
        **kwargs,
    ) -> None:
        if activation_config is None:
            raise ValueError("IntxObservedLinear requires an activation config")
        _validate_activation_config(activation_config)
        super().__init__(
            in_features,
            out_features,
            bias,
            activation_config,
            weight_config,
            *args,
            **kwargs,
        )
        torch._C._log_api_usage_once(
            "torchao.prototype.quantization.IntxObservedLinear"
        )
        self.activation_observer = IntxMinMaxObserver(
            mapping_type=activation_config.mapping_type,
            target_dtype=activation_config.dtype,
            granularity=activation_config.granularity,
            quant_min=activation_config.quant_min,
            quant_max=activation_config.quant_max,
            eps=activation_config.eps,
            scale_dtype=activation_config.scale_precision,
            zero_point_dtype=activation_config.zero_point_precision,
            zero_point_domain=activation_config.zero_point_domain,
        )
        self.calibration_enabled = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.calibration_enabled:
            self.activation_observer(input)
            if self.weight_fake_quantizer is not None:
                weight = self.weight_fake_quantizer(self.weight)
            else:
                weight = self.weight
            return F.linear(input, weight, self.bias)
        return super().forward(input)

    def enable_calibration(self) -> None:
        """Reset activation ranges and enable calibration pass-through."""
        self.activation_observer.reset_min_max()
        self.calibration_enabled = True

    def finalize_calibration(self) -> None:
        """Install observed activation qparams and exit calibration mode."""
        if not self.calibration_enabled:
            raise ValueError(
                "Calibration must be enabled before calling finalize_calibration"
            )
        if not hasattr(self.activation_observer, "min_val") or not hasattr(
            self.activation_observer, "max_val"
        ):
            self.calibration_enabled = False
            raise ValueError("No calibration data was collected")
        try:
            scale, zero_point = self.activation_observer.calculate_qparams()
        except Exception:
            self.calibration_enabled = False
            raise
        if self.activation_fake_quantizer is None:
            raise AssertionError("activation fake quantizer is missing")
        self.activation_fake_quantizer.scale = scale
        self.activation_fake_quantizer.zero_point = zero_point
        self.calibration_enabled = False

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        activation_config: IntxFakeQuantizeConfig,
        weight_config: Optional[IntxFakeQuantizeConfig] = None,
    ) -> "IntxObservedLinear":
        observed_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            activation_config=activation_config,
            weight_config=weight_config,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        observed_linear.weight = linear.weight
        observed_linear.bias = linear.bias
        observed_linear.train(linear.training)
        return observed_linear


@dataclass
class IntxObservedLinearConfig(AOBaseConfig):
    """Configuration that prepares linear modules for observed integer QAT."""

    activation_config: IntxFakeQuantizeConfig
    weight_config: Optional[IntxFakeQuantizeConfig] = None

    def __post_init__(self) -> None:
        _validate_activation_config(self.activation_config)


@register_quantize_module_handler(IntxObservedLinearConfig)
def _intx_observed_linear_transform(
    module: torch.nn.Module,
    config: IntxObservedLinearConfig,
) -> torch.nn.Module:
    if not isinstance(module, torch.nn.Linear):
        raise ValueError(
            "IntxObservedLinearConfig only supports torch.nn.Linear modules"
        )
    return IntxObservedLinear.from_linear(
        module,
        config.activation_config,
        config.weight_config,
    )
