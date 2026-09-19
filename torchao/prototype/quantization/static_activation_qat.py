# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from torchao.core.config import AOBaseConfig
from torchao.quantization.granularity import PerTensor
from torchao.quantization.observer import AffineQuantizedMinMaxObserver
from torchao.quantization.qat.fake_quantize_config import (
    IntxFakeQuantizeConfig,
)
from torchao.quantization.qat.linear import FakeQuantizedLinear
from torchao.quantization.quant_primitives import ZeroPointDomain
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)


def _validate_activation_config(config: IntxFakeQuantizeConfig) -> None:
    if config.zero_point_domain != ZeroPointDomain.INT:
        raise ValueError("Calibration requires an integer zero-point domain")
    if (
        config.is_dynamic
        or config.range_learning
        or not isinstance(config.granularity, PerTensor)
    ):
        raise ValueError(
            "Calibration is only supported for static per-tensor activation "
            "quantization"
        )


class IntxStaticActQATLinear(FakeQuantizedLinear):
    """Static-activation QAT linear with repeatable calibration.

    This module calibrates activations only. During calibration, it collects
    activation ranges and bypasses activation fake quantization. Calibration
    does not bypass weight fake quantization and honors its current enabled
    state. Finalization installs fixed activation quantization parameters for
    the subsequent QAT forwards.

    Example::

        from torchao.quantization.quant_primitives import MappingType

        activation_config = IntxFakeQuantizeConfig(
            torch.int16,
            PerTensor(),
            MappingType.SYMMETRIC_NO_CLIPPING_ERR,
            is_dynamic=False,
        )
        linear = IntxStaticActQATLinear(
            128,
            64,
            activation_config=activation_config,
        )
        linear.enable_calibration()
        with torch.no_grad():
            linear(calibration_batch_1)
            linear(calibration_batch_2)
        linear.finalize_calibration()
        output = linear(training_batch)

    ``enable_calibration`` starts a new collection cycle and clears temporary
    ranges. ``finalize_calibration`` ends that cycle and returns to the normal
    QAT path. Neither method changes a fake quantizer's enabled state.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *args,
        activation_config: IntxFakeQuantizeConfig,
        weight_config: Optional[IntxFakeQuantizeConfig] = None,
        **kwargs,
    ) -> None:
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
            "torchao.prototype.quantization.IntxStaticActQATLinear"
        )
        self.activation_observer = AffineQuantizedMinMaxObserver(
            mapping_type=activation_config.mapping_type,
            target_dtype=activation_config.dtype,
            granularity=activation_config.granularity,
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
            self.calibration_enabled = False
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
    ) -> "IntxStaticActQATLinear":
        qat_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            activation_config=activation_config,
            weight_config=weight_config,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        # In distributed training, the model may be instantiated on the meta
        # device. Copying meta parameters into the new module raises an error.
        if linear.weight.device != torch.device("meta"):
            qat_linear.weight = linear.weight
            qat_linear.bias = linear.bias
        qat_linear.train(linear.training)
        return qat_linear


@dataclass
class IntxStaticActQATConfig(AOBaseConfig):
    """Prepare linear modules for static activation calibration and integer QAT.

    The activation config must use static per-tensor quantization. Calibration
    observes activations only. Weight fake quantization remains active during
    calibration when ``weight_config`` is set.
    """

    activation_config: IntxFakeQuantizeConfig
    weight_config: Optional[IntxFakeQuantizeConfig] = None

    def __post_init__(self) -> None:
        _validate_activation_config(self.activation_config)


@register_quantize_module_handler(IntxStaticActQATConfig)
def _intx_static_activation_qat_transform(
    module: torch.nn.Module,
    config: IntxStaticActQATConfig,
) -> torch.nn.Module:
    if not isinstance(module, torch.nn.Linear):
        raise ValueError("IntxStaticActQATConfig only supports torch.nn.Linear modules")
    return IntxStaticActQATLinear.from_linear(
        module,
        config.activation_config,
        config.weight_config,
    )
