# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import torch
import torch.nn.functional as F

from torchao.dtypes.utils import is_device
from torchao.quantization.GPTQ import (
    Int8DynActInt4WeightLinear,
    WeightOnlyInt4Linear,
    _check_linear_int4_k,
    _replace_linear_8da4w,
    _replace_linear_int4,
    groupwise_affine_quantize_tensor,
)
from torchao.quantization.quant_primitives import (
    TorchAODType,
    ZeroPointDomain,
)
from torchao.quantization.unified import TwoStepQuantizer
from torchao.quantization.utils import get_group_qparams_symmetric
from torchao.utils import TORCH_VERSION_AT_LEAST_2_6

from .api import FakeQuantizeConfig
from .fake_quantizer import FakeQuantizer
from .utils import (
    _get_qmin_qmax,
)


class FakeQuantizedLinear(torch.nn.Linear):
    """
    General linear layer with fake quantized weights and activations.

    Specific target dtypes, granularity, schemes etc. are specified
    through separate configs for weights and activations.

    Example usage::

        activation_config = FakeQuantizeConfig(
            dtype=torch.int8,
            granularity="per_token",
            is_symmetric=False,
        )
        weight_config = FakeQuantizeConfig(
            dtype=torch.int4,
            group_size=8,
            is_symmetric=True,
        )
        fq_linear = FakeQuantizedLinear(
            16, 32, False, activation_config, weight_config,
        )
        fq_linear(torch.randn(16))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_config: Optional[FakeQuantizeConfig] = None,
        weight_config: Optional[FakeQuantizeConfig] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            *args,
            **kwargs,
        )
        # initialize activation fake quantizer
        if activation_config is not None:
            self.activation_fake_quantizer = FakeQuantizer(activation_config)
        else:
            self.activation_fake_quantizer = None

        # initialize weight fake quantizer
        if weight_config is not None:
            group_size = weight_config.group_size
            if group_size is not None and in_features % group_size != 0:
                raise ValueError(
                    "in_features (%s) %% group_size (%s) must be == 0"
                    % (in_features, group_size)
                )
            self.weight_fake_quantizer = FakeQuantizer(weight_config)
        else:
            self.weight_fake_quantizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_fake_quantizer is not None:
            x = self.activation_fake_quantizer(x)
        if self.weight_fake_quantizer is not None:
            w = self.weight_fake_quantizer(self.weight)
        else:
            w = self.weight
        return F.linear(x, w, self.bias)

    def to_linear(self) -> torch.nn.Linear:
        new_linear = torch.nn.Linear(
            self.in_features,
            self.out_features,
            self.bias is not None,
            device=self.weight.device,
        )
        # In distributed training, the model may be instantiated
        # on the meta device, in which case there is no need to
        # copy the weights, and doing so will result in an error
        if self.weight.device != torch.device("meta"):
            new_linear.weight = self.weight
            new_linear.bias = self.bias
        return new_linear

    @classmethod
    def from_linear(
        cls,
        mod: torch.nn.Linear,
        activation_config: Optional[FakeQuantizeConfig] = None,
        weight_config: Optional[FakeQuantizeConfig] = None,
    ):
        new_linear = FakeQuantizedLinear(
            mod.in_features,
            mod.out_features,
            mod.bias is not None,
            activation_config=activation_config,
            weight_config=weight_config,
            device=mod.weight.device,
        )
        # In distributed training, the model may be instantiated
        # on the meta device, in which case there is no need to
        # copy the weights, and doing so will result in an error
        if mod.weight.device != torch.device("meta"):
            new_linear.weight = mod.weight
            new_linear.bias = mod.bias
        return new_linear


class _LegacyQATQuantizer(TwoStepQuantizer):
    """
    Base class for sharing common methods across legacy QAT quantizers.
    """

    def get_activation_fake_quantize_config(self) -> Optional[FakeQuantizeConfig]:
        return None

    def get_weight_fake_quantize_config(self) -> Optional[FakeQuantizeConfig]:
        return None


# =========================================================
# |   Linear int8 dynamic activations + int4 weight QAT   |
# =========================================================


class Int8DynActInt4WeightQATQuantizer(_LegacyQATQuantizer):
    """
    Quantizer for performing QAT on a model, where linear layers have int8
    dynamic per token fake quantized activations and int4 fake quantized
    grouped per channel weights.
    """

    def __init__(
        self,
        groupsize: int = 256,
        padding_allowed: bool = False,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.groupsize: int = groupsize
        self.padding_allowed: bool = padding_allowed
        self.precision: torch.dtype = precision
        self.scales_precision: torch.dtype = scales_precision
        # TODO: generalize this
        self.activation_scales_precision = torch.float32

    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        _replace_linear_8da4w(
            model,
            self.groupsize,
            self.padding_allowed,
            self.precision,
            self.scales_precision,
            Int8DynActInt4WeightQATLinear,
            copy_weights=True,
        )
        return model

    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        self._convert_qat_linear_8da4w(model)
        return model

    def _convert_qat_linear_8da4w(self, module: torch.nn.Module):
        """
        Replace all `Int8DynActInt4WeightQATLinear` with `Int8DynActInt4WeightLinear`.
        """
        for name, child in module.named_children():
            if isinstance(child, Int8DynActInt4WeightQATLinear):
                config = child.weight_fake_quantizer.config
                quantized_linear = Int8DynActInt4WeightLinear(
                    child.in_features,
                    child.out_features,
                    child.bias is not None,
                    groupsize=config.group_size,
                    precision=child.weight.dtype,
                    scales_precision=config.scale_precision,
                )
                setattr(module, name, quantized_linear)

                # Load weights and qparams into quantized linear
                n_bit = 4
                (qmin, qmax) = _get_qmin_qmax(n_bit)
                (s, zp) = get_group_qparams_symmetric(
                    child.weight,
                    n_bit,
                    config.group_size,
                    precision=config.scale_precision,
                )
                zp = zp.to(config.zero_point_precision)
                from torchao._executorch_ops import (
                    _quantized_decomposed_quantize_per_channel_group_wrapper,
                )

                q_weight = _quantized_decomposed_quantize_per_channel_group_wrapper(
                    child.weight,
                    s,
                    zp,
                    qmin,
                    qmax,
                    torch.int8,
                    config.group_size,
                )
                quantized_linear.weight = q_weight
                quantized_linear.scales = s
                quantized_linear.zeros = zp
                if child.bias is not None:
                    quantized_linear.bias = child.bias
            else:
                self._convert_qat_linear_8da4w(child)

    def get_activation_fake_quantize_config(self) -> Optional[FakeQuantizeConfig]:
        return _get_8da4w_activation_config(self.activation_scales_precision)

    def get_weight_fake_quantize_config(self) -> Optional[FakeQuantizeConfig]:
        return _get_8da4w_weight_config(self.groupsize, self.scales_precision)


class Int8DynActInt4WeightQATLinear(FakeQuantizedLinear):
    """
    This module implements a linear layer with int8 dynamic per token fake
    quantized activations with int4 fake quantized grouped per channel weights.

    args:
        groupsize: the number of elements in each quantized group for weights
        precision: precision of weights
        scales_precision: precision of per group scales and zero points

    Note: we hardcode activation scales to use torch.fp32, but allow users to specify the weight scales (defaults to torch.fp32).
    To get an exact numerical match with Int8DynamicActivationInt4WeightConfig, users must use the same dtype for both the weights
    and the scales. Here scales_precision refers specifically to the weight scales only, not the activation scales.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: torch.device = None,
        groupsize: int = 256,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
    ) -> None:
        # Use torch.float32 to match torchao.quantization.quant_api._int8_asymm_per_token_quant,
        # which is used in PTQ routines
        # TODO: generalize this
        activation_config = _get_8da4w_activation_config(torch.float32)
        weight_config = _get_8da4w_weight_config(groupsize, scales_precision)
        super().__init__(
            in_features,
            out_features,
            bias,
            activation_config,
            weight_config,
            device=device,
            dtype=precision,
        )

    def enable_fake_quant(self, enabled: bool = True):
        self.activation_fake_quantizer.enabled = enabled
        self.weight_fake_quantizer.enabled = enabled

    def disable_fake_quant(self):
        self.enable_fake_quant(False)


def enable_8da4w_fake_quant(mod: torch.nn.Module):
    """
    Enable fake quantization for `Int8DynActInt4WeightQATLinear`.
    """
    if isinstance(mod, Int8DynActInt4WeightQATLinear):
        mod.enable_fake_quant()


def disable_8da4w_fake_quant(mod: torch.nn.Module):
    """
    Disable fake quantization for `Int8DynActInt4WeightQATLinear`.
    """
    if isinstance(mod, Int8DynActInt4WeightQATLinear):
        mod.disable_fake_quant()


def _get_8da4w_activation_config(qparams_precision: torch.dtype) -> FakeQuantizeConfig:
    """
    Return the activation `FakeQuantizeConfig` for `Int8DynActInt4WeightQATQuantizer`.
    """
    # TODO: generalize this
    assert qparams_precision == torch.float32
    return FakeQuantizeConfig(
        dtype=torch.int8,
        granularity="per_token",
        is_symmetric=False,
        is_dynamic=True,
        scale_precision=qparams_precision,
        zero_point_precision=qparams_precision,
        eps=torch.finfo(qparams_precision).eps,
    )


def _get_8da4w_weight_config(
    group_size: int,
    qparams_precision: torch.dtype,
) -> FakeQuantizeConfig:
    """
    Return the weight `FakeQuantizeConfig` for `Int8DynActInt4WeightQATQuantizer`.
    """
    return FakeQuantizeConfig(
        dtype=TorchAODType.INT4,
        group_size=group_size,
        is_symmetric=True,
        is_dynamic=True,
        scale_precision=qparams_precision,
        zero_point_precision=qparams_precision,
    )


# ===================================
# |   Linear int4 weight-only QAT   |
# ===================================


class Int4WeightOnlyQATQuantizer(_LegacyQATQuantizer):
    """
    Quantizer for performing QAT on a model, where linear layers have
    int4 fake quantized grouped per channel weights.
    """

    def __init__(
        self,
        groupsize: int = 256,
        inner_k_tiles: Optional[int] = 8,
        precision: torch.dtype = torch.bfloat16,
        scales_precision: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        assert inner_k_tiles in [2, 4, 8]
        assert groupsize in [32, 64, 128, 256]
        self.inner_k_tiles = inner_k_tiles
        self.groupsize = groupsize
        self.precision = precision
        self.scales_precision = scales_precision

    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        _replace_linear_int4(
            model,
            self.groupsize,
            self.inner_k_tiles,
            padding_allowed=True,
            precision=self.precision,
            scales_precision=self.scales_precision,
            linear_class=Int4WeightOnlyQATLinear,
            copy_weights=True,
        )
        return model

    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        self._convert_qat_linear_4w(model)
        return model

    def _convert_qat_linear_4w(self, module: torch.nn.Module):
        """
        Replace all `Int4WeightOnlyQATLinear` with `WeightOnlyInt4Linear`.
        """
        for name, child in module.named_children():
            if isinstance(child, Int4WeightOnlyQATLinear):
                in_features = child.in_features
                out_features = child.out_features
                inner_k_tiles = child.inner_k_tiles
                config = child.weight_fake_quantizer.config
                quantized_linear = WeightOnlyInt4Linear(
                    in_features,
                    out_features,
                    bias=False,
                    groupsize=config.group_size,
                    inner_k_tiles=inner_k_tiles,
                    precision=child.weight.dtype,
                    scales_precision=config.scale_precision,
                    device=next(child.parameters()).device,
                )
                setattr(module, name, quantized_linear)

                # Load weights and qparams into quantized linear
                n_bit = 4
                (q_weight, scales_and_zeros) = groupwise_affine_quantize_tensor(
                    child.weight,
                    n_bit,
                    config.group_size,
                )
                if (
                    is_device(q_weight.device.type, "cpu")
                    and TORCH_VERSION_AT_LEAST_2_6
                ):
                    q_weight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                        q_weight.to(child.weight.device),
                        child.inner_k_tiles,
                    )
                else:
                    q_weight = torch.ops.aten._convert_weight_to_int4pack(
                        q_weight.to(child.weight.device),
                        child.inner_k_tiles,
                    )
                quantized_linear.weight = q_weight
                quantized_linear.scales_and_zeros = scales_and_zeros
            else:
                self._convert_qat_linear_4w(child)

    def get_weight_fake_quantize_config(self) -> Optional[FakeQuantizeConfig]:
        return _get_4w_weight_config(self.groupsize, self.scales_precision)


class Int4WeightOnlyQATLinear(FakeQuantizedLinear):
    """
    This module implements a linear layer with int4 fake quantized grouped
    per channel weights, with forward numerics matching `WeightOnlyInt4Linear`,
    which uses the efficient int4 tinygemm kernel.

    args:
        groupsize: the number of elements in each quantized group for weights
        precision: precision of weights
        scales_precision: precision of per group scales and zero points
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: torch.device = None,
        groupsize: int = 256,
        inner_k_tiles: int = 8,
        precision: torch.dtype = torch.bfloat16,
        scales_precision: torch.dtype = torch.bfloat16,
    ) -> None:
        assert scales_precision == torch.bfloat16, "only bf16 is supported for scales"
        if not _check_linear_int4_k(in_features, groupsize, inner_k_tiles):
            raise ValueError("Padding for QAT 4w is not supported yet")
        self.inner_k_tiles = inner_k_tiles
        weight_config = _get_4w_weight_config(groupsize, scales_precision)
        super().__init__(
            in_features,
            out_features,
            bias,
            activation_config=None,
            weight_config=weight_config,
            device=device,
            dtype=precision,
        )

    def enable_fake_quant(self, enabled: bool = True):
        self.activation_fake_quantizer.enabled = enabled
        self.weight_fake_quantizer.enabled = enabled

    def disable_fake_quant(self):
        self.enable_fake_quant(False)


def enable_4w_fake_quant(mod: torch.nn.Module):
    """
    Enable fake quantization for `Int4WeightOnlyQATLinear`.
    """
    if isinstance(mod, Int4WeightOnlyQATLinear):
        mod.enable_fake_quant()


def disable_4w_fake_quant(mod: torch.nn.Module):
    """
    Disable fake quantization for `Int4WeightOnlyQATLinear`.
    """
    if isinstance(mod, Int4WeightOnlyQATLinear):
        mod.disable_fake_quant()


def _get_4w_weight_config(
    group_size: int,
    qparams_precision: torch.dtype,
) -> FakeQuantizeConfig:
    """
    Return the weight `FakeQuantizeConfig` for `Int4WeightOnlyQATQuantizer`.
    """
    return FakeQuantizeConfig(
        dtype=torch.uint4,
        group_size=group_size,
        is_symmetric=False,
        is_dynamic=True,
        scale_precision=qparams_precision,
        zero_point_precision=qparams_precision,
        zero_point_domain=ZeroPointDomain.FLOAT,
    )
