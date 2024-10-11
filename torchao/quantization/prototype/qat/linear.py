# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import torch
import torch.nn.functional as F

from torchao.quantization.GPTQ import (
    _check_linear_int4_k,
    _replace_linear_int4,
    _replace_linear_8da4w,
    get_groupwise_affine_qparams,
    groupwise_affine_quantize_tensor,
    Int8DynActInt4WeightLinear,
    WeightOnlyInt4Linear,
)
from torchao.quantization.quant_primitives import ZeroPointDomain
from torchao.quantization.unified import TwoStepQuantizer
from torchao.quantization.utils import get_group_qparams_symmetric
from .utils import (
    _choose_qparams_per_token_asymmetric,
    _fake_quantize_per_channel_group,
    _fake_quantize_per_token,
    _get_qmin_qmax,
)


# =========================================================
# |   Linear int8 dynamic activations + int4 weight QAT   |
# =========================================================


class Int8DynActInt4WeightQATQuantizer(TwoStepQuantizer):
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

    def prepare(
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any
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
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any
    ) -> torch.nn.Module:
        _convert_qat_linear_8da4w(model)
        return model


def _convert_qat_linear_8da4w(module: torch.nn.Module):
    """
    Replace all `Int8DynActInt4WeightQATLinear` with `Int8DynActInt4WeightLinear`.
    """
    for name, child in module.named_children():
        if isinstance(child, Int8DynActInt4WeightQATLinear):
            quantized_linear = Int8DynActInt4WeightLinear(
                child.in_features,
                child.out_features,
                bias=False,
                groupsize=child.groupsize,
                precision=child.precision,
                scales_precision=child.scales_precision,
            )
            setattr(module, name, quantized_linear)

            # Load weights and qparams into quantized linear
            n_bit = 4
            (qmin, qmax) = _get_qmin_qmax(n_bit)
            (s, zp) = get_group_qparams_symmetric(child.weight, n_bit, child.groupsize)
            from torchao._executorch_ops import _quantized_decomposed_quantize_per_channel_group_wrapper
            q_weight = _quantized_decomposed_quantize_per_channel_group_wrapper(
                child.weight, s, zp, qmin, qmax, torch.int8, child.groupsize,
            )
            quantized_linear.weight = q_weight
            quantized_linear.scales = s
            quantized_linear.zeros = zp
        else:
            _convert_qat_linear_8da4w(child)


class Int8DynActInt4WeightQATLinear(torch.nn.Linear):
    """
    This module implements a linear layer with int8 dynamic per token fake
    quantized activations with int4 fake quantized grouped per channel weights.

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
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            device=device,
            dtype=precision,
        )
        assert (
            in_features % groupsize == 0
        ), f"require in_features:{in_features} % groupsize:{groupsize} == 0"
        assert not bias, "require bias=False"
        self.groupsize = groupsize
        self.precision = precision
        self.scales_precision = scales_precision
        # TODO: make this configurable?
        self.zero_points_precision = torch.int32
        self._fake_quant_enabled = True

    def enable_fake_quant(self, enabled: bool = True):
        self._fake_quant_enabled = enabled

    def disable_fake_quant(self):
        self.enable_fake_quant(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # activations: int8 dynamic asymmetric quant
        if self._fake_quant_enabled:
            (act_scales, act_zp) = _choose_qparams_per_token_asymmetric(
                x, self.scales_precision, self.zero_points_precision,
            )
            (act_qmin, act_qmax) = _get_qmin_qmax(8)
            x_fq = _fake_quantize_per_token(
                x, act_scales, act_zp, act_qmin, act_qmax,
            )
        else:
            x_fq = x

        # weights: int4 grouped per channel symmetric quant
        if self._fake_quant_enabled:
            (weight_scales, weight_zp) = get_group_qparams_symmetric(
                self.weight, 4, self.groupsize, self.scales_precision,
            )
            # TODO: pass zp dtype to `get_group_qparams_symmetric` instead
            weight_zp = weight_zp.to(self.zero_points_precision)
            (weight_qmin, weight_qmax) = _get_qmin_qmax(4)
            w_fq = _fake_quantize_per_channel_group(
                self.weight,
                weight_scales,
                weight_zp,
                weight_qmin,
                weight_qmax,
                self.groupsize,
            )
        else:
            w_fq = self.weight
        return F.linear(x_fq, w_fq)


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


# ===================================
# |   Linear int4 weight-only QAT   |
# ===================================


class Int4WeightOnlyQATQuantizer(TwoStepQuantizer):
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
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any
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
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any
    ) -> torch.nn.Module:
        _convert_qat_linear_4w(model)
        return model


def _convert_qat_linear_4w(module: torch.nn.Module):
    """
    Replace all `Int4WeightOnlyQATLinear` with `WeightOnlyInt4Linear`.
    """
    for name, child in module.named_children():
        if isinstance(child, Int4WeightOnlyQATLinear):
            in_features = child.in_features
            out_features = child.out_features
            groupsize = child.groupsize
            inner_k_tiles = child.inner_k_tiles
            quantized_linear = WeightOnlyInt4Linear(
                in_features,
                out_features,
                bias=False,
                groupsize=groupsize,
                inner_k_tiles=inner_k_tiles,
                precision=child.precision,
                scales_precision=child.scales_precision,
            )
            setattr(module, name, quantized_linear)

            # Load weights and qparams into quantized linear
            n_bit = 4
            (q_weight, scales_and_zeros) = groupwise_affine_quantize_tensor(
                child.weight, n_bit, child.groupsize,
            )
            q_weight = torch.ops.aten._convert_weight_to_int4pack(
                q_weight.to(child.weight.device), child.inner_k_tiles,
            )
            quantized_linear.weight = q_weight
            quantized_linear.scales_and_zeros = scales_and_zeros
        else:
            _convert_qat_linear_4w(child)


class Int4WeightOnlyQATLinear(torch.nn.Linear):
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
        super().__init__(
            in_features,
            out_features,
            bias,
            device=device,
            dtype=precision,
        )
        assert not bias, "require bias=False"
        assert scales_precision == torch.bfloat16, "only bf16 is supported for scales"
        if not _check_linear_int4_k(in_features, groupsize, inner_k_tiles):
            raise ValueError("Padding for QAT 4w is not supported yet")
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.precision = precision
        self.scales_precision = scales_precision
        self._fake_quant_enabled = True

    def enable_fake_quant(self, enabled: bool = True):
        self._fake_quant_enabled = enabled

    def disable_fake_quant(self):
        self.enable_fake_quant(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_bit = 4
        qmin = 0
        qmax = 2 ** n_bit - 1
        scales, zero_points = get_groupwise_affine_qparams(
            self.weight, n_bit, self.groupsize, self.scales_precision,
        )
        w_fq = _fake_quantize_per_channel_group(
            self.weight,
            scales,
            zero_points,
            qmin,
            qmax,
            self.groupsize,
            ZeroPointDomain.FLOAT,
        )
        return F.linear(x, w_fq)


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
