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
from torchao.quantization.linear_activation_quantized_tensor import (
    to_linear_activation_quantized,
)
from torchao.quantization.quant_api import (
    _get_linear_subclass_inserter,
    _replace_with_custom_fn_if_matches_filter,
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    quantize_,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization.unified import TwoStepQuantizer
from torchao.quantization.utils import (
    _get_per_token_block_size,
    get_group_qparams_symmetric,
)
from .affine_fake_quantized_tensor import to_affine_fake_quantized
from .utils import (
    _choose_qparams_per_token_asymmetric,
    _enable_fake_quant,
    _fake_quantize_per_channel_group,
    _fake_quantize_per_token,
    _is_linear_with_fq_weight,
    _unwrap_affine_fake_quantized_tensor,
)


# =================
# |   8da4w QAT   |
# =================

def int8_dynamic_activation_int4_weight_fake_quantize(group_size=32):
    """
    Applies int8 dynamic per token asymmetric activation fake quantization and
    int4 per group weight symmetric fake quantization to linear. Please see
    :func:`~torchao.quantization.int8_dynamic_activation_int4_weight` for more details.

    Example usage:
        from torchao.quantization import quantize_
        quantize_(model, int8_dynamic_activation_int4_weight_fake_quantize(group_size=32))
    """
    def _apply_fake_quant(weight):
        # avoid circular dep
        from torchao.dtypes import to_affine_quantized

        # weight settings
        mapping_type = MappingType.SYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        quant_min = -8
        quant_max = 7

        # input settings
        input_mapping_type = MappingType.ASYMMETRIC
        input_target_dtype = torch.int8

        def input_quant_func(x: torch.Tensor):
            return to_affine_fake_quantized(
                x,
                input_mapping_type,
                _get_per_token_block_size(x),
                input_target_dtype,
            )

        weight = to_affine_fake_quantized(
            weight,
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
        )
        weight = to_linear_activation_quantized(weight, input_quant_func)
        return weight

    return _get_linear_subclass_inserter(_apply_fake_quant, requires_grad=True)

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
        quantize_(
            model,
            int8_dynamic_activation_int4_weight_fake_quantize(group_size=self.groupsize),
        )
        return model

    def convert(
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any
    ) -> torch.nn.Module:
        unwrap_fn = _get_linear_subclass_inserter(_unwrap_affine_fake_quantized_tensor)
        filter_fn = _is_linear_with_fq_weight
        model = _replace_with_custom_fn_if_matches_filter(model, unwrap_fn, filter_fn)
        quantize_fn = int8_dynamic_activation_int4_weight(self.groupsize)
        quantize_(model, quantize_fn)
        return model

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
            (act_qmin, act_qmax) = self._get_qmin_qmax(8)
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
            (weight_qmin, weight_qmax) = self._get_qmin_qmax(4)
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

    # TODO: move this to common util
    def _get_qmin_qmax(self, n_bit: int):
        qmin = -(2 ** (n_bit - 1))
        qmax = 2 ** (n_bit - 1) - 1
        return (qmin, qmax)

def enable_8da4w_fake_quant(mod: torch.nn.Module):
    """
    Enable fake quantization for int8 dynamic activations + int4 weight.
    """
    _enable_fake_quant(mod, enable=True)

def disable_8da4w_fake_quant(mod: torch.nn.Module):
    """
    Disable fake quantization for int8 dynamic activations + int4 weight.
    """
    _enable_fake_quant(mod, enable=False)


# ==================
# |   int4wo QAT   |
# ==================

def int4_weight_only_fake_quantize(group_size=128):
    """
    Applies uint4 weight-only asymmetric per-group fake quantization to linear layers.
    Please see :func:`~torchao.quantization.int4_weight_only` for more details.

    Example usage:
        from torchao.quantization import quantize_
        quantize_(model, int4_weight_only_fake_quantize(group_size=32))
    """
    def _apply_fake_quant(weight):
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        eps = 1e-6
        preserve_zero = False
        zero_point_dtype = torch.bfloat16
        zero_point_domain = ZeroPointDomain.FLOAT
        return to_affine_fake_quantized(
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
        )
    return _get_linear_subclass_inserter(_apply_fake_quant, requires_grad=True)

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
        quantize_(model, int4_weight_only_fake_quantize(group_size=self.groupsize))
        return model

    def convert(
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any
    ) -> torch.nn.Module:
        unwrap_fn = _get_linear_subclass_inserter(_unwrap_affine_fake_quantized_tensor)
        filter_fn = _is_linear_with_fq_weight
        model = _replace_with_custom_fn_if_matches_filter(model, unwrap_fn, filter_fn)
        quantize_fn = int4_weight_only(self.groupsize, self.inner_k_tiles)
        quantize_(model, quantize_fn)
        return model


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
    Enable fake quantization for int4 weight only.
    """
    _enable_fake_quant(mod, enable=True)

def disable_4w_fake_quant(mod: torch.nn.Module):
    """
    Disable fake quantization for int4 weight only.
    """
    _enable_fake_quant(mod, enable=False)
