# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from torchao.dtypes import (
    TensorCoreTiledLayoutType,
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
from torchao.quantization.utils import _get_per_token_block_size
from .affine_fake_quantized_tensor import to_affine_fake_quantized
from .utils import (
    _enable_fake_quant,
    _get_qat_linear_subclass_inserter,
    _is_linear_with_fq_weight,
    _unwrap_affine_fake_quantized_tensor,
)


class ComposableQATQuantizer(TwoStepQuantizer):
    """
    Composable quantizer that users can use to apply multiple QAT quantizers easily.
    Quantizers will be applied in the order they are specified in the constructor.

    Note: the quantizers provided must apply to different modules in the model,
    e.g. nn.Linear and nn.Embedding, otherwise the behavior will be undefined.

    Example usage::

        my_quantizer = ComposableQATQuantizer([
            QATQuantizer1(),
            QATQuantizer2(),
            QATQuantizer3(),
        ])
        model = my_quantizer.prepare(model)
        train(model)
        model = my_quantizer.convert(model)
    """

    def __init__(self, quantizers: List[TwoStepQuantizer]):
        self.quantizers = quantizers

    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        for quantizer in self.quantizers:
            model = quantizer.prepare(model)
        return model

    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        for quantizer in self.quantizers:
            model = quantizer.convert(model)
        return model


# =================
# |   8da4w QAT   |
# =================

def int8_dynamic_activation_int4_weight_fake_quantize(group_size=32):
    """
    Applies int8 dynamic per token asymmetric activation fake quantization and
    int4 per group weight symmetric fake quantization to linear. Please see
    :func:`~torchao.quantization.int8_dynamic_activation_int4_weight` for more details.

    Example usage::

        from torchao.quantization import quantize_
        quantize_(model, int8_dynamic_activation_int4_weight_fake_quantize(group_size=32))
    """
    # avoid circular dep
    from torchao.dtypes import to_affine_quantized_intx

    def _apply_weight_fake_quant(weight: torch.Tensor):
        mapping_type = MappingType.SYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        quant_min = -8
        quant_max = 7
        return to_affine_fake_quantized(
            weight,
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
        )

    def _apply_input_activation_fake_quant(x: torch.Tensor):
        mapping_type = MappingType.ASYMMETRIC
        target_dtype = torch.int8
        return to_affine_fake_quantized(
            x,
            mapping_type,
            _get_per_token_block_size(x),
            target_dtype,
        )

    return _get_qat_linear_subclass_inserter(
        _apply_weight_fake_quant,
        _apply_input_activation_fake_quant,
    )

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

    Example usage::

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
    return _get_qat_linear_subclass_inserter(_apply_fake_quant)

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
        layout_type = TensorCoreTiledLayoutType(self.inner_k_tiles)
        quantize_fn = int4_weight_only(self.groupsize, layout_type)
        quantize_(model, quantize_fn)
        return model

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
