# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

import torch
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib
from torch.library import impl

from torchao.quantization.utils import TORCH_VERSION_AFTER_2_3
from torchao.quantization.quant_primitives import get_group_qparams_symmetric
from torchao.quantization.unified import TwoStepQuantizer


if TORCH_VERSION_AFTER_2_3:
    from torchao.quantization.GPTQ import (
        _replace_linear_8da4w,
        Int8DynActInt4WeightLinear,
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
            _replace_linear_8da4w(
                model,
                self.groupsize,
                self.padding_allowed,
                self.precision,
                self.scales_precision,
                Int8DynActInt4WeightQATLinear,
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
                (qmin, qmax) = child._get_qmin_qmax(n_bit)
                (s, zp) = get_group_qparams_symmetric(child.weight, n_bit, child.groupsize)
                q_weight = torch.ops.quantized_decomposed.quantize_per_channel_group(
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
            groupsize: int = 256,
            precision: torch.dtype = torch.float32,
            scales_precision: torch.dtype = torch.float32,
        ) -> None:
            super().__init__(
                in_features,
                out_features,
                bias,
                device=None,
                dtype=precision,
            )
            assert (
                in_features % groupsize == 0
            ), f"require in_features:{in_features} % groupsize:{groupsize} == 0"
            assert not bias, "require bias=False"
            self.groupsize = groupsize
            self.precision = precision
            self.scales_precision = scales_precision

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # activations: int8 dynamic asymmetric quant
            (act_qmin, act_qmax) = self._get_qmin_qmax(8)
            (act_scales, act_zp) = _choose_qparams_per_token_asymmetric(
                x, torch.int8,  # dtype not used
            )
            x_fq = fake_quantize_per_token(
                x, act_scales, act_zp, act_qmin, act_qmax,
            )

            # weights: int4 grouped per channel symmetric quant
            (weight_qmin, weight_qmax) = self._get_qmin_qmax(4)
            (weight_scales, weight_zp) = get_group_qparams_symmetric(
                self.weight, 4, self.groupsize, self.scales_precision,
            )
            w_fq = fake_quantize_per_channel_group(
                self.weight,
                weight_scales,
                weight_zp,
                weight_qmin,
                weight_qmax,
                self.groupsize,
            )
            return torch.nn.functional.linear(x_fq, w_fq)

        # TODO: move this to common util
        def _get_qmin_qmax(self, n_bit: int):
            qmin = -(2 ** (n_bit - 1))
            qmax = 2 ** (n_bit - 1) - 1
            return (qmin, qmax)


# ========================
# |   QUANT PRIMITIVES   |
# ========================

class _GenericFakeQuantize(torch.autograd.Function):
    """
    Implementation of generic fake quantize with backward STE.

    With the appropriate input tensor shape, this can be used to express
    grouped per channel fake quantize or per token fake quantize.
    """

    @staticmethod
    def forward(ctx, input, scales, zero_points, quant_min, quant_max):
        # Note: this diverges from `torch.fake_quantize_per_channel_affine`,
        # which rounds first before adding the zero points. However, this
        # is what `quantize_per_channel_group` and `quantize_per_token`
        # do and here we try to match that behavior as closely as possible.
        q = input.div(scales).add(zero_points).round()
        dq = q.clamp(quant_min, quant_max).sub(zero_points).mul(scales)
        # TODO: do we need this mask?
        mask = torch.logical_and((q >= quant_min), (q <= quant_max))
        ctx.save_for_backward(mask)
        return dq

    @staticmethod
    def backward(ctx, gy):
        (mask,) = ctx.saved_tensors
        return gy * mask, None, None, None, None, None

# TODO: move this to core
quantized_decomposed_lib.define(
    "fake_quantize_per_channel_group(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, int group_size) -> Tensor"
)

@impl(quantized_decomposed_lib, "fake_quantize_per_channel_group", "CompositeImplicitAutograd")
def fake_quantize_per_channel_group(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    group_size: int,
) -> torch.Tensor:
    assert group_size > 1
    assert input.shape[-1] % group_size == 0
    assert input.dim() == 2
    assert torch.isnan(input).sum() == 0
    grouped_input = input.reshape(-1, group_size)
    scales = scales.reshape(-1, 1)
    zero_points = zero_points.reshape(-1, 1)
    fq = _GenericFakeQuantize.apply(
        grouped_input, scales, zero_points, quant_min, quant_max,
    )
    return fq.reshape_as(input)

# TODO: move this to core
quantized_decomposed_lib.define(
    "fake_quantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max) -> Tensor"
)

@impl(quantized_decomposed_lib, "fake_quantize_per_token", "CompositeImplicitAutograd")
def fake_quantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
) -> torch.Tensor:
    # TODO: we won't need this import anymore once we move this to core
    from torch.ao.quantization.fx._decomposed import _per_token_quant_qparam_dim_check

    _per_token_quant_qparam_dim_check(input, scales, zero_points)
    return _GenericFakeQuantize.apply(
        input, scales, zero_points, quant_min, quant_max,
    )

# TODO: This is copied from torch/ao/quantization/fx/_decomposed.py.
# The version in pytorch does not have backward support yet so we add
# it here for now until https://github.com/pytorch/pytorch/pull/123452
# is landed.
def _choose_qparams_per_token_asymmetric(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """
    # Based on https://github.com/google/XNNPACK/blob/df156f0cf3db5a4576cc711123eeb54915f82ffc/src/xnnpack/quantization.h#L18
    qmin, qmax = -128, 127
    min_val = torch.amin(input, dim=-1, keepdim=True)
    max_val = torch.amax(input, dim=-1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    eps = torch.finfo(torch.float32).eps  # use xnnpack eps?

    # scale
    scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
    scale = scale.clamp(min=eps)

    # zero point
    descaled_min = min_val_neg / scale
    descaled_max = max_val_pos / scale
    zero_point_from_min_error = qmin + descaled_min
    zero_point_from_max_error = qmax + descaled_max
    zero_point = torch.where(
        zero_point_from_min_error + zero_point_from_max_error > 0,
        qmin - descaled_min,
        qmax - descaled_max,
    )
    zero_point = torch.clamp(zero_point, qmin, qmax).round()

    return scale.to(torch.float32), zero_point.to(torch.float32)
