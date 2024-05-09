# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Tuple

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
                copy_weights = True,
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
            self._fake_quant_enabled = True

        def enable_fake_quant(self, enabled: bool = True):
            self._fake_quant_enabled = enabled

        def disable_fake_quant(self):
            self.enable_fake_quant(False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # activations: int8 dynamic asymmetric quant
            if self._fake_quant_enabled:
                (
                    act_scales,
                    act_zp
                ) = torch.ops.quantized_decomposed._choose_qparams_per_token_asymmetric_impl(
                    x, torch.int8,  # dtype not used
                )
                (act_qmin, act_qmax) = self._get_qmin_qmax(8)
                x_fq = fake_quantize_per_token(
                    x, act_scales, act_zp, act_qmin, act_qmax,
                )
            else:
                x_fq = x

            # weights: int4 grouped per channel symmetric quant
            if self._fake_quant_enabled:
                (weight_scales, weight_zp) = get_group_qparams_symmetric(
                    self.weight, 4, self.groupsize, self.scales_precision,
                )
                (weight_qmin, weight_qmax) = self._get_qmin_qmax(4)
                w_fq = fake_quantize_per_channel_group(
                    self.weight,
                    weight_scales,
                    weight_zp,
                    weight_qmin,
                    weight_qmax,
                    self.groupsize,
                )
            else:
                w_fq = self.weight
            return torch.nn.functional.linear(x_fq, w_fq)

        # TODO: move this to common util
        def _get_qmin_qmax(self, n_bit: int):
            qmin = -(2 ** (n_bit - 1))
            qmax = 2 ** (n_bit - 1) - 1
            return (qmin, qmax)

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
