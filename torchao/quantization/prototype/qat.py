# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional, Tuple

import torch
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib
from torch.library import impl

from torchao.quantization.utils import TORCH_VERSION_AFTER_2_4
from torchao.quantization.quant_primitives import get_group_qparams_symmetric
from torchao.quantization.unified import TwoStepQuantizer


if TORCH_VERSION_AFTER_2_4:
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
            """
            Prepare a model for QAT by swapping `torch.nn.Linear` with
            `Int8DynActInt4WeightQATLinear`, which performs 8da4w fake quantize.

            Users may optionally specify `skip_quantize_filter`, a function
            that takes in a fully-qualified name corresponding to a submodule,
            and returns True if we should skip quantizing that submodule.
            """
            skip_quantize_filter = kwargs.get("skip_quantize_filter")
            _replace_linear_8da4w(
                model,
                self.groupsize,
                self.padding_allowed,
                self.precision,
                self.scales_precision,
                Int8DynActInt4WeightQATLinear,
                copy_weights=True,
                skip_quantize_filter = skip_quantize_filter,
            )
            return model

        def convert(
            self,
            model: torch.nn.Module,
            *args: Any,
            **kwargs: Any
        ) -> torch.nn.Module:
            """
            Convert a float QAT model to a 8da4w quantized model.

            Users may optionally specify `skip_quantize_filter`, a function
            that takes in a fully-qualified name corresponding to a submodule,
            and returns True if we should skip quantizing that submodule.
            """
            skip_quantize_filter = kwargs.get("skip_quantize_filter")
            _convert_qat_linear_8da4w(model, skip_quantize_filter)
            return model

    def _convert_qat_linear_8da4w(
        module: torch.nn.Module,
        skip_quantize_filter: Optional[Callable] = None,
        fqn: str = "",
    ):
        """
        Replace all `Int8DynActInt4WeightQATLinear` with `Int8DynActInt4WeightLinear`.
        """
        for name, child in module.named_children():
            new_fqn = (fqn + "." + name).lstrip(".")
            if skip_quantize_filter is not None and skip_quantize_filter(new_fqn):
                continue
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
                _convert_qat_linear_8da4w(child, skip_quantize_filter, new_fqn)
    
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
                # TODO: pass zp dtype to `get_group_qparams_symmetric` instead
                weight_zp = weight_zp.to(self.zero_points_precision)
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

else:  # not TORCH_VERSION_AFTER_2_4

    class Int8DynActInt4WeightQATQuantizer:
        def __init__(*args, **kwargs):
            raise ValueError(
                "Int8DynActInt4WeightQATQuantizer is only supported after PyTorch 2.4+"
            )

    class Int8DynActInt4WeightQATLinear:
        def __init__(*args, **kwargs):
            raise ValueError(
                "Int8DynActInt4WeightQATLinear is only supported after PyTorch 2.4+"
            )


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
        # Note: for bf16 inputs, casting them to fp32 has the unexpected
        # side effect of reducing memory footprint significantly, presumably
        # because bf16 * fp32 kernels are not as memory efficient
        assert input.dtype == torch.float32
        assert scales.dtype == torch.float32
        assert zero_points.dtype == torch.int32
        q = input.mul(1.0 / scales).round().add(zero_points)
        dq = q.clamp(quant_min, quant_max).sub(zero_points).mul(scales)
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
    grouped_input = input.reshape(-1, group_size).to(torch.float32)
    scales = scales.reshape(-1, 1)
    zero_points = zero_points.reshape(-1, 1)
    fq = _GenericFakeQuantize.apply(
        grouped_input, scales, zero_points, quant_min, quant_max,
    )
    return fq.reshape_as(input).to(input.dtype)

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
    fq_input = input.to(torch.float32)
    fq = _GenericFakeQuantize.apply(
        fq_input, scales, zero_points, quant_min, quant_max,
    )
    return fq.reshape_as(input).to(input.dtype)

# TODO: This is copied from torch/ao/quantization/fx/_decomposed.py.
# The version in pytorch does not have backward support yet so we add
# it here for now until https://github.com/pytorch/pytorch/pull/123452
# is landed.
def _choose_qparams_per_token_asymmetric(
    input: torch.Tensor,
    scales_precision: torch.dtype = torch.float32,
    zero_points_precision: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       scales_precision (torch.dtype): precision of returned scales
       zero_points_precision (torch.dtype): precision of returned zero points

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

    return scale.to(scales_precision), zero_point.to(zero_points_precision)
