# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any

import torch

from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
    _fake_quantize_affine,
)
from torchao.quantization.utils import (
    _get_per_token_block_size,
)


class _Float8RowwiseFakeQuantize(torch.autograd.Function):
    """
    Implementation of float8 rowwise fake quantize with backward STE.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        float8_dtype: torch.dtype,
        axiswise_dim: int,
    ):
        # compute rowwise scale based on `torchao.float8.float8_utils.tensor_to_scale`
        eps = 1e-12
        amax = torch.amax(torch.abs(x), dim=axiswise_dim, keepdim=True)
        amax = amax.to(torch.float64)
        scale = torch.finfo(float8_dtype).max / torch.clamp(amax, min=eps)
        scale = scale.to(torch.float32)

        # fake quantize
        max_value = torch.finfo(float8_dtype).max
        x_fq = x.to(torch.float32) * scale
        x_fq = x_fq.clamp(min=-max_value, max=max_value)
        x_fq = x_fq.to(float8_dtype).to(x.dtype)
        x_fq = x_fq / scale
        return x_fq.to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        return gy, None, None


def _fake_quantize_per_channel_group(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    group_size: int,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
) -> torch.Tensor:
    assert group_size > 1
    assert input.shape[-1] % group_size == 0
    assert input.dim() == 2
    block_size = (1, group_size)
    return _fake_quantize_affine(
        input,
        block_size,
        scales,
        zero_points,
        quant_dtype=torch.int32,
        quant_min=quant_min,
        quant_max=quant_max,
        zero_point_domain=zero_point_domain,
    )


def _fake_quantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
) -> torch.Tensor:
    from torch.ao.quantization.fx._decomposed import _per_token_quant_qparam_dim_check

    _per_token_quant_qparam_dim_check(input, scales, zero_points)
    block_size = _get_per_token_block_size(input)
    fq = _fake_quantize_affine(
        input,
        block_size,
        scales,
        zero_points,
        quant_dtype=torch.int32,
        quant_min=quant_min,
        quant_max=quant_max,
    )
    return fq.reshape_as(input).to(input.dtype)


def _get_qmin_qmax(n_bit: int, symmetric: bool = True):
    if symmetric:
        qmin = -(2 ** (n_bit - 1))
        qmax = 2 ** (n_bit - 1) - 1
    else:
        qmin = 0
        qmax = 2**n_bit - 1
    return (qmin, qmax)


def _log_deprecation_warning(old_api_object: Any):
    """
    Log a helpful deprecation message pointing users to the new QAT API,
    only once per deprecated class.
    """
    warnings.warn(
        """'%s' is deprecated and will be removed in a future release. Please use the following API instead:

    base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
    quantize_(model, QATConfig(base_config, step="prepare"))
    # train (not shown)
    quantize_(model, QATConfig(base_config, step="convert"))

Alternatively, if you prefer to pass in fake quantization configs:

    activation_config = IntxFakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
    weight_config = IntxFakeQuantizeConfig(torch.int4, group_size=32)
    qat_config = QATConfig(
        activation_config=activation_config,
        weight_config=weight_config,
        step="prepare",
    )
    quantize_(model, qat_config)

Please see https://github.com/pytorch/ao/issues/2630 for more details.
        """
        % old_api_object.__class__.__name__
    )
