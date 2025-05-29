# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
    fake_quantize_affine_cachemask,
)
from torchao.quantization.utils import (
    _get_per_token_block_size,
)


class _GenericFakeQuantize(torch.autograd.Function):
    """
    Implementation of generic fake quantize with backward STE.

    With the appropriate input tensor shape, this can be used to express
    grouped per channel fake quantize or per token fake quantize.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        block_size: List[int],
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        quant_min: int,
        quant_max: int,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
    ) -> torch.Tensor:
        # avoid circular dependencies
        from torchao.quantization.qat.affine_fake_quantized_tensor import (
            AffineFakeQuantizedTensor,
        )

        if isinstance(input, AffineFakeQuantizedTensor):
            _input = input.original_tensor
        else:
            _input = input

        (fq, mask) = fake_quantize_affine_cachemask(
            _input,
            block_size,
            scales,
            zero_points,
            torch.int32,
            quant_min,
            quant_max,
            zero_point_domain,
        )

        ctx.save_for_backward(mask)
        return fq

    @staticmethod
    def backward(ctx, gy):
        (mask,) = ctx.saved_tensors
        return gy * mask, None, None, None, None, None, None


class _UnwrapAffineFakeQuantizedTensor(torch.autograd.Function):
    """
    Helper autograd function to unwrap `AffineFakeQuantizedTensor` while ensuring
    gradients are still passed to the tensor subclass. This is used in place of
    `_GenericFakeQuantize` when fake quant is disabled.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
    ) -> torch.Tensor:
        # avoid circular dependencies
        from torchao.quantization.qat.affine_fake_quantized_tensor import (
            AffineFakeQuantizedTensor,
        )

        assert isinstance(input, AffineFakeQuantizedTensor)
        return input.original_tensor

    @staticmethod
    def backward(ctx, gy):
        return (gy,)


class _Round(torch.autograd.Function):
    """
    Implementation of generic round operation with backward STE.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, gy: torch.Tensor) -> torch.Tensor:
        return gy


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
    return _GenericFakeQuantize.apply(
        input,
        block_size,
        scales,
        zero_points,
        quant_min,
        quant_max,
        zero_point_domain,
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
    fq = _GenericFakeQuantize.apply(
        input,
        block_size,
        scales,
        zero_points,
        quant_min,
        quant_max,
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
