# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch

from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
    fake_quantize_affine,
)
from torchao.quantization.utils import (
    _get_per_token_block_size,
)


# TODO: delete?
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
    return fake_quantize_affine(
        input,
        block_size,
        scales,
        zero_points,
        torch.int32,
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
    fq = fake_quantize_affine(
        input,
        block_size,
        scales,
        zero_points,
        torch.int32,
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
