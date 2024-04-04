# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib
from torch.library import impl
from .quant_primitives import get_group_qparams_symmetric


# TODO: move this to core and merge with quantized_decomposed.fake_quant_per_channel
quantized_decomposed_lib.define(
    "fake_quantize_per_channel_group(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, int group_size) -> Tensor")

class _FakeQuantizePerChannel(torch.autograd.Function):
    """
    Implementation of fake quantize per channel that simulates the numerics
    of `quantize_per_channel_group` + `dequantize_per_channel_group`.
    """
    @staticmethod
    def forward(ctx, input, scales, zero_points, quant_min, quant_max):
        # Note: this diverges from `torch.fake_quantize_per_channel_affine`,
        # which rounds first before adding the zero points. However, this
        # is what `torch.ops.quantized_decomposed.quantize_per_channel_group`
        # does and we try to match that behavior here as closely as possible.
        q = input.div(scales).add(zero_points).round()
        dq = q.clamp(quant_min, quant_max).sub(zero_points).mul(scales)
        # TODO: do we need this mask?
        mask = torch.logical_and((q >= quant_min), (dq <= quant_max))
        ctx.save_for_backward(mask)
        return dq.reshape_as(input)

    @staticmethod
    def backward(ctx, gy):
        mask, = ctx.saved_tensors
        return gy * mask, None, None, None, None, None

@impl(quantized_decomposed_lib, "fake_quantize_per_channel_group", "Autograd")
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
    input = input.reshape(-1, group_size)
    scales = scales.reshape(-1, 1)
    zero_points = zero_points.reshape(-1, 1)
    return _FakeQuantizePerChannel.apply(
        input, scales, zero_points, quant_min, quant_max,
    )

@impl(quantized_decomposed_lib, "fake_quantize_per_channel_group", "Meta")
def fake_quantize_per_channel_group_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    group_size: int,
) -> torch.Tensor:
    return torch.empty_like(input)

def group_fake_quantize_tensor_symmetric(
    w,  
    n_bit=4,
    group_size=128,
    precision=torch.float32,
):
    scales, zeros = get_group_qparams_symmetric(w, n_bit, group_size, precision)
    qmin = -(2 ** (n_bit - 1))
    qmax = 2 ** (n_bit - 1) - 1
    w_fq = torch.ops.quantized_decomposed.fake_quantize_per_channel_group(
        w, scales, zeros, qmin, qmax, group_size,
    )
    return w_fq, scales, zeros
