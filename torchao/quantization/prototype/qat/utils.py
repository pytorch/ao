# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch

from torchao.quantization.quant_primitives import (
    fake_quantize_affine_cachemask,
    ZeroPointDomain,
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
        ## Note: for bf16 inputs, casting them to fp32 has the unexpected
        ## side effect of reducing memory footprint significantly, presumably
        ## because bf16 * fp32 kernels are not as memory efficient
        #assert input.dtype == torch.float32
        #assert scales.dtype == torch.float32
        #assert zero_points.dtype == torch.int32

        (fq, mask) = fake_quantize_affine_cachemask(
            input,
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
        input, block_size, scales, zero_points, quant_min, quant_max, zero_point_domain,
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
    fq_input = input.to(torch.float32)
    fq = _GenericFakeQuantize.apply(
        fq_input, block_size, scales, zero_points, quant_min, quant_max,
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

def _unwrap_affine_fake_quantized_tensor(t: torch.Tensor):
    """
    Return the original, non-fake-quantized float tensor from a `AffineFakeQuantizedTensor`.
    """
    # avoid circular dependencies
    from torchao.quantization.prototype.qat.affine_fake_quantized_tensor import (
        AffineFakeQuantizedTensor,
    )
    assert isinstance(t, AffineFakeQuantizedTensor)
    return t.original_tensor

def _is_linear_with_fq_weight(mod: torch.nn.Module, *args):
    """
    Return whether this is a nn.Linear module with `AffineFakeQuantizeTensor` weights.
    """
    # avoid circular dependencies
    from torchao.quantization.linear_activation_quantized_tensor import (
        LinearActivationQuantizedTensor,
    )
    from torchao.quantization.prototype.qat.affine_fake_quantized_tensor import (
        AffineFakeQuantizedTensor,
    )
    if not isinstance(mod, torch.nn.Linear) or not hasattr(mod, "weight"):
        return False
    weight = mod.weight
    if isinstance(weight, LinearActivationQuantizedTensor):
        weight = weight.original_weight_tensor
    return isinstance(weight, AffineFakeQuantizedTensor)

def _enable_fake_quant(mod: torch.nn.Module, enable: bool):
    """
    Enable or disable fake quantization in the activations and weights of a `nn.Linear` module.
    """
    from torchao.quantization.linear_activation_quantized_tensor import (
        LinearActivationQuantizedTensor,
    )
    from torchao.quantization.prototype.qat.affine_fake_quantized_tensor import (
        AffineFakeQuantizedTensor,
    )
    if not _is_linear_with_fq_weight(mod):
        return
    weight = mod.weight
    if isinstance(weight, LinearActivationQuantizedTensor):
        weight.input_quant_func_enabled = enable
        weight = weight.original_weight_tensor
    assert isinstance(weight, AffineFakeQuantizedTensor)
    weight.fake_quant_enabled = enable
