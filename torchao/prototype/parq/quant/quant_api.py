# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union

import torch

from torchao.quantization import (
    MappingType,
)
from torchao.quantization.quant_primitives import (
    _SUB_BYTE_UINT_BOUNDS,
    _get_reduction_params,
)


def choose_qparams_stretched_affine(
    input_float: torch.Tensor,
    mapping_type: MappingType,
    block_size: Tuple[int, ...],
    target_dtype: torch.dtype,
    b: int,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale_dtype is None:
        scale_dtype = input_float.dtype
    if eps is None:
        eps = torch.finfo(input_float.dtype).eps
    if zero_point_dtype is None:
        zero_point_dtype = input_float.dtype

    assert len(block_size) == input_float.dim(), f"Got {input.dim()=}, {block_size=}"
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input_float.size()
    )
    input_float = input_float.view(shape_for_reduction)

    q_abs = input_float.abs()
    max_val = torch.minimum(
        b * q_abs.mean(dim=reduction_dims, keepdim=True),
        torch.amax(q_abs, dim=reduction_dims, keepdim=True),
    ).clamp_(min=eps)

    scale = max_val / quant_max
    scale = scale.to(dtype=scale_dtype, device=input_float.device)
    zero_point = torch.full_like(scale, -0.5, dtype=zero_point_dtype)
    return scale, zero_point


def quantize_stretched_affine(
    input_float: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    target_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
) -> torch.Tensor:
    if target_dtype in _SUB_BYTE_UINT_BOUNDS:
        target_dtype = torch.uint8
    assert input_float.dtype in (torch.float32, torch.float16, torch.bfloat16), (
        f"Unsupported input_float dtype: {input_float.dtype}"
    )
    assert len(block_size) == input_float.dim(), (
        f"Got {input_float.dim()=}, {block_size=}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input_float.size()
    )
    original_shape = input_float.shape
    input_float = input_float.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)

    if zero_point is not None and zero_point.numel() > 0:
        zero_point = zero_point.view(shape_after_reduction)
    else:
        zero_point = None

    max_val = scale.mul(quant_max)
    input_float = input_float.clamp(min=-max_val, max=max_val)
    with torch.no_grad():
        # difference from quantize_affine: add zero_point before rounding
        quant = torch.round(input_float / scale + zero_point)
    quant = quant.to(dtype=target_dtype).view(original_shape)
    return quant
