# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum, auto
from typing import List, Optional, Tuple, Dict
import torch

from torchao.kernel.intmm import int_scaled_matmul
from torchao.kernel.intmm import safe_int_mm
from torchao.utils import (
    TORCH_VERSION_AFTER_2_3,
    TORCH_VERSION_AFTER_2_5,
)
from torchao.utils import _register_custom_op


__all__ = [
    "safe_int_mm",
    "int_scaled_matmul",
    "choose_qparams_affine",
    "quantize_affine",
    "dequantize_affine",
]

class MappingType(Enum):
    """How floating point number is mapped to integer number

    symmetric mapping means floating point range is symetrically mapped to integer range
    let's say we have floating point range (-3.5, 10.2) and integer range (-8, 7) (int4)
    we'll use (-10.2, 10.2) as the range for floating point and map that to (-8, 7)
    e.g. scale = (10.2 - (-10.2)) / (7 - (-8))

    asymmetric mapping means we just directly map the floating point range to integer range,
    for the above example, we will map (-3.5, 10.2) to (-8, 7) and calculate quantization parameter
    based on this mapping
    e.g. scale = (10.2 - (-3.5)) / (7 - (-8))
    """
    SYMMETRIC = auto()
    ASYMMETRIC = auto()

class ZeroPointDomain(Enum):
    """Enum that indicate whether zero_point is in integer domain or floating point domain

    integer domain: quantized_val = (float_val / scale) (integer) + zero_point (integer)
    float domain: quantized_val = (float_val - (zero_point (float) - scale * mid_point)) / scale
    """
    INT = auto()
    FLOAT = auto()

"""
Map from dtype to the bound value of integers
TODO: maybe can replace this with call to torch.iinfo
"""
_DTYPE_TO_QVALUE_BOUNDS: Dict[torch.dtype, Tuple[int, int]] = {
    torch.uint8: (0, 255),
    torch.int8: (-128, 127),
    torch.int16: (-(2**15), 2**15 - 1),
    torch.int32: (-(2**31), 2**31 - 1),
}

if TORCH_VERSION_AFTER_2_3:
    _DTYPE_TO_QVALUE_BOUNDS.update({
        torch.uint1: (0, 2**1-1),
        torch.uint2: (0, 2**2-1),
        torch.uint3: (0, 2**3-1),
        torch.uint4: (0, 2**4-1),
        torch.uint5: (0, 2**5-1),
        torch.uint6: (0, 2**6-1),
        torch.uint7: (0, 2**7-1),
    })


quant_lib = torch.library.Library("quant", "FRAGMENT")

register_custom_op = _register_custom_op(quant_lib)

# TODO: decide on if we want to allow custom quant_min/quant_max here
def _get_and_check_qmin_qmax(dtype, quant_min, quant_max):
    """Get quant_min and quant_max args based on dtype and also
    verify that they are within the range of possible quant_min/quant_max
    for dtype
    """
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]
    if quant_min is None:
        quant_min = quant_min_lower_bound
    if quant_max is None:
        quant_max = quant_max_upper_bound

    assert quant_min >= quant_min_lower_bound, \
        "quant_min out of bound for dtype, " \
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"

    assert quant_max <= quant_max_upper_bound, \
        "quant_max out of bound for dtype, " \
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"
    return quant_min, quant_max

def _get_reduction_params(block_size, input_size):
    """Given block_size and input size find the parameters for reduction:

    Output:
        shape_for_reduction: the shape we use to `view` input to prepare it for reduction
        reduction_dims: the dims we'll do reduction over

    Example::
        Input:
          block_size: (3, 3, 2, 10)
          input_size: (3, 3, 10, 10)

        Output:
          shape_for_reduction: (3, 3, 5, 2, 10)
          reduction_dim: [0, 1, 3, 4]
    """
    assert len(block_size) == len(input_size)
    shape_for_reduction = []
    reduction_dims = []
    cur_dim = 0
    for i in range(len(block_size)):
        if block_size[i] != input_size[i] and block_size[i] > 1:
            assert input_size[i] % block_size[i] == 0, f"Expecting input size at {i} dimension: {input_size[i]} to be divisible by block_size at {i} dimension: {block_size[i]}"
            shape_for_reduction.append(input_size[i] // block_size[i])
            shape_for_reduction.append(block_size[i])
            # reduce over the block_size[i] dim
            reduction_dims.append(cur_dim + 1)
            cur_dim += 2
        else:
            # block_size[i] == input_size[i] or block_size[i] == 1
            shape_for_reduction.append(input_size[i])
            # we only need to reduce over the dimension if block_size is greater than 1
            # otherwise it's already the same as reduced dimension
            if block_size[i] != 1:
                reduction_dims.append(cur_dim)
            cur_dim += 1
    return shape_for_reduction, reduction_dims


def quantize_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
) -> torch.Tensor:
    """
    Args:
      input (torch.Tensor): original float32, float16 or bfloat16 Tensor
      block_size: (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
           e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (float): quantization parameter for affine quantization
      zero_point (int): quantization parameter for affine quantization
      output_dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for output Tensor, if not specified, it will be derived from dtype
      quant_max (Optional[int]): maximum quantized value for output Tensor, if not specified, it will be derived from dtype
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be eitehr integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT

    Note:
      How can block_size represent different granularities?
      let's say we have a Tensor of size: (3, 3, 10, 10), here is the table showing how block_size represents different
      granularities:

       granularity type       |     block_size
         per_tensor           |    (3, 3, 10, 10)
         per_axis (axis=0)    |    (1, 3, 10, 10)
         per_axis (axis=1)    |    (3, 1, 10, 10)
     per_group (groupsize=2)  |    (3, 3, 10, 2)
     per_group (groupsize=2) for axis = 3 | (3, 3, 2, 10)


    Output:
      quantized tensor with requested dtype
    """
    return _quantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        output_dtype,
        quant_min,
        quant_max,
        zero_point_domain.name,
    )


@register_custom_op
def _quantize_affine(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
    zero_point_domain: str = "INT",
) -> torch.Tensor:
    """op definition that has compatible signatures with custom op library
    """
    # TODO: validations
    # TODO: validate scale/zero_point dimensions are compatible with block_size
    assert input.dtype in [torch.float32, torch.float16, torch.bfloat16], f"Unsupported input dtype: {input.dtype}"
    quant_min, quant_max = _get_and_check_qmin_qmax(output_dtype, quant_min, quant_max)
    shape_for_reduction, reduction_dims = _get_reduction_params(block_size, input.size())
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)
    if zero_point is not None:
        zero_point = zero_point.view(shape_after_reduction)

    if zero_point_domain == ZeroPointDomain.INT.name:
        quant = torch.clamp(
            torch.round(input * (1.0 / scale)) + zero_point, quant_min, quant_max
        ).to(output_dtype)
    else:
        assert zero_point_domain == ZeroPointDomain.FLOAT.name
        mid_point = (quant_max + quant_min + 1) / 2
        min_val = zero_point - scale * mid_point
        quant = (
            torch.clamp(
                torch.round((input - min_val) / scale),
                quant_min, quant_max)
        ).to(output_dtype)
    quant = quant.view(original_shape)

    return quant

def dequantize_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    input_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Args:
      input (torch.Tensor): quantized tensor, should match the dtype `dtype` argument
      block_size: (List[int]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
                               e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (Tensor): quantization parameter for affine quantization
      zero_point (Tensor): quantization parameter for affine quantization
      dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for input Tensor
      quant_max (Optional[int]): maximum quantized value for input Tensor
      output_dtype (torch.dtype): dtype for output Tensor, default is fp32
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be eitehr integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT

    Output:
      dequantized Tensor, with requested dtype or fp32
    """
    return _dequantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        input_dtype,
        quant_min,
        quant_max,
        zero_point_domain.name,
        output_dtype=output_dtype,
    )

@register_custom_op
def _dequantize_affine(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    input_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
    zero_point_domain: str = "INT",
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """op definition that has compatible signatures with custom op library
    """

    # TODO: validations
    # TODO: validate scale/zero_point dimensions are compatible with block_size
    assert input.dtype == input_dtype, f"Expected: {input_dtype}, got: {input.dtype}"
    assert output_dtype in [torch.float32, torch.float16, torch.bfloat16], f"Unsupported output dtype: {output_dtype}"
    quant_min, quant_max = _get_and_check_qmin_qmax(input_dtype, quant_min, quant_max)

    shape_for_reduction, reduction_dims = _get_reduction_params(block_size, input.size())
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)
    if zero_point is not None:
        zero_point = zero_point.view(shape_after_reduction)

    if zero_point_domain == ZeroPointDomain.INT.name:
        # Force a copy to avoid input modification due
        # to upcoming in-place operations.
        dequant = input.to(torch.int32, copy=True)
        if zero_point is not None:
            dequant = dequant - zero_point.to(torch.int32)
        dequant = dequant.to(output_dtype)
        dequant = dequant * scale
    else:
        assert zero_point_domain == ZeroPointDomain.FLOAT.name, f"Unexpected zero point domain: {zero_point_domain}"
        mid_point = (quant_max + quant_min + 1) / 2
        # This should allocate new memory and avoid input modification
        dequant = input - mid_point
        dequant = dequant.to(output_dtype)
        dequant *= scale
        if zero_point is not None:
            dequant += zero_point

    return dequant.view(original_shape).to(output_dtype)

def choose_qparams_affine(
   input: torch.Tensor,
   mapping_type: MappingType,
   block_size: Tuple[int, ...],
   target_dtype: torch.dtype,
   quant_min: Optional[int] = None,
   quant_max: Optional[int] = None,
   eps: Optional[float] = None,
   scale_dtype: Optional[torch.dtype] = None,
   zero_point_dtype: Optional[torch.dtype] = None,
   preserve_zero: bool = True,
   zero_point_domain = ZeroPointDomain.INT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        input (torch.Tensor): fp32, bf16, fp16 input Tensor
        mapping_type (MappingType): determines how the qparams are calculated, symmetric or asymmetric
        block_size: (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
          e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
        target_dtype (torch.dtype): dtype for target quantized Tensor
        quant_min (Optional[int]): minimum quantized value for target quantized Tensor
        quant_max (Optioanl[int]): maximum quantized value for target quantized Tensor
        eps (Optional[float]): minimum scale, if not provided, default to eps of input.dtype
        scale_dtype (torch.dtype): dtype for scale Tensor
        zero_point_dtype (torch.dtype): dtype for zero_point Tensor
        preserve_zero (bool): a flag to indicate whether we need zero to be exactly
          representable or not, this is typically required for ops that needs zero padding, like convolution
          it's less important for ops that doesn't have zero padding in the op itself, like linear.

          For example, given a floating point Tensor [1.2, 0.1, 3.0, 4.0, 0.4, 0], if `preserve_zero` is True,
          we'll make sure there is a integer value corresponding to the floating point 0, e.g. [-3, -8, 3, 7, -7, -8], 0 will be mapped to `-8` without loss. But if `preserve_zero` is not True, there won't be such
          gurantee.

          If we don't need zero to be exactly representable, we won't do rounding and clamping for zero_point

        zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be eitehr integer or float
            if zero_point is in integer domain, zero point is added to the quantized integer value during
            quantization
            if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
            value during quantization
            default is ZeroPointDomain.INT

    Output:
        Tuple of scales and zero_points Tensor with requested dtype
    """
    return _choose_qparams_affine(
        input,
        mapping_type.name,
        block_size,
        target_dtype,
        quant_min,
        quant_max,
        eps,
        scale_dtype,
        zero_point_dtype,
        preserve_zero,
        zero_point_domain.name
    )

@register_custom_op
def _choose_qparams_affine(
   input: torch.Tensor,
   mapping_type: str,
   block_size: List[int],
   target_dtype: torch.dtype,
   quant_min: Optional[int] = None,
   quant_max: Optional[int] = None,
   eps: Optional[float] = None,
   scale_dtype: Optional[torch.dtype] = None,
   zero_point_dtype: Optional[torch.dtype] = None,
   preserve_zero: bool = True,
   zero_point_domain: str = "INT",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """op definition that has compatible signatures with custom op library
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
    assert mapping_type in [MappingType.SYMMETRIC.name, MappingType.ASYMMETRIC.name], f"Unsupported mapping type: {mapping_type}"

    if scale_dtype is None:
        scale_dtype = input.dtype
    if zero_point_dtype is None:
        zero_point_dtype = input.dtype

    assert len(block_size) == input.dim()
    shape_for_reduction, reduction_dims = _get_reduction_params(block_size, input.size())
    input = input.view(shape_for_reduction)

    min_val = torch.amin(input, dim=reduction_dims, keepdim=False)
    max_val = torch.amax(input, dim=reduction_dims, keepdim=False)

    if preserve_zero:
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    else:
        min_val_neg = min_val
        max_val_pos = max_val

    if mapping_type == MappingType.SYMMETRIC.name:
        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        if not preserve_zero:
            raise ValueError("preserve_zero == False is not supported for symmetric quantization")
        if zero_point_domain != ZeroPointDomain.INT.name:
            raise ValueError("zero_point_domain != ZeroPointDomain.INT is not supported for symmetric quantization")
        zero_point = torch.full_like(scale, int((quant_max + quant_min + 1) / 2))
    else:
        assert mapping_type == MappingType.ASYMMETRIC.name
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        if preserve_zero:
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        else:
            assert zero_point_domain == ZeroPointDomain.FLOAT.name, "if not preserve_zero, zero_point must be in FLOAT domain"
            mid_point = (quant_max + quant_min + 1) / 2
            zero_point = min_val_neg + scale * mid_point

    if eps is None:
        eps = torch.finfo(input.dtype).eps
    scale = torch.clamp(scale, min=eps)

    return scale.to(dtype=scale_dtype), zero_point.to(dtype=zero_point_dtype)
