# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from torchao.prototype.custom_fp_utils import (
    _f32_to_floatx_unpacked,
    _floatx_unpacked_to_f32,
    _n_ones,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_3,
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
    _is_float8_type,
    _register_custom_op,
)

__all__ = [
    "choose_qparams_affine",
    "choose_qparams_affine_with_min_max",
    "choose_qparams_affine_floatx",
    "quantize_affine",
    "dequantize_affine",
    "quantize_affine_floatx",
    "dequantize_affine_floatx",
    "fake_quantize_affine",
    "fake_quantize_affine_cachemask",
    "choose_qparams_and_quantize_affine_hqq",
    "choose_qparams_and_quantize_affine_qqq",
    "dequantize_affine_qqq",
    "MappingType",
    "ZeroPointDomain",
    "TorchAODType",
    "choose_qparams_affine_float8",
    "quantize_affine_float8",
    "dequantize_affine_float8",
    "choose_qparams_gguf",
    "quantize_gguf",
    "dequantize_gguf",
]


class MappingType(Enum):
    """How floating point number is mapped to integer number

    symmetric mapping means floating point range is symmetrically mapped to integer range
    let's say we have floating point range (-3.5, 10.2) and integer range (-8, 7) (int4)
    we'll use (-10.2, 10.2) as the range for floating point and map that to (-8, 7)
    e.g. scale = (10.2 - (-10.2)) / (7 - (-8))

    SYMMETRIC_NO_CLIPPING_ERR is a variant of symmetric mapping, where the scale is the max of smin
    and smax, where smin = min_val_neg / quant_min, and smax = max_val_pos / quant_max. By calculating
    smin and smax individually, there can be less round error on negative values, and no out-of-range
    of all floating point values.

    asymmetric mapping means we just directly map the floating point range to integer range,
    for the above example, we will map (-3.5, 10.2) to (-8, 7) and calculate quantization parameter
    based on this mapping
    e.g. scale = (10.2 - (-3.5)) / (7 - (-8))
    """

    SYMMETRIC = auto()
    SYMMETRIC_NO_CLIPPING_ERR = auto()
    ASYMMETRIC = auto()


class ZeroPointDomain(Enum):
    """Enum that indicate whether zero_point is in integer domain or floating point domain

    integer domain: quantized_val = (float_val / scale) (integer) + zero_point (integer)
    float domain: quantized_val = (float_val - (zero_point (float) - scale * mid_point)) / scale
    none domain: quantized_val = (float_val / scale)
    """

    INT = auto()
    FLOAT = auto()
    NONE = auto()


class TorchAODType(Enum):
    """
    Placeholder for dtypes that do not exist in PyTorch core yet.
    """

    # torch.int1 to torch.int7 will be added to PyTorch 2.6
    # These will remain here for BC with older PyTorch versions
    INT1 = auto()
    INT2 = auto()
    INT3 = auto()
    INT4 = auto()
    INT5 = auto()
    INT6 = auto()
    INT7 = auto()


if TORCH_VERSION_AT_LEAST_2_5:
    torch.serialization.add_safe_globals([MappingType, ZeroPointDomain])

FP8_TYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
}

"""
Map from dtype to the bound value of integers
TODO: maybe can replace this with call to torch.iinfo
"""
_DTYPE_TO_QVALUE_BOUNDS: Dict[Union[torch.dtype, TorchAODType], Tuple[int, int]] = {
    torch.uint8: (0, 255),
    torch.int8: (-128, 127),
    torch.int16: (-(2**15), 2**15 - 1),
    torch.int32: (-(2**31), 2**31 - 1),
}
_DTYPE_TO_BIT_WIDTH: Dict[Union[torch.dtype, TorchAODType], Tuple[int, int]] = {
    TorchAODType.INT1: 1,
    TorchAODType.INT2: 2,
    TorchAODType.INT3: 3,
    TorchAODType.INT4: 4,
    TorchAODType.INT5: 5,
    TorchAODType.INT6: 6,
    TorchAODType.INT7: 7,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.int32: 32,
}

_SUB_BYTE_UINT_BOUNDS: Dict[Union[torch.dtype, TorchAODType], Tuple[int, int]] = {}
_SUB_BYTE_INT_BOUNDS: Dict[Union[torch.dtype, TorchAODType], Tuple[int, int]] = {
    TorchAODType.INT1: (-(2**0), 2**0 - 1),
    TorchAODType.INT2: (-(2**1), 2**1 - 1),
    TorchAODType.INT3: (-(2**2), 2**2 - 1),
    TorchAODType.INT4: (-(2**3), 2**3 - 1),
    TorchAODType.INT5: (-(2**4), 2**4 - 1),
    TorchAODType.INT6: (-(2**5), 2**5 - 1),
    TorchAODType.INT7: (-(2**6), 2**6 - 1),
}

# torch.uintX available only in PyTorch 2.3+
if TORCH_VERSION_AT_LEAST_2_3:
    _SUB_BYTE_UINT_BOUNDS = {
        torch.uint1: (0, 2**1 - 1),
        torch.uint2: (0, 2**2 - 1),
        torch.uint3: (0, 2**3 - 1),
        torch.uint4: (0, 2**4 - 1),
        torch.uint5: (0, 2**5 - 1),
        torch.uint6: (0, 2**6 - 1),
        torch.uint7: (0, 2**7 - 1),
    }
    _DTYPE_TO_BIT_WIDTH.update(
        {
            torch.uint1: 1,
            torch.uint2: 2,
            torch.uint3: 3,
            torch.uint4: 4,
            torch.uint5: 5,
            torch.uint6: 6,
            torch.uint7: 7,
        }
    )

# torch.intX available only in PyTorch 2.6+
if TORCH_VERSION_AT_LEAST_2_6:
    _SUB_BYTE_INT_BOUNDS.update(
        {
            torch.int1: (-(2**0), 2**0 - 1),
            torch.int2: (-(2**1), 2**1 - 1),
            torch.int3: (-(2**2), 2**2 - 1),
            torch.int4: (-(2**3), 2**3 - 1),
            torch.int5: (-(2**4), 2**4 - 1),
            torch.int6: (-(2**5), 2**5 - 1),
            torch.int7: (-(2**6), 2**6 - 1),
        }
    )
    _DTYPE_TO_BIT_WIDTH.update(
        {
            torch.int1: 1,
            torch.int2: 2,
            torch.int3: 3,
            torch.int4: 4,
            torch.int5: 5,
            torch.int6: 6,
            torch.int7: 7,
        }
    )

_DTYPE_TO_QVALUE_BOUNDS.update(_SUB_BYTE_UINT_BOUNDS)
_DTYPE_TO_QVALUE_BOUNDS.update(_SUB_BYTE_INT_BOUNDS)
assert _DTYPE_TO_BIT_WIDTH.keys() == _DTYPE_TO_QVALUE_BOUNDS.keys()

_GGUF_QK_K = 256

_ONES_TABLE = [_n_ones(i) for i in range(8)]

quant_lib = torch.library.Library("torchao", "FRAGMENT")

register_custom_op = _register_custom_op(quant_lib)


# TODO: decide on if we want to allow custom quant_min/quant_max here
def _get_and_check_qmin_qmax(dtype, quant_min, quant_max):
    """Get quant_min and quant_max args based on dtype and also
    verify that they are within the range of possible quant_min/quant_max
    for dtype
    """
    if dtype in FP8_TYPES:
        quant_min_lower_bound, quant_max_upper_bound = (
            torch.finfo(dtype).min,
            torch.finfo(dtype).max,
        )
    elif dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    else:
        quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]
    if quant_min is None:
        quant_min = quant_min_lower_bound
    if quant_max is None:
        quant_max = quant_max_upper_bound

    assert quant_min >= quant_min_lower_bound, (
        "quant_min out of bound for dtype, "
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"
    )

    assert quant_max <= quant_max_upper_bound, (
        "quant_max out of bound for dtype, "
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"
    )
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
            assert input_size[i] % block_size[i] == 0, (
                f"Expecting input size at {i} dimension: {input_size[i]} to be divisible by block_size at {i} dimension: {block_size[i]}"
            )
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


@torch.no_grad()
def quantize_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
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
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be either integer or float
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
    if zero_point_domain is None:
        raise ValueError("Please use ZeroPointDomain.NONE instead of None")
    elif zero_point_domain is ZeroPointDomain.NONE and zero_point is not None:
        raise ValueError("zero_point should be None when zero_point_domain is NONE")
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
    quant_min: Optional[Union[int, float, bool]] = None,
    quant_max: Optional[Union[int, float, bool]] = None,
    zero_point_domain: str = ZeroPointDomain.INT.name,
) -> torch.Tensor:
    """op definition that has compatible signatures with custom op library

    Note:
        zero_point_domain is optional specifies how we quantize the floating point to quantized data:
        INT: quantized_val = (float_val / scale) (integer) + zero_point (integer)
        FLOAT: quantized_val = (float_val - (zero_point (float) - scale * mid_point)) / scale
        None: quantized_val = (float_val / scale) | this is primarily used for floatx quantization
            Where we do not want to round values to nearest integer and instead scale and cast.
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(output_dtype, quant_min, quant_max)
    # workaround for uintx dtypes, since we don't have native Uintx dtype connected with
    # torch.uintx dtypes yet
    if output_dtype in _SUB_BYTE_UINT_BOUNDS:
        output_dtype = torch.uint8
    if block_size == [-1]:
        # per-tensor quantization
        block_size = [-1] * input.dim()
    return _quantize_affine_no_dtype_cast(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
        output_dtype,
        zero_point_domain,
    ).to(output_dtype)


def _quantize_affine_no_dtype_cast(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Union[int, float],
    quant_max: Union[int, float],
    quant_dtype: torch.dtype,
    zero_point_domain: str = ZeroPointDomain.INT.name,
) -> torch.Tensor:
    """
    The op does the following:
    1. figure out the dimension for reduction based on block_size, also reshape the input to align with
       the shape after reduction
    2. quantize the input based on the quantization parameters scale and zero_point and args like zero_point_domain
    3. reshape the quantized result to origianl shape
    """
    # TODO: validations
    # TODO: validate scale/zero_point dimensions are compatible with block_size
    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported input dtype: {input.dtype}"
    assert len(block_size) == input.dim(), (
        f"Got input dim:{input.dim()}, block_size: {block_size}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)

    if zero_point is not None and zero_point.numel() > 0:
        zero_point = zero_point.view(shape_after_reduction)
    else:
        # in some cases zero_point being a non-value shows as a tensor
        # with numel=0 which we handle by unifying the two
        zero_point = None

    if zero_point_domain == ZeroPointDomain.INT.name:
        quant = torch.clamp(
            torch.round(input * (1.0 / scale)) + zero_point, quant_min, quant_max
        )
    elif zero_point_domain == ZeroPointDomain.NONE.name:
        assert zero_point is None, (
            "zero_point should be None when zero_point_domain is NONE"
        )
        if _is_float8_type(quant_dtype):
            quant = torch.clamp(input * scale.reciprocal(), quant_min, quant_max)
        else:
            quant = torch.clamp(
                torch.round(input * (1.0 / scale)), quant_min, quant_max
            )
    else:
        assert zero_point_domain == ZeroPointDomain.FLOAT.name
        mid_point = (quant_max + quant_min + 1) / 2
        min_val = zero_point - scale * mid_point
        quant = torch.clamp(
            torch.round((input - min_val) / scale), quant_min, quant_max
        )
    quant = quant.view(original_shape)

    return quant


def dequantize_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    input_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
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
      input_dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for input Tensor
      quant_max (Optional[int]): maximum quantized value for input Tensor
      output_dtype (torch.dtype): dtype for output Tensor, default is fp32
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be either integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT

    Output:
      dequantized Tensor, with requested dtype or fp32
    """
    if zero_point_domain is None:
        raise ValueError("Please use ZeroPointDomain.NONE instead of None")
    elif zero_point_domain is ZeroPointDomain.NONE and zero_point is not None:
        raise ValueError("zero_point should be None when zero_point_domain is NONE")
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
    quant_min: Optional[Union[int, float, bool]] = None,
    quant_max: Optional[Union[int, float, bool]] = None,
    zero_point_domain: Optional[str] = ZeroPointDomain.INT.name,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """op definition that has compatible signatures with custom op library"""
    # TODO: validate scale/zero_point dimensions are compatible with block_size
    if input_dtype not in _SUB_BYTE_UINT_BOUNDS:
        assert input.dtype == input_dtype, (
            f"Expected: {input_dtype}, got: {input.dtype}"
        )
    assert output_dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported output dtype: {output_dtype}"
    if block_size == [-1]:
        # per-tensor quantization
        block_size = [-1] * input.dim()
    quant_min, quant_max = _get_and_check_qmin_qmax(input_dtype, quant_min, quant_max)
    return _dequantize_affine_no_dtype_check(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
        zero_point_domain,
        output_dtype,
    )


def _dequantize_affine_no_dtype_check(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Union[int, float],
    quant_max: Union[int, float],
    zero_point_domain: Optional[str] = ZeroPointDomain.INT.name,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """This function converts AQT tensors to their high precision floating point representation

    The op does the following:
    1. figure out the dimension for reduction based on block_size, also reshape the input to align with
       the shape after reduction
    2. dequantize the input based on the quantization parameters scale and zero_point and args like zero_point_domain
    3. reshape the quantized result to origianl shape and change dtype to the output_dtype
    """
    assert len(block_size) == input.dim(), (
        f"Got input dim:{input.dim()}, block_size: {block_size}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
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
    elif zero_point_domain == ZeroPointDomain.NONE.name:
        assert zero_point is None, (
            "zero_point should be None when zero_point_domain is NONE"
        )
        dequant = input.to(output_dtype)
        dequant = dequant * scale
    else:
        assert zero_point_domain == ZeroPointDomain.FLOAT.name, (
            f"Unexpected zero point domain: {zero_point_domain}"
        )
        # TODO: this seems to be a detail for tinygemm (converting from uint to int, probably need to refactor this)
        mid_point = (quant_max + quant_min + 1) / 2
        # This should allocate new memory and avoid input modification
        dequant = input - mid_point
        dequant = dequant.to(output_dtype)
        dequant *= scale
        if zero_point is not None:
            dequant += zero_point

    return dequant.view(original_shape).to(output_dtype)


def fake_quantize_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
) -> torch.Tensor:
    """
    General fake quantize op for quantization-aware training (QAT).
    This is equivalent to calling `quantize_affine` + `dequantize_affine`
    but without the dtype casts.

    Args:
      input (torch.Tensor): original float32, float16 or bfloat16 Tensor
      block_size: (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
           e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (float): quantization parameter for affine quantization
      zero_point (int): quantization parameter for affine quantization
      quant_dtype (torch.dtype): desired quantized dtype for determining and validating quant_min and quant_max values.
      quant_min (Optional[int]): minimum quantized value for output Tensor, if not specified, it will be derived from dtype
      quant_max (Optional[int]): maximum quantized value for output Tensor, if not specified, it will be derived from dtype
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be either integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT
    """
    if zero_point_domain is None:
        raise ValueError("Please use ZeroPointDomain.NONE instead of None")
    elif zero_point_domain is ZeroPointDomain.NONE and zero_point is not None:
        raise ValueError("zero_point should be None when zero_point_domain is NONE")
    (_, fq) = _do_fake_quantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        quant_dtype,
        quant_min,
        quant_max,
        zero_point_domain,
    )
    return fq


def fake_quantize_affine_cachemask(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    General fake quantize op for quantization-aware training (QAT).
    This is equivalent to calling `quantize_affine` + `dequantize_affine`
    but without the dtype casts.

    Note: Compared to :func:`~torchao.quantization.quant_primitives.fake_quantize_affine`,
    this consumes more memory and returns an additional outlier mask for
    intermediate quantized values.

    Args:
      Same as :func:`~torchao.quantization.quant_primitives.fake_quantize_affine`.

    Returns:
      A 2-tuple of (
          final fake quantized values,
          outlier mask for intermediate quantized values
      )

    """
    if zero_point_domain is None:
        raise ValueError("Please use ZeroPointDomain.NONE instead of None")
    elif zero_point_domain is None and zero_point is not None:
        raise ValueError("zero_point should be None when zero_point_domain is NONE")
    (q, dq) = _do_fake_quantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        quant_dtype,
        quant_min,
        quant_max,
        zero_point_domain,
    )
    mask = torch.logical_and((q >= quant_min), (q <= quant_max))
    return (dq, mask)


def _do_fake_quantize_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function for `fake_quantize_affine` that returns both the
    intermediate quantized values and the final dequantized values.
    """
    input_dtype = input.dtype
    quant_min, quant_max = _get_and_check_qmin_qmax(quant_dtype, quant_min, quant_max)
    q = _quantize_affine_no_dtype_cast(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
        quant_dtype,
        zero_point_domain.name,
    )
    dq = _dequantize_affine_no_dtype_check(
        q,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
        zero_point_domain.name,
        output_dtype=input_dtype,
    )
    return (q, dq)


@torch.no_grad()
def choose_qparams_affine(
    input: torch.Tensor,
    mapping_type: MappingType,
    block_size: Tuple[int, ...],
    target_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
    preserve_zero: bool = True,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
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

        zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be either integer or float
            if zero_point is in integer domain, zero point is added to the quantized integer value during
            quantization
            if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
            value during quantization
            default is ZeroPointDomain.INT

    Output:
        Tuple of scales and zero_points Tensor with requested dtype
    """
    if zero_point_domain is None:
        raise ValueError("Please use ZeroPointDomain.NONE instead of None")

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
        zero_point_domain.name,
    )


def choose_qparams_affine_with_min_max(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    mapping_type: MappingType,
    block_size: Tuple[int, ...],
    target_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
    preserve_zero: bool = True,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A variant of :func:`~torchao.quantization.quant_primitives.choose_qparams_affine`
    operator that pass in min_val and max_val directly instead of deriving these from a single input.
    This is used for observers in static quantization where min_val and max_val may be obtained through
    tracking all the data in calibration data set.

    Args:
      Mostly same as :func:`~torchao.quantization.quant_primitives.choose_qparams_affine`. with one
      difference: instead of passing in `input` Tensor and use that to calculate min_val/max_val
      and then scale/zero_point, we pass in min_val/max_val directly
    """
    if zero_point_domain is None:
        raise ValueError("Please use ZeroPointDomain.NONE instead of None")
    return _choose_qparams_affine(
        None,
        mapping_type.name,
        block_size,
        target_dtype,
        quant_min,
        quant_max,
        eps,
        scale_dtype,
        zero_point_dtype,
        preserve_zero,
        zero_point_domain.name,
        min_val,
        max_val,
    )


@register_custom_op
def _choose_qparams_affine(
    input: Optional[torch.Tensor],
    mapping_type: str,
    block_size: List[int],
    target_dtype: torch.dtype,
    quant_min: Optional[Union[int, float, bool]] = None,
    quant_max: Optional[Union[int, float, bool]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
    preserve_zero: bool = True,
    zero_point_domain: Optional[str] = "INT",
    min_val: Optional[torch.Tensor] = None,
    max_val: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """op definition that has compatible signatures with custom op library

    The op does the following:
    1. figure out the dimension for reduction based on block_size
    2. find min_val/max_val based on the dimension for reduction
    3. calculate quantization parameters based on min_val/max_val based on args like `preserve_zero`
       and `zero_point_domain`
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
    assert mapping_type in [
        MappingType.SYMMETRIC.name,
        MappingType.SYMMETRIC_NO_CLIPPING_ERR.name,
        MappingType.ASYMMETRIC.name,
    ], f"Unsupported mapping type: {mapping_type}"
    if target_dtype in FP8_TYPES:
        assert mapping_type == MappingType.SYMMETRIC.name, (
            f"Only symmetric quantization is supported for FP8 types, got {mapping_type}"
        )

    if input is not None:
        if scale_dtype is None:
            scale_dtype = input.dtype
        if eps is None:
            eps = torch.finfo(input.dtype).eps
        if block_size == [-1]:
            # per-tensor quantization
            block_size = [-1] * input.dim()

        assert len(block_size) == input.dim(), (
            f"Got input dim:{input.dim()}, block_size: {block_size}"
        )
        shape_for_reduction, reduction_dims = _get_reduction_params(
            block_size, input.size()
        )
        input = input.view(shape_for_reduction)

        min_val = torch.amin(input, dim=reduction_dims, keepdim=False)
        max_val = torch.amax(input, dim=reduction_dims, keepdim=False)
    else:
        assert min_val is not None and max_val is not None, (
            "Need to provide `min_val` and `max_val` when `input` is None, got: {min_val, max_val}"
        )
        assert min_val.dtype == max_val.dtype, (
            "Expecting `min_val` and `max_val` to have the same dtype, got: {min_val.dtype, max_val.dtype}"
        )

        if scale_dtype is None:
            scale_dtype = min_val.dtype
        if eps is None:
            eps = torch.finfo(min_val.dtype).eps

    if preserve_zero:
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    else:
        min_val_neg = min_val
        max_val_pos = max_val

    if (
        mapping_type == MappingType.SYMMETRIC.name
        or mapping_type == MappingType.SYMMETRIC_NO_CLIPPING_ERR.name
    ):
        # scales
        if mapping_type == MappingType.SYMMETRIC.name:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
        else:
            assert mapping_type == MappingType.SYMMETRIC_NO_CLIPPING_ERR.name
            # calculate smin and smax individually and choose the larger one. For example, if quant_min = -8 and
            # quant_max = 7.
            # - If smin is bigger: There would be coverage on negative values down to -8, and less rounding
            # error than the existing SYMMETRIC case.
            # - If smax is bigger: it covers the positive values up to 7. The round
            # error may be bigger than the existing SYMMETRIC case. Either way, there's no out-of-range fp values after
            # quantization.
            smin = min_val_neg / float(quant_min)
            smax = max_val_pos / float(quant_max)
            mask = smin > smax
            scale = torch.where(mask, smin, smax)
        # zeros
        if not preserve_zero:
            raise ValueError(
                "preserve_zero == False is not supported for symmetric quantization"
            )
        if zero_point_domain == ZeroPointDomain.FLOAT.name:
            # TODO INT should not be a valid ZeroPointDomain for symmetric quantization since
            # symmetric quant doesn't have a zero_point
            raise ValueError(
                "zero_point_domain should be ZeroPointDomain.INT or ZeroPointDomain.NONE for symmetric quantization"
            )
        if zero_point_domain == ZeroPointDomain.NONE.name:
            zero_point = None
        else:
            zero_point = torch.full_like(scale, int((quant_max + quant_min + 1) / 2))
        scale = torch.clamp(scale, min=eps)
    else:
        assert mapping_type == MappingType.ASYMMETRIC.name
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.clamp(scale, min=eps)
        if zero_point_domain == ZeroPointDomain.NONE.name:
            zero_point = None
        elif zero_point_domain == ZeroPointDomain.INT.name:
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
            if zero_point_dtype is None:
                zero_point_dtype = torch.int32
        else:
            assert zero_point_domain == ZeroPointDomain.FLOAT.name, (
                "zero_point must be in FLOAT/INT/None domain for asymmetric quantization"
            )
            mid_point = (quant_max + quant_min + 1) / 2
            # this is not preserving zero_point, this is converting to TensorCoreTiledFormat
            # TODO move the conversion of zero_point out of quant_primitives
            # and into TensorCoreTiledLayout.from_plain
            zero_point = min_val_neg + scale * mid_point

    if zero_point is not None:
        zero_point = zero_point.to(dtype=zero_point_dtype)
    return scale.to(dtype=scale_dtype), zero_point


def choose_qparams_and_quantize_affine_qqq(
    w: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert num_bits == 4, f"Unsupported num_bits = {num_bits}"
    size_n, size_k = w.shape
    assert group_size in [-1, 128, size_k], f"Unsupported groupsize = {group_size}"
    orig_device = w.device
    if group_size == -1:
        group_size = size_k

    if group_size < size_k:
        # Reshape to [-1, group_size]
        w = w.reshape((-1, group_size))

        max_q_val = 2**num_bits - 1
        half_q_val = (max_q_val + 1) // 2

        # Compute scale for each group
        s_group = torch.amax(torch.abs(w), -1, keepdim=True)
        s_group *= 2 / max_q_val  # 2 => symmetric

        # Quantize
        q_w = torch.round(w / s_group).int()
        q_w += half_q_val
        q_w = torch.clamp(q_w, 0, max_q_val)
        # Compute ref (dequantized)
        w_ref = (q_w - half_q_val).half() * s_group

        # Restore original shapes
        def reshape_w(w):
            w = w.reshape((size_n, size_k)).contiguous()
            return w

        q_w = reshape_w(q_w)
        w_ref = reshape_w(w_ref)

        # Compute int8 quantization scale for each channel
        s_channel = torch.amax(torch.abs(w_ref), -1, keepdim=True)
        s_channel /= 127.0
        t_int8 = (w_ref / s_channel).round().clamp(-128, 127).to(torch.int8)
        w_ref = t_int8.half() * s_channel
        s_channel = s_channel.reshape(-1, 1).to(dtype=torch.float)

        # Fuse scales
        s_group = (s_group.reshape(size_n, -1).contiguous() / s_channel).to(
            dtype=torch.half
        )
    else:
        max_q_val = 2 ** (num_bits - 1) - 1

        # Compute scale for each channel
        s_channel = torch.amax(torch.abs(w), -1, keepdim=True)
        s_channel /= max_q_val

        # Quantize
        q_w = torch.round(w / s_channel).int()
        q_w = torch.clamp(q_w, -max_q_val, max_q_val)
        # Compute ref (dequantized)
        w_ref = q_w.half() * s_channel

        s_group = torch.tensor([], dtype=torch.half, device=orig_device)
        # div 2 ** (8 - self.bits)) to offset right shift in unpacking
        s_channel /= 2 ** (8 - num_bits)
        s_channel = s_channel.reshape(size_n, -1).contiguous().to(torch.float)

    return q_w, s_group, s_channel, w_ref


def choose_qparams_gguf(
    input: Optional[torch.Tensor],
    block_size: List[int],
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    There are two sets of qparams: quantized_block_scale, quantized_block_min and super_block_scale_scale and super_block_min_scale
    the relationship is the following:
    block_scale = quantized_block_scale * super_block_sclae
    block_min = quantized_block_min * super_block_min
    quantized_val = (float_val - block_min) / block_scale + quant_min
    first we calculate block_scale and block_min
    then we calculate super_block_scale_scale and super_block_min_scale
    after that we can calculate quantized_block_scale and quantized_min_scale
    the returned values are: super_block_scale_scale, super_block_min_scale, quantized_block_scale
    and quantized_min_scale
    """
    dtype = input.dtype

    # 1. get block_scale block_min
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    input = input.view(shape_for_reduction)
    min_val = torch.amin(input, dim=reduction_dims, keepdim=False)
    max_val = torch.amax(input, dim=reduction_dims, keepdim=False)
    quant_max = 15
    quant_min = 0
    # asymmetric quant to fully utilize the range
    block_scale = max_val / (float(quant_max - quant_min) / 2)
    block_scale = (max_val - min_val) / float(quant_max - quant_min)
    block_min = min_val

    # 2. get super_block_scale_scale and super_block_min_scale
    assert _GGUF_QK_K % block_size[-1] == 0
    super_block_size = (1, _GGUF_QK_K // block_size[-1])
    shape_for_reduction, reduction_dims = _get_reduction_params(
        super_block_size, block_scale.size()
    )
    block_scale = block_scale.view(shape_for_reduction)
    block_min = block_min.view(shape_for_reduction)

    shape_after_reduction = shape_for_reduction.copy()
    for i in reduction_dims:
        shape_after_reduction[i] = 1

    block_scale_absmax = torch.amax(
        torch.abs(block_scale), dim=reduction_dims, keepdim=False
    )
    block_min_absmax = torch.amax(
        torch.abs(block_min), dim=reduction_dims, keepdim=False
    )

    # 2. get super_block_scale_scale and super_block_min_scale
    # TODO: make this configurable
    # we also quantize the quantization parameters (scale and min) for each block to 6 bit
    # for Q4_K
    qparam_quant_max = 2**6 - 1
    qparam_quant_min = 0
    super_block_scale_scale = block_scale_absmax / float(
        qparam_quant_max - qparam_quant_min
    )
    super_block_min_scale = block_min_absmax / float(
        qparam_quant_max - qparam_quant_min
    )
    super_block_scale_scale_view = super_block_scale_scale.view(shape_after_reduction)
    super_block_min_scale_view = super_block_min_scale.view(shape_after_reduction)

    # 3. quantize block scale and min are stored in 6 bits using super_block_scale_scale and super_block_min_scale
    quantized_block_scale = torch.clamp(
        block_scale / super_block_scale_scale_view, qparam_quant_min, qparam_quant_max
    )
    quantized_block_min = torch.clamp(
        block_min / super_block_min_scale_view, qparam_quant_min, qparam_quant_max
    )
    return (
        super_block_scale_scale.to(dtype),
        super_block_min_scale.to(dtype),
        quantized_block_scale.to(dtype),
        quantized_block_min.to(dtype),
    )


def quantize_gguf(
    input: torch.Tensor,
    block_size: List[int],
    target_dtype: torch.dtype,
    super_block_scale_scale: torch.Tensor,
    super_block_min_scale: torch.Tensor,
    quantized_block_scale: torch.Tensor,
    quantized_block_min: torch.Tensor,
) -> torch.Tensor:
    assert target_dtype == torch.uint4

    # step 1: first order quantization
    # just going through shape calculation for block_scale and block_min to get the correct shape
    input_shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    block_qparam_shape_after_reduction = input_shape_for_reduction.copy()
    for i in reduction_dims:
        block_qparam_shape_after_reduction[i] = 1
    original_shape = input.shape
    input = input.view(input_shape_for_reduction)
    quantized_block_scale = quantized_block_scale.view(
        block_qparam_shape_after_reduction
    )
    quantized_block_min = quantized_block_min.view(block_qparam_shape_after_reduction)

    # step 2: second order quantization, recover unquantized block_scale and block_min
    super_block_size = (1, _GGUF_QK_K // block_size[-1], 1)
    super_block_input_shape_for_reduction, reduction_dims = _get_reduction_params(
        super_block_size, quantized_block_scale.size()
    )
    super_block_qparam_shape_after_reduction = (
        super_block_input_shape_for_reduction.copy()
    )
    for i in reduction_dims:
        super_block_qparam_shape_after_reduction[i] = 1

    quantized_block_scale = quantized_block_scale.view(
        super_block_input_shape_for_reduction
    )
    quantized_block_min = quantized_block_min.view(
        super_block_input_shape_for_reduction
    )
    super_block_scale_scale = super_block_scale_scale.view(
        super_block_qparam_shape_after_reduction
    )
    super_block_min_scale = super_block_min_scale.view(
        super_block_qparam_shape_after_reduction
    )

    block_scale = super_block_scale_scale * quantized_block_scale
    block_min = super_block_min_scale * quantized_block_min

    # step 3: quantization with the unquantized block_scale and block_min
    block_scale = block_scale.view(block_qparam_shape_after_reduction)
    block_min = block_min.view(block_qparam_shape_after_reduction)
    int_data = (input - block_min) / block_scale
    int_data = int_data.view(original_shape)

    return int_data


def dequantize_gguf(
    input: torch.Tensor,
    block_size: List[int],
    target_dtype: torch.dtype,
    super_block_scale_scale: torch.Tensor,
    super_block_min_scale: torch.Tensor,
    quantized_block_scale: torch.Tensor,
    quantized_block_min: torch.Tensor,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    # step 1. reshape input and quantized block scale and min to the shape
    # after first quantization
    input_shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    block_qparam_shape_after_reduction = input_shape_for_reduction.copy()
    for i in reduction_dims:
        block_qparam_shape_after_reduction[i] = 1

    original_shape = input.shape
    input = input.view(input_shape_for_reduction)
    quantized_block_scale = quantized_block_scale.view(
        block_qparam_shape_after_reduction
    )
    quantized_block_min = quantized_block_min.view(block_qparam_shape_after_reduction)

    # step 2. calculate and reshape block_qparams for second quantization step
    super_block_size = (1, _GGUF_QK_K // block_size[-1], 1)
    super_block_input_shape_for_reduction, reduction_dims = _get_reduction_params(
        super_block_size, quantized_block_scale.size()
    )
    super_block_qparam_shape_after_reduction = (
        super_block_input_shape_for_reduction.copy()
    )
    for i in reduction_dims:
        super_block_qparam_shape_after_reduction[i] = 1
    quantized_block_scale = quantized_block_scale.view(
        super_block_input_shape_for_reduction
    )
    quantized_block_min = quantized_block_min.view(
        super_block_input_shape_for_reduction
    )
    super_block_scale_scale = super_block_scale_scale.view(
        super_block_qparam_shape_after_reduction
    )
    super_block_min_scale = super_block_min_scale.view(
        super_block_qparam_shape_after_reduction
    )

    block_scale = super_block_scale_scale * quantized_block_scale
    block_min = super_block_min_scale * quantized_block_min

    # step 3. dequantize with block_scale and block_min
    block_scale = block_scale.view(block_qparam_shape_after_reduction)
    block_min = block_min.view(block_qparam_shape_after_reduction)
    dequant = input * block_scale + block_min
    dequant = dequant.view(original_shape)
    if output_dtype is not None:
        dequant = dequant.to(output_dtype)

    return dequant


def dequantize_affine_qqq(
    w: torch.Tensor,
    s_group: torch.Tensor,
    s_channel: torch.Tensor,
    num_bits: int,
    group_size: int,
    output_dtype: Optional[torch.dtype] = None,
):
    assert num_bits == 4, f"Unsupported num_bits = {num_bits}"
    size_n, size_k = w.shape
    assert group_size in [-1, 128, size_k], f"Unsupported groupsize = {group_size}"
    if group_size == -1:
        group_size = size_k

    if group_size < size_k:
        # Reshape to [-1, group_size]
        w = w.reshape((-1, group_size))

        max_q_val = 2**num_bits - 1
        half_q_val = (max_q_val + 1) // 2

        s_group = s_group * s_channel.half()
        w_dq = (w - half_q_val).half() * s_group.reshape(-1, 1)

        # Restore original shapes
        def reshape_w(w):
            w = w.reshape((size_n, size_k)).contiguous()
            return w

        w_dq = reshape_w(w_dq)

    else:
        s_channel = s_channel * (2 ** (8 - num_bits))
        w_dq = w.half() * s_channel

    if output_dtype is None:
        w_dq = w_dq.to(torch.float16)
    else:
        w_dq = w_dq.to(output_dtype)

    return w_dq


# HQQ
############################################################################
# Shrinking operator (proximal operator for the lp norm)
def _shrink_lp_op(x: torch.Tensor, beta: float, lp_norm: float) -> torch.Tensor:
    if lp_norm == 1:
        return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
    else:
        return torch.sign(x) * torch.nn.functional.relu(
            torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1)
        )


# Proximal solver || W - dequantize(quantize(W))||_p^p
@torch.inference_mode()
def optimize_weights_proximal_legacy(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    min_max: list,
    axis: int = 0,
    dtype: Union[torch.dtype, None] = None,
    device: Union[str, None] = None,
    verbose: bool = False,
    opt_params: dict = {
        "lp_norm": 0.7,
        "beta": 1e1,
        "kappa": 1.01,
        "iters": 20,
        "early_stop": True,
    },
) -> tuple:
    lp_norm, beta, kappa, iters, early_stop = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
        opt_params["early_stop"],
    )

    device = tensor.device if (device is None) else torch.device(device)

    if dtype is None:
        dtype = torch.float16 if (device.type == "cuda") else torch.float32

    W_f = tensor.to(dtype=dtype, device=device)
    scale = scale.to(dtype=dtype, device=device)
    zero = zero.to(dtype=dtype, device=device)

    best_error = 1e4
    for i in range(iters):
        W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale
        W_e = _shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        beta *= kappa

        current_error = float(torch.abs(W_f - W_r).mean())
        if verbose:
            print("Iter " + str(i + 1), " | Error: " + str(current_error))
        if early_stop:
            if current_error < best_error:
                best_error = current_error
            else:
                break

    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r, W_e
    torch.cuda.empty_cache()

    W_q = torch.round(tensor * scale + zero).clamp(min_max[0], min_max[1])
    return W_q, scale, zero


# Mainly used to check if the group-size is divisible by numel()
def _is_divisible(val1: int, val2: int) -> bool:
    return int(val2 * math.ceil(val1 / val2)) == val1


# Converts hqq format W_dequant = (W_q - zero)*scale into affinequantized format: (W_q - mid_point)*scale_ao + zero_ao
def _convert_to_affinequantized_format(
    W_q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    nbits: int,
    shape: Union[List, Tuple, torch.Size],
) -> Tuple:
    quant_min = 0
    quant_max = 2**nbits - 1
    mid_point = (quant_max + quant_min + 1) / 2
    zero_ao = ((mid_point - zero.float()) * scale.float()).to(zero.dtype)
    scale_ao = scale
    W_q_ao = W_q.view(shape)
    return W_q_ao, scale_ao, zero_ao


# Main hqq quantizer function
def choose_qparams_and_quantize_affine_hqq(
    tensor: torch.Tensor,
    nbits: float = 4,
    group_size: int = 64,
    optimize: bool = True,
    axis: int = 1,
    compute_dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    verbose: bool = False,  # to check the optimizer error
    raw_output: bool = False,  # If True, it will return the quant params in hqq lib format
    optimize_weights: Callable = optimize_weights_proximal_legacy,  # weights proximal optimizer function
) -> tuple:
    assert axis in [0, 1], "axis should be either 0 or 1"
    if group_size is not None:
        assert _is_divisible(tensor.numel(), group_size), (
            "group_size should be divisble by the total tensor dimensions. shape: "
            + str(tensor.shape)
            + ", group_size: "
            + str(group_size)
        )

    # It's better to work with float32 here
    W = tensor.to(device=device, dtype=torch.float32)
    shape = W.shape

    # Reshape for grouping
    if group_size is not None:
        W = W.reshape([-1, group_size]) if (axis == 1) else W.reshape([group_size, -1])

    # Get min/max values
    _min = W.min(axis=axis, keepdim=True)[0]
    _max = W.max(axis=axis, keepdim=True)[0]

    max_v = round(2**nbits - 1)
    min_v = 0
    min_max = [min_v, max_v]

    # Clamp to avoid fp16 issues
    scale = (max_v / (_max - _min)).clamp(max=2e4)
    zero = -_min * scale

    # Round zero as in: https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py#L42C9-L42C14
    if nbits in [4]:
        zero = torch.round(zero)

    # Fine-tune weights
    if optimize:
        W_q, scale, zero = optimize_weights(
            tensor=W,
            scale=scale,
            zero=zero,
            min_max=min_max,
            axis=axis,
            device=device,
            verbose=verbose,
        )
    else:
        zero = zero.to(compute_dtype)
        scale = scale.to(compute_dtype)
        W_q = torch.round(W * scale + zero).clamp(min_max[0], min_max[1])

    # Store meta-data (we invert the scale for dequantization)
    scale = 1.0 / scale

    # Convert to TensorCoreTiled format
    # TODO move the conversion of zero_point out of quant_primitives
    # and into TensorCoreTiledLayout.from_plain and rename this
    # helper function correctly.
    if raw_output is False:
        W_q, scale, zero = _convert_to_affinequantized_format(
            W_q, scale, zero, nbits, shape
        )
    else:
        # this path was not used before, the way hqq sets up scale/zero is transposed
        # compared to the rest of our utils so we need to reshape them acccordingly.
        W_q = W_q.reshape(shape)
        if axis == 1:
            scale = scale.reshape(shape[0], -1)
            zero = zero.reshape(shape[0], -1)
        else:
            scale = scale.reshape(-1, shape[-1])
            zero = zero.reshape(-1, shape[-1])
    # Make sure all the weights are in the right compute_dtype/device
    W_q = W_q.to(dtype=torch.uint8, device=device)
    scale = scale.to(dtype=compute_dtype, device=device)
    zero = zero.to(dtype=compute_dtype, device=device)

    # cleanup
    del W, _min, _max
    torch.cuda.empty_cache()

    return W_q, scale, zero, shape


def choose_qparams_affine_floatx(
    tensor: torch.Tensor, ebits: int, mbits: int
) -> torch.Tensor:
    # _n_ones() is not compatible with torch.compile() due to << operator
    # https://github.com/pytorch/pytorch/issues/119152
    # exp_bias = _n_ones(ebits - 1)
    # max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2 ** mbits))

    # workaround: global lookup table
    exp_bias = _ONES_TABLE[ebits - 1]
    max_normal = 2 ** (_ONES_TABLE[ebits] - exp_bias) * (
        _ONES_TABLE[mbits + 1] / (2**mbits)
    )

    dtype = tensor.dtype
    tensor = tensor.float()
    scale = tensor.abs().amax(1).clamp(min=1e-12) / max_normal
    return scale.to(dtype)


def quantize_affine_floatx(
    tensor: torch.Tensor, scale: torch.Tensor, ebits: int, mbits: int
) -> torch.Tensor:
    """Quantizes the float32 high precision floating point tensor to low precision floating point number and
    converts the result to unpacked floating point format with the format of 00SEEEMM (for fp6_e3m2) where S means sign bit, e means exponent bit and m means mantissa bit
    """
    tensor = tensor.float()
    tensor_floatx = _f32_to_floatx_unpacked(tensor / scale.view(-1, 1), ebits, mbits)
    return tensor_floatx


def dequantize_affine_floatx(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    ebits: int,
    mbits: int,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensor = _floatx_unpacked_to_f32(tensor, ebits, mbits)
    tensor = tensor * scale.float().view(-1, 1)
    tensor = tensor.to(dtype=output_dtype)
    return tensor


def choose_qparams_affine_float8(
    tensor: torch.Tensor,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """
    Calculates float8 scaling factor for the given high precision tensor, using tensorwise granularity.

    Args:
        tensor (torch.Tensor): Input tensor to be quantized.
        float8_dtype (torch.dtype): Data type of the quantized tensor (e.g., torch.float8_e4m3fn, torch.float8_e5m2).
    """
    # only tensorwise scaling is supported for now:
    quant_min, quant_max = torch.finfo(float8_dtype).min, torch.finfo(float8_dtype).max
    min_val_neg = torch.min(tensor)
    max_val_pos = torch.max(tensor)
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    return scale.to(dtype=torch.float32)


def quantize_affine_float8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """
    Quantizes the high precision floating point tensor to a float8 tensor, using the given scaling factor.

    Args:
        tensor (torch.Tensor): Input tensor to be quantized.
        scale (torch.Tensor): Scaling factor for the quantization.
        float8_dtype (torch.dtype): Data type of the quantized tensor (e.g., torch.float8_e4m3fn, torch.float8_e5m2).
    """
    # Note: when the line below is compiled with `torch.compile`, `tensor` is automatically
    # upcasted to `float32` to multiply with the scale, since scale is a fp32 tensor in float8 quantization.
    # In order to match numerics between eager and compile, we upcast manually here.
    tensor_scaled = tensor.to(torch.float32) / scale
    max_value = torch.finfo(float8_dtype).max
    tensor_clamped = tensor_scaled.clamp(min=-max_value, max=max_value)
    fp8_tensor = tensor_clamped.to(float8_dtype)
    return fp8_tensor


def dequantize_affine_float8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantizes the float8 tensor to high precision tensor.

    Args:
        tensor (torch.Tensor): Input float8 tensor to be dequantized.
        scale (torch.Tensor): Scaling factor for the dequantization.
        output_dtype (torch.dtype): Data type of the output tensor (e.g., torch.float32).
    """
    # Note: when the line below is compiled with `torch.compile`, `tensor` is automatically
    # upcasted to `float32` to divide by the scale, since scale is a fp32 for float8 quantization.
    # In order to match numerics between eager and compile, we upcast manually here.
    fp8_tensor = tensor.to(torch.float32)
    hp_tensor = fp8_tensor * scale
    return hp_tensor.to(output_dtype)
