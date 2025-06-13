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
    _register_custom_op,
)

__all__ = [
    "choose_qparams_affine",
    "choose_qparams_affine_tinygemm",
    "choose_qparams_affine_dont_preserve_zero",
    "choose_qparams_affine_with_min_max",
    "choose_qparams_affine_floatx",
    "quantize_affine",
    "quantize_affine_no_zero_point",
    "quantize_affine_tinygemm",
    "dequantize_affine",
    "dequantize_affine_no_zero_point",
    "dequantize_affine_tinygemm",
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
    """Defines how floating point values are mapped to quantized integer values.

    This enum specifies the quantization mapping strategy used to convert high-precision
    floating point numbers to low-precision integer representations.

    Attributes:
        SYMMETRIC: Symmetric mapping where the floating point range is symmetrically
            mapped to the integer range. The scale is calculated as:
            ``scale = max(|min_val|, |max_val|) / (quant_max - quant_min) * 2``

            Example:
                - Float range: (-3.5, 10.2), int4 range: (-8, 7)
                - Symmetric range: (-10.2, 10.2) mapped to (-8, 7)
                - Scale: (10.2 - (-10.2)) / (7 - (-8)) = 1.36

        SYMMETRIC_NO_CLIPPING_ERR: Variant of symmetric mapping that reduces clipping
            errors by calculating separate scales for positive and negative values:
            ``smin = min_val_neg / quant_min``, ``smax = max_val_pos / quant_max``
            ``scale = max(smin, smax)``
            This approach provides better coverage and less rounding error.

        ASYMMETRIC: Asymmetric mapping where the floating point range is directly
            mapped to the integer range without symmetry constraints:
            ``scale = (max_val - min_val) / (quant_max - quant_min)``

            Example:
                - Float range: (-3.5, 10.2) directly mapped to int4 range: (-8, 7)
                - Scale: (10.2 - (-3.5)) / (7 - (-8)) = 0.913

    Note:
        Symmetric quantization typically preserves zero exactly but may waste quantization
        levels for asymmetric data. Asymmetric quantization uses the full quantization
        range but may not preserve zero exactly.
    """

    SYMMETRIC = auto()
    SYMMETRIC_NO_CLIPPING_ERR = auto()
    ASYMMETRIC = auto()


class ZeroPointDomain(Enum):
    """Specifies the domain (integer or float) in which the zero_point is represented.

    This enum determines how the zero_point parameter is interpreted during quantization
    and dequantization operations, affecting the mathematical formulation used.

    Attributes:
        INT: Zero point is in the integer domain. The quantization formula is:
            ``quantized_val = round(float_val / scale) + zero_point``
            where both quantized_val and zero_point are integers.

            Dequantization formula:
            ``float_val = (quantized_val - zero_point) * scale``

        FLOAT: Zero point is in the floating point domain. Used by specialized kernels
            like TinyGemm. The quantization formula is:
            ``quantized_val = round((float_val - zero_point_offset) / scale)``
            where ``zero_point_offset = zero_point - scale * mid_point``

            Dequantization formula:
            ``float_val = quantized_val * scale + zero_point_offset``

        NONE: No zero point is used (symmetric quantization). The formula is:
            ``quantized_val = round(float_val / scale)``

            Dequantization formula:
            ``float_val = quantized_val * scale``

    Note:
        INT domain is most common and straightforward. FLOAT domain is used for
        specialized CUDA kernels that expect the zero point in floating point form.
        NONE is used for symmetric quantization schemes.
    """

    INT = auto()
    FLOAT = auto()
    NONE = auto()


class TorchAODType(Enum):
    """Placeholder dtypes for sub-byte integer types not yet available in PyTorch core.

    This enum provides integer data types with bit widths from 1 to 7 bits, which are
    commonly used in quantization but may not be natively supported in all PyTorch
    versions. These are used internally for quantization parameter calculations and
    bound checking.

    Attributes:
        INT1: 1-bit signed integer, range: [-1, 0]
        INT2: 2-bit signed integer, range: [-2, 1]
        INT3: 3-bit signed integer, range: [-4, 3]
        INT4: 4-bit signed integer, range: [-8, 7]
        INT5: 5-bit signed integer, range: [-16, 15]
        INT6: 6-bit signed integer, range: [-32, 31]
        INT7: 7-bit signed integer, range: [-64, 63]

    Note:
        Starting from PyTorch 2.6, torch.int1 to torch.int7 are natively supported.
        These TorchAODType enums will remain for backward compatibility with older
        PyTorch versions. The actual quantized tensors are typically stored using
        standard PyTorch dtypes like torch.uint8 with appropriate packing.
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


class _Round(torch.autograd.Function):
    """Differentiable rounding operation using Straight-Through Estimator (STE).

    This class implements a custom autograd function that performs rounding in the
    forward pass while using the straight-through estimator for the backward pass.
    The STE assumes the derivative of the rounding function is the identity function,
    allowing gradients to flow through quantization operations during training.

    The forward pass computes: ``output = round(input)``
    The backward pass uses STE: ``grad_input = grad_output`` (identity)

    This is commonly used in quantization-aware training where we need gradients
    to flow through the quantization process despite the non-differentiable
    rounding operation.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: apply rounding operation.

        Args:
            ctx: Context object for storing information for backward pass
            x: Input tensor to be rounded

        Returns:
            Rounded tensor with same shape and device as input
        """
        return torch.round(x)

    @staticmethod
    def backward(ctx, gy: torch.Tensor) -> torch.Tensor:
        """Backward pass: apply straight-through estimator.

        Args:
            ctx: Context object (not used in this implementation)
            gy: Gradient of the output

        Returns:
            Gradient with respect to input (same as output gradient)
        """
        return gy


# TODO: decide on if we want to allow custom quant_min/quant_max here
def _get_and_check_qmin_qmax(dtype, quant_min, quant_max):
    """Get and validate quantization bounds based on dtype.

    This function determines the valid quantization range for a given dtype and
    validates that user-provided quant_min/quant_max values are within acceptable
    bounds. If quant_min or quant_max are None, they are set to the dtype's
    natural bounds.

    Args:
        dtype: Target quantization dtype (e.g., torch.int8, torch.uint4, TorchAODType.INT4)
        quant_min: Optional minimum quantization value. If None, uses dtype minimum.
        quant_max: Optional maximum quantization value. If None, uses dtype maximum.

    Returns:
        tuple: (validated_quant_min, validated_quant_max)

    Raises:
        ValueError: If dtype is not supported
        AssertionError: If provided quant_min/quant_max are outside valid bounds

    Example:
        >>> _get_and_check_qmin_qmax(torch.int8, None, None)
        (-128, 127)
        >>> _get_and_check_qmin_qmax(TorchAODType.INT4, -7, 7)
        (-7, 7)
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
    """Calculate tensor reshaping parameters for block-wise quantization operations.

    This function determines how to reshape a tensor to enable block-wise operations
    (like computing min/max values within blocks) and which dimensions to reduce over.
    It's used internally by quantization functions to support different granularities.

    The function splits dimensions where block_size < input_size into two dimensions:
    one for the number of blocks and one for the block size itself.

    Args:
        block_size: Tuple specifying the granularity of quantization blocks
        input_size: Tuple specifying the input tensor dimensions

    Returns:
        tuple: (shape_for_reduction, reduction_dims) where:
            - shape_for_reduction: New shape for tensor.view() to enable block operations
            - reduction_dims: List of dimension indices to reduce over

    Example:
        For per-group quantization with group_size=2 along the last dimension:

        >>> _get_reduction_params((3, 3, 2, 10), (3, 3, 10, 10))
        ([3, 3, 5, 2, 10], [1, 3])

        This reshapes a (3,3,10,10) tensor to (3,3,5,2,10) where:
        - Dimension 2: 10 -> (5,2) for 5 blocks of size 2
        - Dimensions [1,3] will be reduced (corresponding to the block dimensions)

    Note:
        This function is essential for implementing various quantization granularities:
        - Per-tensor: block_size equals input_size
        - Per-channel: block_size has 1 in the channel dimension
        - Per-group: block_size specifies group size in relevant dimensions
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
) -> torch.Tensor:
    """Quantize a floating point tensor using affine quantization.

    This function performs affine (linear) quantization, converting high-precision
    floating point values to low-precision integer values using the formula:
    ``quantized_val = clamp(round(input / scale) + zero_point, quant_min, quant_max)``

    The quantization can be applied at different granularities (per-tensor, per-channel,
    or per-group) as specified by the block_size parameter.

    Args:
        input: Input tensor to quantize. Must be float32, float16, or bfloat16.
        block_size: Granularity of quantization. Specifies the size of tensor elements
            that share the same quantization parameters (scale and zero_point).
        scale: Scaling factor for quantization. Shape must be compatible with block_size.
        zero_point: Zero point offset for quantization. If None, assumes zero offset.
            Shape must be compatible with block_size when provided.
        output_dtype: Target quantization dtype (e.g., torch.int8, torch.uint4).
        quant_min: Minimum allowed quantized value. If None, derived from output_dtype.
        quant_max: Maximum allowed quantized value. If None, derived from output_dtype.

    Returns:
        Quantized tensor with the specified output_dtype and same shape as input.

    Example:
        >>> # Per-tensor quantization of a 2D tensor
        >>> input = torch.randn(4, 8, dtype=torch.float32)
        >>> scale = torch.tensor(0.1)
        >>> zero_point = torch.tensor(128)
        >>> quantized = quantize_affine(
        ...     input, (4, 8), scale, zero_point, torch.uint8
        ... )

        >>> # Per-channel quantization along axis 0
        >>> scale = torch.randn(4) * 0.1
        >>> zero_point = torch.full((4,), 128)
        >>> quantized = quantize_affine(
        ...     input, (1, 8), scale, zero_point, torch.uint8
        ... )

    Note:
        **Block Size Granularity Examples:**
        For a tensor of size (3, 3, 10, 10), block_size represents different granularities:

        =====================================  ==================
        Granularity Type                       block_size
        =====================================  ==================
        per_tensor                            (3, 3, 10, 10)
        per_axis (axis=0)                     (1, 3, 10, 10)
        per_axis (axis=1)                     (3, 1, 10, 10)
        per_group (groupsize=2) on axis=2     (3, 3, 2, 10)
        per_group (groupsize=2) on axis=3     (3, 3, 10, 2)
        =====================================  ==================

        This function uses integer domain zero points (ZeroPointDomain.INT).
        For floating point domain zero points, use :func:`quantize_affine_tinygemm`.
    """
    return _quantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        output_dtype,
        quant_min,
        quant_max,
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
) -> torch.Tensor:
    """op definition that has compatible signatures with custom op library

    Note:
        zero_point_domain is pre-defined specifies how we quantize the floating point to quantized data:
        INT: quantized_val = (float_val / scale) (integer) + zero_point (integer)
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(output_dtype, quant_min, quant_max)
    # workaround for uintx dtypes, since we don't have native Uintx dtype connected with
    # torch.uintx dtypes yet
    if output_dtype in _SUB_BYTE_UINT_BOUNDS:
        output_dtype = torch.uint8
    return _quantize_affine_no_dtype_cast(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
    ).to(output_dtype)


def _quantize_affine_no_dtype_cast(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Union[int, float],
    quant_max: Union[int, float],
) -> torch.Tensor:
    """
    The op does the following:
    1. figure out the dimension for reduction based on block_size, also reshape the input to align with
       the shape after reduction
    2. quantize the input based on the quantization parameters scale and zero_point and zero_point_domain = INT
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

    quant = torch.clamp(
        _Round.apply(input * (1.0 / scale)) + zero_point, quant_min, quant_max
    )
    quant = quant.view(original_shape)

    return quant


def quantize_affine_tinygemm(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_min: Optional[Union[int, float, bool]] = None,
    quant_max: Optional[Union[int, float, bool]] = None,
) -> torch.Tensor:
    """Quantize tensor using TinyGemm-compatible affine quantization with float zero points.

    This function performs affine quantization optimized for TinyGemm CUDA kernels.
    Unlike standard affine quantization, it uses floating point domain zero points,
    which is required by certain optimized matrix multiplication kernels.

    The quantization formula is:
    ``quantized_val = clamp(round((input - zero_point_offset) / scale), quant_min, quant_max)``
    where ``zero_point_offset = zero_point - scale * mid_point``
    and ``mid_point = (quant_max + quant_min + 1) / 2``

    Args:
        input: Input tensor to quantize. Must be float32, float16, or bfloat16.
        block_size: Granularity of quantization block sizes.
        scale: Scaling factor for quantization. Shape must be compatible with block_size.
        zero_point: Zero point in floating point domain. If None, assumes zero offset.
        output_dtype: Target quantization dtype (e.g., torch.int4, torch.uint8).
        quant_min: Minimum allowed quantized value. If None, derived from output_dtype.
        quant_max: Maximum allowed quantized value. If None, derived from output_dtype.

    Returns:
        Quantized tensor with specified output_dtype and same shape as input.

    Note:
        This function uses floating point domain zero points (ZeroPointDomain.FLOAT),
        which differs from the standard integer domain used in :func:`quantize_affine`.
        This format is specifically designed for compatibility with optimized CUDA
        kernels like TinyGemm that expect zero points in floating point format.

        The zero point transformation accounts for the mid-point offset to ensure
        proper quantization behavior compatible with tensor core operations.
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(output_dtype, quant_min, quant_max)
    # workaround for uintx dtypes, since we don't have native Uintx dtype connected with
    # torch.uintx dtypes yet
    if output_dtype in _SUB_BYTE_UINT_BOUNDS:
        output_dtype = torch.uint8
    return _quantize_affine_tinygemm_no_dtype_cast(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
    ).to(output_dtype)


def _quantize_affine_tinygemm_no_dtype_cast(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
) -> torch.Tensor:
    """
    The op does the following:
    1. figure out the dimension for reduction based on block_size, also reshape the input to align with
       the shape after reduction
    2. quantize the input based on the quantization parameters scale and zero_point and zero_point_domain = FLOAT
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

    mid_point = (quant_max + quant_min + 1) / 2
    min_val = zero_point - scale * mid_point
    quant = torch.clamp(_Round.apply((input - min_val) / scale), quant_min, quant_max)
    quant = quant.view(original_shape)

    return quant


def quantize_affine_no_zero_point(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_min: Optional[Union[int, float, bool]] = None,
    quant_max: Optional[Union[int, float, bool]] = None,
) -> torch.Tensor:
    """Quantize tensor using symmetric (zero point-free) affine quantization.

    This function performs symmetric quantization without zero points, suitable for
    cases where the data distribution is symmetric around zero or when zero point
    adjustment is not needed. This is commonly used for floating point quantization
    formats where exact integer representation is not required.

    The quantization formula is:
    ``quantized_val = clamp(round(input / scale), quant_min, quant_max)``

    Args:
        input: Input tensor to quantize. Must be float32, float16, or bfloat16.
        block_size: Granularity of quantization block sizes.
        scale: Scaling factor for quantization. Shape must be compatible with block_size.
        zero_point: Ignored parameter (maintained for API compatibility). Should be None.
        output_dtype: Target quantization dtype (e.g., torch.int8, custom float formats).
        quant_min: Minimum allowed quantized value. If None, derived from output_dtype.
        quant_max: Maximum allowed quantized value. If None, derived from output_dtype.

    Returns:
        Quantized tensor with specified output_dtype and same shape as input.

    Note:
        This function uses ZeroPointDomain.NONE, meaning no zero point adjustment
        is applied. This is primarily used for:
        - Symmetric quantization where data is centered around zero
        - Custom floating point formats (floatx) where exact integer mapping isn't needed
        - Cases where computational simplicity is preferred over quantization accuracy

        For asymmetric data that requires zero point adjustment, use :func:`quantize_affine`
        or :func:`quantize_affine_tinygemm` instead.
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(output_dtype, quant_min, quant_max)
    # workaround for uintx dtypes, since we don't have native Uintx dtype connected with
    # torch.uintx dtypes yet
    if output_dtype in _SUB_BYTE_UINT_BOUNDS:
        output_dtype = torch.uint8
    return _quantize_affine_no_zero_point_no_dtype_cast(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
    ).to(output_dtype)


def _quantize_affine_no_zero_point_no_dtype_cast(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
) -> torch.Tensor:
    """
    The op does the following:
    1. figure out the dimension for reduction based on block_size, also reshape the input to align with
       the shape after reduction
    2. quantize the input based on the quantization parameters scale and zero_point and zero_point_domain = NONE
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

    quant = torch.clamp(_Round.apply(input * (1.0 / scale)), quant_min, quant_max)
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
    *,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a quantized tensor back to floating point using affine dequantization.

    This function performs the inverse of :func:`quantize_affine`, converting low-precision
    quantized values back to high-precision floating point values using the formula:
    ``dequantized_val = (quantized_val - zero_point) * scale``

    Args:
        input: Quantized input tensor to dequantize. Should match input_dtype.
        block_size: Granularity of quantization that was used during quantization.
            Specifies the size of tensor elements that share the same quantization parameters.
        scale: Scaling factor used during quantization. Shape must be compatible with block_size.
        zero_point: Zero point offset used during quantization. If None, assumes zero offset.
            Shape must be compatible with block_size when provided.
        input_dtype: Original quantized dtype (e.g., torch.int8, torch.uint4) for validation.
        quant_min: Minimum quantized value that was used. If None, derived from input_dtype.
        quant_max: Maximum quantized value that was used. If None, derived from input_dtype.
        output_dtype: Target floating point dtype for output. Defaults to torch.float32.

    Returns:
        Dequantized tensor with specified output_dtype and same shape as input.

    Example:
        >>> # Dequantize a per-tensor quantized tensor
        >>> quantized = torch.randint(0, 255, (4, 8), dtype=torch.uint8)
        >>> scale = torch.tensor(0.1)
        >>> zero_point = torch.tensor(128)
        >>> dequantized = dequantize_affine(
        ...     quantized, (4, 8), scale, zero_point, torch.uint8
        ... )

        >>> # Dequantize per-channel quantized tensor
        >>> scale = torch.randn(4) * 0.1
        >>> zero_point = torch.full((4,), 128)
        >>> dequantized = dequantize_affine(
        ...     quantized, (1, 8), scale, zero_point, torch.uint8
        ... )

    Note:
        This function assumes integer domain zero points (ZeroPointDomain.INT).
        For floating point domain zero points, use :func:`dequantize_affine_tinygemm`.
        For symmetric quantization without zero points, use :func:`dequantize_affine_no_zero_point`.

        The block_size must match the granularity used during the original quantization
        for correct dequantization results.
    """
    return _dequantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        input_dtype,
        quant_min,
        quant_max,
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
    quant_min, quant_max = _get_and_check_qmin_qmax(input_dtype, quant_min, quant_max)
    return _dequantize_affine_no_dtype_check(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
        output_dtype,
    )


def _dequantize_affine_no_dtype_check(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Union[int, float],
    quant_max: Union[int, float],
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

    # Force a copy to avoid input modification due
    # to upcoming in-place operations.
    dequant = input.to(output_dtype, copy=True)
    if zero_point is not None:
        dequant = dequant - zero_point.to(output_dtype)
    dequant = dequant * scale

    return dequant.view(original_shape).to(output_dtype)


def _dequantize_affine_no_zero_point_no_dtype_check(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Union[int, float],
    quant_max: Union[int, float],
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

    assert zero_point is None, (
        "zero_point should be None for dequantize_affine_no_zero_point"
    )
    dequant = input.to(output_dtype)
    dequant = dequant * scale

    return dequant.view(original_shape).to(output_dtype)


def dequantize_affine_no_zero_point(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    input_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Args:
      input (torch.Tensor): quantized tensor, should match the dtype `dtype` argument
      block_size: (List[int]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
                               e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (Tensor): quantization parameter for affine quantization
      zero_point (Tensor): quantization parameter for affine quantization, no zero point is used for this op
      input_dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for input Tensor
      quant_max (Optional[int]): maximum quantized value for input Tensor
      output_dtype (torch.dtype): dtype for output Tensor, default is fp32

      Default value for zero_point is in integer domain, zero point is added to the quantized integer value during quantization

    Output:
      dequantized Tensor, with requested dtype or fp32
    """
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
    quant_min, quant_max = _get_and_check_qmin_qmax(input_dtype, quant_min, quant_max)
    return _dequantize_affine_no_zero_point_no_dtype_check(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
        output_dtype,
    )


def _dequantize_affine_tinygemm_no_dtype_check(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Union[int, float],
    quant_max: Union[int, float],
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

    # TODO: this seems to be a detail for tinygemm (converting from uint to int, probably need to refactor this)
    mid_point = (quant_max + quant_min + 1) / 2
    # This should allocate new memory and avoid input modification
    dequant = input - mid_point
    dequant = dequant.to(output_dtype)
    dequant *= scale
    if zero_point is not None:
        dequant += zero_point

    return dequant.view(original_shape).to(output_dtype)


def dequantize_affine_tinygemm(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    input_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
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

      Default value for zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)

    Output:
      dequantized Tensor, with requested dtype or fp32
    """
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
    quant_min, quant_max = _get_and_check_qmin_qmax(input_dtype, quant_min, quant_max)
    return _dequantize_affine_tinygemm_no_dtype_check(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
        output_dtype,
    )


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
    """Simulate quantization for Quantization-Aware Training (QAT) without dtype conversion.

    This function performs "fake quantization" by applying quantization and dequantization
    operations while maintaining the original floating point dtype. This is essential for
    QAT where gradients must flow through the quantization process during training.

    The operation is equivalent to:
    ``dequantize_affine(quantize_affine(input, ...), ...)``
    but keeps the tensor in floating point format throughout.

    Args:
        input: Input tensor to fake quantize. Must be float32, float16, or bfloat16.
        block_size: Granularity of quantization. Specifies the size of tensor elements
            that share the same quantization parameters.
        scale: Scaling factor for quantization. Shape must be compatible with block_size.
        zero_point: Zero point offset for quantization. If None when zero_point_domain
            is not NONE, assumes zero offset.
        quant_dtype: Quantization dtype used for determining valid quantization bounds.
        quant_min: Minimum allowed quantized value. If None, derived from quant_dtype.
        quant_max: Maximum allowed quantized value. If None, derived from quant_dtype.
        zero_point_domain: Domain of the zero point (INT, FLOAT, or NONE).
            - INT: Standard integer domain zero points
            - FLOAT: Floating point domain (TinyGemm compatible)
            - NONE: No zero point (symmetric quantization)

    Returns:
        Fake quantized tensor with same dtype and shape as input.

    Example:
        >>> # Basic fake quantization for QAT
        >>> input = torch.randn(4, 8, requires_grad=True)
        >>> scale = torch.tensor(0.1, requires_grad=True)
        >>> zero_point = torch.tensor(128.0)
        >>> fake_quant = fake_quantiza_affine(
        ...     input, (4, 8), scale, zero_point, torch.uint8
        ... )
        >>> # Gradients can flow through fake_quant for training

    Note:
        This function is primarily used during quantization-aware training to simulate
        the effects of quantization while allowing gradient flow. The output maintains
        the same dtype as the input, enabling standard backpropagation.

        For actual quantization (dtype conversion), use :func:`quantize_affine`.
        For outlier detection during QAT, use :func:`fake_quantize_affine_cachemask`.
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
    if zero_point_domain == ZeroPointDomain.INT:
        _quantize_affine = _quantize_affine_no_dtype_cast
        _dequantize_affine = _dequantize_affine_no_dtype_check
    elif zero_point_domain == ZeroPointDomain.FLOAT:
        _quantize_affine = _quantize_affine_tinygemm_no_dtype_cast
        _dequantize_affine = _dequantize_affine_tinygemm_no_dtype_check
    elif ZeroPointDomain == ZeroPointDomain.NONE:
        _quantize_affine = _quantize_affine_no_zero_point_no_dtype_cast
        _dequantize_affine = _dequantize_affine_no_zero_point_no_dtype_check
    else:
        raise ValueError(f"Unrecognized zero point domain: {zero_point_domain}")
    q = _quantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
    )
    dq = _dequantize_affine(
        q,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
        output_dtype=input_dtype,
    )
    return (q, dq)


@torch.no_grad()
def choose_qparams_affine(
    input: torch.Tensor,
    mapping_type: MappingType,
    block_size: Tuple[int],
    target_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = torch.int32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate quantization parameters (scale and zero_point) for affine quantization.

    This function analyzes input tensor statistics and computes optimal scale and zero_point
    values for quantization. The calculation method depends on the mapping_type:

    - **SYMMETRIC**: ``scale = max(|min_val|, |max_val|) / (quant_max - quant_min) * 2``
    - **ASYMMETRIC**: ``scale = (max_val - min_val) / (quant_max - quant_min)``
    - **SYMMETRIC_NO_CLIPPING_ERR**: Uses separate scales for positive/negative values

    Args:
        input: Input tensor for which to calculate quantization parameters.
            Must be float32, float16, or bfloat16.
        mapping_type: Strategy for mapping floating point values to quantized values.
            Determines whether to use symmetric or asymmetric quantization.
        block_size: Granularity of quantization. Specifies the size of tensor elements
            that share the same quantization parameters.
        target_dtype: Target quantization dtype (e.g., torch.int8, torch.uint4).
        quant_min: Minimum allowed quantized value. If None, derived from target_dtype.
        quant_max: Maximum allowed quantized value. If None, derived from target_dtype.
        eps: Minimum scale value to prevent division by zero. If None, uses input.dtype eps.
        scale_dtype: Desired dtype for scale tensor. If None, uses input.dtype.
        zero_point_dtype: Desired dtype for zero_point tensor. Defaults to torch.int32.

    Returns:
        tuple: (scale, zero_point) tensors with specified dtypes and shapes compatible
        with the block_size granularity.

    Example:
        >>> # Per-tensor quantization parameters
        >>> input = torch.randn(4, 8)
        >>> scale, zero_point = choose_qparams_affine(
        ...     input, MappingType.ASYMMETRIC, (4, 8), torch.int8
        ... )

        >>> # Per-channel quantization parameters
        >>> scale, zero_point = choose_qparams_affine(
        ...     input, MappingType.SYMMETRIC, (1, 8), torch.int8
        ... )

    Note:
        This function preserves zero in the quantized representation by default,
        ensuring that zero in the input maps exactly to an integer in the quantized range.
        The calculated parameters can be used with :func:`quantize_affine` and
        :func:`dequantize_affine` for the quantization process.

        For specialized quantization schemes, see:
        - :func:`choose_qparams_affine_tinygemm` for TinyGemm compatibility
        - :func:`choose_qparams_affine_with_min_max` for custom min/max values
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
    )


# TODO: lower this op to custom op library
@torch.no_grad()
def choose_qparams_affine_tinygemm(
    input: torch.Tensor,
    mapping_type: MappingType,
    block_size: Tuple[int],
    target_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Specialized version of choose_qparams_affine

    This is used for tinygemm int4mm kernel where zero point is in floating point domain
    and zero does not have to be exactly representable.

    Args:
        input (torch.Tensor): fp32, bf16, fp16 input Tensor
        mapping_type (MappingType): determines how the qparams are calculated, symmetric or asymmetric
        block_size: (Tuple[int]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
        target_dtype (torch.dtype): dtype for target quantized Tensor
        quant_min (Optional[int]): minimum quantized value for target quantized Tensor
        quant_max (Optioanl[int]): maximum quantized value for target quantized Tensor
        eps (Optional[float]): minimum scale, if not provided, default to eps of input.dtype
        scale_dtype (torch.dtype): dtype for scale Tensor
        zero_point_dtype (torch.dtype): dtype for zero_point Tensor

    Output:
        Tuple of scales and zero_points Tensor with requested dtype
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
    assert mapping_type is MappingType.ASYMMETRIC, (
        f"Unsupported mapping type: {mapping_type}"
    )
    if scale_dtype is None:
        scale_dtype = input.dtype
    if eps is None:
        eps = torch.finfo(input.dtype).eps

    assert len(block_size) == input.dim(), (
        f"Got input dim:{input.dim()}, block_size: {block_size}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    input = input.view(shape_for_reduction)

    min_val = torch.amin(input, dim=reduction_dims, keepdim=False)
    max_val = torch.amax(input, dim=reduction_dims, keepdim=False)

    # For preserve_zero=False, we don't ensure zero is exactly representable
    min_val_neg = min_val
    max_val_pos = max_val

    scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
    scale = torch.clamp(scale, min=eps)

    # For zero_point_domain=FLOAT in asymmetric quantization
    mid_point = (quant_max + quant_min + 1) / 2
    # this is not preserving zero_point, this is converting to TensorCoreTiledFormat
    zero_point = min_val_neg + scale * mid_point

    if zero_point_dtype is None:
        zero_point_dtype = input.dtype

    zero_point = zero_point.to(dtype=zero_point_dtype)
    return scale.to(dtype=scale_dtype, device=input.device), zero_point


# TODO: lower this op to custom op library
def choose_qparams_affine_dont_preserve_zero(
    input: torch.Tensor,
    mapping_type: MappingType,
    block_size: Tuple[int],
    target_dtype: torch.dtype,
    quant_min: Optional[Union[int, float, bool]] = None,
    quant_max: Optional[Union[int, float, bool]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Specialized version of choose_qparams_affine with zero_point_domain=ZeroPointDomain.INT and preserve_zero=False.

    Args:
        input (torch.Tensor): fp32, bf16, fp16 input Tensor
        mapping_type (MappingType): determines how the qparams are calculated, asymmetric only
        block_size: (Tuple[int]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
        target_dtype (torch.dtype): dtype for target quantized Tensor
        quant_min (Optional[int]): minimum quantized value for target quantized Tensor
        quant_max (Optioanl[int]): maximum quantized value for target quantized Tensor
        eps (Optional[float]): minimum scale, if not provided, default to eps of input.dtype
        scale_dtype (torch.dtype): dtype for scale Tensor
        zero_point_dtype (torch.dtype): dtype for zero_point Tensor
        Now removed params default values:
            zero_point_domain (ZeroPointDomain): the domain that zero_point is in, defaults to Integer
            preserve_zero (bool): whether to preserve zero in the quantized Tensor, defaults to False

    Output:
        Tuple of scales and zero_points Tensor with requested dtype
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
    assert mapping_type == MappingType.ASYMMETRIC, (
        f"Unsupported mapping type: {mapping_type}"
    )

    if scale_dtype is None:
        scale_dtype = input.dtype
    if eps is None:
        eps = torch.finfo(input.dtype).eps

    assert len(block_size) == input.dim(), (
        f"Got input dim:{input.dim()}, block_size: {block_size}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    input = input.view(shape_for_reduction)

    min_val = torch.amin(input, dim=reduction_dims, keepdim=False)
    max_val = torch.amax(input, dim=reduction_dims, keepdim=False)

    # For no preserve zero, we don't ensure zero is exactly representable
    min_val_neg = min_val
    max_val_pos = max_val

    scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
    scale = torch.clamp(scale, min=eps)
    # Zero point is int
    zero_point = quant_min - _Round.apply(min_val_neg / scale)
    zero_point = torch.clamp(zero_point, quant_min, quant_max)
    if zero_point_dtype is None:
        zero_point_dtype = torch.int32
    return scale.to(dtype=scale_dtype, device=input.device), zero_point.to(
        dtype=zero_point_dtype
    )


# TODO: lower this op to custom op library
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
    quant_min, quant_max = _get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
    assert mapping_type in [
        MappingType.SYMMETRIC,
        MappingType.SYMMETRIC_NO_CLIPPING_ERR,
        MappingType.ASYMMETRIC,
    ], f"Unsupported mapping type: {mapping_type}"

    assert min_val is not None and max_val is not None, (
        "Need to provide `min_val` and `max_val`, got: {min_val, max_val}"
    )
    assert min_val.dtype == max_val.dtype, (
        "Expecting `min_val` and `max_val` to have the same dtype, got: {min_val.dtype, max_val.dtype}"
    )

    if scale_dtype is None:
        scale_dtype = min_val.dtype
    if eps is None:
        eps = torch.finfo(min_val.dtype).eps

    scale_device = min_val.device

    if preserve_zero:
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    else:
        min_val_neg = min_val
        max_val_pos = max_val

    if (
        mapping_type == MappingType.SYMMETRIC
        or mapping_type == MappingType.SYMMETRIC_NO_CLIPPING_ERR
    ):
        # scales
        if mapping_type == MappingType.SYMMETRIC:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
        else:
            assert mapping_type == MappingType.SYMMETRIC_NO_CLIPPING_ERR
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
        if zero_point_domain == ZeroPointDomain.FLOAT:
            # TODO INT should not be a valid ZeroPointDomain for symmetric quantization since
            # symmetric quant doesn't have a zero_point
            raise ValueError(
                "zero_point_domain should be ZeroPointDomain.INT or ZeroPointDomain.NONE for symmetric quantization"
            )
        if zero_point_domain == ZeroPointDomain.NONE:
            zero_point = None
        else:
            zero_point = torch.full_like(scale, int((quant_max + quant_min + 1) / 2))
        scale = torch.clamp(scale, min=eps)
    else:
        assert mapping_type == MappingType.ASYMMETRIC
        scale = (max_val_pos - min_val_neg) / torch.tensor(
            float(quant_max - quant_min), dtype=scale_dtype, device=scale_device
        )
        scale = torch.clamp(scale, min=eps)
        if zero_point_domain == ZeroPointDomain.NONE:
            zero_point = None
        elif zero_point_domain == ZeroPointDomain.INT:
            zero_point = quant_min - _Round.apply(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
            if zero_point_dtype is None:
                zero_point_dtype = torch.int32
        else:
            assert zero_point_domain == ZeroPointDomain.FLOAT, (
                "zero_point must be in FLOAT/INT/None domain for asymmetric quantization"
            )
            mid_point = (quant_max + quant_min + 1) / 2
            # this is not preserving zero_point, this is converting to TensorCoreTiledFormat
            # TODO move the conversion of zero_point out of quant_primitives
            # and into TensorCoreTiledLayout.from_plain
            zero_point = min_val_neg + scale * mid_point

    if zero_point is not None:
        zero_point = zero_point.to(dtype=zero_point_dtype)
    return scale.to(dtype=scale_dtype, device=min_val.device), zero_point


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

    if scale_dtype is None:
        scale_dtype = input.dtype
    if eps is None:
        eps = torch.finfo(input.dtype).eps

    assert len(block_size) == input.dim(), (
        f"Got input dim:{input.dim()}, block_size: {block_size}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    input = input.view(shape_for_reduction)

    min_val = torch.amin(input, dim=reduction_dims, keepdim=False)
    max_val = torch.amax(input, dim=reduction_dims, keepdim=False)

    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

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
        zero_point = torch.full_like(scale, int((quant_max + quant_min + 1) / 2))
        scale = torch.clamp(scale, min=eps)
    else:
        assert mapping_type == MappingType.ASYMMETRIC.name
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.clamp(scale, min=eps)
        zero_point = quant_min - _Round.apply(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        if zero_point_dtype is None:
            zero_point_dtype = torch.int32

    return scale.to(dtype=scale_dtype, device=input.device), zero_point.to(
        dtype=zero_point_dtype
    )


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
        q_w = _Round.apply(w / s_group).int()
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
        q_w = _Round.apply(w / s_channel).int()
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
    """HQQ (Half-Quadratic Quantization) - advanced quantization with optimization.

    This function implements HQQ quantization, which uses proximal optimization to
    minimize quantization error through iterative refinement of quantization parameters.
    HQQ typically achieves better accuracy than standard quantization methods by
    optimizing the trade-off between quantization error and sparsity.

    The method alternates between:
    1. Quantizing weights with current parameters
    2. Optimizing scale and zero point to minimize reconstruction error
    3. Applying proximal operator for regularization

    Args:
        tensor: Input tensor to quantize (typically neural network weights).
        nbits: Number of bits for quantization (typically 4).
        group_size: Size of groups for block-wise quantization. None for per-channel.
        optimize: Whether to apply proximal optimization for better accuracy.
        axis: Axis along which to apply quantization (0 or 1).
        compute_dtype: Working precision for computations (torch.float16/float32).
        device: Device for computation ("cuda" or "cpu").
        verbose: Whether to print optimization progress.
        raw_output: If True, returns parameters in HQQ library format.
        optimize_weights: Optimization function to use (defaults to proximal method).

    Returns:
        tuple: (quantized_weights, scale, zero_point, original_shape)
        - quantized_weights: Quantized tensor in uint8 format
        - scale: Optimized scaling factors
        - zero_point: Optimized zero points
        - original_shape: Original tensor shape for reconstruction

    Note:
        HQQ quantization provides superior accuracy compared to standard methods by:
        - Using iterative optimization to refine quantization parameters
        - Applying proximal operators for regularization
        - Supporting both grouped and per-channel quantization

        This method is particularly effective for weight quantization in large language
        models where maintaining accuracy is critical.
    """
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
        zero = _Round.apply(zero)

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
        W_q = _Round.apply(W * scale + zero).clamp(min_max[0], min_max[1])

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
    scale_dtype: torch.dtype = torch.float32,
    block_size: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Calculates float8 scaling factor for the given high precision tensor, using tensorwise granularity.

    Args:
        tensor (torch.Tensor): Input tensor to be quantized.
        float8_dtype (torch.dtype): Data type of the quantized tensor (e.g., torch.float8_e4m3fn, torch.float8_e5m2).
        scale_dtype (torch.dtype): Data type of the scaling factor (e.g., torch.float32).
        block_size (Optional[Tuple[int, ...]]): Block size for block-wise quantization. If None, tensorwise quantization is used.
    """
    quant_max = torch.finfo(float8_dtype).max
    # only tensorwise scaling is supported for now:
    if block_size is None:
        max_abs = tensor.abs().max()
        scale = max_abs / quant_max
    else:
        shape_for_reduction, reduction_dims = _get_reduction_params(
            block_size, tensor.shape
        )
        tensor_reshaped = tensor.view(shape_for_reduction)
        max_abs = tensor_reshaped.abs().amax(dim=reduction_dims, keepdim=True)

        scale = max_abs / quant_max
        # Reshape scale back to match the expected output shape
        # The scale tensor should have the same shape as the input divided by block_size
        output_shape = [
            input_size // block_size[i] for i, input_size in enumerate(tensor.shape)
        ]
        scale = scale.reshape(output_shape)

    if scale_dtype is not torch.float32:
        # Shielding for Version > 2.8
        assert scale_dtype is torch.float8_e8m0fnu, "Only float8_e8m0fnuz is supported"
        scale = torch.exp2(_Round.apply(torch.log2(scale)))
    return scale.to(dtype=torch.float32)


def _expand_scale_to_tensor_shape(
    scale: torch.Tensor, target_shape: torch.Size
) -> torch.Tensor:
    """
    Expand a scale tensor to match the target tensor shape for block-wise quantization.

    Args:
        scale (torch.Tensor): Scale tensor with shape corresponding to block structure
        target_shape (torch.Size): Target tensor shape to expand to

    Returns:
        torch.Tensor: Scale tensor expanded to match target_shape
    """
    if scale.shape == target_shape:
        # Scale already matches target shape
        return scale

    if scale.numel() == 1:
        # Scalar scale - can broadcast naturally
        return scale

    # Calculate block sizes from shape difference
    if len(scale.shape) != len(target_shape):
        raise ValueError(
            f"Scale tensor has {len(scale.shape)} dimensions but target has {len(target_shape)}"
        )

    block_sizes = tuple(
        target_shape[i] // scale.shape[i] for i in range(len(target_shape))
    )

    # Verify that target_shape is evenly divisible by scale.shape
    for i, (target_dim, scale_dim, block_size) in enumerate(
        zip(target_shape, scale.shape, block_sizes)
    ):
        if target_dim != scale_dim * block_size:
            raise ValueError(
                f"Dimension {i}: target size {target_dim} is not evenly divisible "
                f"by scale size {scale_dim} (block size would be {target_dim / scale_dim})"
            )

    # Expand scale using repeat_interleave
    expanded_scale = scale
    for i, block_size in enumerate(block_sizes):
        if block_size > 1:
            expanded_scale = expanded_scale.repeat_interleave(block_size, dim=i)

    return expanded_scale


def quantize_affine_float8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Quantize a high-precision tensor to float8 format.

    This function converts float32/float16/bfloat16 tensors to float8 format using
    the provided scaling factor. Float8 quantization offers a good balance between
    model size reduction and accuracy preservation, especially for transformer models.

    The quantization formula is:
    ``float8_val = clamp(tensor / scale, -max_val, max_val).to(float8_dtype)``

    Args:
        tensor: Input high-precision tensor to quantize.
        scale: Scaling factor(s) for quantization. Can be scalar or tensor with
            shape compatible for broadcasting with input tensor.
        float8_dtype: Target float8 dtype. Common options:
            - torch.float8_e4m3fn: 4-bit exponent, 3-bit mantissa (default)
            - torch.float8_e5m2: 5-bit exponent, 2-bit mantissa

    Returns:
        Quantized tensor in the specified float8 format with same shape as input.

    Example:
        >>> tensor = torch.randn(4, 8, dtype=torch.float32)
        >>> scale = torch.tensor(0.1)  # Per-tensor scaling
        >>> quantized = quantize_affine_float8(tensor, scale)

        >>> # Per-channel scaling
        >>> scale = torch.randn(4) * 0.1
        >>> quantized = quantize_affine_float8(tensor, scale)

    Note:
        Float8 quantization is particularly effective for:
        - Large language model inference and training
        - Maintaining numerical stability in transformer attention
        - GPU acceleration with tensor core operations

        The scale tensor is automatically expanded to match input dimensions
        for block-wise quantization support.
    """
    tensor_fp32 = tensor.to(torch.float32)

    # Expand scale to match tensor dimensions for block-wise quantization
    scale_expanded = _expand_scale_to_tensor_shape(scale, tensor.shape)

    tensor_scaled = tensor_fp32 / scale_expanded
    max_value = torch.finfo(float8_dtype).max
    tensor_clamped = tensor_scaled.clamp(min=-max_value, max=max_value)
    fp8_tensor = tensor_clamped.to(float8_dtype)
    return fp8_tensor


def dequantize_affine_float8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a float8 tensor back to high-precision format.

    This function converts float8 tensors back to higher precision formats
    (float32/float16/bfloat16) using the provided scaling factor. This is the
    inverse operation of :func:`quantize_affine_float8`.

    The dequantization formula is:
    ``dequantized_val = tensor.to(float32) * scale``

    Args:
        tensor: Float8 tensor to dequantize.
        scale: Scaling factor(s) used during quantization. Must be compatible
            for broadcasting with tensor dimensions.
        output_dtype: Target high precision dtype for output. Defaults to float32.

    Returns:
        Dequantized tensor in specified high precision format with same shape as input.

    Example:
        >>> # Dequantize float8 tensor back to float32
        >>> float8_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn)
        >>> scale = torch.tensor(0.1)
        >>> dequantized = dequantize_affine_float8(float8_tensor, scale)

        >>> # Dequantize to float16 with per-channel scale
        >>> scale = torch.tensor([0.1, 0.2])
        >>> dequantized = dequantize_affine_float8(
        ...     float8_tensor, scale, output_dtype=torch.float16
        ... )

    Note:
        This function is typically used in conjunction with :func:`quantize_affine_float8`
        to restore quantized weights or activations during inference or training.

        The scale parameter must match the scaling used during the original
        quantization for accurate reconstruction of the original values.
    """
    fp8_tensor = tensor.to(torch.float32)

    # Expand scale to match tensor dimensions for block-wise quantization
    scale_expanded = _expand_scale_to_tensor_shape(scale, tensor.shape)

    hp_tensor = fp8_tensor * scale_expanded
    return hp_tensor.to(output_dtype)
