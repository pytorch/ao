import torch
from torchao.dtypes.utils import PlainLayout
from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
)
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from torchao.float8.inference import (
    preprocess_data,
    Float8MMConfig,
    addmm_float8_unwrapped_inference,
    _is_rowwise_scaled
)
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
    is_traceable_wrapper_subclass,
)
from torchao.quantization.quant_primitives import (
    int_scaled_matmul,
    ZeroPointDomain,
)

def _aqt_is_int8(aqt):
    """Check if an AffineQuantizedTensor is int8 quantized Tensor"""
    return (
        aqt.tensor_impl.dtype == torch.int8 and
        (aqt.quant_min is None or aqt.quant_min == -128) and
        (aqt.quant_max is None or aqt.quant_max == 127)
    )

def _aqt_is_int8_reduced_range(aqt):
    return (
        aqt.tensor_impl.dtype == torch.int8 and
        aqt.quant_min == -127 and
        (aqt.quant_max is None or aqt.quant_max == 127)
    )


def _linear_fp_act_int8_weight_check(input_tensor, weight_tensor, bias):
    return (
        # input is native float tensor
        not is_traceable_wrapper_subclass(input_tensor) and
        input_tensor.is_floating_point() and
        # weight is int8 per channel quantized affine quantized tensor
        isinstance(weight_tensor, AffineQuantizedTensor) and
        _aqt_is_int8(weight_tensor) and
        len(weight_tensor.shape) == 2 and
        len(weight_tensor.block_size) == 2 and
        weight_tensor.block_size[0] == 1 and
        weight_tensor.block_size[1] == weight_tensor.shape[1] and
        weight_tensor.zero_point_domain == ZeroPointDomain.INT and
        isinstance(weight_tensor._layout, PlainLayout)
    )


def _linear_fp_act_int8_weight_impl(input_tensor, weight_tensor, bias):
    # TODO: enable cpu and mps efficient path
    # is_cpu and is_mps only, some issue with is_contiguous() currently
    # return torch.ops.aten._weight_int8pack_mm(input_tensor.contiguous(), w_vals_int8_t, weight_tensor.tensor_impl.scale)

    # per channel int8 weight only quantizated mm
    w_vals_int8_t = weight_tensor.tensor_impl.int_data.t()
    scale = weight_tensor.tensor_impl.scale
    orig_dtype = input_tensor.dtype
    m = torch.mm(
        input_tensor.reshape(-1, input_tensor.shape[-1]),
        w_vals_int8_t.to(input_tensor.dtype),
    )
    y = m * scale.to(m.dtype)
    y = y.reshape(*input_tensor.shape[:-1], y.shape[-1])
    if bias is not None:
        y += bias.to(m.dtype)
    return y


def _linear_int8_act_int8_weight_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(input_tensor, AffineQuantizedTensor) and
        _aqt_is_int8_reduced_range(input_tensor) and
        isinstance(weight_tensor, AffineQuantizedTensor) and
        input_tensor.dtype == weight_tensor.dtype and
        isinstance(input_tensor._layout, PlainLayout) and
        isinstance(weight_tensor._layout, PlainLayout)
    )

def _linear_int8_act_int8_weight_impl(input_tensor, weight_tensor, bias):
    #
    # 1. do the matrix form of dot(X_i, W_j)
    #
    #
    # 2. rescale the output
    #
    # in cases with large matrices, y_dot_int32 can grow sufficiently
    # large that y_dot_int32 * a float16 scale is greater than the maximum
    # value of a float 16, (which results in a value of inf even if multiplying
    # by the other scale would bring it within the expected range)

    x_vals_int8 = input_tensor.tensor_impl.int_data
    x_scales = input_tensor.tensor_impl.scale
    w_vals_int8_t = weight_tensor.tensor_impl.int_data.contiguous().t()
    w_scales = weight_tensor.tensor_impl.scale
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    x_scales_dtype = x_scales.dtype
    # Cast fp16 scale to float to avoid overflow in int_scaled_matmul
    intermediate_dtype = torch.float if x_scales_dtype == torch.half else x_scales_dtype
    y_dot_scaled = int_scaled_matmul(tmp, w_vals_int8_t, x_scales.reshape(-1, 1).to(intermediate_dtype))
    y_dot_scaled = y_dot_scaled.to(x_scales_dtype)

    y = (y_dot_scaled * w_scales).reshape(
        *x_vals_int8.shape[:-1], y_dot_scaled.shape[-1]
    )

    # can downcast only at the very end
    output_dtype = input_tensor.dtype
    y = y.to(output_dtype)
    if bias is not None:
        y += bias
    return y
