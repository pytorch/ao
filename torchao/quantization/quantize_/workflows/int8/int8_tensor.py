# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
    quantize_affine,
)
from torchao.quantization.quantize_.common import (
    QuantizeTensorKwargs,
    _choose_quant_func_and_quantize_tensor,
)
from torchao.utils import TorchAOBaseTensor

__all__ = ["Int8Tensor", "QuantizeTensorToInt8Kwargs"]

aten = torch.ops.aten


@dataclass
class QuantizeTensorToInt8Kwargs(QuantizeTensorKwargs):
    """Tensor kwargs for creating int8 tensor (either activation or weight)

    Args:
        block_size (Optional[list[int]]): block size for quantization granularity
    """

    block_size: Optional[list[int]] = None


class Int8Tensor(TorchAOBaseTensor):
    """
    int8 quantized tensor with plain layout

    Tensor Attributes:
        qdata: (N, K) int8 quantized weight data
        scale: scale factors for dequantization
        zero_point: zero points for dequantization

    Non-Tensor Attributes:
        block_size: block size for quantization granularity
        shape: original tensor shape
        act_quant_kwargs: flags for static/dynamic activation quantization
    """

    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["block_size", "_shape"]
    optional_tensor_attribute_names = [
        "act_quant_kwargs",
        "dtype",
    ]

    def __new__(
        cls,
        qdata,
        scale,
        zero_point,
        block_size,
        shape,
        act_quant_kwargs=None,
        dtype=None,
    ):
        kwargs = {
            "device": qdata.device,
            "dtype": dtype or scale.dtype,
            "requires_grad": False,
        }
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(
        self,
        qdata,
        scale,
        zero_point,
        block_size,
        shape,
        act_quant_kwargs=None,
        dtype=None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size
        self._shape = shape
        self.act_quant_kwargs = act_quant_kwargs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.act_quant_kwargs=}, {self.qdata=}, {self.scale=}, "
            f"{self.zero_point=}, {self.block_size=}, "
            f"{self.shape=}, {self.device=}, {self.dtype=})"
        )

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: list[int],
        act_quant_kwargs: Optional[QuantizeTensorToInt8Kwargs] = None,
    ):
        if w.dim() != 2 or len(block_size) != 2:
            raise ValueError("Expected 2D tensor and block_size length 2")

        scale, zero_point = choose_qparams_affine(
            input=w,
            mapping_type=MappingType.SYMMETRIC,
            block_size=tuple(block_size),
            target_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            scale_dtype=w.dtype,
            zero_point_dtype=torch.int8,
        )

        int_data = quantize_affine(
            w,
            block_size=tuple(block_size),
            scale=scale,
            zero_point=zero_point,
            output_dtype=torch.int8,
        )

        return cls(
            int_data,
            scale,
            zero_point,
            block_size,
            w.shape,
            act_quant_kwargs=act_quant_kwargs,
            dtype=w.dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize int8 tensor to floating point"""
        dtype = output_dtype or self.dtype or self.scale.dtype

        qdata_fp = self.qdata.to(dtype)
        scale = self.scale.to(dtype)
        zero_point = self.zero_point.to(dtype)

        # Reshape 1D scale/zero_point to [N, 1] for broadcasting with [N, K] qdata
        if scale.ndim == 1:
            scale = scale.unsqueeze(1)
        if zero_point.ndim == 1:
            zero_point = zero_point.unsqueeze(1)

        return (qdata_fp - zero_point) * scale


implements = Int8Tensor.implements


@implements([aten.dequantize.self])
def _(func, types, args, kwargs):
    """dequantization: int8 -> float"""
    return args[0].dequantize()


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    """quantization: float -> int8"""
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    assert isinstance(weight_tensor, Int8Tensor), (
        f"Expected weight to be Int8Tensor, got {type(weight_tensor)}"
    )

    if weight_tensor.act_quant_kwargs is not None:
        # INT8 × INT8 (dynamic)
        # Quantize input if it's not already quantized
        if not isinstance(input_tensor, Int8Tensor):
            input_tensor = _choose_quant_func_and_quantize_tensor(
                input_tensor, weight_tensor.act_quant_kwargs
            )

        x_vals_int8 = input_tensor.qdata
        x_scales = input_tensor.scale
        w_vals_int8_t = weight_tensor.qdata.contiguous().t()
        w_scales = weight_tensor.scale

        tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
        x_scales_dtype = x_scales.dtype

        # Cast fp16 scale to float
        intermediate_dtype = (
            torch.float if x_scales_dtype == torch.half else x_scales_dtype
        )
        y_dot_int64 = torch.mm(tmp.to(torch.int64), w_vals_int8_t.to(torch.int64))
        y_dot_scaled = y_dot_int64.to(intermediate_dtype) * x_scales.reshape(-1, 1).to(
            intermediate_dtype
        )
        y_dot_scaled = y_dot_scaled.to(x_scales_dtype)

        result = (y_dot_scaled * w_scales).reshape(
            *x_vals_int8.shape[:-1], y_dot_scaled.shape[-1]
        )
        result = result.to(input_tensor.dtype)
    else:
        # FP × INT8 (weight-only)
        input_tensor = input_tensor.dequantize()

        result = func(input_tensor, weight_tensor.dequantize(input_tensor.dtype), None)

    return result + bias if bias is not None else result


@implements([aten.slice.Tensor])
def _(func, types, args, kwargs):
    """Slice operation for Int8Tensor"""
    tensor, dim, start, end, step = (
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] if len(args) > 4 else 1,
    )

    # Slice scale and zero_point along dimension 0 if slicing rows
    sliced_scale = tensor.scale
    sliced_zero_point = tensor.zero_point

    if dim == 0 and tensor.scale.ndim >= 1:
        sliced_scale = aten.slice.Tensor(tensor.scale, 0, start, end, step)
        sliced_zero_point = aten.slice.Tensor(tensor.zero_point, 0, start, end, step)

    sliced_shape = list(
        aten.slice.Tensor(torch.empty(tensor.shape), dim, start, end, step).shape
    )

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Int8Tensor(
            aten.slice.Tensor(tensor.qdata, dim, start, end, step),
            sliced_scale,
            sliced_zero_point,
            tensor.block_size,
            sliced_shape,
            tensor.act_quant_kwargs,
            tensor.dtype,
        ),
    )


@implements(aten.transpose.int)
def _(func, types, args, kwargs):
    """Dimension transposer for Int8Tensor"""
    self, dim0, dim1 = args
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Int8Tensor(
            self.qdata.transpose(dim0, dim1),
            self.scale,
            self.zero_point,
            [self.block_size[dim1], self.block_size[dim0]],
            [self._shape[dim1], self._shape[dim0]],
            self.act_quant_kwargs,
            self.dtype,
        ),
    )


@implements(aten.select.int)
def _(func, types, args, kwargs):
    """Index selector for Int8Tensor"""
    self, dim, index = args
    assert dim == 0, f"Only dim=0 supported, got {dim}"

    # Handle 0-dim scale/zero_point (per-tensor quantization)
    if self.scale.ndim == 0:
        selected_scale = self.scale
        selected_zero_point = self.zero_point
    else:
        selected_scale = self.scale[index]
        selected_zero_point = self.zero_point[index]

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Int8Tensor(
            self.qdata[index],
            selected_scale,
            selected_zero_point,
            self.block_size,
            list(self.qdata[index].shape),
            self.act_quant_kwargs,
            self.dtype,
        ),
    )


Int8Tensor.__module__ = "torchao.quantization"
torch.serialization.add_safe_globals([Int8Tensor, QuantizeTensorToInt8Kwargs])
