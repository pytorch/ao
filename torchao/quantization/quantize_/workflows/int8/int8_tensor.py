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

    block_size: list[int]


class Int8Tensor(TorchAOBaseTensor):
    """
    int8 quantized tensor with plain layout

    Tensor Attributes:
        qdata: (N, K) int8 quantized weight data
        scale: scale factors for dequantization

    Non-Tensor Attributes:
        block_size: block size for quantization granularity
        act_quant_kwargs: flags for static/dynamic activation quantization
    """

    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = ["block_size"]
    optional_tensor_attribute_names = [
        "act_quant_kwargs",
        "dtype",
    ]

    def __new__(
        cls: type,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: list[int],
        act_quant_kwargs=None,
        dtype=None,
    ):
        kwargs = {
            "device": qdata.device,
            "dtype": dtype or scale.dtype,
            "requires_grad": False,
        }
        return torch.Tensor._make_wrapper_subclass(cls, list(qdata.shape), **kwargs)

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: list[int],
        act_quant_kwargs=None,
        dtype=None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.block_size = block_size
        self.act_quant_kwargs = act_quant_kwargs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.act_quant_kwargs=}, {self.qdata=}, {self.scale=}, "
            f"{self.block_size=}, {self.shape=}, {self.device=}, {self.dtype=})"
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
            block_size,
            act_quant_kwargs=act_quant_kwargs,
            dtype=w.dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize int8 tensor to floating point"""
        dtype = output_dtype or self.dtype or self.scale.dtype

        qdata_fp = self.qdata.to(dtype)
        scale = self.scale.to(dtype)

        # Reshape scale to broadcast
        if scale.numel() > 1 and scale.shape != qdata_fp.shape:
            scale = scale.view(*scale.shape, *[1] * (qdata_fp.ndim - scale.ndim))

        return qdata_fp * scale


implements = Int8Tensor.implements


@implements([aten.dequantize.self])
def _(func, types, args, kwargs):
    """dequantization: int8 -> float"""
    return args[0].dequantize()


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    """quantization: float -> int8"""
    activation_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    assert isinstance(weight_tensor, Int8Tensor), (
        f"Expected weight to be Int8Tensor, got {type(weight_tensor)}"
    )

    if weight_tensor.act_quant_kwargs is not None:
        # INT8 × INT8 (dynamic)
        # Quantize activation if it's not already quantized
        if not isinstance(activation_tensor, Int8Tensor):
            activation_tensor = _choose_quant_func_and_quantize_tensor(
                activation_tensor, weight_tensor.act_quant_kwargs
            )

        x_vals_int8 = activation_tensor.qdata
        x_scales = activation_tensor.scale
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
        result = result.to(activation_tensor.dtype)
    else:
        # FP × INT8 (weight-only)
        activation_tensor = activation_tensor.dequantize()

        result = func(
            activation_tensor, weight_tensor.dequantize(activation_tensor.dtype), None
        )

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

    assert dim in (0, 1), f"Only dim 0 or 1 supported, got {dim}"

    if end >= tensor.shape[dim]:
        end = tensor.shape[dim]

    # Always slice the qdata
    sliced_qdata = func(tensor.qdata, dim, start, end, step)

    if tensor.scale.numel() == 1:
        # Per-tensor quantization - scale doesn't change
        sliced_scale = tensor.scale
    elif dim < tensor.scale.ndim and tensor.scale.shape[dim] > 1:
        # Block-wise quantization - need to slice the scale appropriately
        sliced_scale = func(tensor.scale, dim, start, end, step)
    else:
        sliced_scale = tensor.scale

    # adjust block_size since the shape has changed, block_size[i] should not be greater than shape[i]
    block_size = list(tensor.block_size)

    for i in range(len(block_size)):
        block_size[i] = min(block_size[i], sliced_qdata.shape[i])

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Int8Tensor(
            sliced_qdata,
            sliced_scale,
            block_size,
            tensor.act_quant_kwargs,
            tensor.dtype,
        ),
    )


@implements(aten.transpose.int)
def _(func, types, args, kwargs):
    self, dim0, dim1 = args
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Int8Tensor(
            self.qdata.transpose(dim0, dim1),
            self.scale,
            [self.block_size[dim1], self.block_size[dim0]],
            self.act_quant_kwargs,
            self.dtype,
        ),
    )


@implements(aten.select.int)
def _(func, types, args, kwargs):
    self, dim, index = args
    assert dim == 0, f"Only dim=0 supported, got {dim}"

    selected_scale = self.scale if self.scale.ndim == 0 else self.scale[index]

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Int8Tensor(
            self.qdata[index],
            selected_scale,
            self.block_size,
            self.act_quant_kwargs,
            self.dtype,
        ),
    )


Int8Tensor.__module__ = "torchao.quantization"
torch.serialization.add_safe_globals([Int8Tensor, QuantizeTensorToInt8Kwargs])
