# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.float8.inference import _slice_scale_for_dimension
from torchao.quantization.quant_primitives import (
    MappingType,
    _maybe_expand_scale_to_tensor_shape,
    choose_qparams_affine,
    quantize_affine,
)
from torchao.quantization.quantize_.common import (
    QuantizeTensorKwargs,
    _choose_quant_func_and_quantize_tensor,
)
from torchao.utils import TorchAOBaseTensor, fill_defaults

__all__ = ["Int8Tensor", "QuantizeTensorToInt8Kwargs"]

aten = torch.ops.aten


@dataclass
class QuantizeTensorToInt8Kwargs(QuantizeTensorKwargs):
    """Tensor kwargs for creating int8 tensor (either activation or weight)

    Args:
        block_size (list[int]): block size for quantization granularity
        static_scale (Optional[torch.Tensor]): pre-computed scale for static quantization
    """

    block_size: list[int]
    static_scale: Optional[torch.Tensor] = None


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
        return torch.Tensor._make_wrapper_subclass(cls, qdata.shape, **kwargs)

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

        if act_quant_kwargs is not None and act_quant_kwargs.static_scale is not None:
            # INT8 × INT8 (static)
            scale = act_quant_kwargs.static_scale
            zero_point = torch.zeros_like(scale, dtype=torch.int8)
        else:
            # INT8 × INT8 (dynamic): compute scale at runtime
            scale, zero_point = choose_qparams_affine(
                input=w,
                mapping_type=MappingType.SYMMETRIC,
                block_size=block_size,
                target_dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
                scale_dtype=w.dtype,
                zero_point_dtype=torch.int8,
            )

        int_data = quantize_affine(
            w,
            block_size=block_size,
            scale=scale,
            zero_point=zero_point,
            output_dtype=torch.int8,
        )

        if tuple(block_size) == w.shape:
            # per-tensor
            pass
        elif len(scale.shape) == 1:
            # per-row, 1D -> 2D
            scale = scale.unsqueeze(-1)

        return cls(
            int_data,
            scale,
            block_size,
            act_quant_kwargs=act_quant_kwargs,
            dtype=w.dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize int8 tensor to floating point"""

        if output_dtype is None:
            output_dtype = self.dtype

        qdata_fp = self.qdata.to(output_dtype)
        # Reshape scale to broadcast if granularity is block-wise
        scale_expanded = _maybe_expand_scale_to_tensor_shape(
            self.scale, self.qdata.shape
        )
        return qdata_fp * scale_expanded.to(output_dtype)


implements = Int8Tensor.implements
implements_torch_function = Int8Tensor.implements_torch_function


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    """quantization: dynamic, static, weight-only int8 quantization"""
    activation_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    assert isinstance(weight_tensor, Int8Tensor), (
        f"Expected weight to be Int8Tensor, got {type(weight_tensor)}"
    )

    if weight_tensor.act_quant_kwargs is not None:
        if not isinstance(activation_tensor, Int8Tensor):
            # Activation quantization
            activation_tensor = _choose_quant_func_and_quantize_tensor(
                activation_tensor, weight_tensor.act_quant_kwargs
            )

        x_vals = activation_tensor.qdata
        x_scales = activation_tensor.scale
        w_vals_t = weight_tensor.qdata.contiguous().t()
        w_scales = weight_tensor.scale

        tmp_shape = (-1, x_vals.shape[-1])
        tmp = x_vals.view(tmp_shape)

        # Cast fp16 scale to float
        intermediate_dtype = (
            torch.float if x_scales.dtype == torch.half else x_scales.dtype
        )
        # Note: CUDA doesn't support int32/int64 matmul, so we convert to float
        # Error message is NotImplementedError: "addmm_cuda" not implemented for 'Int'
        # This may introduce minor numerical differences compared to int arithmetic
        y_dot = torch.mm(tmp.to(intermediate_dtype), w_vals_t.to(intermediate_dtype))

        # Apply activation scale
        is_per_tensor_act = x_scales.numel() == 1
        if is_per_tensor_act:
            y_dot.mul_(x_scales.to(intermediate_dtype))
        else:
            # For block-wise activation scale, reshape to match y_dot
            x_scales_reshaped = x_scales.view(y_dot.shape[0], -1)
            y_dot.mul_(x_scales_reshaped.to(intermediate_dtype))

        # Apply weight scale
        is_per_tensor_weight = w_scales.numel() == 1
        if is_per_tensor_weight:
            result = y_dot.mul_(w_scales.to(intermediate_dtype))
        else:
            # Per-row weight scale - transpose and broadcast
            w_scales_broadcast = w_scales.t().expand_as(y_dot)
            result = y_dot.mul_(w_scales_broadcast.to(intermediate_dtype))

        # Reshape back to original shape
        result = result.view(*x_vals.shape[:-1], result.shape[-1])
        result = result.to(activation_tensor.dtype)
    else:
        # FP × INT8 (weight-only)
        result = func(
            activation_tensor, weight_tensor.dequantize(activation_tensor.dtype), None
        )

    return result + bias if bias is not None else result


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    """Slice operation for Int8Tensor"""
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])

    if step != 1:
        raise NotImplementedError("Slicing with step > 1 is not supported")

    if end >= self.shape[dim]:
        end = self.shape[dim]

    sliced_qdata = aten.slice.Tensor(self.qdata, dim, start, end, step)

    if self.scale.numel() == 1:
        # Per-tensor quantization - scale doesn't change
        sliced_scale = self.scale
    elif dim < self.scale.ndim and self.scale.shape[dim] > 1:
        # Block-wise quantization - need to slice the scale appropriately
        sliced_scale = aten.slice.Tensor(self.scale, dim, start, end, step)
    else:
        # Block-wise quantization - need to slice the scale appropriately
        sliced_scale = _slice_scale_for_dimension(
            self.scale, self.qdata.shape, dim, start, end, step
        )

    block_size = list(self.block_size)
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
            self.act_quant_kwargs,
            dtype=self.dtype,
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
