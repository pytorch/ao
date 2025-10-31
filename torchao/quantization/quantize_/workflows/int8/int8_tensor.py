# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.float8.inference import (
    _slice_scale_for_dimension,
    preprocess_scale,
)
from torchao.kernel import int_scaled_matmul
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
        # TODO: Static quantization support using `static_scale`, `static_zero_point`
    """

    block_size: list[int]


class Int8Tensor(TorchAOBaseTensor):
    """
    int8 quantized tensor with plain layout

    Tensor Attributes:
        qdata: (N, K) or (B, N, K) int8 quantized weight data (2D or 3D)
        scale: scale factors for dequantization

    Non-Tensor Attributes:
        block_size: block size for quantization granularity
        act_quant_kwargs: flags for dynamic activation quantization
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
        if w.dim() not in [2, 3] or len(block_size) != w.dim():
            raise ValueError("Expected 2D or 3D tensor with same block_size length")

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
    """INT8 quantization: dynamic activation or weight-only"""
    activation_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    assert isinstance(weight_tensor, Int8Tensor), (
        f"Expected weight to be Int8Tensor, got {type(weight_tensor)}"
    )

    # Store original shape for reshaping result
    original_weight_shape = weight_tensor.qdata.shape

    # Reshape 3D weights to 2D: (B, N, K) -> (B*N, K)
    if weight_tensor.qdata.dim() == 3:
        w_q_2d = weight_tensor.qdata.reshape(-1, original_weight_shape[-1])
        w_scale_2d = (
            weight_tensor.scale.reshape(-1)
            if weight_tensor.scale.numel() > 1
            else weight_tensor.scale
        )
    else:
        w_q_2d = weight_tensor.qdata
        w_scale_2d = weight_tensor.scale

    if weight_tensor.act_quant_kwargs is not None:
        if not isinstance(activation_tensor, Int8Tensor):
            # Dynamic activation quantization
            act_kwargs = weight_tensor.act_quant_kwargs
            input_ndim = activation_tensor.ndim

            # Ensure block_size matches input tensor dimensions
            if len(act_kwargs.block_size) != input_ndim:
                if input_ndim == 3 and len(act_kwargs.block_size) == 2:
                    block_size_updated = [1] + list(act_kwargs.block_size)
                else:
                    block_size_updated = list(act_kwargs.block_size)[-input_ndim:]
                act_kwargs = QuantizeTensorToInt8Kwargs(block_size=block_size_updated)

            activation_tensor = _choose_quant_func_and_quantize_tensor(
                activation_tensor, act_kwargs
            )

        x_vals = activation_tensor.qdata.reshape(-1, activation_tensor.qdata.shape[-1])
        x_scales = preprocess_scale(activation_tensor.scale, x_vals.shape)
        w_vals_t = w_q_2d.contiguous().t()
        intermediate_dtype = (
            torch.float if x_scales.dtype == torch.half else x_scales.dtype
        )

        y_dot_scaled = int_scaled_matmul(
            x_vals, w_vals_t, x_scales.to(intermediate_dtype)
        )
        y_dot_scaled = y_dot_scaled.to(activation_tensor.scale.dtype)

        result = (y_dot_scaled * w_scale_2d).reshape(
            *activation_tensor.shape[:-1], *original_weight_shape[:-1]
        )
        result = result.to(activation_tensor.dtype)
    else:
        # FP × INT8 (weight-only)
        w_vals_int8_t = w_q_2d.t()
        m = torch.mm(
            activation_tensor.reshape(-1, activation_tensor.shape[-1]),
            w_vals_int8_t.to(activation_tensor.dtype),
        )
        result = m * w_scale_2d.to(m.dtype)
        result = result.reshape(
            *activation_tensor.shape[:-1], *original_weight_shape[:-1]
        )

    return result + bias if bias is not None else result


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    """Slice operation for Int8Tensor"""
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])

    if step != 1:
        raise NotImplementedError("Slicing with step > 1 is not supported")

    assert dim in [0, 1, 2], f"Only dim=0,1,2 are supported, got: dim={dim}"
    assert self.qdata.ndim in [2, 3], (
        f"Expected qdata to have dim=2,3 got: dim={self.qdata.ndim}"
    )

    if end >= self.shape[dim]:
        end = self.shape[dim]

    sliced_qdata = aten.slice.Tensor(self.qdata, dim, start, end, step)
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
            self.block_size[1:],
            self.act_quant_kwargs,
            self.dtype,
        ),
    )


Int8Tensor.__module__ = "torchao.quantization"
torch.serialization.add_safe_globals([Int8Tensor, QuantizeTensorToInt8Kwargs])
