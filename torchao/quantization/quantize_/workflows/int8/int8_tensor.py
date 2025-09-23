# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch

from torchao.quantization.quantize_.common import (
    KernelPreference,
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
        kernel_preference (KernelPreference): kernel preference for ops like matmul, grouped matmul etc.
        block_size (Optional[list[int]]): block size for quantization granularity
    """

    kernel_preference: KernelPreference = KernelPreference.AUTO
    block_size: Optional[list[int]] = None


# TODO: Implement block-wise quantization using block_size
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
        kernel_preference: kernel preference for operations
    """

    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["block_size"]
    optional_tensor_attribute_names = [
        "act_quant_kwargs",
        "kernel_preference",
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
        kernel_preference=KernelPreference.AUTO,
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
        kernel_preference=KernelPreference.AUTO,
        dtype=None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size
        self.act_quant_kwargs = act_quant_kwargs
        self.kernel_preference = kernel_preference

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.act_quant_kwargs=}, {self.qdata=}, {self.scale=}, "
            f"{self.zero_point=}, {self.block_size=}, {self.kernel_preference=}, "
            f"{self.shape=}, {self.device=}, {self.dtype=})"
        )

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: list[int],
        act_quant_kwargs: Optional[QuantizeTensorToInt8Kwargs] = None,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
    ):
        if w.dim() != 2 or len(block_size) != 2:
            raise ValueError("Expected 2D tensor and block_size length 2")

        # Rounding function from high precision dtype
        scale = w.abs().max(dim=-1, keepdim=True)[0] / 127.0
        scale = scale.clamp(min=1e-6)

        int_data = torch.round(w / scale).clamp(-128, 127).to(torch.int8)

        return cls(
            int_data,
            scale.squeeze(-1),
            torch.zeros_like(scale.squeeze(-1), dtype=torch.int8),
            block_size,
            w.shape,
            act_quant_kwargs=act_quant_kwargs,
            kernel_preference=kernel_preference,
            dtype=w.dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize int8 tensor to floating point"""
        dtype = output_dtype or self.dtype or self.scale.dtype
        return (
            self.qdata.to(dtype) - self.zero_point.to(dtype).unsqueeze(1)
        ) * self.scale.to(dtype).unsqueeze(1)


implements = Int8Tensor.implements


@implements([aten.dequantize.self])
def _(func, types, args, kwargs):
    """dequantization: int8 -> float"""
    tensor = args[0]
    dtype = tensor.dtype or tensor.scale.dtype
    return (
        tensor.qdata.to(dtype) - tensor.zero_point.to(dtype).unsqueeze(1)
    ) * tensor.scale.to(dtype).unsqueeze(1)


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

    # Dynamic activation quantization if enabled
    if weight_tensor.act_quant_kwargs is not None:
        input_tensor = _choose_quant_func_and_quantize_tensor(
            input_tensor, weight_tensor.act_quant_kwargs
        )

    if isinstance(input_tensor, Int8Tensor):
        # INT8 × INT8 (dynamic)
        x_int32 = input_tensor.qdata.to(torch.int32)
        w_int32 = weight_tensor.qdata.to(torch.int32).t()

        result = torch.mm(x_int32.view(-1, x_int32.size(-1)), w_int32)
        scale = input_tensor.scale.view(-1, 1) * weight_tensor.scale.unsqueeze(0)
        result = result.to(scale.dtype) * scale
        result = result.view(*input_tensor.shape[:-1], -1)
    else:
        # FP × INT8 (static)
        result = torch.nn.functional.linear(
            input_tensor, weight_tensor.dequantize(), None
        )

    return result + bias if bias is not None else result


Int8Tensor.__module__ = "torchao.quantization"
torch.serialization.add_safe_globals([Int8Tensor, QuantizeTensorToInt8Kwargs])
