# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import torch

from torchao.utils import TorchAOBaseTensor

__all__ = ["Int8PlainInt8Tensor"]

aten = torch.ops.aten


# TODO: Implement block-wise quantization using block_size
class Int8PlainInt8Tensor(TorchAOBaseTensor):
    """
    int8 quantized tensor with plain layout

    Tensor Attributes:
        qdata: (N, K) int8 quantized weight data
        scale: scale factors for dequantization
        zero_point: zero points for dequantization

    Non-Tensor Attributes:
        block_size: block size for quantization granularity
        shape: original tensor shape
    """

    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["block_size"]

    def __new__(cls, qdata, scale, zero_point, block_size, shape):
        kwargs = {"device": qdata.device, "dtype": scale.dtype, "requires_grad": False}
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, qdata, scale, zero_point, block_size, shape):
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size

    @classmethod
    def from_hp(cls, w: torch.Tensor, block_size: list[int]):
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
        )


implements = Int8PlainInt8Tensor.implements


@implements([aten.dequantize.self])
def _(func, types, args, kwargs):
    """dequantization: int8 -> float"""
    tensor = args[0]
    return (
        tensor.qdata.to(tensor.scale.dtype)
        - tensor.zero_point.to(tensor.scale.dtype).unsqueeze(1)
    ) * tensor.scale.unsqueeze(1)


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    """quantization: float -> int8"""
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    if isinstance(input_tensor, Int8PlainInt8Tensor):
        # INT8 × INT8
        x_int32 = input_tensor.qdata.to(torch.int32)
        w_int32 = weight_tensor.qdata.to(torch.int32).t()

        result = torch.mm(x_int32.view(-1, x_int32.size(-1)), w_int32)
        scale = input_tensor.scale.view(-1, 1) * weight_tensor.scale.unsqueeze(0)
        result = result.to(scale.dtype) * scale
        result = result.view(*input_tensor.shape[:-1], -1)
    else:
        # FP × INT8
        result = torch.nn.functional.linear(
            input_tensor, weight_tensor.dequantize(), None
        )

    return result + bias if bias is not None else result


Int8PlainInt8Tensor.__module__ = "torchao.quantization"
torch.serialization.add_safe_globals([Int8PlainInt8Tensor])
