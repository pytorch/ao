# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import torch

from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
    quantize_affine,
)
from torchao.utils import (
    TorchAOBaseTensor,
)

__all__ = [
    "Int4PlainInt32",
]

aten = torch.ops.aten


class Int4PlainInt32(TorchAOBaseTensor):
    """
    int4 weight-only quantization on XPU with oneDNN as backend (groupwise quantization only)

    Tensor Attributes:
        qdata: (N, K/8), packed int4 weight, the data type is int32 here with 4*(int4*2)
        scale: (K/group_size, N), dtype is the same as the original Tensor dtype
        zero_point: (K/group_size, N), dtype is int8

    Non-Tensor Attributes:
        block_size: the block size for quantization, representing the granularity.
        shape: shape of the original Tensor

    """

    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["block_size", "shape"]

    def __new__(
        cls,
        qdata,
        scale,
        zero_point,
        block_size,
        shape,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, qdata, scale, zero_point, block_size, shape):
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}"

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: List[int],
    ):
        assert w.ndim == 2 and w.device.type == "xpu", (
            f"Expecting 2D tensor on XPU, but got: {w.shape} on {w.device.type}"
        )
        assert len(block_size) == w.ndim

        original_shape = w.shape
        mapping_type = MappingType.ASYMMETRIC
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        eps = 1e-6
        scale_dtype = None
        zero_point_dtype = torch.int32
        scale, zero_point = choose_qparams_affine(
            w,
            mapping_type.name,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
            scale_dtype,
            zero_point_dtype,
        )
        int_data = quantize_affine(
            w,
            block_size,
            scale,
            zero_point,
            target_dtype,
            quant_min,
            quant_max,
        )
        assert int_data.dtype == torch.int32, (
            "torch.ops.aten._convert_weight_to_int4pack expects `int32` dtype"
        )
        packed_weight = (int_data[::, 1::2] << 4 | int_data[::, ::2]).to(torch.uint8)
        packed_weight = torch.ops.aten._convert_weight_to_int4pack(
            packed_weight.contiguous(), 8
        )
        scale = scale.reshape(int_data.shape[0], -1)
        zero_point = zero_point.reshape(int_data.shape[0], -1)
        return Int4PlainInt32(
            packed_weight,
            scale.transpose(0, 1).contiguous(),
            zero_point.transpose(0, 1).contiguous().to(torch.int8),
            block_size,
            original_shape,
        )


implements = Int4PlainInt32.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    assert input_tensor.device.type == "xpu", (
        f"For XPU device only but got: {input_tensor.device}"
    )
    assert isinstance(weight_tensor, Int4PlainInt32), (
        f"Expected weight_tensor to be Int4PlainInt32, got: {type(weight_tensor)}"
    )
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"Shapes of input and weight do not match, input:{input_tensor.shape}, weight: {weight_tensor.shape}"
    )

    act_mat = input_tensor
    packed_weight = weight_tensor.qdata
    scale = weight_tensor.scale
    zero_point = weight_tensor.zero_point

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype

    # reshape to 2D
    act_mat = act_mat.reshape(-1, act_mat.shape[-1])

    # groupwise int4 quantization
    groupsize = weight_tensor.block_size[1]
    y = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
        act_mat, packed_weight, groupsize, scale, zero_point
    )

    # remove out_feature padding
    assert weight_tensor.ndim == 2
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y += bias
    return y.to(orig_dtype)


Int4PlainInt32.__module__ = "torchao.quantization"

# Allow a model with Int4PlainInt32 weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4PlainInt32])
