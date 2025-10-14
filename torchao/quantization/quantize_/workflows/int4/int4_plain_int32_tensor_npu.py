# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch

from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
    quantize_affine,
)
from torchao.utils import (
    TorchAOBaseTensor,
)

__all__ = ["Int4PlainInt32TensorNPU"]

aten = torch.ops.aten


class Int4PlainInt32TensorNPU(TorchAOBaseTensor):
    """
    int4 weight-only quantization on Ascend NPU backend (groupwise quantization only)

    Tensor Attributes:
        qdata: (N, K/8), packed int4 weight, the data type is int32 here with 8*int4, the original dtype can be float16 or bfloat16
        scale: (K/group_size, N), dtype is the same as the original Tensor type (float16 or bfloat16)
        zero_point: (K/group_size, N), dtype is the same as the original Tensor type (float16 or bfloat16)

    Non-Tensor Attributes:
        block_size: the block size for quantization, representing the granularity
        shape: shape of the original Tensor

    Optional Tensor Data Attributes:
        act_pre_scale (Optional[Tensor]): Optional scale for activation Tensor, if present,
               we'll multiply activation Tensor with act_pre_scale before applying dynamic
               quantization to activation or running quantized mm op

    """

    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["block_size", "shape"]
    optional_tensor_data_names = ["act_pre_scale"]

    def __new__(
        cls,
        qdata,
        scale,
        zero_point,
        block_size,
        shape,
        act_pre_scale: Optional[torch.Tensor] = None,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata,
        scale,
        zero_point,
        block_size,
        shape,
        act_pre_scale: Optional[torch.Tensor] = None,
    ):
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size
        self.act_pre_scale = act_pre_scale

    def _quantization_type(self):
        s = f"shape={self.shape}, block_size={self.block_size}, device={self.device}"
        if self.act_pre_scale is not None:
            s += f", act_pre_scale.shape={self.act_pre_scale.shape}"
        return s

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: List[int],
    ):
        assert w.ndim == 2 and w.device.type == "npu", (
            f"Expecting 2D tensor on NPU, but got: {w.shape} on {w.device.type}"
        )
        assert len(block_size) == w.ndim
        assert w.dtype in [torch.float16, torch.bfloat16], (
            f"Expecting float16 or bfloat16 weight tensor, but got: {w.dtype}"
        )

        original_shape = w.shape
        mapping_type = MappingType.ASYMMETRIC
        target_dtype = torch.int32
        quant_min = -8
        quant_max = 7
        eps = 1e-6
        scale_dtype = w.dtype
        zero_point_dtype = w.dtype

        scale, zero_point = choose_qparams_affine(
            w,
            mapping_type,
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
            f"torch_npu.npu_convert_weight_to_int4pack expects `int32` dtype"
        )

        assert int_data.shape[-1] % 8 == 0, (
            f"torch_npu.npu_convert_weight_to_int4pack expects last dim must be aligned to 8,but got {int_data.shape[-1]}"
        )

        packed_weight = torch.ops.npu.npu_convert_weight_to_int4pack(
            int_data.contiguous(), 0
        )

        scale = scale.reshape(int_data.shape[0], -1)
        zero_point = zero_point.reshape(int_data.shape[0], -1)

        return Int4PlainInt32TensorNPU(
            packed_weight,
            scale.transpose(0, 1).contiguous(),
            zero_point.transpose(0, 1).contiguous(),
            block_size,
            original_shape,
            act_pre_scale=None,
        )


implements = Int4PlainInt32TensorNPU.implements
implements_torch_function = Int4PlainInt32TensorNPU.implements_torch_function


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):

    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    assert input_tensor.device.type == "npu", (
        f"For NPU device only but got: {input_tensor.device.type}"
    )
    assert isinstance(weight_tensor, Int4PlainInt32TensorNPU), (
        f"Expected weight_tensor to be Int4PlainInt32NPUTensor, got: {type(weight_tensor)}"
    )
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"Shapes of input and weight do not match, input:{input_tensor.shape}, weight: {weight_tensor.shape}"
    )

    if weight_tensor.act_pre_scale is not None:
        input_tensor = input_tensor * weight_tensor.act_pre_scale

    act_mat = input_tensor
    packed_weight = weight_tensor.qdata
    scale = weight_tensor.scale
    zero_point = weight_tensor.zero_point

    orig_act_size = act_mat.shape
    orig_dtype = act_mat.dtype

    # dtype alignment
    if act_mat.dtype == torch.float16:
        scale = scale.to(torch.float16)
        zero_point = zero_point.to(torch.float16)
        if bias is not None:
            bias = bias.to(torch.float16)
    elif act_mat.dtype == torch.bfloat16:
        scale = scale.to(torch.bfloat16)
        zero_point = zero_point.to(torch.bfloat16)
        if bias is not None:
            bias = bias.to(torch.float32)

    # reshape to 2D
    act_mat = act_mat.reshape(-1, act_mat.shape[-1])

    # groupwise int4 quantization
    groupsize = weight_tensor.block_size[1]

    y = torch.ops.npu.npu_weight_quant_batchmatmul(
        x=act_mat,
        weight=packed_weight.contiguous().transpose(-1, -2),
        antiquant_scale=scale,
        antiquant_offset=zero_point,
        antiquant_group_size=groupsize,
        bias=bias,
    )
    
    # remove out_feature padding
    assert weight_tensor.ndim == 2
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)
    
    return y.to(orig_dtype)


Int4PlainInt32TensorNPU.__module__ = "torchao.quantization"

# Allow a model with Int4PlainInt32TensorNPU weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4PlainInt32TensorNPU])
