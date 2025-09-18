# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional

import torch

from torchao.quantization.quant_primitives import (
    MappingType,
    _choose_qparams_affine_tinygemm,
    _quantize_affine_tinygemm,
)
from torchao.utils import (
    TorchAOBaseTensor,
)

__all__ = [
    "Int4OpaqueTensor",
]

aten = torch.ops.aten


class Int4OpaqueTensor(TorchAOBaseTensor):
    """
    int4 weight-only quantization on CPU with tinygemm (groupwise quantization only). The packing format is determined on ISA and shape.
    This is an opaque tensor subclass, the packing format is not exposed to the rest of the system. See the note below for more details.

    Tensor Attributes:
        qdata: preshuffled and packed int4 weight for CPU tinygemm kernel, always viewed as a 2D (N, K/2) tensor, last dimension is packed
               preshuffling is specific to CPU kernels based on ISA and shape, see Note below.
        scale_and_zero: (K/group_size, N, 2), dtype is the same as the original Tensor dtype

    Non-Tensor Attributes:
        block_size: the block size for quantization, representing the granularity, for groupwise quantization, will have block_size (1, group_size).
                    we only support group_size = 32/64/128.
        shape: shape of the original Tensor

    Optional Tensor Data Attributes:
        act_pre_scale (Optional[Tensor]): Optional scale for activation Tensor, if present,
               we'll multiply activation Tensor with act_pre_scale before applying dynamic
               quantization to activation or running quantized mm op

    Note on Details for data layout for CPU tinygemm kernel:

      We use AVX512 to compute TINYGEMM on CPU. We can also leverage AVX512_VNNI and AMX instructions with torch.compile and max-autotune.
      For data locality, we preshuffle the data in plain layout (N, K/2) to (N/block_n, K, block_n/2), where block_n = 64/32/16.
      See https://github.com/pytorch/pytorch/blob/32eee8ed225d9f10fbbcb38c24b8b44c24c0c97c/aten/src/ATen/native/cpu/int4mm_kernel.cpp#L583 for more details.
    """

    tensor_data_names = ["qdata", "scale_and_zero"]
    tensor_attribute_names = ["block_size", "shape"]
    optional_tensor_data_names = ["act_pre_scale"]

    def __new__(
        cls,
        qdata,
        scale_and_zero,
        block_size,
        shape,
        act_pre_scale: Optional[torch.Tensor] = None,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = scale_and_zero.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        scale_and_zero: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
        act_pre_scale: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale_and_zero = scale_and_zero
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
        assert w.ndim == 2 and w.device.type == "cpu", (
            f"Expecting 2D tensor on CPU, but got: {w.shape} on {w.device.type}"
        )
        assert len(block_size) == w.ndim
        assert block_size[0] == 1 and block_size[1] in (32, 64, 128), (
            f"Expecting groupwise quantization with group size = 32/64/128, but got block_size: {block_size}"
        )
        original_shape = w.shape
        mapping_type = MappingType.ASYMMETRIC
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        eps = 1e-6
        scale_dtype = None
        zero_point_dtype = w.dtype
        scale, zero_point = _choose_qparams_affine_tinygemm(
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
        int_data = _quantize_affine_tinygemm(
            w,
            block_size,
            scale,
            zero_point,
            target_dtype,
            quant_min,
            quant_max,
        )
        assert int_data.dtype == torch.int32, (
            "torch.ops.aten._convert_weight_to_int4pack_for_cpu expects `int32` dtype"
        )
        packed_weight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            int_data,
            1,  # innerKTiles is not needed for CPU
        )

        scale = scale.reshape(int_data.shape[0], -1)
        zero_point = zero_point.reshape(int_data.shape[0], -1)
        from torchao.quantization.utils import pack_tinygemm_scales_and_zeros

        scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero_point, scale.dtype)
        return Int4OpaqueTensor(
            qdata=packed_weight,
            scale_and_zero=scale_and_zero,
            block_size=block_size,
            shape=original_shape,
            act_pre_scale=None,
        )


implements = Int4OpaqueTensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    assert input_tensor.device.type == "cpu", (
        f"For CPU device only but got: {input_tensor.device}"
    )
    assert isinstance(weight_tensor, Int4OpaqueTensor), (
        f"Expected weight_tensor to be Int4OpaqueTensor, got: {type(weight_tensor)}"
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
    scale_and_zero = weight_tensor.scale_and_zero

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype

    # reshape to 2D
    act_mat = act_mat.reshape(-1, act_mat.shape[-1])

    # groupwise int4 quantization
    groupsize = weight_tensor.block_size[1]
    y = torch.ops.aten._weight_int4pack_mm_for_cpu(
        act_mat.contiguous(), packed_weight, groupsize, scale_and_zero
    )

    # remove out_feature padding
    assert weight_tensor.ndim == 2
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y += bias
    return y.to(orig_dtype)


Int4OpaqueTensor.__module__ = "torchao.quantization"

# Allow a model with Int4OpaqueTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4OpaqueTensor])
