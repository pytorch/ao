# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

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
    "Int4WoqCpuTensor",
]

aten = torch.ops.aten


class Int4WoqCpuTensor(TorchAOBaseTensor):
    """
    int4 weight-only quantization on CPU (groupwise quantization only)

    Tensor Attributes:
        qdata: preshuffled and packed int4 weight, always viewed as a 2D (N, K/2) tensor, last dimension is packed
               preshuffling is specific to CPU kernels, see Note below.
        qscale_and_zero: (K/group_size, N, 2), dtype is the same as the original Tensor dtype

    Non-Tensor Attributes:
        block_size: the block size for quantization, representing the granularity, for groupwise quantization, will have block_size (1, group_size).
                    we only support group_size = 32/64/128.
        shape: shape of the original Tensor

    Note on Details for data layout for CPU kernel:

      We use AVX512, AVX512_VNNI and AMX instructions (torch.compile and max-autotune needed for the latter two) to compute GEMM on CPU.
      For data locality, we preshuffle the data in plain layout (N, K/2) to (N/block_n, K, block_n/2), where block_n = 64. And when packing
      the last dimension, data are shuffled by lanes before packing two int4 to one int8:
      block_n = 64 = 16 * 4, so we have 4 lanes, each lane has 16 int4s = [lane0, lane1, lane2, lane3]. We pack them as [lane0|lane2, lane1|lane3].
      See https://github.com/pytorch/pytorch/blob/32eee8ed225d9f10fbbcb38c24b8b44c24c0c97c/aten/src/ATen/native/cpu/int4mm_kernel.cpp#L583 for more details.
    """

    tensor_data_names = ["qdata", "qscale_and_zero"]
    optional_tensor_data_names = []
    tensor_attribute_names = ["block_size", "shape"]

    def __new__(
        cls,
        qdata,
        qscale_and_zero,
        block_size,
        shape,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = qscale_and_zero.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        qscale_and_zero: torch.Tensor,
        block_size: List[int],
        shape: List[int],
    ):
        self.qdata = qdata
        self.qscale_and_zero = qscale_and_zero
        self.block_size = block_size

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}"

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: List[int],
    ):
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
            1,  # TODO:remove
        )

        scale = scale.reshape(int_data.shape[0], -1)
        zero_point = zero_point.reshape(int_data.shape[0], -1)
        from torchao.quantization.utils import pack_tinygemm_scales_and_zeros

        scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero_point, scale.dtype)
        return Int4WoqCpuTensor(
            qdata=packed_weight,
            qscale_and_zero=scale_and_zero,
            block_size=block_size,
            shape=original_shape,
        )


implements = Int4WoqCpuTensor.implements


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
    assert isinstance(weight_tensor, Int4WoqCpuTensor), (
        f"Expected weight_tensor to be Int4WoqCpuTensor, got: {type(weight_tensor)}"
    )
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"need input_tensor shape: {input_tensor.shape} final"
        f"dim to match weight_tensor shape: {weight_tensor.shape} second dim "
    )

    act_mat = input_tensor
    packed_weight = weight_tensor.qdata.contiguous()
    scale_and_zero = weight_tensor.qscale_and_zero.contiguous()

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
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y += bias
    return y.to(orig_dtype)


Int4WoqCpuTensor.__module__ = "torchao.quantization"

# Allow a model with Int4WoqCpuTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4WoqCpuTensor])
