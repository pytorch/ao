# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import List

import torch

from torchao.quantization.quant_primitives import (
    MappingType,
    _choose_qparams_affine_tinygemm,
    _choose_qparams_and_quantize_affine_hqq,
    _quantize_affine_tinygemm,
)
from torchao.quantization.utils import pack_tinygemm_scales_and_zeros
from torchao.utils import TorchAOBaseTensor, fill_defaults, find_multiple

from .int4_choose_qparams_algorithm import Int4ChooseQParamsAlgorithm

__all__ = [
    "Int4TilePackedTo4dTensor",
]

aten = torch.ops.aten


class Int4TilePackedTo4dTensor(TorchAOBaseTensor):
    """
    int4 quantization with tile packed to 4d packing format for groupwise quantization

    Tensor Attributes:
        qdata: tile packed to 4d int4 weight, 4-d tensor of dimension:
               [n / 8][k / (inner_k_tiles * 16)][32][inner_k_tiles / 2]
               (unpacked Tensor shape is n * k)
               (inner_k_tiles is fixed to 8 for Int4TilePackedTo4dTensor)
        scale_and_zero: combined scale and zero point tensor packed for tinygemm kernels

    Non-Tensor Attributes:
        block_size: the block size for quantization, representing the granularity,
                   for example groupwise quantization will have block_size (1, group_size)
        shape: shape of the original Tensor

    Note on Details for tile packed to 4d packing format:

      This is used by tinygemm kernels `_weight_int4pack_mm`. The weight is stored as
      a 4-d packed tensor with specific packing format for efficient computation on tensor cores.
      The packing format optimizes for tensor core matrix multiplication performance.
    """

    tensor_data_names = ["qdata", "scale_and_zero"]
    tensor_attribute_names = ["block_size", "shape"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale_and_zero: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = torch.bfloat16  # This tensor subclass only supports bfloat16
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        scale_and_zero: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
    ):
        self.qdata = qdata
        self.scale_and_zero = scale_and_zero
        self.block_size = block_size

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}"

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        block_size: List[int],
        int4_choose_qparams_algorithm: Int4ChooseQParamsAlgorithm = Int4ChooseQParamsAlgorithm.TINYGEMM,
    ):
        assert len(block_size) == hp_tensor.ndim, (
            f"Expecting the length of block_size to be equal to the dimension of the weight, got {block_size=} and {hp_tensor.ndim=}"
        )

        assert all(x == 1 for x in block_size[:-1]), (
            f"Only per group quantization is supported, got block_size: {block_size}"
        )

        assert hp_tensor.dtype == torch.bfloat16, (
            f"Only bfloat16 is supported for Int4TilePackedTo4dTensor, got {hp_tensor.dtype}"
        )

        original_shape = hp_tensor.shape
        # use a fixed inner_k_tiles value to simplify the argument list and config
        # for Int4TilePackedTo4dTensor
        inner_k_tiles = 8

        # Validate kernel requirements
        orig_out_features, orig_in_features = hp_tensor.shape[-2:]
        # TODO: relax checks to enable quantizing in other platoforms and run in A100
        if not torch.cuda.get_device_capability()[0] >= 8:
            raise ValueError(
                f"Cannot use tinygemm int4 kernel with a device of compute capability {torch.cuda.get_device_capability()}, the minimum compute capability is 8.0 for tensor core kernels."
            )

        # Pre-process: pad to required dimensions
        in_features = find_multiple(orig_in_features, 1024)
        out_features = find_multiple(orig_out_features, 8)
        hp_tensor_padded = torch.nn.functional.pad(
            hp_tensor,
            (0, in_features - orig_in_features, 0, out_features - orig_out_features),
        )

        # Quantize
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15

        # we support two paths for constructing a Int4TilePackedTo4dTensor
        # 1. use [hqq](https://mobiusml.github.io/hqq_blog/) algorithm to compute
        # scale and zero_point, then convert to the format that's compatible with tinygemm kernels
        # 2. don't use hqq, use default tinygemm algorithm to compute scale and zero_point
        #
        # both approach should have the same speed since both are using tinygemm kernel for gemm
        # 1. typically will have higher accuracy compared to 2.
        if int4_choose_qparams_algorithm == Int4ChooseQParamsAlgorithm.HQQ:
            nbits = int(math.log2(quant_max + 1))
            axis = 1
            group_size = block_size[-1]
            compute_dtype = hp_tensor_padded.dtype
            device = hp_tensor_padded.device
            int_data, scale, zero_point, _ = _choose_qparams_and_quantize_affine_hqq(
                hp_tensor_padded,
                nbits=nbits,
                group_size=group_size,
                axis=axis,
                compute_dtype=compute_dtype,
                device=device,
                verbose=False,
                raw_output=False,
                # raw_output=False is basically the 'convert to tinygemm zero_point version' option (add scale*midpoint) that's used in TilePackedTo4d
                # note _choose_qparams_affine_tinygemm does this same thing
            )
            int_data = int_data.to(target_dtype)
        else:
            assert (
                int4_choose_qparams_algorithm == Int4ChooseQParamsAlgorithm.TINYGEMM
            ), (
                f"Unsupported Int4ChooseQParamsAlgorithm: {int4_choose_qparams_algorithm}"
            )
            # Calculate scale and zero_point for tinygemm
            scale, zero_point = _choose_qparams_affine_tinygemm(
                hp_tensor_padded,
                mapping_type=MappingType.ASYMMETRIC,
                block_size=tuple(block_size),
                target_dtype=target_dtype,
                quant_min=quant_min,
                quant_max=quant_max,
                scale_dtype=hp_tensor.dtype,
                zero_point_dtype=hp_tensor.dtype,
            )

            # Quantize for tinygemm
            int_data = _quantize_affine_tinygemm(
                hp_tensor_padded,
                block_size,
                scale,
                zero_point,
                target_dtype,
                quant_min=quant_min,
                quant_max=quant_max,
            )

        # Convert to packed format
        def quant_2d(int_data_2d):
            int_data_2d = (int_data_2d[::, ::2] << 4 | int_data_2d[::, 1::2]).to(
                torch.uint8
            )
            return torch.ops.aten._convert_weight_to_int4pack(
                int_data_2d.contiguous(), inner_k_tiles
            )

        if int_data.dim() == 3:  # for moe quant
            num_experts = int_data.shape[0]
            packed_weight_list = []
            for expert in range(num_experts):
                packed_weight_list.append(quant_2d(int_data[expert]).unsqueeze(0))
            packed_weight = torch.cat(packed_weight_list, dim=0)
            scale = scale.reshape(int_data.shape[0], int_data.shape[-2], -1)
            zero_point = (
                zero_point.reshape(int_data.shape[0], int_data.shape[-2], -1)
                if zero_point is not None
                else None
            )
        else:
            assert int_data.dim() == 2
            packed_weight = quant_2d(int_data)
            scale = scale.reshape(int_data.shape[0], -1)
            zero_point = (
                zero_point.reshape(int_data.shape[0], -1)
                if zero_point is not None
                else None
            )

        scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero_point, scale.dtype)

        return cls(
            qdata=packed_weight,
            scale_and_zero=scale_and_zero,
            block_size=block_size,
            shape=original_shape,
        )


implements = Int4TilePackedTo4dTensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    assert weight_tensor.qdata.is_contiguous(), "Expected qdata to be contiguous"
    assert weight_tensor.scale_and_zero.is_contiguous(), (
        "Expected scale_and_zero to be contiguous"
    )

    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"need input_tensor shape: {input_tensor.shape} final"
        f"dim to match weight_tensor shape: {weight_tensor.shape} second dim "
    )

    # weight is packed from padded (out_features, in_features) weight tensor
    # (same dimension requirement as F.linear weight)
    packed_weight = weight_tensor.qdata
    scale_and_zero = weight_tensor.scale_and_zero
    original_shape = weight_tensor.shape

    orig_act_size = input_tensor.size()
    orig_dtype = input_tensor.dtype

    # Folds batch dimension into the first dimension
    act_mat = input_tensor.reshape(-1, input_tensor.shape[-1]).to(torch.bfloat16)
    pad_size = find_multiple(act_mat.shape[-1], 1024)
    act_mat = torch.nn.functional.pad(act_mat, (0, pad_size - act_mat.shape[-1]))

    # groupwise int4 quantization
    groupsize = weight_tensor.block_size[-1]
    if act_mat.numel() == 0:  # handling for empty input
        y = act_mat
    else:
        y = torch.ops.aten._weight_int4pack_mm(
            act_mat, packed_weight, groupsize, scale_and_zero
        )
    # remove out_feature padding
    orig_out_features = original_shape[-2]
    y = y[:, :orig_out_features]

    # Unfold the batch dimension
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y += bias.to(y.dtype)
    return y.to(orig_dtype)


@implements(aten.slice.Tensor)
def _(func, _types, args, _kwargs):
    """Slice operation for tensor core tiled packed tensor"""
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    cur_shape = self.shape

    assert len(cur_shape) == 2
    assert self.qdata.dim() == 4
    # qdata has shape [n/8, k/(inner_k_tiles*16), 32, inner_k_tiles/2]
    n_by_8, k_by_inner_tiles, _, _ = self.qdata.shape
    sz_dim1, sz_dim0, _ = self.scale_and_zero.shape

    data_len = cur_shape[dim]
    assert dim in [
        0,
        1,
    ], (
        f"Int4TilePackedTo4dTensor slice: attempting to run {func}, with dim={dim}, that is not supported"
    )

    if dim == 0:
        pw_len = n_by_8
        sz_len = sz_dim0
    else:
        pw_len = k_by_inner_tiles
        sz_len = sz_dim1

    if pw_len == 0 or sz_len == 0:
        return Int4TilePackedTo4dTensor(
            self.qdata,
            self.scale_and_zero,
            self.block_size,
            self.shape,
        )

    pw_ratio = data_len / pw_len
    start_pw = int(start / pw_ratio)
    end_pw = int(end / pw_ratio)

    sz_ratio = data_len / sz_len
    start_sz = int(start / sz_ratio)
    end_sz = int(end / sz_ratio)

    qdata = aten.slice(self.qdata, dim, start_pw, end_pw, step)
    scale_and_zero = aten.slice(self.scale_and_zero, 1 - dim, start_sz, end_sz, step)

    # Calculate new shape after slicing
    new_shape = list(self.shape)
    new_shape[dim] = end - start

    block_size = list(self.block_size)
    block_size[dim] = min(block_size[dim], new_shape[dim])

    return Int4TilePackedTo4dTensor(
        qdata,
        scale_and_zero,
        block_size,
        new_shape,
    )


Int4TilePackedTo4dTensor.__module__ = "torchao.quantization"

# Allow a model with Int4TilePackedTo4dTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4TilePackedTo4dTensor])
