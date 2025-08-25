# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
from typing import List, Optional

import torch

from torchao.utils import (
    TorchAOBaseTensor,
)

__all__ = [
    "Int4PreshuffledTensor",
]

aten = torch.ops.aten


if (
    importlib.util.find_spec("fbgemm_gpu") is None
    or importlib.util.find_spec("fbgemm_gpu.experimental") is None
):
    quantize_int4_preshuffle = None
    quantize_fp8_row = None
else:
    from fbgemm_gpu.experimental.gen_ai.quantize import (
        quantize_fp8_row,
        quantize_int4_preshuffle,
    )


class Int4PreshuffledTensor(TorchAOBaseTensor):
    """
    int4 quantization with preshuffled packing format (for all granularities)

    Tensor Attributes:
        qdata: preshuffled and packed int4 weight, either 2D (N, K/2) or 3D (B, N, K/2), last dimension is packed
               preshuffling is specific to fbgemm kernels, see Note for motivation, detailed layout doc is WIP
        for bf16 activation:
            group_scale: (K/group_size, N) for 2D Tensor, (B, K/group_size, N) for 3D Tensor, where B is batch size,
                   dtype is the same as the original Tensor dtype
            group_zero: (K/group_size, N) for 2D Tensor, (B, K/group_size, N) for 3D Tensor, where B is batch size,
                   dtype is the same as the original Tensor dtype
        for float8 activation:
            group_scale: (K/group_size/8, 8, N) for 2D Tensor, (B, K/group_size/8, 8, N) for 3D Tensor
                   dtype is float8
            row_scale: (N,) for 2D Tensor, (B, N) for 3D Tensor
                   dtype is the same as the original Tensor dtype

    Non-Tensor Attributes:
        block_size: the block size for quantization, representing the granularity, for example groupwise quantization will have block_size (1, group_size)
        shape: shape of the original Tensor

    Note on Details for preshuffle for fbgemm kernel:

      We use WGMMA instruction for efficient matrix multiplication in H100 Tensor Core.
      To address a major inefficiency in how WGMMA tiles are loaded into shared memory before
      dispatching to tensor cores, Each thread of an FP8 WGMMA reads 4 groups for 4 elements
      (or 4 groups of 2 elements for BF16) into local registers. Each of those groups thus
      contains a total 32 bits, which can be efficiently loaded using a single 32-bit load instruction.
      However, weights are loaded using the same format. As the INT4 weights are only 4-bits each,
      one group has a total of 16 bits. Unfortunately, 16 bit loads are not any faster than 32 bit
      loads so having to load all four groups is wasteful. We can optimize weight loading by shuffling
      the order of elements such that all 4 groups are sequential in memory. This allows us to
      perform a single 64 bit load to move all needed weights for the thread into register memory.

    Note for float8 activation int4 weight kernel:
      float8 activation int4 weight kernel doesn't work with zero_point, since it use table lookup approach which
      requires symmetric quantization
    """

    tensor_data_names = ["qdata", "group_scale"]
    tensor_attribute_names = ["block_size", "shape"]
    optional_tensor_data_names = ["group_zero", "row_scale"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        group_scale: torch.Tensor,
        block_size: List[int],
        shape: List[int],
        group_zero: Optional[torch.Tensor] = None,
        row_scale: Optional[torch.Tensor] = None,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = group_scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        group_scale: torch.Tensor,
        block_size: List[int],
        shape: List[int],
        group_zero: Optional[torch.Tensor] = None,
        row_scale: Optional[torch.Tensor] = None,
    ):
        # one and only one of group_scale and group_zero should be None
        assert group_zero is None or row_scale is None
        assert not (group_zero is not None and row_scale is not None)
        self.qdata = qdata
        self.row_scale = row_scale
        self.block_size = block_size
        self.group_scale = group_scale
        self.group_zero = group_zero

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}"

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: List[int],
        activation_dtype: torch.dtype = torch.bfloat16,
    ):
        assert len(block_size) == w.ndim, (
            f"Expecting the length of block_size to be equal to the dimension of the weight, got {block_size=} and {w.ndim=}"
        )

        assert all(x == 1 for x in block_size[:-1]), (
            f"Only per group quantization is supported, got block_size: {block_size}"
        )

        _SUPPORTED_DTYPE_TO_STR = {
            torch.bfloat16: "bf16",
            torch.float8_e4m3fn: "fp8",
        }
        assert activation_dtype in _SUPPORTED_DTYPE_TO_STR, (
            f"activation dtype {activation_dtype} is not supported, supported ones are: {_SUPPORTED_DTYPE_TO_STR.keys()}"
        )

        if quantize_int4_preshuffle is None:
            raise ImportError("Requires fbgemm-gpu-genai >= 1.2.0")

        assert all(x == 1 for x in block_size[:-1]) and block_size[-1] != 1, (
            "Only groupwise quant is supported right now"
        )
        original_shape = w.shape
        group_size = block_size[-1]

        activation_dtype_str = _SUPPORTED_DTYPE_TO_STR[activation_dtype]

        if w.ndim >= 3:
            wq, scales = zip(
                *[
                    quantize_int4_preshuffle(
                        i.cuda(), group_size=group_size, dtype=activation_dtype_str
                    )
                    for i in w
                ]
            )
            wq = torch.stack(wq, dim=0)
            group_scale, group_zero_or_row_scale = zip(*scales)
            group_zero_or_row_scale = torch.stack(
                group_zero_or_row_scale, dim=0
            ).contiguous()
            group_scale = torch.stack(group_scale, dim=0).contiguous()
        else:
            wq, (group_scale, group_zero_or_row_scale) = quantize_int4_preshuffle(
                w.cuda(), group_size=group_size, dtype=activation_dtype_str
            )

        if activation_dtype == torch.bfloat16:
            group_zero = group_zero_or_row_scale
            row_scale = None
        else:
            group_zero = None
            row_scale = group_zero_or_row_scale

        return Int4PreshuffledTensor(
            qdata=wq,
            group_scale=group_scale,
            block_size=block_size,
            shape=original_shape,
            group_zero=group_zero,
            row_scale=row_scale,
        )


implements = Int4PreshuffledTensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    orig_input_size = input_tensor.size()
    orig_out_features = weight_tensor.shape[-2]

    wq = weight_tensor.qdata.contiguous()
    group_scale = weight_tensor.group_scale.contiguous()
    if weight_tensor.group_zero is not None:
        # bf16 activation
        group_zero = weight_tensor.group_zero.contiguous()
        res = torch.ops.fbgemm.bf16i4bf16_shuffled(
            input_tensor, wq, group_scale, group_zero
        )
    else:
        # dynamically quantizes activation to fp8
        assert weight_tensor.row_scale is not None
        row_scale = weight_tensor.row_scale.contiguous()
        xq, x_scale = quantize_fp8_row(input_tensor)
        res = torch.ops.fbgemm.f8i4bf16_shuffled(
            xq, wq, x_scale, row_scale, group_scale
        )

    res = res.reshape(*orig_input_size[:-1], orig_out_features)
    if bias is not None:
        res = res + bias
    return res


@implements(torch.bmm)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = (
        args[0],
        args[1],
    )
    orig_input_size = input_tensor.size()
    orig_out_features = weight_tensor.shape[-2]

    wq = weight_tensor.qdata.contiguous()
    group_scale = weight_tensor.group_scale.contiguous()
    if weight_tensor.group_zero is not None:
        # bfloat16 activation
        group_zero = weight_tensor.group_zero.contiguous()
        res = torch.ops.fbgemm.bf16i4bf16_shuffled_batched(
            input_tensor, wq, group_scale, group_zero
        )
    else:
        # dynamically quantizes activation to fp8
        assert weight_tensor.row_scale is not None
        row_scale = weight_tensor.row_scale.contiguous()
        xq, x_scale = quantize_fp8_row(input_tensor)
        # From: https://github.com/pytorch/FBGEMM/blob/ba8f2b7adb90e096cff8818716f7cc3587030f70/fbgemm_gpu/experimental/gen_ai/bench/quantize_ops.py#L1654
        assert xq.dim() == 3
        B, M, _ = xq.shape
        _, N, _ = wq.shape
        res = torch.empty((B, M, N), device=xq.device, dtype=torch.bfloat16)
        for i in range(B):
            res[i] = torch.ops.fbgemm.f8i4bf16_shuffled(
                xq[i], wq[i], x_scale[i], row_scale[i], group_scale[i]
            )

    res = res.reshape(*orig_input_size[:-1], orig_out_features)
    return res


Int4PreshuffledTensor.__module__ = "torchao.quantization"

# Allow a model with Int4PreshuffledTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4PreshuffledTensor])
