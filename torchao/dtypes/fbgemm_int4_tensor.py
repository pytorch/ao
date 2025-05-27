# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TorchAOBaseTensor

__all__ = [
    "to_fbgemm_int4",
]

aten = torch.ops.aten


# copied from https://github.com/pytorch/FBGEMM/blob/2bf4d9aa739b3e78362ca801a72dacb16c67346f/fbgemm_gpu/experimental/gen_ai/gen_ai/quantize.py#L60
def int4_row_quantize(
    x: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_bit = 4  # Number of target bits.
    to_quant = x.reshape(-1, group_size).to(torch.float)

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int

    zeros = min_val + scales * (2 ** (n_bit - 1))

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)

    # Recenter output and move to int8.
    out = (out - 2 ** (n_bit - 1)).to(dtype=torch.int8).reshape(x.shape)

    # Cutlass expects column major layout for scale and zero point,
    # so we transpose here and make them contiguous.
    scales = scales.view(x.shape[0], -1).t().contiguous()
    zeros = zeros.view(x.shape[0], -1).t().contiguous()

    return out, scales.to(x.dtype), zeros.to(x.dtype)


# copied from https://github.com/pytorch/FBGEMM/blob/2bf4d9aa739b3e78362ca801a72dacb16c67346f/fbgemm_gpu/experimental/gen_ai/gen_ai/quantize.py#L18
def pack_int4(x: torch.Tensor) -> torch.Tensor:
    # Given int8 x, pack adjacent int4 values into a single int8.
    low_x = x[:, ::2]
    high_x = x[:, 1::2]

    # High bits need to left shift, this also masks off extra bits.
    high_x = torch.bitwise_left_shift(high_x, 4)
    # Low bits need to have sign bits removed.
    low_x = torch.bitwise_and(low_x, 0xF)

    # Recombine into a single value with bitwise or.
    return torch.bitwise_or(low_x, high_x).contiguous()


class FbgemmInt4Tensor(TorchAOBaseTensor):
    tensor_data_attrs = ["packed_weight", "scale", "zero_point"]
    tensor_attributes = ["group_size"]

    def __new__(cls, packed_weight, scale, zero_point, group_size):
        shape = packed_weight.shape
        kwargs = {}
        kwargs["device"] = packed_weight.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, packed_weight, scale, zero_point, group_size):
        self.packed_weight = packed_weight
        self.scale = scale
        self.zero_point = zero_point
        self.group_size = group_size

    def __tensor_flatten__(self):
        return self.tensor_data_attrs, [
            getattr(self, attr) for attr in self.tensor_attributes
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        return cls(
            *[tensor_data_dict[name] for name in cls.tensor_data_attrs],
            *tensor_attributes,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            *[fn(getattr(self, attr)) for attr in self.tensor_data_attrs],
            *[getattr(self, attr) for attr in self.tensor_attributes],
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(weight={self.packed_weight}, group_size={self.group_size}, "
            f"shape={self.shape}, device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    @classmethod
    def from_float(cls, w: torch.Tensor, group_size: int = 128):
        if w.ndim >= 3:
            wq, scale, zero_point = zip(
                *[int4_row_quantize(i, group_size) for i in w], strict=False
            )
            wq = torch.stack([pack_int4(i) for i in wq], dim=0)
            scale = torch.stack(scale, dim=0)
            zero_point = torch.stack(zero_point, dim=0)
        else:
            wq, scale, zero_point = int4_row_quantize(w, group_size)
            wq = pack_int4(wq)
        del w
        return FbgemmInt4Tensor(
            packed_weight=wq,
            scale=scale,
            zero_point=zero_point,
            group_size=group_size,
        )


implements = FbgemmInt4Tensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if not input_tensor.is_floating_point():
        raise NotImplementedError(
            f"{func} is not implemented for non floating point input"
        )

    orig_act_size = input_tensor.size()
    orig_out_features = weight_tensor.shape[-2]

    res = torch.ops.fbgemm.bf16i4bf16_rowwise(
        input_tensor,
        weight_tensor.packed_weight,
        weight_tensor.scale,
        weight_tensor.zero_point,
    )
    if bias is not None:
        res = res + bias
    return res.reshape(*orig_act_size[:-1], orig_out_features)


@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@implements([aten.clone.default, aten.copy_.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


to_fbgemm_int4 = FbgemmInt4Tensor.from_float
