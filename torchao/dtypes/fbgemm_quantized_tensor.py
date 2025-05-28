# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
from typing import List

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TorchAOBaseTensor

__all__ = [
    "to_fbgemm_quantized",
]

aten = torch.ops.aten


if importlib.util.find_spec("fbgemm_gpu") is None:
    int4_row_quantize_zp = None
    pack_int4 = None
else:
    from fbgemm_gpu.experimental.gen_ai.quantize import int4_row_quantize_zp, pack_int4


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
    def from_float(
        cls,
        w: torch.Tensor,
        input_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        output_dtype: torch.dtype,
        block_size: List[int],
    ):
        assert len(block_size) == w.ndim, (
            f"Expecting the length of block_size to be equal to the dimension of the weight, got {block_size=} and {w.ndim=}"
        )
        group_size = block_size[-1]

        assert (input_dtype, weight_dtype, output_dtype) == (
            torch.bfloat16,
            torch.int4,
            torch.bfloat16,
        )

        if w.ndim >= 3:
            wq, scale, zero_point = zip(
                *[int4_row_quantize_zp(i, group_size) for i in w], strict=False
            )
            wq = torch.stack([pack_int4(i) for i in wq], dim=0)
            scale = torch.stack(scale, dim=0)
            zero_point = torch.stack(zero_point, dim=0)
        else:
            wq, scale, zero_point = int4_row_quantize_zp(w, group_size)
            wq = pack_int4(wq)

        scale = scale.to(w.dtype)
        zero_point = zero_point.to(w.dtype)

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


# We can have `to_fbgemm_tensor` to dispatch to different Fbgemm tensors later
to_fbgemm_quantized = FbgemmInt4Tensor.from_float
