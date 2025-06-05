# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
)

__all__ = [
    "to_fbgemm_fp8",
]

aten = torch.ops.aten


class FbgemmFp8Tensor(TorchAOBaseTensor):
    tensor_data_attrs = ["float8_data", "scale", "activation_scale_ub"]
    tensor_attributes = ["dtype"]

    def __new__(cls, float8_data, scale, activation_scale_ub, dtype):
        shape = float8_data.shape
        kwargs = {}
        kwargs["device"] = float8_data.device
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, float8_data, scale, activation_scale_ub, dtype):
        self.float8_data = float8_data
        self.scale = scale
        self.activation_scale_ub = activation_scale_ub

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
            f"{self.__class__.__name__}(weight={self.float8_data}, scale={self.scale}, "
            f"activation_scale_ub={self.activation_scale_ub}, "
            f"shape={self.shape}, device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def _quantization_type(self):
        return f"shape={self.shape}, activation_scale_ub={self.activation_scale_ub}, device={self.device}"

    @classmethod
    def from_float(
        cls,
        w: torch.Tensor,
        activation_scale_ub: Optional[float] = None,
    ):
        if activation_scale_ub is None:
            activation_scale_ub = 1200.0

        activation_scale_ub = torch.tensor(
            [activation_scale_ub],
            dtype=torch.float,
            device=w.device,
        )
        wq, w_scale = torch.ops.triton.quantize_fp8_row(w)
        # wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
        dtype = w.dtype
        del w
        return FbgemmFp8Tensor(
            wq,
            w_scale,
            activation_scale_ub=activation_scale_ub,
            dtype=dtype,
        )


implements = FbgemmFp8Tensor.implements


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

    # not used
    num_tokens = torch.empty([input_tensor.size(0)], device=input_tensor.device)
    xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
        input_tensor, num_tokens, weight_tensor.activation_scale_ub
    )
    res = torch.ops.fbgemm.f8f8bf16_rowwise(
        xq,
        weight_tensor.float8_data,
        x_scale,
        weight_tensor.scale,
        use_fast_accum=True,
    )
    res = res.reshape(*orig_act_size[:-1], orig_out_features)
    if bias is not None:
        res = res + bias

    return res


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


to_fbgemm_fp8 = FbgemmFp8Tensor.from_float


if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with FbgemmFp8Tensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([FbgemmFp8Tensor])
