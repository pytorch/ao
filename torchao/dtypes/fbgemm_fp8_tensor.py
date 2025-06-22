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
    fill_defaults,
)

__all__ = [
    "to_fbgemm_fp8",
    "FbgemmFp8Tensor",
]

aten = torch.ops.aten


class FbgemmFp8Tensor(TorchAOBaseTensor):
    """
    TODO: needs padding for cutlass kernels
    """

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

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.float8_data.to(device),
            self.scale.to(device),
            self.activation_scale_ub.to(device),
            self.dtype,
        )

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
    orig_act_size = input_tensor.size()
    orig_out_features = weight_tensor.shape[-2]

    # not used
    num_tokens = torch.empty([input_tensor.size(0)], device=input_tensor.device)
    xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
        input_tensor, num_tokens, weight_tensor.activation_scale_ub
    )

    a_data = xq
    b_data = weight_tensor.float8_data

    res = torch.ops.fbgemm.f8f8bf16_rowwise(
        a_data,
        b_data,
        x_scale,
        weight_tensor.scale,
        use_fast_accum=True,
    )
    res = res.reshape(*orig_act_size[:-1], orig_out_features)
    if bias is not None:
        res = res + bias

    return res


@implements(torch.bmm)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = (
        args[0],
        args[1],
    )
    orig_act_size = input_tensor.size()
    # not used
    num_tokens = torch.empty([input_tensor.size(0)], device=input_tensor.device)
    xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
        input_tensor, num_tokens, weight_tensor.activation_scale_ub
    )

    a_data = xq
    b_data = weight_tensor.float8_data
    orig_out_features = b_data.shape[-2]

    res = torch.ops.fbgemm.f8f8bf16_rowwise_batched(
        a_data,
        b_data,
        x_scale,
        weight_tensor.scale,
    )
    res = res.reshape(*orig_act_size[:-1], orig_out_features)
    return res


@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


def _same_metadata(self: "FbgemmFp8Tensor", src: "FbgemmFp8Tensor") -> bool:
    return (
        isinstance(self, FbgemmFp8Tensor)
        and isinstance(src, FbgemmFp8Tensor)
        and self.shape == src.shape
        and self.float8_data.shape == src.float8_data.shape
        and self.scale.shape == src.scale.shape
        and self.activation_scale_ub.shape == src.activation_scale_ub.shape
        and self.dtype == src.dtype
    )


@implements(aten.copy_.default)
def _(func, types, args, kwargs):
    self = args[0]
    src = args[1]
    if _same_metadata(self, src):
        self_tensors = self.__tensor_flatten__()[0]
        for tensor_name in self_tensors:
            getattr(self, tensor_name).copy_(getattr(src, tensor_name))
        return
    raise ValueError(
        f"Not supported args for copy_ due to metadata mismatch: {args[0], args[1]}"
    )


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    """Only supports slicing for dim == 1 and dim == 2
    original tensor shape has dimension (N, K)
    float8_data has dimension (N, K)
    scale (per row quantization) has dimension: (N,)

    since float8_data has the same dimension as original tensor, we can directly slice that
    for scale, we'll do a slice when dim is 0, and don't need to do anything for dim 1

    Note that we need to call slice on the float8_data and scale directly because slice
    is an operation that need to preserve aliasing, see `test_slice_and_copy_` in `test_fbgemm_fp8`
    for
    """
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1
    assert dim == 0 or dim == 1, f"Only dim==0 or 1 are supported, got: {dim}"
    if end >= self.shape[dim]:
        end = self.shape[dim]

    assert self.float8_data.ndim == 2, (
        f"Expected packed weight to have dim 2, got {self.float8_data.dim}"
    )

    # Always slice the float8_data
    sliced_data = aten.slice.Tensor(
        self.float8_data, dim, start, end, step
    ).contiguous()

    if dim == 0:
        # scale has dimension (N,) where N is the dim 0 of `self`
        # so we do the same slice on scale for dimension 0
        sliced_scale = aten.slice.Tensor(self.scale, 0, start, end, step)
    else:
        # since scale is per row, slicing along the dim == 1 dimension does
        # not change the scale
        sliced_scale = self.scale

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        FbgemmFp8Tensor(
            sliced_data, sliced_scale, self.activation_scale_ub, dtype=self.dtype
        ),
    )


to_fbgemm_fp8 = FbgemmFp8Tensor.from_float


if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with FbgemmFp8Tensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([FbgemmFp8Tensor])
