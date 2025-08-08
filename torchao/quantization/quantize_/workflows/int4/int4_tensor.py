# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
    fill_defaults,
)

__all__ = [
    "Int4Tensor",
]

aten = torch.ops.aten


try:
    from fbgemm_gpu.experimental.gen_ai.quantize import int4_row_quantize_zp, pack_int4
except:
    int4_row_quantize_zp = None
    pack_int4 = None


class Int4Tensor(TorchAOBaseTensor):
    """
    int4 quantization with plain (default) packing format (for all granularities)

    Tensor Attributes:
        _data: packed int4 weight, either 2D (N, K/2) or 3D (B, N, K/2), last dimension is packed
        scale: (K/group_size, N) for 2D Tensor, (B, N, K/group_size) for 3D Tensor, where B is batch size,
               dtype is the same as the original Tensor dtype
        zero_point: (K/group_size, N) for 2D Tensor, (B, N, K/group_size) for 3D Tensor, where B is batch size,
               dtype is the same as the original Tensor dtype

    Non-Tensor Attributes:
        block_size: the block size for quantization, representing the granularity, for example groupwise quantization will have block_size (1, group_size)
        shape: the shape of the original Tensor
    """

    tensor_data_attrs = ["_data", "scale", "zero_point"]
    tensor_attributes = ["block_size", "shape"]

    def __new__(cls, _data, scale, zero_point, block_size, shape):
        kwargs = {}
        kwargs["device"] = _data.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, _data, scale, zero_point, block_size, shape):
        self._data = _data
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size

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
            f"{self.__class__.__name__}(weight={self._data}, block_size={self.block_size}, "
            f"shape={self.shape}, device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}"

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self._data.to(device),
            self.scale.to(device),
            self.zero_point.to(device),
            self.block_size,
            self.shape,
        )

    @classmethod
    def from_float(
        cls,
        w: torch.Tensor,
        block_size: List[int],
    ):
        assert len(block_size) == w.ndim, (
            f"Expecting the length of block_size to be equal to the dimension of the weight, got {block_size=} and {w.ndim=}"
        )
        if int4_row_quantize_zp is None:
            raise ImportError("Requires fbgemm-gpu-genai >= 1.2.0")

        assert all(x == 1 for x in block_size[:-1]) and block_size[-1] != 1, (
            "Only groupwise quant is supported right now"
        )

        group_size = block_size[-1]
        original_shape = w.shape

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
        return Int4Tensor(
            _data=wq,
            scale=scale,
            zero_point=zero_point,
            block_size=block_size,
            shape=original_shape,
        )


implements = Int4Tensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    orig_act_size = input_tensor.size()
    orig_out_features = weight_tensor.shape[-2]

    res = torch.ops.fbgemm.bf16i4bf16_rowwise(
        input_tensor,
        weight_tensor._data.contiguous(),
        weight_tensor.scale.contiguous(),
        weight_tensor.zero_point.contiguous(),
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
    orig_out_features = weight_tensor.shape[-2]

    res = torch.ops.fbgemm.bf16i4bf16_rowwise_batched(
        input_tensor,
        weight_tensor._data.contiguous(),
        weight_tensor.scale,
        weight_tensor.zero_point,
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


def _same_metadata(self: "Int4Tensor", src: "Int4Tensor") -> bool:
    return (
        isinstance(self, Int4Tensor)
        and isinstance(src, Int4Tensor)
        and self.shape == src.shape
        and self._data.shape == src._data.shape
        and self.scale.shape == src.scale.shape
        and self.zero_point.shape == src.zero_point.shape
        and self.block_size == src.block_size
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
    _data has dimension: (N, K/2)
    scale and zero_point has dimension: (K/groups, N)

    dim, start, end, step are args that's referring to the original tensor shape
    which is (N, K), and we need to map that to the transformed weight shape of _data,
    scale and zero_point

    when dim == 0: we do a slice on _data dim 0, and on dim 1 of scale and zero_point,
    also adjust the start and end indexes based on the ratio between original shape and the shape
    of _data and scale/zero_point

    when dim == 1: we do a slice on _data dim 1 and dim 0 of scale and zero_point and do the
    same adjustment based on ratio

    Note that we need to call slice on the _data, scale and zero_point directly because slice
    is an operation that need to preserve aliasing, see `test_slice_and_copy_` in `test_fbgemm_int4`
    for
    """
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1
    assert dim == 0 or dim == 1, f"Only dim==0 or 1 are supported, got: {dim}"
    if end >= self.shape[dim]:
        end = self.shape[dim]

    assert self._data.ndim == 2, (
        f"Expected packed weight to have dim 2, got {self._data.dim}"
    )
    N, K_by_2 = self._data.shape
    sz_dim0, sz_dim1 = self.scale.shape

    data_len = self.shape[dim]

    if dim == 0:
        pw_len = N
        sz_len = sz_dim1
    else:
        pw_len = K_by_2
        sz_len = sz_dim0

    sz_dim = 1 - dim
    if pw_len == 0 or sz_len == 0:
        return return_and_correct_aliasing(
            func,
            args,
            kwargs,
            self.__class__(
                self._data,
                self.scale,
                self.zero_point,
                block_size=self.block_size,
                shape=self.shape,
            ),
        )

    pw_ratio = data_len / pw_len
    start_pw = int(start / pw_ratio)
    end_pw = int(end / pw_ratio)

    sz_ratio = data_len / sz_len
    start_sz = int(start / sz_ratio)
    end_sz = int(end / sz_ratio)

    _data = aten.slice.Tensor(self._data, dim, start_pw, end_pw, step)
    scale = aten.slice.Tensor(self.scale, sz_dim, start_sz, end_sz, step)
    zero_point = aten.slice.Tensor(self.zero_point, sz_dim, start_sz, end_sz, step)
    packed_shape0, packed_shape1 = _data.shape
    new_shape = (packed_shape0, packed_shape1 * 2)
    new = self.__class__(
        _data, scale, zero_point, block_size=self.block_size, shape=new_shape
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


Int4Tensor.__module__ = "torchao.quantization"

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with Int4Tensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([Int4Tensor])
