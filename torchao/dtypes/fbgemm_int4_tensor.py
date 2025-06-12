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
    "to_fbgemm_int4",
    "FbgemmInt4Tensor",
]

aten = torch.ops.aten


try: 
    from fbgemm_gpu.experimental.gen_ai.quantize import int4_row_quantize_zp, pack_int4
except:
    int4_row_quantize_zp = None
    pack_int4 = None


class FbgemmInt4Tensor(TorchAOBaseTensor):
    tensor_data_attrs = ["packed_weight", "scale", "zero_point"]
    tensor_attributes = ["group_size", "shape"]

    def __new__(cls, packed_weight, scale, zero_point, group_size, shape):
        kwargs = {}
        kwargs["device"] = packed_weight.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, packed_weight, scale, zero_point, group_size, shape):
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

    def _quantization_type(self):
        return f"shape={self.shape}, group_size={self.group_size}, device={self.device}"

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.packed_weight.to(device),
            self.scale.to(device),
            self.zero_point.to(device),
            self.group_size,
            self.shape,
        )

    @classmethod
    def from_float(
        cls,
        w: torch.Tensor,
        block_size: List[int],
        transpose_input: bool = False,
    ):
        assert len(block_size) == w.ndim, (
            f"Expecting the length of block_size to be equal to the dimension of the weight, got {block_size=} and {w.ndim=}"
        )
        if int4_row_quantize_zp is None:
            raise ImportError("Requires fbgemm-gpu-genai >= 1.2.0")

        if transpose_input:
            if w.ndim == 3:
                w = w.transpose(-1, -2)
            else:
                w = w.t()

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
        return FbgemmInt4Tensor(
            packed_weight=wq,
            scale=scale,
            zero_point=zero_point,
            group_size=group_size,
            shape=original_shape,
        )


implements = FbgemmInt4Tensor.implements


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
        weight_tensor.packed_weight.contiguous(),
        weight_tensor.scale,
        weight_tensor.zero_point,
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
        weight_tensor.packed_weight.contiguous(),
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


def _same_metadata(self: "FbgemmInt4Tensor", src: "FbgemmInt4Tensor") -> bool:
    return (
        isinstance(self, FbgemmInt4Tensor)
        and isinstance(src, FbgemmInt4Tensor)
        and self.shape == src.shape
        and self.packed_weight.shape == src.packed_weight.shape
        and self.scale.shape == src.scale.shape
        and self.zero_point.shape == src.zero_point.shape
        and self.group_size == src.group_size
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
    packed_weight has dimension: (N, K/2)
    scale and zero_point has dimension: (K/groups, N)

    dim, start, end, step are args that's referring to the original tensor shape
    which is (N, K), and we need to map that to the transformed weight shape of packed_weight,
    scale and zero_point

    when dim == 0: we do a slice on packed_weight dim 0, and on dim 1 of scale and zero_point,
    also adjust the start and end indexes based on the ratio between original shape and the shape
    of packed_weight and scale/zero_point

    when dim == 1: we do a slice on packed_weight dim 1 and dim 0 of scale and zero_point and do the
    same adjustment based on ratio

    Note that we need to call slice on the packed_weight, scale and zero_point directly because slice
    is an operation that need to preserve aliasing, see `test_slice_and_copy_` in `test_fbgemm_int4`
    for
    """
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1
    assert dim == 0 or dim == 1, f"Only dim==0 or 1 are supported, got: {dim}"
    if end >= self.shape[dim]:
        end = self.shape[dim]

    assert self.packed_weight.ndim == 2, (
        f"Expected packed weight to have dim 2, got {self.packed_weight.dim}"
    )
    N, K_by_2 = self.packed_weight.shape
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
                self.packed_weight,
                self.scale,
                self.zero_point,
                group_size=self.group_size,
                shape=self.shape,
            ),
        )

    pw_ratio = data_len / pw_len
    start_pw = int(start / pw_ratio)
    end_pw = int(end / pw_ratio)

    sz_ratio = data_len / sz_len
    start_sz = int(start / sz_ratio)
    end_sz = int(end / sz_ratio)

    packed_weight = aten.slice.Tensor(self.packed_weight, dim, start_pw, end_pw, step)
    scale = aten.slice.Tensor(self.scale, sz_dim, start_sz, end_sz, step)
    zero_point = aten.slice.Tensor(self.zero_point, sz_dim, start_sz, end_sz, step)
    packed_shape0, packed_shape1 = packed_weight.shape
    new_shape = (packed_shape0, packed_shape1 * 2)
    new = self.__class__(
        packed_weight, scale, zero_point, group_size=self.group_size, shape=new_shape
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


to_fbgemm_int4 = FbgemmInt4Tensor.from_float


if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with FbgemmInt4Tensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([FbgemmInt4Tensor])
