# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
from typing import List

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
    fill_defaults,
)

__all__ = [
    "to_int4_groupwise_preshuffle",
    "Int4GroupwisePreshuffleTensor",
]

aten = torch.ops.aten


if importlib.util.find_spec("fbgemm_gpu") is None:
    quantize_int4_preshuffle = None
else:
    from fbgemm_gpu.experimental.gen_ai.quantize import quantize_int4_preshuffle


class Int4GroupwisePreshuffleTensor(TorchAOBaseTensor):
    """
    Args:
        shape_multiplier: is the multipler from packed_weight to the real weight, since
        we pack the weight for int4, for example, when we pack the last dimension for
        a 2D tensor, the shape_multiplier will be [1, 2]
    """

    tensor_data_attrs = ["packed_weight", "group_scale", "row_scale"]
    tensor_attributes = ["group_size", "shape_multiplier", "shape"]

    def __new__(
        cls, packed_weight, group_scale, row_scale, group_size, shape_multiplier, shape
    ):
        kwargs = {}
        kwargs["device"] = packed_weight.device
        kwargs["dtype"] = group_scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self, packed_weight, group_scale, row_scale, group_size, shape_multiplier, shape
    ):
        self.packed_weight = packed_weight
        self.group_scale = group_scale
        self.row_scale = row_scale
        self.shape_multiplier = shape_multiplier
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
            f"shape_multiplier={self.shape_multiplier}, shape={self.shape}, device={self.device}, dtype={self.dtype}, "
            f"requires_grad={self.requires_grad})"
        )

    def _quantization_type(self):
        return f"shape={self.shape}, group_size={self.group_size}, device={self.device}"

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.packed_weight.to(device),
            self.group_scale.to(device),
            self.row_scale.to(device),
            self.group_size,
            self.shape_multiplier,
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
        if quantize_int4_preshuffle is None:
            raise ImportError("Requires fbgemm-gpu-genai >= 1.2.0")

        group_size = block_size[-1]
        original_shape = w.shape

        if w.ndim >= 3:
            wq, scales = zip(
                *[quantize_int4_preshuffle(i.cuda(), dtype="bf16") for i in w]
            )
            wq = torch.stack(wq, dim=0)
            group_scale, row_scale = zip(*scales)
            row_scale = torch.stack(row_scale, dim=0)
            group_scale = torch.stack(group_scale, dim=0)
        else:
            wq, (group_scale, row_scale) = quantize_int4_preshuffle(
                w.cuda(), dtype="bf16"
            )

        shape_multiplier = [1] * wq.ndim
        shape_multiplier[-1] = 2

        del w
        return Int4GroupwisePreshuffleTensor(
            packed_weight=wq,
            group_scale=group_scale,
            row_scale=row_scale,
            group_size=group_size,
            shape_multiplier=shape_multiplier,
            shape=original_shape,
        )


implements = Int4GroupwisePreshuffleTensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    orig_input_size = input_tensor.size()
    orig_out_features = weight_tensor.shape[-2]

    wq = weight_tensor.packed_weight
    group_scale = weight_tensor.group_scale
    row_scale = weight_tensor.row_scale

    if input_tensor.dim() == 3:
        B, M, _ = input_tensor.shape
        _, N, _ = wq.shape
        res = torch.empty((B, M, N), device=input_tensor.device, dtype=torch.bfloat16)
        for i in range(B):
            res[i] = torch.ops.fbgemm.bf16i4bf16_shuffled(
                input_tensor[i], wq[i], group_scale[i], row_scale[i]
            )
    else:
        # Otherwise run gemm normally.
        res = torch.ops.fbgemm.bf16i4bf16_shuffled(
            input_tensor, wq, group_scale, row_scale
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
    assert weight_tensor.shape_multiplier[-1] == 2

    wq = weight_tensor.packed_weight
    group_scale = weight_tensor.group_scale
    row_scale = weight_tensor.row_scale
    B, M, _ = input_tensor.shape
    _, N, _ = wq.shape
    res = torch.empty((B, M, N), device=input_tensor.device, dtype=torch.bfloat16)
    for i in range(B):
        res[i] = torch.ops.fbgemm.bf16i4bf16_shuffled(
            input_tensor[i], wq[i], group_scale[i], row_scale[i]
        )
    res = res.reshape(*orig_input_size[:-1], orig_out_features)
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


def _same_metadata(
    self: "Int4GroupwisePreshuffleTensor", src: "Int4GroupwisePreshuffleTensor"
) -> bool:
    return (
        isinstance(self, Int4GroupwisePreshuffleTensor)
        and isinstance(src, Int4GroupwisePreshuffleTensor)
        and self.shape == src.shape
        and self.packed_weight.shape == src.packed_weight.shape
        and self.group_scale.shape == src.group_scale.shape
        and self.row_scale.shape == src.row_scale.shape
        and self.group_size == src.group_size
        and self.shape_multiplier == src.shape_multiplier
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
    group_scale and row_scale has dimension: (K/groups, N)

    dim, start, end, step are args that's referring to the original tensor shape
    which is (N, K), and we need to map that to the transformed weight shape of packed_weight,
    group_scale and row_scale

    when dim == 0: we do a slice on packed_weight dim 0, and on dim 1 of group_scale and row_scale,
    also adjust the start and end indexes based on the ratio between original shape and the shape
    of packed_weight and group_scale/row_scale

    when dim == 1: we do a slice on packed_weight dim 1 and dim 0 of group_scale and row_scale and do the
    same adjustment based on ratio

    Note that we need to call slice on the packed_weight, group_scale and row_scale directly because slice
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
    sz_dim0, sz_dim1 = self.group_scale.shape

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
                self.group_scale,
                self.row_scale,
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
    group_scale = aten.slice.Tensor(self.group_scale, sz_dim, start_sz, end_sz, step)
    row_scale = aten.slice.Tensor(self.row_scale, sz_dim, start_sz, end_sz, step)
    packed_shape0, packed_shape1 = packed_weight.shape
    new_shape = (packed_shape0, packed_shape1 * 2)
    new = self.__class__(
        packed_weight,
        group_scale,
        row_scale,
        group_size=self.group_size,
        shape_multiplier=self.shape_multiplier,
        shape=new_shape,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.cat.default)
def _(func, types, args, kwargs):
    tensors, dim = fill_defaults(args, 2, [[], 0])
    tensor_0 = tensors[0]
    if dim < 0:
        dim = dim + tensor_0.ndim

    for i in range(1, len(tensors)):
        assert tensor_0.packed_weight.ndim == tensors[i].packed_weight.ndim
        assert tensor_0.group_scale.ndim == tensors[i].group_scale.ndim
        assert tensor_0.row_scale.ndim == tensors[i].row_scale.ndim
        assert tensor_0.group_size == tensors[i].group_size
        assert tensor_0.shape_multiplier == tensors[i].shape_multiplier

    packed_weight = [t.packed_weight for t in tensors]
    group_scale = [t.group_scale for t in tensors]
    row_scale = [t.row_scale for t in tensors]

    # with group wise quantization, dimension of group_scale, packed_weight and
    # origianl shape will be the same, so original dim argument applies
    # to both packed_weight and group_scale
    cat_packed_weight = aten.cat.default(packed_weight, dim)
    if cat_packed_weight.ndim == 2:
        sz_dim = 1 - dim
    else:
        sz_dim = dim

    cat_group_scale = aten.cat.default(group_scale, sz_dim)
    cat_row_scale = aten.cat.default(row_scale, sz_dim)
    new_shape = list(cat_packed_weight.shape)
    for i in range(len(tensor_0.shape_multiplier)):
        new_shape[i] *= tensor_0.shape_multiplier[i]
    new_shape = tuple(new_shape)
    new = tensor_0.__class__(
        cat_packed_weight,
        cat_group_scale,
        cat_row_scale,
        group_size=tensor_0.group_size,
        shape_multiplier=tensor_0.shape_multiplier,
        shape=new_shape,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.transpose.int)
def _(func, types, args, kwargs):
    self, dim0, dim1 = args
    packed_weight = self.packed_weight.transpose(dim0, dim1).contiguous()
    shape_multiplier = self.shape_multiplier.copy()
    shape_multiplier[dim0], shape_multiplier[dim1] = (
        shape_multiplier[dim1],
        shape_multiplier[dim0],
    )

    tensor_shape = list(packed_weight.shape)
    for i in range(len(shape_multiplier)):
        tensor_shape[i] *= shape_multiplier[i]
    tensor_shape = tuple(tensor_shape)
    new = self.__class__(
        packed_weight,
        self.group_scale,
        self.row_scale,
        self.group_size,
        shape_multiplier,
        tensor_shape,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


to_int4_groupwise_preshuffle = Int4GroupwisePreshuffleTensor.from_float


if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with Int4GroupwisePreshuffleTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([Int4GroupwisePreshuffleTensor])
