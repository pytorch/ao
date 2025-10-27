# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TorchAOBaseTensor, fill_defaults

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

    Tensor Data Attributes:
        qdata: packed int4 weight, either 2D (N, K/2) or 3D (B, N, K/2), last dimension is packed
        scale: (K/group_size, N) for 2D Tensor, (B, K/group_size, N) for 3D Tensor, where B is batch size,
               dtype is the same as the original Tensor dtype
        zero_point: (K/group_size, N) for 2D Tensor, (B, K/group_size, N) for 3D Tensor, where B is batch size,
               dtype is the same as the original Tensor dtype

    Non-Tensor Data Attributes:
        block_size: the block size for quantization, representing the granularity, for example groupwise quantization will have block_size (1, group_size)
        shape: the shape of the original Tensor

    Optional Tensor Data Attributes:
        act_pre_scale (Optional[Tensor]): Optional scale for activation Tensor, if present,
               we'll multiply activation Tensor with act_pre_scale before applying dynamic
               quantization to activation or running quantized mm op
    """

    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["block_size", "shape"]
    optional_tensor_data_names = ["act_pre_scale"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
        act_pre_scale: Optional[torch.Tensor] = None,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
        act_pre_scale: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size
        self.act_pre_scale = act_pre_scale

    def _quantization_type(self):
        s = f"shape={self.shape}, block_size={self.block_size}, device={self.device}"
        if self.act_pre_scale is not None:
            s += f", act_pre_scale.shape={self.act_pre_scale.shape}"
        return s

    @classmethod
    def from_hp(
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

        return Int4Tensor(
            qdata=wq,
            scale=scale,
            zero_point=zero_point,
            block_size=block_size,
            shape=original_shape,
            act_pre_scale=None,
        )


implements = Int4Tensor.implements
implements_torch_function = Int4Tensor.implements_torch_function


@implements([aten.linear.default])
@implements_torch_function([torch.nn.functional.linear])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    assert isinstance(weight_tensor, Int4Tensor)

    assert weight_tensor.qdata.is_contiguous(), "Expected qdata to be contiguous"
    assert weight_tensor.scale.is_contiguous(), "Expected scale to be contiguous"
    assert weight_tensor.zero_point.is_contiguous(), (
        "Expected zero_point to be contiguous"
    )

    if weight_tensor.act_pre_scale is not None:
        input_tensor = input_tensor * weight_tensor.act_pre_scale

    orig_act_size = input_tensor.size()
    orig_out_features = weight_tensor.shape[-2]

    input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
    res = torch.ops.fbgemm.bf16i4bf16_rowwise(
        input_tensor,
        weight_tensor.qdata,
        weight_tensor.scale,
        weight_tensor.zero_point,
    )
    res = res.reshape(*orig_act_size[:-1], orig_out_features)
    if bias is not None:
        res = res + bias
    return res


@implements_torch_function(torch.bmm)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = (
        args[0],
        args[1],
    )
    assert weight_tensor.qdata.is_contiguous(), "Expected qdata to be contiguous"
    assert weight_tensor.scale.is_contiguous(), "Expected scale to be contiguous"
    assert weight_tensor.zero_point.is_contiguous(), (
        "Expected zero_point to be contiguous"
    )

    orig_act_size = input_tensor.size()
    orig_out_features = weight_tensor.shape[-2]
    res = torch.ops.fbgemm.bf16i4bf16_rowwise_batched(
        input_tensor,
        weight_tensor.qdata,
        weight_tensor.scale,
        weight_tensor.zero_point,
    )
    res = res.reshape(*orig_act_size[:-1], orig_out_features)
    return res


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    """Only supports slicing for dim == 1 and dim == 2
    qdata has dimension: (N, K/2)
    scale and zero_point has dimension: (K/groups, N)

    dim, start, end, step are args that's referring to the original tensor shape
    which is (N, K), and we need to map that to the transformed weight shape of qdata,
    scale and zero_point

    when dim == 0: we do a slice on qdata dim 0, and on dim 1 of scale and zero_point,
    also adjust the start and end indexes based on the ratio between original shape and the shape
    of qdata and scale/zero_point

    when dim == 1: we do a slice on qdata dim 1 and dim 0 of scale and zero_point and do the
    same adjustment based on ratio

    Note that we need to call slice on the qdata, scale and zero_point directly because slice
    is an operation that need to preserve aliasing, see `test_slice_preserves_aliasing` and
    `test_slice_and_copy_similar_to_vllm` in `test_int4_tensor` for more details
    """
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1
    assert dim == 0 or dim == 1, f"Only dim==0 or 1 are supported, got: {dim}"
    if end >= self.shape[dim]:
        end = self.shape[dim]

    assert self.qdata.ndim == 2, (
        f"Expected packed weight to have dim 2, got {self.qdata.dim}"
    )
    N, K_by_2 = self.qdata.shape
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
            Int4Tensor(
                self.qdata,
                self.scale,
                self.zero_point,
                block_size=self.block_size,
                shape=self.shape,
                act_pre_scale=self.act_pre_scale,
            ),
        )

    pw_ratio = data_len / pw_len
    start_pw = int(start / pw_ratio)
    end_pw = int(end / pw_ratio)

    sz_ratio = data_len / sz_len
    start_sz = int(start / sz_ratio)
    end_sz = int(end / sz_ratio)

    qdata = aten.slice.Tensor(self.qdata, dim, start_pw, end_pw, step)
    scale = aten.slice.Tensor(self.scale, sz_dim, start_sz, end_sz, step)
    zero_point = aten.slice.Tensor(self.zero_point, sz_dim, start_sz, end_sz, step)
    packed_shape0, packed_shape1 = qdata.shape
    new_shape = (packed_shape0, packed_shape1 * 2)
    new = Int4Tensor(
        qdata,
        scale,
        zero_point,
        self.block_size,
        new_shape,
        act_pre_scale=self.act_pre_scale,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.cat.default)
def _(func, types, args, kwargs):
    """Concatenate multiple Int4 quantized tensors

    For Int4Tensor, we need to concatenate qdata, scale, and zero_point tensors.
    The concatenation behavior depends on the dimension and block_size configuration.

    If the concatenation dimension is not the same as the packed dimension, then we can just concatenate the
    qdata, scale and zero_point directly, note that scale and zero_point has reversed dimension order in 2D
    If the concatention dimension is the same as block_size, we'll check that scales from all
    tensors are equal and use the first scale
    """
    tensors, dim = fill_defaults(args, 2, [[], 0])
    if not tensors:
        raise ValueError("Cannot concatenate empty list of tensors")

    tensor_0 = tensors[0]
    dim = dim % tensor_0.ndim

    # Validate that all tensors have compatible properties
    for i in range(1, len(tensors)):
        assert tensor_0.qdata.ndim == tensors[i].qdata.ndim
        assert tensor_0.scale.ndim == tensors[i].scale.ndim
        assert tensor_0.zero_point.ndim == tensors[i].zero_point.ndim
        assert tensor_0.block_size == tensors[i].block_size

    qdatas = [t.qdata for t in tensors]
    scales = [t.scale for t in tensors]
    zero_points = [t.zero_point for t in tensors]

    # Concatenate the quantized data along the specified dimension
    cat_qdata = aten.cat.default(qdatas, dim=dim)

    # if concatenation happens in the non-packed dimension, we need to concatenation
    # scale and zero_point
    if tensor_0.block_size[dim] == 1:
        # For scale and zero_point, the concatenation dimension depends on the dimension
        # Int4Tensor has scale and zero_point with shape (K/group_size, N) for 2D or (B, K/group_size, N) for 3D
        if cat_qdata.ndim == 2:  # 2D case
            sz_dim = (
                1 - dim
            )  # If concatenating dim 0 (N), use dim 1 for scale; if dim 1 (K), use dim 0
        else:  # 3D case
            assert cat_qdata.ndim == 3
            if dim in [1, 2]:
                sz_dim = 3 - dim
            else:
                sz_dim = dim

        cat_scale = aten.cat.default(scales, dim=sz_dim)
        cat_zero_point = aten.cat.default(zero_points, dim=sz_dim)

    else:
        # if concatenation happens in the packed dimension, we just need to verify
        # that all scale and zero_points match
        for i in range(1, len(tensors)):
            assert torch.equal(tensor_0.scale, tensors[i].scale)
            assert torch.equal(tensor_0.zero_point, tensors[i].zero_point)
        cat_scale = scales[0]
        cat_zero_point = zero_points[0]

    # Calculate new shape based on the concatenated qdata shape
    new_shape = list(cat_qdata.shape)
    new_shape[-1] *= 2
    new_shape = list(new_shape)

    new = Int4Tensor(
        cat_qdata,
        cat_scale,
        cat_zero_point,
        tensor_0.block_size,
        new_shape,
        act_pre_scale=tensor_0.act_pre_scale,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.transpose.int)
def _(func, types, args, kwargs):
    self, dim0, dim1 = args

    # Transpose the quantized data
    qdata = self.qdata.transpose(dim0, dim1).contiguous()
    if self.scale.ndim == 3:
        # since scale/zero_point dimension order is different
        # (B, K/group_size, N), we'll need to remap the dim
        remapped_dim0 = dim0
        if dim0 in [1, 2]:
            remapped_dim0 = 3 - dim0

        remapped_dim1 = dim1
        if dim1 in [1, 2]:
            remapped_dim1 = 3 - dim1

        scale = self.scale.transpose(remapped_dim0, remapped_dim1)
        zero_point = self.zero_point.transpose(remapped_dim0, remapped_dim1)
    else:
        assert scale.ndim == 2, f"Only support ndim == 2 or 3, got: {scale.ndim}"
        remapped_dim0 = 1 - dim0
        remapped_dim1 = 1 - dim1
        scale = self.scale.transpose(remapped_dim0, remapped_dim1)
        zero_point = self.zero_point.transpose(remapped_dim0, remapped_dim1)

    # Update block_size by swapping the dimensions
    block_size = self.block_size.copy()
    block_size[dim0], block_size[dim1] = block_size[dim1], block_size[dim0]

    # Update shape by swapping the dimensions
    new_shape = list(self.shape)
    new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]

    new = Int4Tensor(
        qdata,
        scale,
        zero_point,
        block_size,
        new_shape,
        act_pre_scale=self.act_pre_scale,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.view.default)
def _(func, types, args, kwargs):
    self, size = args
    original_shape = self.shape
    original_packing_dim = None
    for i in range(len(original_shape)):
        if original_shape[i] == (self.qdata.shape[i] * 2):
            original_packing_dim = i
    assert original_packing_dim is not None, "Didn't find a packing_dim"

    if len(original_shape) == 3 and len(size) == 2:
        # only support combining the dim 0 and dim1 together
        assert original_shape[-1] == size[-1], (
            f"Only support reshaping when last dimension matches, requested: reshaping from {original_shape} to {size}"
        )
        # the dim that int4 packing happens
        if original_packing_dim in [0, 1]:
            packing_dim = 0
        else:
            packing_dim = 1

        block_size = self.block_size.copy()
        block_size = [block_size[0] * block_size[1], block_size[2]]

        qdata_shape = size.copy()
        qdata_shape[packing_dim] //= 2
        qdata = self.qdata.reshape(*qdata_shape)
        sz_shape = []
        for i in range(len(size)):
            sz_shape.append(size[i] // block_size[i])
        # scale and zero_point have reversed dimensions
        sz_shape[0], sz_shape[1] = sz_shape[1], sz_shape[0]

        scale = self.scale.reshape(*sz_shape)
        zero_point = self.zero_point.reshape(*sz_shape)
    elif len(original_shape) == 2 and len(size) == 3:
        # only support extending the dim 0 to 2, `t.unflatten(0, (num_experts, -1))`
        assert original_shape[-1] == size[-1], (
            f"Only support reshaping when last dimension matches, requested: reshaping from {original_shape} to {size}"
        )
        if original_packing_dim == 0:
            packing_dim = 1
        else:
            # original_packing_dim is 1
            packing_dim = 2

        block_size = self.block_size.copy()
        block_size = [1, block_size[0], block_size[1]]

        qdata_shape = size.copy()
        qdata_shape[packing_dim] //= 2
        qdata = self.qdata.reshape(*qdata_shape)

        sz_shape = []
        for i in range(len(size)):
            sz_shape.append(size[i] // block_size[i])

        # scale and zero_point have reversed dimensions
        sz_shape[1], sz_shape[2] = sz_shape[2], sz_shape[1]

        scale = self.scale.reshape(*sz_shape)
        zero_point = self.zero_point.reshape(*sz_shape)
    elif len(original_shape) == len(size):
        assert all(x == y or y == -1 for x, y in zip(original_shape, size)), (
            f"Only support viewing with match dimensions or -1, got: {original_shape}, {size}"
        )
        packing_dim = original_packing_dim
        block_size = self.block_size
    else:
        assert len(original_shape) == 2 and len(size) == 3, (
            f"Only support reshaping from 2D to 3D or from 3D to 2D or between sam ranges, requested: reshaping from {original_shape} to {size}"
        )

    shape = list(qdata.shape)
    for i in range(len(shape)):
        if i == packing_dim:
            shape[i] *= 2

    new = Int4Tensor(
        qdata,
        scale,
        zero_point,
        block_size,
        shape,
        act_pre_scale=self.act_pre_scale,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


@implements(aten.squeeze.dim)
def _(func, types, args, kwargs):
    self, dim = args

    # Squeeze qdata
    qdata = self.qdata.squeeze(dim=dim)

    # For scale and zero_point, we need to squeeze based on the tensor layout
    # Int4Tensor has scale and zero_point with shape (K/group_size, N) for 2D or (B, N, K/group_size) for 3D
    if self.qdata.ndim == 2:  # 2D case
        # qdata is (N, K/2), scale/zero_point is (K/group_size, N)
        # When squeezing qdata dim, we need to squeeze scale/zero_point in reverse order
        sz_dim = 1 - dim
    else:  # 3D case
        # qdata is (B, N, K/2), scale/zero_point is (B, N, K/group_size)
        sz_dim = dim

    scale = self.scale.squeeze(dim=sz_dim)
    zero_point = self.zero_point.squeeze(dim=sz_dim)

    # Update block_size by removing the squeezed dimension
    new_block_size = list(self.block_size)
    if len(qdata.shape) < len(new_block_size):
        new_block_size.pop(dim)

    # Update shape by removing the squeezed dimension
    new_shape = list(self.shape)
    if len(qdata.shape) < len(new_shape):
        assert new_shape[dim] == 1
        new_shape.pop(dim)

    new = Int4Tensor(
        qdata,
        scale,
        zero_point,
        new_block_size,
        new_shape,
        act_pre_scale=self.act_pre_scale,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


Int4Tensor.__module__ = "torchao.quantization"

# Allow a model with Int4Tensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4Tensor])
