# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.uintx.plain_layout import (
    PlainAQTTensorImpl,
    _aqt_is_int8_reduced_range,
)
from torchao.dtypes.utils import (
    Layout,
    PlainLayout,
)

logger = logging.getLogger(__name__)

aten = torch.ops.aten


@dataclass(frozen=True)
class BlockSparseLayout(Layout):
    """BlockSparseLayout is a data class that represents the layout of a block sparse matrix.

    Attributes:
        blocksize (int): The size of the blocks in the sparse matrix. Default is 64.
    """

    blocksize: int = 64


@register_layout(BlockSparseLayout)
class BlockSparseAQTTensorImpl(PlainAQTTensorImpl):
    bsr_crow_indices: Optional[torch.Tensor]
    bsr_col_indices: Optional[torch.Tensor]
    bsr_values: Optional[torch.Tensor]
    scale: Optional[torch.Tensor]
    zero_point: Optional[torch.Tensor]

    __slots__ = [
        "bsr_crow_indices",
        "bsr_col_indices",
        "bsr_values",
        "scale",
        "zero_point",
    ]

    @staticmethod
    def __new__(  # noqa: PYI034
        cls,
        shape: torch.Size,
        bsr_crow_indices: Optional[torch.Tensor],
        bsr_col_indices: Optional[torch.Tensor],
        bsr_values: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
        requires_grad: bool = False,
    ):
        if bsr_values is None:
            raise ValueError("bsr values must be provided!")
        else:
            previous_tensor = bsr_values

        kwargs = {
            "device": previous_tensor.device,
            "dtype": previous_tensor.dtype,
            "layout": previous_tensor.layout,
            "requires_grad": requires_grad,
        }
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(  # noqa: PYI034
        self,
        shape: torch.Size,
        bsr_crow_indices: Optional[torch.Tensor],
        bsr_col_indices: Optional[torch.Tensor],
        bsr_values: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
        requires_grad: bool = False,
    ):
        self.bsr_crow_indices = bsr_crow_indices
        self.bsr_col_indices = bsr_col_indices
        self.bsr_values = bsr_values
        self.scale = scale
        self.zero_point = zero_point
        self._layout = _layout

    def __tensor_flatten__(self):
        inner_tensors = list(
            filter(lambda x: getattr(self, x) is not None, self.__slots__)
        )
        tensor_meta = (self.shape, self._layout, self.requires_grad)
        return inner_tensors, tensor_meta

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta: Tuple[torch.Size, bool],
        outer_size,
        outer_stride,
    ) -> torch.Tensor:
        shape, _layout, requires_grad = tensor_meta
        return cls(
            shape=shape,
            bsr_crow_indices=inner_tensors.get("bsr_crow_indices", None),
            bsr_col_indices=inner_tensors.get("bsr_col_indices", None),
            bsr_values=inner_tensors.get("bsr_values", None),
            scale=inner_tensors.get("scale", None),
            zero_point=inner_tensors.get("zero_point", None),
            _layout=_layout,
            requires_grad=requires_grad,
        )

    @classmethod
    def from_plain(cls, int_data, scale, zero_point, _layout):
        bsr_tensor = int_data.to_sparse_bsr(_layout.blocksize)
        return cls(
            shape=int_data.shape,
            bsr_crow_indices=bsr_tensor.crow_indices(),
            bsr_col_indices=bsr_tensor.col_indices(),
            bsr_values=bsr_tensor.values(),
            scale=scale,
            zero_point=zero_point,
            _layout=_layout,
            requires_grad=False,
        )

    def get_plain(self):
        int_data_expanded = torch.ops.blocksparse.bsr_to_dense(
            self.crow_indices(),
            self.col_indices(),
            self.values(),
            self.shape[0],
            self.shape[1],
        )
        return int_data_expanded, self.scale, self.zero_point

    def _apply_fn_to_data(self, func):
        return self.__class__(
            shape=self.shape,
            bsr_crow_indices=func(self.bsr_crow_indices),
            bsr_col_indices=func(self.bsr_col_indices),
            bsr_values=func(self.bsr_values),
            scale=self.scale,
            zero_point=self.zero_point,
            _layout=self._layout,
            requires_grad=self.requires_grad,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )
        if func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        # Need the following for bsr specific functions
        if func is aten.crow_indices.default:
            return args[0].bsr_crow_indices.detach()

        if func is aten.col_indices.default:
            return args[0].bsr_col_indices.detach()

        if func is aten.values.default:
            return args[0].bsr_values.detach()

        if func is aten._nnz.default:
            return args[0].bsr_values.shape[0]

        raise NotImplementedError(
            f"BlockSparseAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )


def _linear_int8_act_int8_weight_block_sparse_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(input_tensor, AffineQuantizedTensor)
        and _aqt_is_int8_reduced_range(input_tensor)
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and weight_tensor.is_cuda
        and input_tensor.dtype == weight_tensor.dtype
        and isinstance(input_tensor._layout, PlainLayout)
        and isinstance(weight_tensor._layout, BlockSparseLayout)
    )


def _linear_int8_act_int8_weight_block_sparse_impl(input_tensor, weight_tensor, bias):
    x_vals_int8 = input_tensor.tensor_impl.int_data
    x_scales = input_tensor.tensor_impl.scale
    w_vals = weight_tensor.tensor_impl
    w_scales = weight_tensor.tensor_impl.scale
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    tmp_t = tmp.t()

    y = torch.ops.blocksparse.int_addmm(
        w_vals.crow_indices(),
        w_vals.col_indices(),
        w_vals.values(),
        tmp_t,
        w_scales,
        x_scales.reshape(-1),
    )
    y_shape = (*x_vals_int8.shape[:-1], w_scales.shape[-1])
    y = y.reshape(*y_shape)

    # can downcast only at the very end
    output_dtype = input_tensor.dtype
    y = y.to(output_dtype)
    if bias is not None:
        y += bias
    return y
