# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

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
from torchao.dtypes.utils import Layout, PlainLayout

aten = torch.ops.aten


def _linear_int8_act_int8_weight_semi_structured_sparse_check(
    input_tensor, weight_tensor, bias
):
    return (
        isinstance(input_tensor, AffineQuantizedTensor)
        and _aqt_is_int8_reduced_range(input_tensor)
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and weight_tensor.is_cuda
        and input_tensor.dtype == weight_tensor.dtype
        and isinstance(input_tensor._layout, PlainLayout)
        and isinstance(weight_tensor._layout, SemiSparseLayout)
    )


def _linear_int8_act_int8_weight_semi_structured_sparse_impl(
    input_tensor, weight_tensor, bias
):
    x_vals_int8 = input_tensor.tensor_impl.int_data
    x_scales = input_tensor.tensor_impl.scale
    w_vals_int8 = weight_tensor.tensor_impl.int_data
    w_scales = weight_tensor.tensor_impl.scale
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    # must pad
    row, col = tmp.shape
    from torch.sparse import SparseSemiStructuredTensorCUSPARSELT

    tmp_padded = SparseSemiStructuredTensorCUSPARSELT._pad_dense_input(tmp)
    # we fuse one of the scalar matrix multiplications (w_scales) into the sparse mm
    y_dot_bf16_w_scales_fused = torch._cslt_sparse_mm(
        w_vals_int8,
        tmp_padded.t(),
        alpha=w_scales.to(torch.float32),
        out_dtype=torch.bfloat16,
    ).t()[:row, :]
    y = (y_dot_bf16_w_scales_fused * x_scales.reshape(-1, 1)).reshape(
        *x_vals_int8.shape[:-1], y_dot_bf16_w_scales_fused.shape[-1]
    )
    output_dtype = input_tensor.dtype
    # TODO: waiting for jesse's test/fix
    y = y.to(output_dtype).contiguous()
    if bias is not None:
        y += bias
    return y


@dataclass(frozen=True)
class SemiSparseLayout(Layout):
    """SemiSparseLayout is a layout class for handling semi-structured sparse
    matrices in affine quantized tensors. This layout is specifically designed
    to work with the 2:4 sparsity pattern, where two out of every four elements
    are pruned to zero. This class provides methods for preprocessing input
    tensors to conform to this sparsity pattern.
    """

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        # prune to 2:4 if not already
        temp = input.detach()
        pruning_inds = temp.abs().view(-1, 4).argsort(dim=1)[:, :2]
        temp.view(-1, 4).scatter_(1, pruning_inds, value=0)
        return temp


@register_layout(SemiSparseLayout)
class SemiSparseAQTTensorImpl(PlainAQTTensorImpl):
    """
    TensorImpl for semi_sparse_cusparselt layout for affine quantized tensor
    """

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"SparseAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    def get_plain(self):
        # Currently we don't have cuSPARSELt expansion routines, so we matmul by
        # the identity matrix to get the original dense matrix. This is slow though.
        cols = self.int_data.numel() * 16 // (10 * self.scale.shape[0])
        int_data_expanded = torch._cslt_sparse_mm(
            self.int_data,
            torch.eye(cols, dtype=self.int_data.dtype, device=self.int_data.device).t(),
        )
        return int_data_expanded, self.scale, self.zero_point

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert isinstance(_layout, SemiSparseLayout)
        int_data_compressed = torch._cslt_compress(int_data)
        return cls(int_data_compressed, scale, zero_point, _layout)
