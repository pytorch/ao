# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch

from torchao.ops import to_sparse_semi_structured_cutlass_sm9x_f8
from torchao.quantization.quant_primitives import (
    _choose_scale_float8,
    _quantize_affine_float8,
)
from torchao.utils import TorchAOBaseTensor

__all__ = ["Float8SemiSparseTensor"]
aten = torch.ops.aten


class Float8SemiSparseTensor(TorchAOBaseTensor):
    tensor_data_names = ["sparse", "meta", "scale"]

    def __new__(
        cls,
        sparse: torch.Tensor,
        meta: torch.Tensor,
        scale: torch.Tensor,
    ):
        kwargs = {}
        kwargs["device"] = sparse.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        shape = (sparse.shape[0], 2 * sparse.shape[-1])
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        sparse: torch.Tensor,
        meta: torch.Tensor,
        scale: torch.Tensor,
    ):
        super().__init__()
        self.sparse = sparse
        self.meta = meta
        self.scale = scale

    def _quantization_type(self):
        return f"shape={self.shape}, device={self.device}, dtype={self.dtype}"

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: List[int],
    ):
        from torchao.sparsity.utils import mask_creator

        dense = w * mask_creator(w).bool()

        scale = _choose_scale_float8(
            dense,
            block_size=block_size,
            float8_dtype=torch.float8_e4m3fn,
        )

        w_fp8 = _quantize_affine_float8(
            dense,
            scale=scale,
            float8_dtype=torch.float8_e4m3fn,
        )

        sparse, meta = to_sparse_semi_structured_cutlass_sm9x_f8(w_fp8)

        return cls(
            sparse,
            meta,
            scale,
        )


implements = Float8SemiSparseTensor.implements
implements_torch_function = Float8SemiSparseTensor.implements_torch_function


@implements(aten.t.default)
def _(func, types, args, kwargs):
    from torch.utils._python_dispatch import return_and_correct_aliasing

    self = args[0]
    new = Float8SemiSparseTensor(
        sparse=self.sparse,
        meta=self.meta,
        scale=self.scale,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


def _linear_fp8_semi_sparse(input_tensor, weight_tensor, bias):
    from torchao.ops import rowwise_scaled_linear_sparse_cutlass_f8f8
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
        Float8Tensor,
    )

    if isinstance(input_tensor, Float8Tensor):
        input = input_tensor.qdata
        input_scale = input_tensor.scale
        out_dtype = input_tensor.dtype
    else:
        input = input_tensor.qdata
        input_scale = input_tensor.scale
        out_dtype = input_tensor.dtype

    weight = weight_tensor.sparse
    weight_meta = weight_tensor.meta
    weight_scale = weight_tensor.scale

    # Reshape input_scale if needed: kernel expects scale to match input shape minus last dim
    # For input [B, K], scale should be [B] not [B, 1]
    if input_scale.dim() > input.dim() - 1:
        input_scale = input_scale.squeeze(-1)

    return rowwise_scaled_linear_sparse_cutlass_f8f8(
        input, input_scale, weight, weight_meta, weight_scale, bias, out_dtype
    )


@implements([aten.mm.default, aten.addmm.default])
def _(func, types, args, kwargs):
    if func == aten.addmm.default:
        bias, input_tensor, weight_tensor = args
    else:  # aten.mm.default
        input_tensor, weight_tensor = args
        bias = None

    return _linear_fp8_semi_sparse(input_tensor, weight_tensor, bias)


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    return _linear_fp8_semi_sparse(input_tensor, weight_tensor, bias)


Float8SemiSparseTensor.__module__ = "torchao.quantization"

# Allow a model with Float8SemiSparseTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Float8SemiSparseTensor])
