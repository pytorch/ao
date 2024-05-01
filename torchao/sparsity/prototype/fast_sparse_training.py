# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast

import torch
from torch.sparse import SparseSemiStructuredTensor, SparseSemiStructuredTensorCUTLASS, SparseSemiStructuredTensorCUSPARSELT

def semi_sparse_pointwise_op(func, types, args=(), kwargs=None, allow_sparsify_args_list=()):
    """
    adds pointwise op support for semi-structured tensors
    """
    self = None
    for tensor in args:
        if isinstance(tensor, SparseSemiStructuredTensor):
            self = tensor
    # assert self is not None
    assert isinstance(self, SparseSemiStructuredTensorCUTLASS), "Only implemented for CUTLASS tensors"

    args_updated = [sparsify_if_allowed(i, tensor) for i, tensor in enumerate(args)]

    def sparsify_if_allowed(i, tensor):
        if isinstance(tensor, torch.Tensor):
            if not isinstance(tensor, SparseSemiStructuredTensor):
                if i in allow_sparsify_args_list:
                    tensor = sparsify24_like(tensor, self)
                else:
                    raise ValueError(
                        f"Operation {func.__module__}.{func.__name__} on SparseSemiStructuredTensor requires all operands to "
                        f"be SparseSemiStructuredTensors, but operand {i} is a {type(tensor)}"
                    )
            if (
                tensor.compressed_swizzled_bitmask is None
                or self.compressed_swizzled_bitmask is None
                or tensor.compressed_swizzled_bitmask.data_ptr() != self.compressed_swizzled_bitmask.data_ptr()
                or tensor.compressed_swizzled_bitmask.stride() != self.compressed_swizzled_bitmask.stride()
            ):
                raise ValueError(
                    f"Operation {func.__module__}.{func.__name__} on SparseSemiStructuredTensor requires all operands to be "
                    "SparseSemiStructuredTensors with the same sparsity pattern"
                )
        return tensor

    return SparseSemiStructuredTensorCUTLASS(
        self.shape,
        func(
            *[(x.packed if isinstance(x, SparseSemiStructuredTensor) else x) for x in args_updated]
        ),
        self.meta,
        func(
            *[
                x.packed_t if isinstance(x, SparseSemiStructuredTensor)
                else x for x in args_updated
            ]
        ),
        self.meta_t,
        self.compressed_swizzled_bitmask,
    )

CUTLASS_POINTWISE_OP_DISPATCH_TABLE = {
    torch.ops.aten.relu: semi_sparse_pointwise_op,
    torch.ops.aten.gelu: semi_sparse_pointwise_op,
    torch.ops.aten.silu: semi_sparse_pointwise_op,
    torch.ops.aten.mul: partial(
        # `mul` BW in swiglu
        semi_sparse_pointwise_op,
        allow_sparsify_args_list=(0, 1),
    ),
    torch.ops.aten.add: semi_sparse_pointwise_op,
    # Note: for these ops, we allow the gradient to come in as a `torch.Tensor`
    # and we will run the sparsification right before calling the BW aten func
    torch.ops.aten.gelu_backward: partial(
        semi_sparse_pointwise_op,
        allow_sparsify_args_list=(0,)
    ),
    torch.ops.aten.silu_backward: partial(
        semi_sparse_pointwise_op,
        allow_sparsify_args_list=(0, 1)
    ),
    torch.ops.aten.threshold_backward: partial(  # relu BW
        semi_sparse_pointwise_op,
        allow_sparsify_args_list=(0,),
    ),
}

SparseSemiStructuredTensorCUTLASS._load_dispatch_table(CUTLASS_POINTWISE_OP_DISPATCH_TABLE)

if torch.__version__ >= "2.1.0":
    torch._dynamo.allow_in_graph(SparseSemiStructuredTensorCUSPARSELT)
    torch._dynamo.allow_in_graph(SparseSemiStructuredTensorCUTLASS)


class _Sparsify24Func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, algo: str, backend: str):  # type: ignore[override]
        use_cutlass = (backend == "cutlass")
        if not isinstance(x, SparseSemiStructuredTensor):
            (packed, meta, packed_t, meta_t, bitmask) = torch._sparse_semi_structured_tile(
                x, algorithm=algo, use_cutlass=use_cutlass
            )
            cls = (
                SparseSemiStructuredTensorCUTLASS if use_cutlass else SparseSemiStructuredTensorCUSPARSELT
            )
            out = cls(
                x.shape,
                packed=packed,
                meta=meta,
                packed_t=packed_t,
                meta_t=meta_t,
                compressed_swizzled_bitmask=bitmask,
                requires_grad=False,
                fuse_transpose_cusparselt=True,
            )
        else:
            out = x

        ctx.meta = out.meta
        ctx.meta_t = out.meta_t
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        return grad_out, None, None


class _Sparsify24LikeFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, pattern: SparseSemiStructuredTensor):  # type: ignore[override]
        assert isinstance(pattern, SparseSemiStructuredTensor)

        if not isinstance(pattern, SparseSemiStructuredTensorCUTLASS):
            raise NotImplementedError(
                "`sparsify24_like(x, pattern)` is only implemented for CUTLASS backend"
            )
        if not pattern.compressed_swizzled_bitmask.is_contiguous():
            raise NotImplementedError(
                "`sparsify24_like(x, pattern)` is not implemented when `bitmask` is transposed"
            )

        packed, packed_t = torch._sparse_semi_structured_apply(x, pattern.compressed_swizzled_bitmask)
        return pattern.__class__(
            x.shape,
            packed,
            pattern.meta,
            packed_t,
            pattern.meta_t,
            pattern.compressed_swizzled_bitmask,
            requires_grad=x.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        return grad_out, None

# We want to use `torch._dynamo.allow_in_graph` as a decorator
# (see https://fburl.com/workplace/uimiz0mf) but it breaks mypy.
# This is a hack to work around this
F = TypeVar("F", bound=Callable[..., Any])


def allow_in_graph(func: F) -> F:
    return cast(F, torch._dynamo.allow_in_graph(func))

@allow_in_graph
def sparsify24(
    x: torch.Tensor,
    algo: str = "",
    backend: str = "cutlass",
) -> SparseSemiStructuredTensor:
    return _Sparsify24Func.apply(x, algo, backend)


@allow_in_graph
def sparsify24_like(
    x: torch.Tensor,
    pattern: SparseSemiStructuredTensor,
) -> SparseSemiStructuredTensor:
    if not isinstance(pattern, SparseSemiStructuredTensor):
        raise ValueError(
            f"`pattern` must be a `SparseSemiStructuredTensor` but got a {type(pattern)}"
        )
    return _Sparsify24LikeFunc.apply(x, pattern)
