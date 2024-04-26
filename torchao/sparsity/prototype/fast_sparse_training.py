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

from torch.sparse import SparseSemiStructuredTensorCUSPARSELT as Sparse24TensorCuSparseLt
from torch.sparse import SparseSemiStructuredTensorCUTLASS as Sparse24TensorCutlass
from torch.sparse import SparseSemiStructuredTensor as Sparse24Tensor

def sparse24_pointwise_op(
    func, types, args=(), kwargs=None, allow_sparsify_args_list=()
):
    self = None
    for tensor in args:
        if isinstance(tensor, Sparse24Tensor):
            self = tensor
    assert self is not None
    args_updated = []

    for i, tensor in enumerate(args):
        if isinstance(tensor, torch.Tensor):
            if not isinstance(tensor, Sparse24Tensor):
                if i in allow_sparsify_args_list:
                    tensor = sparsify24_like(tensor, self)
                else:
                    raise ValueError(
                        f"Operation {func.__module__}.{func.__name__} on Sparse24Tensor requires all operands to "
                        f"be Sparse24Tensors, but operand {i} is a {type(tensor)}"
                    )
            if (
                tensor.compressed_swizzled_bitmask is None
                or self.compressed_swizzled_bitmask is None
                or tensor.compressed_swizzled_bitmask.data_ptr() != self.compressed_swizzled_bitmask.data_ptr()
                or tensor.compressed_swizzled_bitmask.stride() != self.compressed_swizzled_bitmask.stride()
            ):
                raise ValueError(
                    f"Operation {func.__module__}.{func.__name__} on Sparse24Tensor requires all operands to be "
                    "Sparse24Tensors with the same sparsity pattern"
                )
        args_updated.append(tensor)
    assert isinstance(
        self, Sparse24TensorCutlass
    ), "Only implemented for CUTLASS tensors"
    return Sparse24TensorCutlass(
        self.shape,
        func(
            *[(x.packed if isinstance(x, Sparse24Tensor) else x) for x in args_updated]
        ),
        self.meta,
        func(
            *[
                (x.packed_t if isinstance(x, Sparse24Tensor) else x)
                for x in args_updated
            ]
        ),
        self.meta_t,
        self.compressed_swizzled_bitmask,
    )

SPARSE24_DISPATCH_CUTLASS = {
    torch.ops.aten.relu: sparse24_pointwise_op,
    torch.ops.aten.gelu: sparse24_pointwise_op,
    torch.ops.aten.silu: sparse24_pointwise_op,
    torch.ops.aten.mul: partial(
        # `mul` BW in swiglu
        sparse24_pointwise_op,
        allow_sparsify_args_list=(
            0,
            1,
        ),
    ),
    torch.ops.aten.add: sparse24_pointwise_op,
    # Note: for these ops, we allow the gradient to come in as a `torch.Tensor`
    # and we will run the sparsification right before calling the BW aten func
    torch.ops.aten.gelu_backward: partial(
        sparse24_pointwise_op, allow_sparsify_args_list=(0,)
    ),
    torch.ops.aten.silu_backward: partial(
        sparse24_pointwise_op, allow_sparsify_args_list=(0, 1)
    ),
    torch.ops.aten.threshold_backward: partial(  # relu BW
        sparse24_pointwise_op,
        allow_sparsify_args_list=(0,),
    ),
}

Sparse24TensorCutlass._load_dispatch_table(SPARSE24_DISPATCH_CUTLASS)

if torch.__version__ >= "2.1.0":
    torch._dynamo.allow_in_graph(Sparse24TensorCuSparseLt)
    torch._dynamo.allow_in_graph(Sparse24TensorCutlass)

GRADIENT_SP24 = "24sparse"
GRADIENT_DENSE = "24dense"

BACKEND_CUTLASS = "cutlass"
BACKEND_CUSPARSELT = "cusparselt"


class _Sparsify24Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, algo: str, backend: str):  # type: ignore[override]
        use_cutlass = (backend == BACKEND_CUTLASS)
        if not isinstance(x, Sparse24Tensor):
            (packed, meta, packed_t, meta_t, bitmask) = torch._sparse_semi_structured_tile(
                x, algorithm=algo, use_cutlass=use_cutlass
            )
            cls = (
                Sparse24TensorCutlass if use_cutlass else Sparse24TensorCuSparseLt
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
    def forward(ctx, x: torch.Tensor, pattern: Sparse24Tensor):  # type: ignore[override]
        assert isinstance(pattern, Sparse24Tensor)
        if not isinstance(pattern, Sparse24TensorCutlass):
            raise NotImplementedError(
                "`sparsify24_like(x, pattern)` is only implemented for CUTLASS backend"
            )
        if not pattern.compressed_swizzled_bitmask.is_contiguous():
            raise NotImplementedError(
                "`sparsify24_like(x, pattern)` is not implemented when `pattern` is transposed"
            )
        ctx.threads_masks = pattern.compressed_swizzled_bitmask
        ctx.meta = pattern.meta
        ctx.meta_t = pattern.meta_t
        ctx.dtype = pattern.dtype
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
    backend: str = BACKEND_CUTLASS,
) -> Sparse24Tensor:
    return _Sparsify24Func.apply(x, algo, backend)


@allow_in_graph
def sparsify24_like(
    x: torch.Tensor, pattern: torch.Tensor
) -> Sparse24Tensor:
    if not isinstance(pattern, Sparse24Tensor):
        raise ValueError(
            f"`pattern` must be a `Sparse24Tensor` but got a {type(pattern)}"
        )
    return _Sparsify24LikeFunc.apply(x, pattern)
