import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from torch.sparse import SparseSemiStructuredTensor, SparseSemiStructuredTensorCUTLASS, SparseSemiStructuredTensorCUSPARSELT
from collections.abc import Iterable
from torch import nn

if torch.__version__ >= "2.1.0":
    torch._dynamo.allow_in_graph(SparseSemiStructuredTensorCUSPARSELT)
    torch._dynamo.allow_in_graph(SparseSemiStructuredTensorCUTLASS)

GRADIENT_STE = "ste"
GRADIENT_DENSE = "dense"
GRADIENT_SPARSE = "sparse"

class _SparsifyFunc(torch.autograd.Function):

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
            out = x.detach()

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        return grad_out, None, None

class _SparsifyLikeFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, pattern: SparseSemiStructuredTensor, gradient=GRADIENT_SPARSE):  # type: ignore[override]
        assert isinstance(pattern, SparseSemiStructuredTensor)

        if not isinstance(pattern, SparseSemiStructuredTensorCUTLASS):
            raise NotImplementedError(
                "`sparsify_like(x, pattern)` is only implemented for CUTLASS backend"
            )
        if not pattern.compressed_swizzled_bitmask.is_contiguous():
            raise NotImplementedError(
                "`sparsify_like(x, pattern)` is not implemented when `bitmask` is transposed"
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
        if ctx.gradient == GRADIENT_STE or isinstance(grad_out, SparseSemiStructuredTensor):
            return grad_out, None, None, None
        assert not isinstance(grad_out, SparseSemiStructuredTensor)
        assert grad_out.dtype == ctx.dtype

        if ctx.gradient == GRADIENT_DENSE:
            assert ctx.threads_masks.is_contiguous()
            return (
                torch._sparse_semi_structured_apply_dense(grad_out, ctx.threads_masks),
                None,
                None,
                None,
            )
        assert ctx.gradient == GRADIENT_SPARSE

        packed, _, packed_t, _ = torch._sparse_semi_structured_tile(
            grad_out, ctx.threads_masks, backend="cutlass"
        )
        return (
            SparseSemiStructuredTensorCUTLASS(
                grad_out.shape,
                packed,
                ctx.meta,
                packed_t,
                ctx.meta_t,
                ctx.threads_masks,
                requires_grad=grad_out.requires_grad,
            ),
            None,
            None,
            None,
        )
        return grad_out, None

# We want to use `torch._dynamo.allow_in_graph` as a decorator
# (see https://fburl.com/workplace/uimiz0mf) but it breaks mypy.
# This is a hack to work around this
F = TypeVar("F", bound=Callable[..., Any])
def allow_in_graph(func: F) -> F:
    return cast(F, torch._dynamo.allow_in_graph(func))

@allow_in_graph
def semi_sparse_sparsify(
    x: torch.Tensor,
    pattern: SparseSemiStructuredTensor = None,
    algo: str = "",
    backend: str = "cutlass",
) -> SparseSemiStructuredTensor:
    """
    Sparsifies a dense tensor into a semi-structured tensor.
    When pattern is provided, we will sparsify the tensor according using the same mask as the provided sparse tensor.
    Otherwise, we sparsify using the algorithm/backend specified
    """
    if pattern is None:
        return _SparsifyFunc.apply(x, algo, backend)
    else:
        if not isinstance(pattern, SparseSemiStructuredTensor):
            raise ValueError(
                f"`pattern` must be a `SparseSemiStructuredTensor` but got a {type(pattern)}"
            )
        return _SparsifyLikeFunc.apply(x, pattern)
