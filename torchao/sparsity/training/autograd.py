# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum

import torch
from torch.sparse import SparseSemiStructuredTensor

from torchao.utils import TORCH_VERSION_AT_LEAST_2_3

if TORCH_VERSION_AT_LEAST_2_3:
    from torch.sparse import (
        SparseSemiStructuredTensorCUSPARSELT,
        SparseSemiStructuredTensorCUTLASS,
    )

    torch._dynamo.allow_in_graph(SparseSemiStructuredTensorCUSPARSELT)
    torch._dynamo.allow_in_graph(SparseSemiStructuredTensorCUTLASS)


GRADIENT_TYPE = Enum("GRADIENT_TYPE", ["DENSE", "SPARSE", "STE"])


class _SparsifyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, algo: str, backend: GRADIENT_TYPE):  # type: ignore[override]
        use_cutlass = backend == "cutlass"
        if not isinstance(x, SparseSemiStructuredTensor):
            (packed, meta, packed_t, meta_t, bitmask) = (
                torch._sparse_semi_structured_tile(
                    x, algorithm=algo, use_cutlass=use_cutlass
                )
            )
            cls = (
                SparseSemiStructuredTensorCUTLASS
                if use_cutlass
                else SparseSemiStructuredTensorCUSPARSELT
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
        # We just return grad_out, since we just use STE - straight through estimation
        return grad_out, None, None


class _SparsifyLikeFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        pattern: SparseSemiStructuredTensor,
        gradient=GRADIENT_TYPE.SPARSE,
    ):  # type: ignore[override]
        assert isinstance(pattern, SparseSemiStructuredTensor)

        if not isinstance(pattern, SparseSemiStructuredTensorCUTLASS):
            raise NotImplementedError(
                "`sparsify_like(x, pattern)` is only implemented for CUTLASS backend"
            )
        if not pattern.compressed_swizzled_bitmask.is_contiguous():
            raise NotImplementedError(
                "`sparsify_like(x, pattern)` is not implemented when `bitmask` is transposed"
            )

        packed, packed_t = torch._sparse_semi_structured_apply(
            x, pattern.compressed_swizzled_bitmask
        )

        # save for backwards
        ctx.meta = pattern.meta
        ctx.meta_t = pattern.meta_t
        ctx.bitmask = pattern.compressed_swizzled_bitmask
        ctx.gradient = gradient

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
        if ctx.gradient == GRADIENT_TYPE.STE or isinstance(
            grad_out, SparseSemiStructuredTensor
        ):
            return grad_out, None, None, None
        assert not isinstance(grad_out, SparseSemiStructuredTensor)
        assert grad_out.dtype == ctx.dtype

        if ctx.gradient == GRADIENT_TYPE.DENSE:
            assert ctx.threads_masks.is_contiguous()
            return (
                torch._sparse_semi_structured_apply_dense(grad_out, ctx.bitmask),
                None,
                None,
                None,
            )
        assert ctx.gradient == GRADIENT_TYPE.SPARSE

        packed, _, packed_t, _ = torch._sparse_semi_structured_tile(
            grad_out, ctx.bitmask, backend="cutlass"
        )
        return (
            SparseSemiStructuredTensorCUTLASS(
                grad_out.shape,
                packed,
                ctx.meta,
                packed_t,
                ctx.meta_t,
                ctx.bitmask,
                requires_grad=grad_out.requires_grad,
            ),
            None,
            None,
            None,
        )
        return grad_out, None


@torch._dynamo.allow_in_graph
def semi_structured_sparsify(
    x: torch.Tensor,
    algo: str = "",
    backend: str = "cutlass",
) -> SparseSemiStructuredTensor:
    """
    Sparsifies a dense tensor into a semi-structured tensor, according to the algo and backend passed.
    """
    return _SparsifyFunc.apply(x, algo, backend)


@torch._dynamo.allow_in_graph
def semi_structured_sparsify_like(
    x: torch.Tensor,
    pattern: SparseSemiStructuredTensor,
    gradient: GRADIENT_TYPE = GRADIENT_TYPE.SPARSE,
) -> SparseSemiStructuredTensor:
    """
    Sparsifies a dense tensor into a semi-structured tensor, using the mask of the provided pattern.
    """
    return _SparsifyLikeFunc.apply(x, pattern, gradient)
