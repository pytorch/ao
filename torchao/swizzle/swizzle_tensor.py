# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils._pytree import tree_map


# copied from float8_utils.py
def _get_min_alignment(size: int, alignment_value: int) -> int:
    return (1 + ((size - 1) // alignment_value)) * alignment_value


class SwizzleTensor(torch.Tensor):
    """
    A Python-only swizzled tensor subclass.

    Intended usage of this abstraction:
    Swizzle weight Tensor to avoid LDS use during GEMMs on ROCm hardware.
    """

    def __new__(
        cls,
        original: torch.Tensor,
        shallow: bool = False,
    ):
        wrapper = torch.empty_like(original, device="meta")
        return torch.Tensor._make_subclass(cls, wrapper)

    def __init__(self, original, shallow=False):
        if shallow:
            return
        # assert original.ndim == 2 or original.ndim == 3  # (M, K) or (B, M, K)
        assert original.ndim == 2, "SwizzleTensor only supports ndim 2"
        assert original.itemsize == 1 or original.itemsize == 2
        kdiv = 32 if original.itemsize == 2 else 64
        lastdim = 8 if original.itemsize == 2 else 16
        if original.ndim == 2:
            M, K = original.shape
            B = 0
        if original.ndim == 3:
            B, M, K = original.shape
        alignedM = _get_min_alignment(M, 16)
        alignedK = _get_min_alignment(K, kdiv)
        paddedM = alignedM - M
        paddedK = alignedK - K
        x = torch.nn.functional.pad(original, (0, paddedK, 0, paddedM), "constant", 0)
        if original.ndim == 2:
            x = x.view(alignedM // 16, 16, alignedK // kdiv, 4, lastdim)
            x = x.permute(0, 2, 3, 1, 4)
        if original.ndim == 3:
            x = x.view(B, alignedM // 16, 16, alignedK // kdiv, 4, lastdim)
            x = x.permute(0, 1, 3, 4, 2, 5)
        self.x = x.contiguous()
        self.B = B
        self.M = M
        self.K = K
        self.alignedM = alignedM
        self.alignedK = alignedK
        self.paddedM = paddedM
        self.paddedK = paddedK
        self.original_ndim = original.ndim
        self.is_transposed = False

    def __repr__(self):
        return f"{self.__class__.__name__}(original={self.unswizzle()})"

    def unswizzle(self):
        undone = None
        if self.original_ndim == 2:
            undone = self.x.permute(0, 3, 1, 2, 4).contiguous()
            undone = undone.reshape(self.alignedM, self.alignedK)
            undone = undone[0 : self.M, 0 : self.K]
            undone = undone.reshape(self.M, self.K)
            if self.is_transposed:
                undone = undone.T
        if self.original_ndim == 3:
            undone = self.x.permute(0, 1, 4, 2, 3, 5).contiguous()
            undone = undone.reshape(self.B, self.alignedM, self.alignedK)
            undone = undone[0 : self.B, 0 : self.M, 0 : self.K]
            undone = undone.reshape(self.B, self.M, self.K)
        return undone

    def as_tensor(self):
        # note the transpose because this causes col major hipblaslt op to be TN
        if self.original_ndim == 2:
            tmp = self.x.reshape(self.alignedM, self.alignedK)
            if self.is_transposed:
                tmp = tmp.T
            return tmp
        if self.original_ndim == 3:
            tmp = self.x.reshape(self.B, self.alignedM, self.alignedK)
            if self.is_transposed:
                tmp = tmp.T
            return tmp

    def shallow_transpose(self):
        shape = (
            (self.M, self.K) if self.original_ndim == 2 else (self.B, self.M, self.K),
        )
        new_obj = SwizzleTensor(
            torch.empty(*shape, dtype=self.dtype, layout=self.layout, device="meta"),
            True,
        )
        new_obj.x = self.x
        new_obj.B = self.B
        new_obj.M = self.M
        new_obj.K = self.K
        new_obj.alignedM = self.alignedM
        new_obj.alignedK = self.alignedK
        new_obj.paddedM = self.paddedM
        new_obj.paddedK = self.paddedK
        new_obj.original_ndim = self.original_ndim
        new_obj.is_transposed = not self.is_transposed
        return new_obj

    @property
    def shape(self):
        return torch.Size((self.K, self.M) if self.is_transposed else (self.M, self.K))

    def stride(self):
        return (1, self.K) if self.is_transposed else (self.K, 1)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # Lazy import to avoid circular dependency
        from torchao.swizzle.swizzle_ops import SWIZZLE_OPS_TABLE

        if func in SWIZZLE_OPS_TABLE:
            return SWIZZLE_OPS_TABLE[func](func, args, kwargs)

        def unwrap(e):
            return e.unswizzle() if isinstance(e, SwizzleTensor) else e

        def wrap(e):
            return SwizzleTensor(e) if isinstance(e, torch.Tensor) else e

        return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))

    # Do not force the SwizzleTensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
