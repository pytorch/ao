# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils._triton import has_triton

from torchao.utils import is_sm_at_least_100, torch_version_at_least

BLOCK_M = 128
BLOCK_N = 128
SAME_BOTH_DIMS = 0
VARYING_FIRST_DIM = 1


if torch_version_at_least("2.10.0") and has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _group_idx_from_range(
        element_offset,
        group_range_ptr,
        num_tensors: tl.constexpr,
    ):
        group_idx = 0
        for i in range(num_tensors):
            start = tl.load(group_range_ptr + i)
            if element_offset >= start:
                group_idx = i
        return group_idx

    @triton.jit
    def _get_group_idx_binary(token_idx, group_range_ptr, num_groups):
        """
        Use binary search to find group_idx for this token_idx.
        Load each group offset individually to support arbitrary number of groups.
        Preloading all offsets requires rounding up to next power of 2, which
        requires host code that is not graph-safe.
        """
        start = 0
        end = num_groups
        while (end - start) != 1:
            mid = (end - start) // 2 + start
            mid_idx = tl.load(group_range_ptr + mid)
            if token_idx >= mid_idx:
                start = mid
            else:
                end = mid
        return start


def _validate_grouped_hadamard_inputs(
    A: torch.Tensor,
    B: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    packed_sequence_length: int,
    hidden_size: int,
    shape_rep: int,
) -> None:
    if not isinstance(A, torch.Tensor):
        raise TypeError("A must be a torch.Tensor")
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got {A.ndim}D")
    if A.dtype != torch.bfloat16:
        raise ValueError("A.dtype must be torch.bfloat16")
    if not A.is_contiguous():
        raise ValueError("A must be row-major (contiguous)")
    if not isinstance(B, torch.Tensor):
        raise TypeError("B must be a torch.Tensor")
    if B.ndim != 2:
        raise ValueError(f"B must be 2D, got {B.ndim}D")
    if B.dtype != torch.bfloat16:
        raise ValueError("B.dtype must be torch.bfloat16")
    if B.shape != (16, 16):
        raise ValueError(f"B must have shape (16, 16), got {tuple(B.shape)}")
    if A.shape[1] % BLOCK_N != 0:
        raise ValueError("A.shape[1] must be divisible by 128")
    if not A.is_cuda:
        raise ValueError("A must be on CUDA")
    if not B.is_cuda:
        raise ValueError("B must be on CUDA")
    if B.device != A.device:
        raise ValueError("B must be on the same device as A")
    if torch.cuda.is_available() and not is_sm_at_least_100():
        raise NotImplementedError(
            "GroupRHT Triton kernel requires SM100 (Blackwell) for FP4 conversion."
        )
    if packed_sequence_length != A.shape[0]:
        raise ValueError("packed_sequence_length must match A.shape[0]")
    if hidden_size != A.shape[1]:
        raise ValueError("hidden_size must match A.shape[1]")
    if packed_sequence_length % BLOCK_M != 0:
        raise ValueError("packed_sequence_length must be divisible by 128")
    if hidden_size % BLOCK_N != 0:
        raise ValueError("hidden_size must be divisible by 128")
    if num_tensors <= 0:
        raise ValueError("num_tensors must be greater than 0")
    if shape_rep not in (SAME_BOTH_DIMS, VARYING_FIRST_DIM):
        raise ValueError(
            "graph-safe TE GroupRHT only supports SAME_BOTH_DIMS or VARYING_FIRST_DIM"
        )
    if shape_rep == SAME_BOTH_DIMS:
        if packed_sequence_length % num_tensors != 0:
            raise ValueError(
                "packed_sequence_length must be divisible by num_tensors "
                "for SAME_BOTH_DIMS"
            )
        if (packed_sequence_length // num_tensors) % BLOCK_M != 0:
            raise ValueError("SAME_BOTH_DIMS group row count must be divisible by 128")
    if not isinstance(offsets, torch.Tensor):
        raise TypeError("offsets must be a torch.Tensor")
    if offsets.ndim != 1:
        raise ValueError(f"offsets must be 1D, got {offsets.ndim}D")
    if offsets.dtype != torch.int64:
        raise ValueError("offsets.dtype must be torch.int64")
    if not offsets.is_cuda:
        raise ValueError("offsets must be on CUDA")
    if offsets.device != A.device:
        raise ValueError("offsets must be on the same device as A")
    if offsets.numel() < num_tensors:
        raise ValueError("offsets must have at least num_tensors elements")
    if shape_rep == VARYING_FIRST_DIM:
        torch.ops.aten._assert_async.msg(
            torch.all(offsets[:num_tensors] % (BLOCK_M * hidden_size) == 0),
            "VARYING_FIRST_DIM offsets must align group boundaries to 128-row tiles",
        )
