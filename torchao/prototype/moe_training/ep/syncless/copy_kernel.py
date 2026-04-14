# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Triton kernels for syncless expert-parallel buffer operations.

``copy_into_buffer_2d`` copies a 2D source tensor ``src[0:num_rows, 0:cols]``
into a flat destination buffer at a GPU-determined row offset, without any
D2H synchronisation.  Both ``offset`` and ``num_rows`` are scalar GPU
tensors (int64).
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _copy_into_buffer_2d_kernel(
    src_ptr,
    dst_ptr,
    offset_ptr,  # scalar int64 GPU tensor: row offset in dst
    num_rows_ptr,  # scalar int64 GPU tensor: number of rows to copy
    cols: tl.constexpr,  # number of columns (compile-time constant)
    BLOCK_SIZE: tl.constexpr,  # elements per program instance
):
    """Copy src[0:num_rows, 0:cols] -> dst[offset:offset+num_rows, 0:cols].

    All pointers are element-typed (e.g. uint8 for FP8 data).  The kernel
    is launched with enough programs to cover ``num_rows * cols`` elements.
    """
    pid = tl.program_id(0)
    offset = tl.load(offset_ptr).to(tl.int64)
    num_rows = tl.load(num_rows_ptr).to(tl.int64)
    total_elems = num_rows * cols

    elem_ids = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = elem_ids < total_elems

    vals = tl.load(src_ptr.to(tl.pointer_type(tl.uint8)) + elem_ids, mask=mask)
    dst_start = offset * cols
    tl.store(
        dst_ptr.to(tl.pointer_type(tl.uint8)) + dst_start + elem_ids, vals, mask=mask
    )


@triton_op("torchao::copy_into_buffer_2d", mutates_args={"dst"})
def copy_into_buffer_2d(
    src: torch.Tensor,
    dst: torch.Tensor,
    offset: torch.Tensor,
    num_rows: torch.Tensor,
    cols: int,
) -> None:
    """Copy ``src[0:num_rows, 0:cols]`` into ``dst[offset:offset+num_rows, 0:cols]``.

    ``dst`` is a flat (1-D) tensor of the same dtype as ``src``.
    ``offset`` and ``num_rows`` are scalar int64 GPU tensors — no D2H sync.

    Args:
        src: Source tensor, contiguous, at least ``num_rows * cols`` elements.
        dst: Destination flat buffer, same dtype as src.
        offset: Scalar int64 GPU tensor — row offset into dst.
        num_rows: Scalar int64 GPU tensor — number of rows to copy.
        cols: Number of columns (Python int, known at launch time).
    """
    BLOCK_SIZE = 1024
    # Upper bound on grid size: use src's full row count since
    # num_rows is on GPU and we can't read it on CPU.
    max_elems = src.shape[0] * cols
    grid = ((max_elems + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    wrap_triton(_copy_into_buffer_2d_kernel)[grid](
        src.view(-1),
        dst,
        offset,
        num_rows,
        cols=cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
