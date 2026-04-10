# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Unified CPU+GPU buffer for saving expert FFN intermediate activations
during the forward pass and restoring them during backward.

Two sub-buffers:
- dispatch_out (FP8 data + uint8 scales): the dispatch output ``x``
- swiglu_input (BF16): the SwiGLU input ``h13``

A single ``CUDAAllocator`` manages token-level offsets across all
sub-buffers.  GPU memory is used first; overflow spills to CPU-pinned
memory transparently (same virtual address space).
"""

import torch

from torchao.prototype.moe_training.ep.syncless.cuda_allocator import CUDAAllocator
from torchao.prototype.moe_training.ep.syncless.unified_buffer_allocator import (
    create_unified_buffer,
)

# VMM allocation granularity — 2 MiB is the recommended granularity on
# modern GPUs (H100, B200).  If the real granularity differs the
# ``create_unified_buffer`` call will raise.
_VMM_ALIGNMENT = 2 * 1024 * 1024


def _align(size: int) -> int:
    """Round *size* up to the VMM allocation granularity."""
    return ((size + _VMM_ALIGNMENT - 1) // _VMM_ALIGNMENT) * _VMM_ALIGNMENT


class SavedActivationsBuffer:
    """Unified GPU+CPU buffer for saving/restoring expert FFN activations.

    Owns three unified-memory sub-buffers (dispatch output FP8 data,
    dispatch output scales, and SwiGLU input in BF16) plus a
    ``CUDAAllocator`` that manages token-level sub-allocations across
    all of them.

    Args:
        gpu_tokens: Number of tokens that fit in GPU memory.
        cpu_tokens: Number of overflow tokens backed by CPU-pinned memory.
        dim: Model dimension (columns of dispatch output).
        hidden_dim: Expert hidden dimension (SwiGLU input has
            ``2 * hidden_dim`` columns).
        device: CUDA device.
    """

    def __init__(
        self,
        gpu_tokens: int,
        cpu_tokens: int,
        dim: int,
        hidden_dim: int,
        device: torch.device,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.gpu_tokens = gpu_tokens
        self.cpu_tokens = cpu_tokens
        max_tokens = gpu_tokens + cpu_tokens

        block_size = 32  # MXFP8 block size
        scale_dim = dim // block_size

        # -- dispatch_out_data: FP8 e4m3, 1 byte per element ---------
        gpu_data_bytes = _align(gpu_tokens * dim)
        cpu_data_bytes = _align(cpu_tokens * dim) if cpu_tokens > 0 else 0
        self._raw_data = create_unified_buffer(cpu_data_bytes, gpu_data_bytes)
        self.dispatch_out_data = (
            self._raw_data[: max_tokens * dim]
            .view(torch.float8_e4m3fn)
            .view(max_tokens, dim)
        )

        # -- dispatch_out_scales: uint8, 1 byte per element -----------
        gpu_scale_bytes = _align(gpu_tokens * scale_dim)
        cpu_scale_bytes = _align(cpu_tokens * scale_dim) if cpu_tokens > 0 else 0
        self._raw_scales = create_unified_buffer(cpu_scale_bytes, gpu_scale_bytes)
        self.dispatch_out_scales = self._raw_scales[: max_tokens * scale_dim].view(
            max_tokens, scale_dim
        )

        # -- swiglu_input: BF16, 2 bytes per element ------------------
        h13_cols = 2 * hidden_dim
        gpu_h13_bytes = _align(gpu_tokens * h13_cols * 2)
        cpu_h13_bytes = _align(cpu_tokens * h13_cols * 2) if cpu_tokens > 0 else 0
        self._raw_h13 = create_unified_buffer(cpu_h13_bytes, gpu_h13_bytes)
        self.swiglu_input = (
            self._raw_h13[: max_tokens * h13_cols * 2]
            .view(torch.bfloat16)
            .view(max_tokens, h13_cols)
        )

        # -- Shared allocator (token offsets) -------------------------
        pools = [gpu_tokens]
        if cpu_tokens > 0:
            pools.append(cpu_tokens)
        self.allocator = CUDAAllocator(self._raw_data, pools)

        # Python-side offset tracker for zero-sync buffer indexing.
        # The CUDAAllocator returns GPU tensors which require .item()
        # (D2H sync) to use as Python slice indices.  This simple
        # bump allocator avoids that: offsets are always Python ints.
        self._py_offset: int = 0

    def alloc_py(self, num_tokens: int) -> int:
        """Claim *num_tokens* rows and return a **Python int** offset.

        Zero device-to-host synchronisation — the offset is tracked
        purely on the CPU side.  Callers must free in LIFO order via
        :meth:`free_py`.
        """
        offset = self._py_offset
        if offset + num_tokens > self.gpu_tokens + self.cpu_tokens:
            raise RuntimeError(
                f"SavedActivationsBuffer overflow: requested {num_tokens} "
                f"at offset {offset}, capacity {self.gpu_tokens + self.cpu_tokens}"
            )
        self._py_offset += num_tokens
        return offset

    def free_py(self, num_tokens: int) -> None:
        """Release *num_tokens* rows (LIFO order)."""
        self._py_offset -= num_tokens

    def free_all_py(self) -> None:
        """Reset the Python-side offset tracker."""
        self._py_offset = 0
