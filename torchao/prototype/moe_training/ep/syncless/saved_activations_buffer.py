# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Unified CPU+GPU buffer for saving expert FFN intermediate activations
during the forward pass and restoring them during backward.

Three sub-buffers, all indexed by the same token offset:
- e4m3_data (FP8 e4m3): requanted dispatch output ``x`` in (dim, max_tokens) col-major storage
- e8m0_scales (uint8): 32×1 MXFP8 scales for ``x`` in (dim, max_tokens//32) layout
- swiglu_input (BF16): the SwiGLU input ``h13``

Two allocation modes:

1. **Python-side bump allocator** (``alloc_py`` / ``free_py``):
   Tracks token offsets on CPU with zero D2H sync.  Used when the
   caller knows ``num_tokens`` on the CPU side.

2. **GPU-side CUDAAllocator** (``allocator.alloc`` / ``allocator.free``):
   Tracks token offsets entirely on-GPU via Triton kernels, enabling
   exact allocation when ``num_tokens`` is a GPU tensor (e.g. from
   ``group_end_offs[-1]``).  Zero D2H sync.

GPU memory is used first; overflow spills to CPU-pinned memory
transparently (same virtual address space via ``DeferredUnifiedBuffer``).
"""

import torch

from torchao.prototype.moe_training.ep.syncless.cuda_allocator import CUDAAllocator
from torchao.prototype.moe_training.ep.syncless.unified_buffer_allocator import (
    DeferredUnifiedBuffer,
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

    Owns three unified-memory sub-buffers:
    - e4m3_data: (dim, max_tokens) col-major FP8 data with 32×1 scaling
    - e8m0_scales: (dim, max_tokens//32) uint8 E8M0 scales for 32×1 scaling
    - swiglu_input: (max_tokens, 2*hidden_dim) BF16 SwiGLU input

    All three are indexed by the same token offset.

    Supports two allocation modes:
    - Python-side bump allocator (``alloc_py`` / ``free_py``)
    - GPU-side CUDAAllocator (``allocator``) for zero-sync exact allocation

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
        assert (
            max_tokens % block_size == 0
        ), f"max_tokens ({max_tokens}) must be divisible by block_size ({block_size})"
        self._scale_dim = dim // block_size
        self._h13_cols = 2 * hidden_dim

        # -- e4m3_data: (dim, max_tokens) FP8 e4m3, 1 byte per element --
        # Transposed storage for col-major 32×1 scaled data written by
        # the fused dequant-requant kernel in the forward pass.
        gpu_data_bytes = _align(gpu_tokens * dim)
        cpu_data_bytes = _align(cpu_tokens * dim) if cpu_tokens > 0 else 0
        self._raw_data = DeferredUnifiedBuffer(gpu_data_bytes, cpu_data_bytes)
        self.e4m3_data = (
            self._raw_data.tensor[: dim * max_tokens]
            .view(torch.float8_e4m3fn)
            .view(dim, max_tokens)
        )

        # -- e8m0_scales: (dim, max_tokens//32) uint8, 1 byte per element --
        # 32×1 E8M0 scales matching e4m3_data layout.
        # The same raw buffer is also used for blocked-format scale output
        # in the backward (via blocked_e8m0_scales flat view).
        scale_cols = max_tokens // block_size
        gpu_scale_bytes = _align(gpu_tokens * self._scale_dim)
        cpu_scale_bytes = _align(cpu_tokens * self._scale_dim) if cpu_tokens > 0 else 0
        self._raw_scales = DeferredUnifiedBuffer(gpu_scale_bytes, cpu_scale_bytes)
        self.e8m0_scales = self._raw_scales.tensor[: dim * scale_cols].view(
            dim, scale_cols
        )

        # Flat view of the same raw scale buffer for the backward's
        # triton_scale_blocked_layout_saved_activation_buffer to write
        # blocked scales at a GPU-resident offset.
        self.blocked_e8m0_scales = self._raw_scales.tensor

        # -- swiglu_input: BF16, 2 bytes per element ------------------
        gpu_h13_bytes = _align(gpu_tokens * self._h13_cols * 2)
        cpu_h13_bytes = _align(cpu_tokens * self._h13_cols * 2) if cpu_tokens > 0 else 0
        self._raw_h13 = DeferredUnifiedBuffer(gpu_h13_bytes, cpu_h13_bytes)
        self.swiglu_input = (
            self._raw_h13.tensor[: max_tokens * self._h13_cols * 2]
            .view(torch.bfloat16)
            .view(max_tokens, self._h13_cols)
        )

        # -- GPU-side CUDAAllocator -----------------------------------
        # Logical token-offset allocator shared across all three
        # sub-buffers.  The allocator tracks offsets in token-row space
        # (not bytes) — each sub-buffer multiplies the offset by its
        # own bytes-per-token to compute byte addresses.
        # The device_tensor arg is only used to infer the CUDA device.
        self.allocator = CUDAAllocator(self._raw_data.tensor, [gpu_tokens, cpu_tokens])

        # Eagerly map all GPU pages so CUDAAllocator's best-fit
        # (non-sequential) allocation always hits mapped memory.
        # CPU pages remain deferred.
        self._raw_data.ensure_mapped(gpu_tokens * dim)
        self._raw_scales.ensure_mapped(gpu_tokens * self._scale_dim)
        self._raw_h13.ensure_mapped(gpu_tokens * self._h13_cols * 2)

    def alloc(self, num_tokens: "torch.Tensor | int") -> torch.Tensor:
        """Allocate *num_tokens* rows and return a scalar int64 GPU tensor offset.

        *num_tokens* may be a scalar GPU tensor (int64) or a Python int.
        The allocation is handled entirely on-GPU via the CUDAAllocator
        — zero D2H sync.

        The returned offset indexes into all three sub-buffers (data,
        scales, h13) uniformly.  Each sub-buffer multiplies the offset
        by its own bytes-per-token to compute byte addresses.
        """
        offset = self.allocator.alloc(num_tokens)
        return offset

    def free(self, offset: "torch.Tensor | int") -> None:
        """Free a previously allocated offset (GPU-side, no sync).

        *offset* may be a scalar GPU tensor (int64) or a Python int.
        """
        self.allocator.free(offset)

    def free_all(self) -> None:
        """Reset the GPU-side CUDAAllocator (all pools to initial state)."""
        self.allocator.free_all()
