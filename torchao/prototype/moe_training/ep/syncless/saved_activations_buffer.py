# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Unified CPU+GPU buffer for saving expert FFN intermediate activations
during the forward pass and restoring them during backward.

Three sub-buffers, all indexed by the same token offset:
- dispatch_out_data (FP8 e4m3): the quantised dispatch output ``x``
- dispatch_out_scales (uint8): the MXFP8 scales for ``x``
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

    Owns three unified-memory sub-buffers (dispatch output FP8 data,
    dispatch output scales, and SwiGLU input in BF16).  All three are
    indexed by the same token offset.

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
        self._scale_dim = dim // block_size
        self._h13_cols = 2 * hidden_dim

        # -- dispatch_out_data: FP8 e4m3, 1 byte per element ---------
        gpu_data_bytes = _align(gpu_tokens * dim)
        cpu_data_bytes = _align(cpu_tokens * dim) if cpu_tokens > 0 else 0
        self._raw_data = DeferredUnifiedBuffer(gpu_data_bytes, cpu_data_bytes)
        self.dispatch_out_data = (
            self._raw_data.tensor[: max_tokens * dim]
            .view(torch.float8_e4m3fn)
            .view(max_tokens, dim)
        )

        # -- dispatch_out_scales: uint8, 1 byte per element -----------
        gpu_scale_bytes = _align(gpu_tokens * self._scale_dim)
        cpu_scale_bytes = _align(cpu_tokens * self._scale_dim) if cpu_tokens > 0 else 0
        self._raw_scales = DeferredUnifiedBuffer(gpu_scale_bytes, cpu_scale_bytes)
        self.dispatch_out_scales = self._raw_scales.tensor[
            : max_tokens * self._scale_dim
        ].view(max_tokens, self._scale_dim)

        # -- swiglu_input: BF16, 2 bytes per element ------------------
        gpu_h13_bytes = _align(gpu_tokens * self._h13_cols * 2)
        cpu_h13_bytes = _align(cpu_tokens * self._h13_cols * 2) if cpu_tokens > 0 else 0
        self._raw_h13 = DeferredUnifiedBuffer(gpu_h13_bytes, cpu_h13_bytes)
        self.swiglu_input = (
            self._raw_h13.tensor[: max_tokens * self._h13_cols * 2]
            .view(torch.bfloat16)
            .view(max_tokens, self._h13_cols)
        )

        # -- Python-side bump allocator (zero D2H sync) ---------------
        self._py_offset: int = 0

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

        self._cpu_mapped: bool = False

    # -- Raw buffer accessors (flat uint8, no shape/view transforms) ---

    @property
    def dispatch_out_data_raw(self) -> torch.Tensor:
        """Flat uint8 tensor for dispatch output FP8 data."""
        return self._raw_data.tensor

    @property
    def dispatch_out_scales_raw(self) -> torch.Tensor:
        """Flat uint8 tensor for dispatch output scales."""
        return self._raw_scales.tensor

    @property
    def swiglu_input_raw(self) -> torch.Tensor:
        """Flat uint8 tensor for SwiGLU input (BF16 data as raw bytes)."""
        return self._raw_h13.tensor

    # -- CPU overflow mapping ------------------------------------------

    def ensure_cpu_mapped(self) -> None:
        """Eagerly map all CPU overflow pages across the three sub-buffers.

        Called when the allocator's GPU pool is exhausted and allocation
        spills to the CPU pool.  No-op after the first call.
        """
        if self._cpu_mapped or self.cpu_tokens == 0:
            return
        total_data = (self.gpu_tokens + self.cpu_tokens) * self.dim
        total_scales = (self.gpu_tokens + self.cpu_tokens) * self._scale_dim
        total_h13 = (self.gpu_tokens + self.cpu_tokens) * self._h13_cols * 2
        self._raw_data.ensure_mapped(total_data)
        self._raw_scales.ensure_mapped(total_scales)
        self._raw_h13.ensure_mapped(total_h13)
        self._cpu_mapped = True

    # -- GPU-side allocator methods ------------------------------------

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

    # -- Python-side bump allocator (backward compat) ------------------

    def alloc_py(self, num_tokens: int) -> int:
        """Claim *num_tokens* rows and return a **Python int** offset.

        Zero device-to-host synchronisation — the offset is tracked
        purely on the CPU side.  Callers must free in LIFO order via
        :meth:`free_py`.

        Physical pages are mapped lazily on the first access via
        ``DeferredUnifiedBuffer.ensure_mapped``.
        """
        offset = self._py_offset
        end = offset + num_tokens
        if end > self.gpu_tokens + self.cpu_tokens:
            raise RuntimeError(
                f"SavedActivationsBuffer overflow: requested {num_tokens} "
                f"at offset {offset}, capacity {self.gpu_tokens + self.cpu_tokens}"
            )
        self._raw_data.ensure_mapped(end * self.dim)
        self._raw_scales.ensure_mapped(end * self._scale_dim)
        self._raw_h13.ensure_mapped(end * self._h13_cols * 2)
        self._py_offset = end
        return offset

    def free_py(self, num_tokens: int) -> None:
        """Release *num_tokens* rows (LIFO order)."""
        self._py_offset -= num_tokens

    def free_all_py(self) -> None:
        """Reset the Python-side offset tracker."""
        self._py_offset = 0
