# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU-resident memory allocator using Triton kernels.

Pure-Python port of ``gb200_moe_sol/csrc/cuda_allocator.cu`` — no C++
extensions, **no CPU synchronisation** on the alloc/free hot path.

Design overview
---------------
Same semantics as the CUDA C++ original:

* Doubly-linked list of memory blocks stored entirely in GPU global memory.
* Best-fit allocation with block splitting.
* Coalescing on free with block-list compaction.
* Multi-pool support (pools tried in order; GPU pool first, CPU fallback).

The state is packed into a single flat ``int32`` tensor on the GPU.  The
Triton kernels are launched with ``grid=(1,)`` — one program instance that
performs the scan in parallel (``tl.arange``) and the mutations serially.

Limitations (same as C++)
-------------------------
* Max 1024 blocks per pool (``_MAX_BLOCKS``).
* NOT thread-safe across streams (caller must synchronise).
* OOM returns addr=0; caller can check ``stats().num_ooms``.
* Double-free / freeing an unknown address is a silent no-op.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# State layout constants
# ---------------------------------------------------------------------------
# The allocator state for *each pool* is stored as a struct-of-arrays inside
# a flat int32 tensor.  Multi-pool states are concatenated along the first
# axis: ``state[pool_idx * _POOL_SIZE + field_offset + block_idx]``.
#
# Layout (offsets in int32 elements):
#   [0]              last_block_id
#   [1]              num_ooms
#   [2..3]           padding
#   [4 .. 4+N)       addrs[N]
#   [4+N .. 4+2N)    lengths[N]
#   [4+2N .. 4+3N)   is_allocated[N]   (0 = free, 1 = allocated)
#   [4+3N .. 4+4N)   prev_ids[N]       (-1 = none)
#   [4+4N .. 4+5N)   next_ids[N]       (-1 = none)
#
# where N = _MAX_BLOCKS = 1024.
# ---------------------------------------------------------------------------

_MAX_BLOCKS: int = 1024

_META_LAST_BLOCK_ID: int = 0
_META_NUM_OOMS: int = 1

_OFF_ADDRS: int = 4
_OFF_LENS: int = _OFF_ADDRS + _MAX_BLOCKS
_OFF_ALLOCS: int = _OFF_LENS + _MAX_BLOCKS
_OFF_PREVS: int = _OFF_ALLOCS + _MAX_BLOCKS
_OFF_NEXTS: int = _OFF_PREVS + _MAX_BLOCKS

_POOL_SIZE: int = _OFF_NEXTS + _MAX_BLOCKS  # 5124 int32 per pool


# ---------------------------------------------------------------------------
# Triton JIT helpers (inlined into the entry-point kernels)
# ---------------------------------------------------------------------------


@triton.jit
def _remove_block_jit(
    s,  # state_ptr (pointer to int32 state tensor)
    pb,  # pool_base offset
    block_id,  # block to remove (swap with last)
    tracked_id,  # block we're tracking (may move during swap)
    # layout offsets (passed as tl.constexpr from caller)
    OA: tl.constexpr,
    OL: tl.constexpr,
    OC: tl.constexpr,
    OP: tl.constexpr,
    ON: tl.constexpr,
):
    """Remove *block_id* by swapping with the last block, then decrement
    ``last_block_id``.  Returns the (possibly updated) *tracked_id*.
    """
    last_id = tl.load(s + pb + 0)  # _META_LAST_BLOCK_ID
    tl.store(s + pb + 0, last_id - 1)

    ret = tracked_id

    if last_id != block_id:
        if ret == last_id:
            ret = block_id

        # Read last block's fields
        lb_a = tl.load(s + pb + OA + last_id)
        lb_l = tl.load(s + pb + OL + last_id)
        lb_c = tl.load(s + pb + OC + last_id)
        lb_p = tl.load(s + pb + OP + last_id)
        lb_n = tl.load(s + pb + ON + last_id)

        # Fix up last block's neighbours to point at the vacated slot
        if lb_p >= 0:
            tl.store(s + pb + ON + lb_p, block_id)
        if lb_n >= 0:
            tl.store(s + pb + OP + lb_n, block_id)

        # Copy last block into the vacated slot
        tl.store(s + pb + OA + block_id, lb_a)
        tl.store(s + pb + OL + block_id, lb_l)
        tl.store(s + pb + OC + block_id, lb_c)
        tl.store(s + pb + OP + block_id, lb_p)
        tl.store(s + pb + ON + block_id, lb_n)

    return ret


# ---------------------------------------------------------------------------
# Alloc kernel
# ---------------------------------------------------------------------------


@triton.jit
def _alloc_kernel(
    state_ptr,
    alloc_sz_ptr,  # pointer to int64 scalar (input: size to allocate)
    return_addr_ptr,  # pointer to int64 scalar (output: allocated address)
    NUM_POOLS: tl.constexpr,
    POOL_STRIDE: tl.constexpr,
    N: tl.constexpr,  # _MAX_BLOCKS
    OA: tl.constexpr,
    OL: tl.constexpr,
    OC: tl.constexpr,
    OP: tl.constexpr,
    ON: tl.constexpr,
):
    alloc_sz = tl.load(alloc_sz_ptr).to(tl.int32)
    allocated: tl.int32 = 0

    for _pool_idx in tl.static_range(NUM_POOLS):
        if allocated == 0:
            pb = _pool_idx * POOL_STRIDE
            last_bid = tl.load(state_ptr + pb + 0)  # last_block_id

            # -- Parallel scan: load all block metadata --
            bids = tl.arange(0, N)
            valid = bids <= last_bid
            lens = tl.load(state_ptr + pb + OL + bids, mask=valid, other=0)
            allocs = tl.load(state_ptr + pb + OC + bids, mask=valid, other=1)

            # Best-fit: smallest free block with length >= alloc_sz
            is_cand = (allocs == 0) & (lens >= alloc_sz) & valid
            search = tl.where(is_cand, lens, 0x7FFFFFFF)
            min_len = tl.min(search)

            if min_len < 0x7FFFFFFF:
                # -- Find winner block index (first with min_len) --
                winner_ids = tl.where(search == min_len, bids, N)
                best_id = tl.min(winner_ids)

                # Read winner block state
                blk_addr = tl.load(state_ptr + pb + OA + best_id)
                blk_len = tl.load(state_ptr + pb + OL + best_id)
                blk_prev = tl.load(state_ptr + pb + OP + best_id)

                if blk_len != alloc_sz:
                    # Split: [..before][ new_free ][ blk (alloc) ][after..]
                    new_id = last_bid + 1
                    tl.store(state_ptr + pb + 0, new_id)  # last_block_id

                    leftover = blk_len - alloc_sz

                    # Write new free block at new_id
                    tl.store(state_ptr + pb + OA + new_id, blk_addr)
                    tl.store(state_ptr + pb + OL + new_id, leftover)
                    tl.store(state_ptr + pb + OC + new_id, 0)  # free
                    tl.store(state_ptr + pb + OP + new_id, blk_prev)
                    tl.store(state_ptr + pb + ON + new_id, best_id)

                    if blk_prev >= 0:
                        tl.store(state_ptr + pb + ON + blk_prev, new_id)

                    # Update allocated block
                    blk_addr = blk_addr + leftover
                    tl.store(state_ptr + pb + OA + best_id, blk_addr)
                    tl.store(state_ptr + pb + OP + best_id, new_id)
                    tl.store(state_ptr + pb + OL + best_id, alloc_sz)

                # Mark allocated
                tl.store(state_ptr + pb + OC + best_id, 1)

                # Write result
                tl.store(return_addr_ptr, blk_addr.to(tl.int64))
                allocated = 1
            else:
                # Pool exhausted — increment OOM for this pool
                old_ooms = tl.load(state_ptr + pb + 1)
                tl.store(state_ptr + pb + 1, old_ooms + 1)

    if allocated == 0:
        # OOM across all pools
        tl.store(return_addr_ptr, tl.load(alloc_sz_ptr) * 0)  # int64 zero


# ---------------------------------------------------------------------------
# Free kernel
# ---------------------------------------------------------------------------


@triton.jit
def _free_kernel(
    state_ptr,
    addr_ptr,  # pointer to int64 scalar (address to free)
    NUM_POOLS: tl.constexpr,
    POOL_STRIDE: tl.constexpr,
    N: tl.constexpr,
    OA: tl.constexpr,
    OL: tl.constexpr,
    OC: tl.constexpr,
    OP: tl.constexpr,
    ON: tl.constexpr,
):
    addr = tl.load(addr_ptr).to(tl.int32)

    for _pool_idx in tl.static_range(NUM_POOLS):
        pb = _pool_idx * POOL_STRIDE
        last_bid = tl.load(state_ptr + pb + 0)

        # -- Parallel search for matching address --
        bids = tl.arange(0, N)
        valid = bids <= last_bid
        addrs = tl.load(state_ptr + pb + OA + bids, mask=valid, other=-1)
        match = (addrs == addr) & valid
        match_ids = tl.where(match, bids, N)
        block_id = tl.min(match_ids)

        if block_id < N:
            # Found — mark as free
            tl.store(state_ptr + pb + OC + block_id, 0)

            # -- Coalesce with previous --
            prev_id = tl.load(state_ptr + pb + OP + block_id)
            if prev_id >= 0:
                prev_alloc = tl.load(state_ptr + pb + OC + prev_id)
                if prev_alloc == 0:
                    prev_addr = tl.load(state_ptr + pb + OA + prev_id)
                    prev_len = tl.load(state_ptr + pb + OL + prev_id)
                    prev_prev = tl.load(state_ptr + pb + OP + prev_id)

                    cur_len = tl.load(state_ptr + pb + OL + block_id)
                    tl.store(state_ptr + pb + OA + block_id, prev_addr)
                    tl.store(state_ptr + pb + OL + block_id, cur_len + prev_len)
                    tl.store(state_ptr + pb + OP + block_id, prev_prev)
                    if prev_prev >= 0:
                        tl.store(state_ptr + pb + ON + prev_prev, block_id)

                    block_id = _remove_block_jit(
                        state_ptr, pb, prev_id, block_id, OA, OL, OC, OP, ON
                    )

            # -- Coalesce with next --
            next_id = tl.load(state_ptr + pb + ON + block_id)
            if next_id >= 0:
                next_alloc = tl.load(state_ptr + pb + OC + next_id)
                if next_alloc == 0:
                    next_len = tl.load(state_ptr + pb + OL + next_id)
                    next_next = tl.load(state_ptr + pb + ON + next_id)

                    cur_len = tl.load(state_ptr + pb + OL + block_id)
                    tl.store(state_ptr + pb + OL + block_id, cur_len + next_len)
                    tl.store(state_ptr + pb + ON + block_id, next_next)
                    if next_next >= 0:
                        tl.store(state_ptr + pb + OP + next_next, block_id)

                    _remove_block_jit(
                        state_ptr, pb, next_id, block_id, OA, OL, OC, OP, ON
                    )


# ---------------------------------------------------------------------------
# Stats (pure PyTorch tensor ops — no CPU sync, results stay on GPU)
# ---------------------------------------------------------------------------


@dataclass
class MemoryStats:
    """Per-pool allocation statistics.

    Each field is a ``torch.Tensor`` of shape ``[num_pools]`` on the
    allocator's device, matching the output format of the C++
    ``cuda_alloc_stats`` kernel.
    """

    last_block_id: torch.Tensor
    num_ooms: torch.Tensor
    sum_allocated: torch.Tensor
    num_free_blocks: torch.Tensor
    largest_free_block: torch.Tensor


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


class CUDAAllocator:
    """GPU-resident best-fit memory allocator with multi-pool support.

    Same semantics as the C++ ``cuda_allocator.cu``:

    * Multiple pools, tried in order (pool 0 first, then pool 1, …).
    * Best-fit allocation: picks the smallest free block that fits.
    * Block splitting when the chosen block is larger than needed.
    * Coalescing on free: merges with adjacent free neighbours.
    * Block-list compaction: removed blocks are swapped with the last entry.

    **Zero CPU synchronisation** on the alloc/free path.  The allocator
    state lives entirely in a GPU ``int32`` tensor, and alloc/free are
    single Triton kernel launches (~5-15 μs each after JIT warm-up).

    Args:
        device_tensor: Any tensor on the target CUDA device (used only to
            infer ``self.device``).
        alloc_sizes: Per-pool sizes.  Pools are laid out contiguously in the
            virtual address space: pool 0 starts at address 0, pool 1 at
            ``alloc_sizes[0]``, etc.  Place faster memory (GPU HBM) first.
    """

    def __init__(self, device_tensor: torch.Tensor, alloc_sizes: list[int]) -> None:
        self.device = device_tensor.device
        self.num_pools = len(alloc_sizes)
        self.alloc_sizes = list(alloc_sizes)

        # Build initial state tensor on GPU
        state = torch.zeros(
            self.num_pools * _POOL_SIZE, dtype=torch.int32, device=self.device
        )
        cumsum = 0
        for i, pool_sz in enumerate(alloc_sizes):
            base = i * _POOL_SIZE
            state[base + _META_LAST_BLOCK_ID] = 0
            state[base + _META_NUM_OOMS] = 0
            # Block 0: one big free block spanning the whole pool
            state[base + _OFF_ADDRS] = cumsum
            state[base + _OFF_LENS] = pool_sz
            state[base + _OFF_ALLOCS] = 0  # free
            state[base + _OFF_PREVS] = -1
            state[base + _OFF_NEXTS] = -1
            cumsum += pool_sz

        self.states = state
        self.states_init = state.clone()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def alloc(
        self, sz: "torch.Tensor | int", *, assert_on_oom: bool = True
    ) -> torch.Tensor:
        """Allocate *sz* contiguous elements.  Returns a scalar ``int64``
        GPU tensor containing the start address (offset).

        *sz* may be a scalar GPU tensor (int64) **or** a Python int (which
        is wrapped into a GPU tensor without sync).

        Pools are tried in order.  On OOM the returned tensor contains ``0``
        and ``stats().num_ooms`` is incremented.
        """
        if isinstance(sz, int):
            sz = torch.tensor(sz, dtype=torch.int64, device=self.device)

        out = torch.empty([], dtype=torch.int64, device=self.device)

        _alloc_kernel[(1,)](
            self.states,
            sz,
            out,
            NUM_POOLS=self.num_pools,
            POOL_STRIDE=_POOL_SIZE,
            N=_MAX_BLOCKS,
            OA=_OFF_ADDRS,
            OL=_OFF_LENS,
            OC=_OFF_ALLOCS,
            OP=_OFF_PREVS,
            ON=_OFF_NEXTS,
        )
        return out

    def free(self, addr: "torch.Tensor | int") -> None:
        """Free a previously allocated address.

        *addr* may be a scalar GPU tensor (int64) or a Python int.
        Double-free / unknown address is a silent no-op.
        """
        if isinstance(addr, int):
            addr = torch.tensor(addr, dtype=torch.int64, device=self.device)

        _free_kernel[(1,)](
            self.states,
            addr,
            NUM_POOLS=self.num_pools,
            POOL_STRIDE=_POOL_SIZE,
            N=_MAX_BLOCKS,
            OA=_OFF_ADDRS,
            OL=_OFF_LENS,
            OC=_OFF_ALLOCS,
            OP=_OFF_PREVS,
            ON=_OFF_NEXTS,
        )

    def free_all(self) -> None:
        """Reset all pools to their initial state (one big free block each).

        Mirrors the C++ ``states.copy_(states_init)`` pattern.
        """
        self.states.copy_(self.states_init)

    def stats(self) -> MemoryStats:
        """Compute per-pool allocation statistics.

        Returns a :class:`MemoryStats` whose fields are ``int64`` tensors
        of shape ``[num_pools]`` on ``self.device``.  All computation
        stays on GPU (no CPU sync); values are only materialised on CPU
        when the caller explicitly reads them.
        """
        pools = self.states.view(self.num_pools, _POOL_SIZE)

        last_bids = pools[:, _META_LAST_BLOCK_ID].to(torch.int64)
        num_ooms = pools[:, _META_NUM_OOMS].to(torch.int64)

        block_ids = torch.arange(_MAX_BLOCKS, device=self.device, dtype=torch.int32)
        valid = block_ids.unsqueeze(0) <= last_bids.unsqueeze(1)

        allocs = pools[:, _OFF_ALLOCS : _OFF_ALLOCS + _MAX_BLOCKS]
        lens = pools[:, _OFF_LENS : _OFF_LENS + _MAX_BLOCKS]

        is_alloc = (allocs == 1) & valid
        is_free = (allocs == 0) & valid

        sum_alloc = (lens * is_alloc.int()).sum(dim=1).to(torch.int64)
        n_free = is_free.sum(dim=1).to(torch.int64)

        # Mask out invalid / allocated blocks before max
        free_lens = torch.where(is_free, lens, torch.zeros_like(lens))
        largest_free = free_lens.max(dim=1).values.to(torch.int64)

        return MemoryStats(
            last_block_id=last_bids,
            num_ooms=num_ooms,
            sum_allocated=sum_alloc,
            num_free_blocks=n_free,
            largest_free_block=largest_free,
        )

    def print_state(self) -> None:
        """Print the block list for every pool (debugging only).

        **This syncs to CPU** — use only for debugging.
        """
        state_cpu = self.states.cpu()
        for pool_idx in range(self.num_pools):
            base = pool_idx * _POOL_SIZE
            last_bid = int(state_cpu[base + _META_LAST_BLOCK_ID])
            n_ooms = int(state_cpu[base + _META_NUM_OOMS])
            print(f"Pool {pool_idx}  [{n_ooms} OOMs, last_block_id={last_bid}]")
            for i in range(last_bid + 1):
                a = int(state_cpu[base + _OFF_ADDRS + i])
                l = int(state_cpu[base + _OFF_LENS + i])
                c = int(state_cpu[base + _OFF_ALLOCS + i])
                p = int(state_cpu[base + _OFF_PREVS + i])
                n = int(state_cpu[base + _OFF_NEXTS + i])
                tag = "USED" if c else "FREE"
                print(f"  [{i}][{tag}][addr={a:6d}] {l:6d}  prev={p} next={n}")
