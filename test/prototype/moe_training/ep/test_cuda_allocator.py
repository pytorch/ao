# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Test requires CUDA", allow_module_level=True)

from torchao.prototype.moe_training.ep.syncless.cuda_allocator import (
    CUDAAllocator,
)


def _make_allocator(pool_sizes: list[int]) -> CUDAAllocator:
    """Create a CUDAAllocator with the given pool sizes."""
    dummy = torch.empty([], device="cuda")
    return CUDAAllocator(dummy, pool_sizes)


# ---------------------------------------------------------------------------
# Single-pool basics
# ---------------------------------------------------------------------------


class TestSinglePool:
    """Tests for a single-pool allocator."""

    def test_alloc_exact_fit(self):
        """Allocating the entire pool should succeed and return addr 0."""
        alloc = _make_allocator([1000])
        addr = alloc.alloc(1000)
        torch.cuda.synchronize()
        assert addr.item() == 0

    def test_alloc_returns_gpu_tensor(self):
        """alloc() returns a scalar int64 CUDA tensor."""
        alloc = _make_allocator([1000])
        addr = alloc.alloc(100)
        assert addr.dtype == torch.int64
        assert addr.is_cuda
        assert addr.dim() == 0

    def test_alloc_int_argument(self):
        """alloc() accepts a plain Python int."""
        alloc = _make_allocator([1000])
        addr = alloc.alloc(100)
        torch.cuda.synchronize()
        assert addr.item() >= 0

    def test_alloc_tensor_argument(self):
        """alloc() accepts a GPU int64 scalar tensor."""
        alloc = _make_allocator([1000])
        sz = torch.tensor(100, dtype=torch.int64, device="cuda")
        addr = alloc.alloc(sz)
        torch.cuda.synchronize()
        assert addr.item() >= 0

    def test_sequential_allocs_non_overlapping(self):
        """Multiple allocations should not overlap."""
        alloc = _make_allocator([1000])
        a1 = alloc.alloc(100)
        a2 = alloc.alloc(200)
        a3 = alloc.alloc(300)
        torch.cuda.synchronize()

        v1, v2, v3 = a1.item(), a2.item(), a3.item()
        # All addresses should be distinct
        assert len({v1, v2, v3}) == 3
        # Ranges should not overlap (addr, addr+size)
        ranges = sorted([(v1, 100), (v2, 200), (v3, 300)])
        for i in range(len(ranges) - 1):
            assert ranges[i][0] + ranges[i][1] <= ranges[i + 1][0]

    def test_alloc_fills_pool(self):
        """Allocating until the pool is full returns correct total."""
        alloc = _make_allocator([500])
        a1 = alloc.alloc(200)
        a2 = alloc.alloc(300)
        torch.cuda.synchronize()

        s = alloc.stats()
        assert s.sum_allocated[0].item() == 500
        assert s.num_free_blocks[0].item() == 0

    def test_oom_returns_zero(self):
        """Allocating more than the pool size should return 0 (OOM)."""
        alloc = _make_allocator([100])
        addr = alloc.alloc(200, assert_on_oom=False)
        torch.cuda.synchronize()
        assert addr.item() == 0

    def test_oom_increments_counter(self):
        """OOM should increment the num_ooms counter."""
        alloc = _make_allocator([100])
        alloc.alloc(200, assert_on_oom=False)
        torch.cuda.synchronize()

        s = alloc.stats()
        assert s.num_ooms[0].item() >= 1

    def test_oom_after_partial_fill(self):
        """OOM after partial allocation: pool has space but not enough."""
        alloc = _make_allocator([500])
        alloc.alloc(400)
        addr = alloc.alloc(200, assert_on_oom=False)
        torch.cuda.synchronize()
        assert addr.item() == 0


# ---------------------------------------------------------------------------
# Best-fit behaviour
# ---------------------------------------------------------------------------


class TestBestFit:
    """Verify best-fit allocation picks the smallest fitting block."""

    def test_best_fit_selects_smallest(self):
        """After creating holes of different sizes, alloc should pick the
        smallest one that fits.
        """
        alloc = _make_allocator([1000])

        # Allocate 4 chunks: [200][100][300][400]
        a1 = alloc.alloc(200)
        a2 = alloc.alloc(100)
        a3 = alloc.alloc(300)
        # Remaining: [400] free

        # Free a1 (200) and a3 (300) to create holes:
        # [FREE:200][USED:100][FREE:300][FREE:400]
        alloc.free(a1)
        alloc.free(a3)
        torch.cuda.synchronize()

        # Allocate 150: should fit in the 200-block (best fit), not 300 or 400
        a4 = alloc.alloc(150)
        torch.cuda.synchronize()

        # The 200-block started at a1's address. After splitting, the
        # allocated 150 should be within that range.
        a1_val = a1.item()
        a4_val = a4.item()
        # Best-fit splits [FREE:50][ALLOC:150] from the 200-block.
        # Allocated block gets the END of the split (addr = a1 + 50).
        assert a4_val == a1_val + 50


# ---------------------------------------------------------------------------
# Free + coalescing
# ---------------------------------------------------------------------------


class TestFreeAndCoalesce:
    """Tests for free() with coalescing of adjacent free blocks."""

    def test_free_simple(self):
        """Free a single allocation and verify pool is fully free again."""
        alloc = _make_allocator([500])
        a = alloc.alloc(200)
        alloc.free(a)
        torch.cuda.synchronize()

        s = alloc.stats()
        assert s.sum_allocated[0].item() == 0
        assert s.largest_free_block[0].item() == 500

    def test_coalesce_with_previous(self):
        """Freeing a block adjacent to a free predecessor should merge."""
        alloc = _make_allocator([1000])
        a1 = alloc.alloc(300)
        a2 = alloc.alloc(200)
        a3 = alloc.alloc(500)

        # Free a1 then a2 → a1 and a2 should coalesce into [FREE:500]
        alloc.free(a1)
        alloc.free(a2)
        torch.cuda.synchronize()

        s = alloc.stats()
        assert s.largest_free_block[0].item() == 500

    def test_coalesce_with_next(self):
        """Freeing a block adjacent to a free successor should merge."""
        alloc = _make_allocator([1000])
        a1 = alloc.alloc(300)
        a2 = alloc.alloc(200)
        a3 = alloc.alloc(500)

        # Free a2 then a1 → a1 and a2 should coalesce into [FREE:500]
        alloc.free(a2)
        alloc.free(a1)
        torch.cuda.synchronize()

        s = alloc.stats()
        assert s.largest_free_block[0].item() == 500

    def test_coalesce_both_neighbours(self):
        """Freeing a block between two free blocks should merge all three."""
        alloc = _make_allocator([1000])
        a1 = alloc.alloc(200)
        a2 = alloc.alloc(300)
        a3 = alloc.alloc(500)

        # Free a1 and a3, then free a2 → should coalesce into [FREE:1000]
        alloc.free(a1)
        alloc.free(a3)
        alloc.free(a2)
        torch.cuda.synchronize()

        s = alloc.stats()
        assert s.sum_allocated[0].item() == 0
        assert s.largest_free_block[0].item() == 1000
        assert s.num_free_blocks[0].item() == 1

    def test_double_free_is_noop(self):
        """Double-free should silently do nothing (no crash)."""
        alloc = _make_allocator([500])
        a = alloc.alloc(200)
        alloc.free(a)
        alloc.free(a)  # no crash
        torch.cuda.synchronize()

        s = alloc.stats()
        assert s.sum_allocated[0].item() == 0

    def test_free_unknown_addr_is_noop(self):
        """Freeing an address that was never allocated should do nothing."""
        alloc = _make_allocator([500])
        alloc.alloc(200)
        alloc.free(99999)  # unknown address, no crash
        torch.cuda.synchronize()

        s = alloc.stats()
        assert s.sum_allocated[0].item() == 200


# ---------------------------------------------------------------------------
# free_all
# ---------------------------------------------------------------------------


class TestFreeAll:
    """Tests for free_all() (reset to initial state)."""

    def test_free_all_resets(self):
        """free_all should restore the pool to a single free block."""
        alloc = _make_allocator([1000])
        alloc.alloc(100)
        alloc.alloc(200)
        alloc.alloc(300)

        alloc.free_all()
        torch.cuda.synchronize()

        s = alloc.stats()
        assert s.sum_allocated[0].item() == 0
        assert s.num_free_blocks[0].item() == 1
        assert s.largest_free_block[0].item() == 1000

    def test_alloc_after_free_all(self):
        """After free_all, allocating the full pool should succeed."""
        alloc = _make_allocator([1000])
        alloc.alloc(1000)

        # Pool is full — OOM expected
        oom = alloc.alloc(1, assert_on_oom=False)
        torch.cuda.synchronize()
        assert oom.item() == 0

        # Reset and try again
        alloc.free_all()
        addr = alloc.alloc(1000)
        torch.cuda.synchronize()
        assert addr.item() == 0


# ---------------------------------------------------------------------------
# Multi-pool
# ---------------------------------------------------------------------------


class TestMultiPool:
    """Tests for multi-pool allocators (GPU + CPU spill)."""

    def test_spill_to_second_pool(self):
        """When pool 0 is full, allocation should spill to pool 1."""
        alloc = _make_allocator([100, 200])
        # Fill pool 0
        a1 = alloc.alloc(100)
        # Next alloc must go to pool 1 (addr starts at 100)
        a2 = alloc.alloc(50)
        torch.cuda.synchronize()

        assert a1.item() == 0  # pool 0 start
        assert a2.item() >= 100  # pool 1 region

    def test_multi_pool_stats(self):
        """Stats should be reported per-pool."""
        alloc = _make_allocator([100, 200])
        alloc.alloc(100)  # fills pool 0
        alloc.alloc(50)  # goes to pool 1

        s = alloc.stats()
        assert s.sum_allocated[0].item() == 100
        assert s.sum_allocated[1].item() == 50
        assert s.largest_free_block[1].item() == 150

    def test_oom_across_all_pools(self):
        """OOM when no pool can satisfy the request."""
        alloc = _make_allocator([100, 200])
        addr = alloc.alloc(500, assert_on_oom=False)
        torch.cuda.synchronize()
        assert addr.item() == 0

    def test_free_in_correct_pool(self):
        """Freeing an addr from pool 1 should only affect pool 1."""
        alloc = _make_allocator([100, 200])
        alloc.alloc(100)  # pool 0
        a2 = alloc.alloc(80)  # pool 1

        alloc.free(a2)
        torch.cuda.synchronize()

        s = alloc.stats()
        assert s.sum_allocated[0].item() == 100  # unchanged
        assert s.sum_allocated[1].item() == 0  # freed


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    """Tests for the stats() method."""

    def test_initial_stats(self):
        """Fresh allocator should have 0 allocated, 1 free block."""
        alloc = _make_allocator([1000])
        s = alloc.stats()
        torch.cuda.synchronize()

        assert s.sum_allocated[0].item() == 0
        assert s.num_free_blocks[0].item() == 1
        assert s.largest_free_block[0].item() == 1000
        assert s.num_ooms[0].item() == 0
        assert s.last_block_id[0].item() == 0

    def test_stats_after_alloc_free(self):
        """Stats should reflect alloc/free operations."""
        alloc = _make_allocator([1000])
        a1 = alloc.alloc(400)
        alloc.alloc(300)

        s = alloc.stats()
        assert s.sum_allocated[0].item() == 700

        alloc.free(a1)
        s = alloc.stats()
        assert s.sum_allocated[0].item() == 300

    def test_stats_returns_gpu_tensors(self):
        """Stats tensors should be on the same device as the allocator."""
        alloc = _make_allocator([100])
        s = alloc.stats()
        assert s.sum_allocated.is_cuda
        assert s.num_free_blocks.is_cuda
        assert s.largest_free_block.is_cuda


# ---------------------------------------------------------------------------
# Stress / alloc-free cycles
# ---------------------------------------------------------------------------


class TestAllocFreeCycles:
    """Test repeated alloc/free patterns to exercise coalescing + splitting."""

    def test_alloc_free_cycle(self):
        """Repeated alloc-free should not leak blocks or corrupt state."""
        alloc = _make_allocator([1000])

        for _ in range(50):
            a = alloc.alloc(100)
            alloc.free(a)

        torch.cuda.synchronize()
        s = alloc.stats()
        assert s.sum_allocated[0].item() == 0
        assert s.largest_free_block[0].item() == 1000

    def test_many_small_allocs_then_free_all(self):
        """Allocate many small blocks, then free_all to reset."""
        alloc = _make_allocator([10000])

        addrs = []
        for _ in range(100):
            addrs.append(alloc.alloc(100))

        torch.cuda.synchronize()
        s = alloc.stats()
        assert s.sum_allocated[0].item() == 10000

        alloc.free_all()
        s = alloc.stats()
        assert s.sum_allocated[0].item() == 0
        assert s.largest_free_block[0].item() == 10000

    def test_interleaved_alloc_free(self):
        """Interleaved alloc/free pattern: alloc 3, free 1, repeat."""
        alloc = _make_allocator([10000])

        addrs = []
        for i in range(30):
            addrs.append(alloc.alloc(100))
            if i % 3 == 2:
                # Free the second of the three
                alloc.free(addrs[-2])
                addrs[-2] = None

        torch.cuda.synchronize()
        s = alloc.stats()
        # 30 allocs of 100, freed 10 → 2000 allocated
        active = sum(1 for a in addrs if a is not None)
        assert s.sum_allocated[0].item() == active * 100


# ---------------------------------------------------------------------------
# Integration: CUDAAllocator + unified GPU+CPU buffer
# ---------------------------------------------------------------------------

try:
    from cuda.bindings import driver as cuda_drv

    _has_cuda_bindings = True
except ImportError:
    _has_cuda_bindings = False

if _has_cuda_bindings:
    from torchao.prototype.moe_training.ep.syncless.unified_buffer_allocator import (
        create_unified_buffer,
        free_unified_buffer,
    )


def _get_granularity() -> int:
    """Return the recommended allocation granularity for the current GPU."""
    device_id = torch.cuda.current_device()
    prop = cuda_drv.CUmemAllocationProp()
    prop.type = cuda_drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    result = cuda_drv.cuMemGetAllocationGranularity(
        prop,
        cuda_drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
    )
    if result[0] != cuda_drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemGetAllocationGranularity failed: {result[0]}")
    return result[1]


@pytest.mark.skipif(not _has_cuda_bindings, reason="requires cuda.bindings")
class TestAllocatorWithUnifiedBuffer:
    """Verify the CUDAAllocator composes with the unified GPU+CPU buffer.

    This mirrors the gb200_moe_sol ``StaticBuffers`` pattern:
    1. ``create_unified_buffer`` allocates physical GPU + CPU memory.
    2. ``CUDAAllocator`` sub-allocates rows within that buffer.
    3. Offsets from the allocator index into the buffer for read/write.
    """

    def test_alloc_offset_indexes_into_buffer(self):
        """Offsets returned by the allocator can be used to index into the
        unified buffer and read/write data correctly.
        """
        g = _get_granularity()
        num_rows = 1024
        row_bytes = 16
        total_bytes = num_rows * row_bytes
        gpu_bytes = (total_bytes // g) * g or g
        # Make sure we have at least gpu_bytes worth of rows
        num_rows = gpu_bytes // row_bytes

        buffer = create_unified_buffer(cpu_bytes=0, gpu_bytes=gpu_bytes)
        buf_2d = buffer.view(num_rows, row_bytes)

        allocator = CUDAAllocator(buffer, [num_rows])

        # Allocate a slice of 10 rows
        offset = allocator.alloc(10)
        torch.cuda.synchronize()
        off_val = offset.item()

        # Write a pattern into the allocated rows
        buf_2d[off_val : off_val + 10].fill_(0xAB)
        torch.cuda.synchronize()
        assert buf_2d[off_val, 0].item() == 0xAB
        assert buf_2d[off_val + 9, 0].item() == 0xAB

        allocator.free(offset)
        free_unified_buffer(buffer.data_ptr())

    def test_gpu_and_cpu_regions_via_allocator(self):
        """Allocator with two pools (GPU + CPU) sub-allocates rows in both
        regions of the unified buffer, and data written at each offset is
        independently correct.
        """
        g = _get_granularity()
        row_bytes = 32
        gpu_rows = g // row_bytes
        cpu_rows = g // row_bytes
        gpu_bytes = gpu_rows * row_bytes
        cpu_bytes = cpu_rows * row_bytes

        buffer = create_unified_buffer(cpu_bytes=cpu_bytes, gpu_bytes=gpu_bytes)
        buf_2d = buffer.view(-1, row_bytes)

        # Pool 0 = GPU rows [0, gpu_rows), Pool 1 = CPU rows [gpu_rows, total)
        allocator = CUDAAllocator(buffer, [gpu_rows, cpu_rows])

        # Fill pool 0 (GPU)
        gpu_offset = allocator.alloc(gpu_rows)
        # Next alloc spills to pool 1 (CPU)
        cpu_offset = allocator.alloc(10)
        torch.cuda.synchronize()

        gpu_off_val = gpu_offset.item()
        cpu_off_val = cpu_offset.item()

        assert gpu_off_val < gpu_rows, "GPU alloc should be in pool 0"
        assert cpu_off_val >= gpu_rows, "CPU alloc should spill to pool 1"

        # Write distinct patterns
        buf_2d[gpu_off_val].fill_(0x11)
        buf_2d[cpu_off_val].fill_(0x22)
        torch.cuda.synchronize()

        assert buf_2d[gpu_off_val, 0].item() == 0x11
        assert buf_2d[cpu_off_val, 0].item() == 0x22

        allocator.free(gpu_offset)
        allocator.free(cpu_offset)
        free_unified_buffer(buffer.data_ptr())

    def test_alloc_free_realloc_preserves_data(self):
        """Allocate rows, write data, free, re-allocate, and verify the new
        allocation can be written independently (no stale data leaks).
        """
        g = _get_granularity()
        row_bytes = 16
        total_rows = g // row_bytes

        buffer = create_unified_buffer(cpu_bytes=0, gpu_bytes=g)
        buf_2d = buffer.view(total_rows, row_bytes)

        allocator = CUDAAllocator(buffer, [total_rows])

        # First allocation: write pattern A
        off1 = allocator.alloc(10)
        torch.cuda.synchronize()
        buf_2d[off1.item() : off1.item() + 10].fill_(0xAA)
        torch.cuda.synchronize()

        # Free and reallocate
        allocator.free(off1)
        off2 = allocator.alloc(10)
        torch.cuda.synchronize()

        # Write pattern B into the new allocation
        buf_2d[off2.item() : off2.item() + 10].fill_(0xBB)
        torch.cuda.synchronize()
        assert buf_2d[off2.item(), 0].item() == 0xBB

        allocator.free(off2)
        free_unified_buffer(buffer.data_ptr())

    def test_multiple_microbatch_simulation(self):
        """Simulate the forward/backward activation saving pattern:
        allocate rows for each microbatch in forward, free in backward.
        """
        g = _get_granularity()
        row_bytes = 8
        total_rows = g // row_bytes

        buffer = create_unified_buffer(cpu_bytes=0, gpu_bytes=g)
        buf_2d = buffer.view(total_rows, row_bytes)

        allocator = CUDAAllocator(buffer, [total_rows])

        # Reserve a sentinel allocation (matches gb200_moe_sol pattern).
        # Note: best-fit places the allocation at the END of the free block,
        # so sentinel_offset = total_rows - sentinel_size.
        sentinel_size = 128
        sentinel = allocator.alloc(sentinel_size)
        torch.cuda.synchronize()

        # Forward: allocate rows for 4 microbatches
        mb_size = 50
        offsets = []
        for i in range(4):
            off = allocator.alloc(mb_size)
            torch.cuda.synchronize()
            off_val = off.item()
            # Write microbatch ID as a tag
            buf_2d[off_val : off_val + mb_size].fill_(i + 1)
            offsets.append(off)

        torch.cuda.synchronize()

        # Verify each microbatch's data is intact
        for i, off in enumerate(offsets):
            assert buf_2d[off.item(), 0].item() == i + 1

        # Backward: free in reverse order (LIFO, like the real training loop)
        for off in reversed(offsets):
            allocator.free(off)

        torch.cuda.synchronize()
        s = allocator.stats()
        # Only the sentinel should remain allocated
        assert s.sum_allocated[0].item() == sentinel_size

        allocator.free(sentinel)
        free_unified_buffer(buffer.data_ptr())

    def test_free_all_resets_composed_allocator(self):
        """free_all on the allocator resets sub-allocation state without
        affecting the underlying unified buffer.
        """
        g = _get_granularity()
        row_bytes = 16
        total_rows = g // row_bytes

        buffer = create_unified_buffer(cpu_bytes=0, gpu_bytes=g)
        buf_2d = buffer.view(total_rows, row_bytes)

        allocator = CUDAAllocator(buffer, [total_rows])
        allocator.alloc(total_rows // 2)
        allocator.alloc(total_rows // 2)

        # Pool is full
        oom = allocator.alloc(1, assert_on_oom=False)
        torch.cuda.synchronize()
        assert oom.item() == 0

        # Reset allocator — buffer is still valid
        allocator.free_all()
        off = allocator.alloc(total_rows)
        torch.cuda.synchronize()
        assert off.item() == 0

        # Buffer is still writable
        buf_2d.fill_(42)
        torch.cuda.synchronize()
        assert buf_2d[0, 0].item() == 42

        allocator.free(off)
        free_unified_buffer(buffer.data_ptr())
