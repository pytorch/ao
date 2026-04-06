# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Test requires CUDA", allow_module_level=True)

try:
    from cuda.bindings import driver as cuda_drv
except ImportError:
    pytest.skip("Test requires cuda.bindings (cuda-python)", allow_module_level=True)

from torchao.prototype.moe_training.ep.syncless.unified_buffer_allocator import (
    _check,
    _g_allocated_cudacpu_buffers,
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
    granularity = _check(
        cuda_drv.cuMemGetAllocationGranularity(
            prop,
            cuda_drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ),
        "cuMemGetAllocationGranularity",
    )
    return granularity


def _align(nbytes: int, granularity: int) -> int:
    """Round *nbytes* up to a multiple of *granularity*."""
    return ((nbytes + granularity - 1) // granularity) * granularity


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateUnifiedBuffer:
    """Tests for create_unified_buffer / free_unified_buffer."""

    def test_gpu_only(self):
        """Allocate with gpu_bytes > 0 and cpu_bytes == 0."""
        g = _get_granularity()
        gpu_bytes = g  # 1 page
        tensor = create_unified_buffer(cpu_bytes=0, gpu_bytes=gpu_bytes)

        assert tensor.dtype == torch.uint8
        assert tensor.is_cuda
        assert tensor.numel() == gpu_bytes
        assert tensor.data_ptr() in _g_allocated_cudacpu_buffers

        free_unified_buffer(tensor.data_ptr())
        assert tensor.data_ptr() not in _g_allocated_cudacpu_buffers

    def test_cpu_only(self):
        """Allocate with cpu_bytes > 0 and gpu_bytes == 0."""
        g = _get_granularity()
        cpu_bytes = g
        tensor = create_unified_buffer(cpu_bytes=cpu_bytes, gpu_bytes=0)

        assert tensor.dtype == torch.uint8
        assert tensor.is_cuda
        assert tensor.numel() == cpu_bytes
        assert tensor.data_ptr() in _g_allocated_cudacpu_buffers

        free_unified_buffer(tensor.data_ptr())

    def test_gpu_and_cpu(self):
        """Allocate with both gpu_bytes and cpu_bytes > 0."""
        g = _get_granularity()
        gpu_bytes = 2 * g
        cpu_bytes = 3 * g
        tensor = create_unified_buffer(cpu_bytes=cpu_bytes, gpu_bytes=gpu_bytes)

        assert tensor.numel() == gpu_bytes + cpu_bytes
        assert tensor.data_ptr() in _g_allocated_cudacpu_buffers

        alloc = _g_allocated_cudacpu_buffers[tensor.data_ptr()]
        assert alloc.gpu_bytes == gpu_bytes
        assert alloc.cpu_bytes == cpu_bytes
        assert alloc.gpu_handle is not None
        assert alloc.cpu_handle is not None

        free_unified_buffer(tensor.data_ptr())

    def test_read_write(self):
        """Verify that the buffer is read-write accessible from GPU kernels."""
        g = _get_granularity()
        gpu_bytes = g
        cpu_bytes = g
        tensor = create_unified_buffer(cpu_bytes=cpu_bytes, gpu_bytes=gpu_bytes)

        total = gpu_bytes + cpu_bytes
        # Write a known pattern
        tensor.fill_(42)
        torch.cuda.synchronize()
        assert tensor[0].item() == 42
        assert tensor[total - 1].item() == 42

        # Write via a CUDA kernel (element-wise add)
        tensor += 1
        torch.cuda.synchronize()
        assert tensor[0].item() == 43

        free_unified_buffer(tensor.data_ptr())

    def test_unaligned_gpu_bytes_raises(self):
        """gpu_bytes not aligned to granularity should raise ValueError."""
        g = _get_granularity()
        with pytest.raises(ValueError, match="gpu_bytes"):
            create_unified_buffer(cpu_bytes=0, gpu_bytes=g + 1)

    def test_unaligned_cpu_bytes_raises(self):
        """cpu_bytes not aligned to granularity should raise ValueError."""
        g = _get_granularity()
        with pytest.raises(ValueError, match="cpu_bytes"):
            create_unified_buffer(cpu_bytes=g + 1, gpu_bytes=0)

    def test_double_free_warns(self, capsys):
        """Calling free_unified_buffer twice should print a warning, not crash."""
        g = _get_granularity()
        tensor = create_unified_buffer(cpu_bytes=0, gpu_bytes=g)
        ptr = tensor.data_ptr()
        free_unified_buffer(ptr)

        # Second free should warn, not raise
        free_unified_buffer(ptr)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "double free" in captured.out.lower()

    def test_multiple_allocations(self):
        """Multiple independent buffers can coexist."""
        g = _get_granularity()
        t1 = create_unified_buffer(cpu_bytes=0, gpu_bytes=g)
        t2 = create_unified_buffer(cpu_bytes=g, gpu_bytes=g)
        t3 = create_unified_buffer(cpu_bytes=g, gpu_bytes=0)

        assert t1.data_ptr() != t2.data_ptr()
        assert t2.data_ptr() != t3.data_ptr()
        assert len(_g_allocated_cudacpu_buffers) >= 3

        # Write different values
        t1.fill_(1)
        t2.fill_(2)
        t3.fill_(3)
        torch.cuda.synchronize()
        assert t1[0].item() == 1
        assert t2[0].item() == 2
        assert t3[0].item() == 3

        free_unified_buffer(t1.data_ptr())
        free_unified_buffer(t2.data_ptr())
        free_unified_buffer(t3.data_ptr())

    def test_tensor_view_and_reshape(self):
        """The returned tensor can be viewed / reshaped as expected."""
        g = _get_granularity()
        total = 2 * g
        tensor = create_unified_buffer(cpu_bytes=g, gpu_bytes=g)

        # Reinterpret as float32 (4 bytes per element)
        float_view = tensor.view(torch.float32)
        assert float_view.numel() == total // 4

        # Reshape into 2D
        reshaped = tensor.view(2, total // 2)
        assert reshaped.shape == (2, total // 2)

        free_unified_buffer(tensor.data_ptr())

    def test_cuda_kernel_on_both_regions(self):
        """Run a CUDA kernel that touches both the GPU and CPU regions."""
        g = _get_granularity()
        gpu_bytes = g
        cpu_bytes = g
        tensor = create_unified_buffer(cpu_bytes=cpu_bytes, gpu_bytes=gpu_bytes)

        # GPU region: first gpu_bytes
        gpu_region = tensor[:gpu_bytes]
        # CPU region: remaining cpu_bytes
        cpu_region = tensor[gpu_bytes:]

        gpu_region.fill_(10)
        cpu_region.fill_(20)
        torch.cuda.synchronize()

        assert gpu_region[0].item() == 10
        assert cpu_region[0].item() == 20

        # Cross-region operation via the full tensor
        tensor += 5
        torch.cuda.synchronize()
        assert tensor[0].item() == 15
        assert tensor[gpu_bytes].item() == 25

        free_unified_buffer(tensor.data_ptr())

    def test_combined_gpu_and_cpu_allocation(self):
        """Allocate a buffer that exceeds GPU memory, proving both GPU and
        CPU-pinned memory are used simultaneously.

        Uses ~50% of GPU memory for the device portion and an equal-sized
        CPU portion, so the total exceeds what would fit on the GPU alone.
        Both regions are independently written and read back to confirm
        they are backed by distinct physical memory.
        """
        g = _get_granularity()
        gpu_mem = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory

        # GPU portion: ~50% of device memory (conservative to avoid OOM
        # when other processes share the GPU).
        gpu_bytes = _align(int(gpu_mem * 0.5), g)
        # CPU portion: ~60% of device memory, so total strictly exceeds GPU.
        cpu_bytes = _align(int(gpu_mem * 0.6), g)
        total_bytes = gpu_bytes + cpu_bytes

        assert total_bytes > gpu_mem, (
            f"Total ({total_bytes / (1024**3):.1f} GB) must exceed "
            f"GPU memory ({gpu_mem / (1024**3):.1f} GB)"
        )

        tensor = create_unified_buffer(cpu_bytes=cpu_bytes, gpu_bytes=gpu_bytes)

        assert tensor.numel() == total_bytes
        assert tensor.is_cuda

        # Both handles must exist.
        alloc = _g_allocated_cudacpu_buffers[tensor.data_ptr()]
        assert alloc.gpu_handle is not None, "GPU handle should be allocated"
        assert alloc.cpu_handle is not None, "CPU handle should be allocated"
        assert alloc.gpu_bytes == gpu_bytes
        assert alloc.cpu_bytes == cpu_bytes

        # Write distinct patterns to each region.
        gpu_region = tensor[:gpu_bytes]
        cpu_region = tensor[gpu_bytes:]

        gpu_region.fill_(0xAA)
        cpu_region.fill_(0xBB)
        torch.cuda.synchronize()

        # Verify GPU-backed region.
        assert gpu_region[0].item() == 0xAA
        assert gpu_region[-1].item() == 0xAA

        # Verify CPU-backed region.
        assert cpu_region[0].item() == 0xBB
        assert cpu_region[-1].item() == 0xBB

        # Cross-region operation: add 1 across the full tensor.
        tensor += 1
        torch.cuda.synchronize()
        assert tensor[0].item() == 0xAB  # was 0xAA
        assert tensor[gpu_bytes].item() == 0xBC  # was 0xBB

        free_unified_buffer(tensor.data_ptr())
