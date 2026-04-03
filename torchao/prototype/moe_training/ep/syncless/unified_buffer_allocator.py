from dataclasses import dataclass
from typing import Optional

import torch
from cuda.bindings import driver as cuda_drv

# ---------------------------------------------------------------------------
# Unified GPU+CPU buffer via cuMemMap (virtual memory management)
# ---------------------------------------------------------------------------

# Global registry: maps device_ptr (int) -> _UnifiedBufferAllocation
_g_allocated_cudacpu_buffers: dict[int, "_UnifiedBufferAllocation"] = {}


@dataclass
class _UnifiedBufferAllocation:
    gpu_handle: object  # CUmemGenericAllocationHandle or None
    cpu_handle: object  # CUmemGenericAllocationHandle or None
    gpu_bytes: int
    cpu_bytes: int
    device_ptr: object  # CUdeviceptr
    # prevent the CUDABuffer (and thus the tensor) from being GC'd while
    # this allocation is alive
    _prevent_gc: object = None


def _check(result: tuple, msg: str = "") -> object:
    """Unwrap a cuda.bindings driver call, raising on error."""
    err = result[0]
    if err != cuda_drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {err}" + (f": {msg}" if msg else ""))
    if len(result) > 1:
        return result[1]
    return None


def free_unified_buffer(device_ptr_int: int) -> None:
    """Free a unified GPU+CPU buffer previously created by create_unified_buffer."""
    alloc = _g_allocated_cudacpu_buffers.get(device_ptr_int)
    if alloc is None:
        print(
            f"WARNING: double free for unified buffer at address 0x{device_ptr_int:x}"
        )
        return

    torch.cuda.synchronize()

    ptr_val = int(alloc.device_ptr)
    if alloc.gpu_bytes > 0:
        _check(
            cuda_drv.cuMemUnmap(ptr_val, alloc.gpu_bytes),
            "cuMemUnmap GPU",
        )
    if alloc.cpu_bytes > 0:
        _check(
            cuda_drv.cuMemUnmap(ptr_val + alloc.gpu_bytes, alloc.cpu_bytes),
            "cuMemUnmap CPU",
        )

    _check(
        cuda_drv.cuMemAddressFree(ptr_val, alloc.gpu_bytes + alloc.cpu_bytes),
        "cuMemAddressFree",
    )

    if alloc.gpu_bytes > 0 and alloc.gpu_handle is not None:
        _check(cuda_drv.cuMemRelease(alloc.gpu_handle), "cuMemRelease GPU")
    if alloc.cpu_bytes > 0 and alloc.cpu_handle is not None:
        _check(cuda_drv.cuMemRelease(alloc.cpu_handle), "cuMemRelease CPU")

    del _g_allocated_cudacpu_buffers[device_ptr_int]


class _CUDAArrayInterfaceWrapper:
    """Exposes a raw CUdeviceptr via ``__cuda_array_interface__`` so that
    ``torch.as_tensor`` can wrap it zero-copy.

    The CUDA Array Interface is a standard protocol for zero-copy interop
    between CUDA-aware Python libraries (CuPy, Numba, PyTorch, etc.).
    See: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    """

    def __init__(self, ptr_int: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",  # uint8
            "data": (ptr_int, False),  # (ptr, read_only)
            "version": 2,
        }


def create_unified_buffer(cpu_bytes: int, gpu_bytes: int) -> torch.Tensor:
    """Allocate a unified GPU+CPU buffer using CUDA virtual memory management.

    Creates a contiguous virtual address range where the first ``gpu_bytes``
    are backed by GPU device memory and the following ``cpu_bytes`` are backed
    by pinned host (NUMA-local) memory.  Both regions are accessible from the
    GPU.  The returned tensor is a ``torch.uint8`` CUDA tensor covering the
    full ``gpu_bytes + cpu_bytes`` range.

    Call :func:`free_unified_buffer` (passing ``tensor.data_ptr()``) to release
    the underlying resources.

    Args:
        cpu_bytes: Size of the CPU-pinned portion (must be aligned to the
            recommended allocation granularity).
        gpu_bytes: Size of the GPU-device portion (must be aligned to the
            recommended allocation granularity).

    Returns:
        A ``torch.uint8`` CUDA tensor of size ``gpu_bytes + cpu_bytes``.
    """
    device_id = torch.cuda.current_device()

    # -- GPU allocation properties --
    gpu_prop = cuda_drv.CUmemAllocationProp()
    gpu_prop.type = cuda_drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    gpu_prop.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    gpu_prop.location.id = device_id

    # -- CPU allocation properties --
    cpu_prop = cuda_drv.CUmemAllocationProp()
    cpu_prop.type = cuda_drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    cpu_prop.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA

    # Retain primary context and set it current so NUMA queries work.
    ctx = _check(
        cuda_drv.cuDevicePrimaryCtxRetain(device_id), "cuDevicePrimaryCtxRetain"
    )
    _check(cuda_drv.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    # Set the CPU allocation to the NUMA node closest to this GPU.
    numa_id = _check(
        cuda_drv.cuDeviceGetAttribute(
            cuda_drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
            device_id,
        ),
        "cuDeviceGetAttribute(HOST_NUMA_ID)",
    )
    cpu_prop.location.id = numa_id

    # -- Granularity checks --
    cpu_page_size = _check(
        cuda_drv.cuMemGetAllocationGranularity(
            cpu_prop,
            cuda_drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ),
        "cuMemGetAllocationGranularity CPU",
    )
    gpu_page_size = _check(
        cuda_drv.cuMemGetAllocationGranularity(
            gpu_prop,
            cuda_drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ),
        "cuMemGetAllocationGranularity GPU",
    )

    if cpu_bytes % cpu_page_size != 0:
        raise ValueError(
            f"cpu_bytes ({cpu_bytes}) must be a multiple of CPU recommended "
            f"granularity ({cpu_page_size})"
        )
    if gpu_bytes % gpu_page_size != 0:
        raise ValueError(
            f"gpu_bytes ({gpu_bytes}) must be a multiple of GPU recommended "
            f"granularity ({gpu_page_size})"
        )

    # -- Allocate physical memory handles --
    gpu_handle: Optional[object] = None
    cpu_handle: Optional[object] = None

    if gpu_bytes > 0:
        result = cuda_drv.cuMemCreate(gpu_bytes, gpu_prop, 0)
        if result[0] != cuda_drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                f"Failed to allocate GPU VMM buffer: {gpu_bytes} bytes "
                f"({gpu_bytes / (1024**3):.3f} GB). CUDA error: {result[0]}"
            )
        gpu_handle = result[1]

    if cpu_bytes > 0:
        result = cuda_drv.cuMemCreate(cpu_bytes, cpu_prop, 0)
        if result[0] != cuda_drv.CUresult.CUDA_SUCCESS:
            if gpu_handle is not None:
                cuda_drv.cuMemRelease(gpu_handle)
            raise RuntimeError(
                f"Failed to allocate CPU VMM buffer: {cpu_bytes} bytes "
                f"({cpu_bytes / (1024**3):.3f} GB). CUDA error: {result[0]}"
            )
        cpu_handle = result[1]

    # -- Reserve virtual address range --
    assert gpu_page_size == cpu_page_size, (
        f"GPU and CPU page sizes differ: {gpu_page_size} vs {cpu_page_size}"
    )
    total_bytes = gpu_bytes + cpu_bytes
    device_ptr = _check(
        cuda_drv.cuMemAddressReserve(total_bytes, gpu_page_size, 0, 0),
        "cuMemAddressReserve",
    )

    # -- Map physical memory onto the virtual range --
    access_desc = cuda_drv.CUmemAccessDesc()
    access_desc.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access_desc.location.id = device_id
    access_desc.flags = cuda_drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    ptr_int = int(device_ptr)

    if gpu_bytes > 0:
        _check(
            cuda_drv.cuMemMap(ptr_int, gpu_bytes, 0, gpu_handle, 0),
            "cuMemMap GPU",
        )
        _check(
            cuda_drv.cuMemSetAccess(ptr_int, gpu_bytes, [access_desc], 1),
            "cuMemSetAccess GPU",
        )

    if cpu_bytes > 0:
        _check(
            cuda_drv.cuMemMap(ptr_int + gpu_bytes, cpu_bytes, 0, cpu_handle, 0),
            "cuMemMap CPU",
        )
        _check(
            cuda_drv.cuMemSetAccess(ptr_int + gpu_bytes, cpu_bytes, [access_desc], 1),
            "cuMemSetAccess CPU",
        )

    # -- Wrap as a PyTorch tensor --
    # We use the CUDA Array Interface protocol (__cuda_array_interface__) to
    # create a PyTorch tensor that points directly at our externally-managed
    # device pointer without copying.  PyTorch keeps the wrapper object alive
    # through its storage, so the memory won't be freed prematurely.
    # Protocol spec: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    wrapper = _CUDAArrayInterfaceWrapper(ptr_int, total_bytes)
    tensor = torch.as_tensor(wrapper, device=f"cuda:{device_id}")

    # Register allocation for later cleanup.
    alloc = _UnifiedBufferAllocation(
        gpu_handle=gpu_handle,
        cpu_handle=cpu_handle,
        gpu_bytes=gpu_bytes,
        cpu_bytes=cpu_bytes,
        device_ptr=device_ptr,
        _prevent_gc=wrapper,
    )
    _g_allocated_cudacpu_buffers[ptr_int] = alloc

    return tensor
