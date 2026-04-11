import logging
import weakref
from dataclasses import dataclass
from typing import Optional

import torch
from cuda.bindings import driver as cuda_drv

# ---------------------------------------------------------------------------
# Unified GPU+CPU buffer via cuMemMap (virtual memory management)
# ---------------------------------------------------------------------------

# Global registry: maps device_ptr (int) -> _UnifiedBufferAllocation
_g_allocated_cudacpu_buffers: dict[int, "_UnifiedBufferAllocation"] = {}

# prevent weak-ref destructor pointers from being GC'd
_g_prevent_destructor_gc: list[weakref.ref] = []


@dataclass
class _UnifiedBufferAllocation:
    gpu_handle: object  # CUmemGenericAllocationHandle or None
    cpu_handle: object  # CUmemGenericAllocationHandle or None
    gpu_bytes: int
    cpu_bytes: int
    device_ptr: object  # CUdeviceptr


def _check(result: tuple, msg: str = "") -> object:
    """Unwrap a cuda.bindings driver call, raising on error."""
    err = result[0]
    if err != cuda_drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {err}" + (f": {msg}" if msg else ""))
    if len(result) > 1:
        return result[1]
    return None


def _set_optimal_cpu_affinity(device_id: int) -> None:
    """Set calling thread's CPU affinity to be NUMA-optimal for the given GPU.

    On multi-socket NUMA systems (e.g. GB200 nodes), physical RAM is split
    across NUMA nodes, each attached to a specific CPU socket.  Each GPU is
    physically closest to one NUMA node.  When cuMemCreate is called with
    CU_MEM_LOCATION_TYPE_HOST_NUMA, the driver pins pages from the NUMA node
    in cpu_prop.location.id — but the Linux kernel's first-touch page policy
    may still allocate pages on whichever NUMA node the *calling thread* is
    running on.  nvmlDeviceSetCpuAffinity pins the calling thread to the CPU
    cores on the NUMA node closest to the target GPU, ensuring page faults
    during pinning land on the correct node.

    Getting this wrong on a multi-tray system means the CPU-overflow memory
    could be pinned on a remote socket, cutting bandwidth by 3-4x (~200 GB/s
    local vs ~50-80 GB/s cross-socket).

    Mirrors the C++ setOptimalCpuAffinity helper.  Uses pynvml if available;
    silently skips if NVML is not installed or the call is unsupported.
    """
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        try:
            pynvml.nvmlDeviceSetCpuAffinity(handle)
        except pynvml.NVMLError_NotSupported:
            logging.warning(
                "nvmlDeviceSetCpuAffinity not supported for device %d",
                device_id,
            )
    except ImportError:
        logging.warning(
            "pynvml not installed; skipping NUMA-optimal CPU affinity for device %d",
            device_id,
        )
    except Exception as e:
        logging.warning(
            "Failed to set optimal CPU affinity for device %d: %s", device_id, e
        )


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


def _make_alloc_prop(
    is_cpu: bool, location_id: int = 0
) -> "cuda_drv.CUmemAllocationProp":
    """Build a fully zero-initialized CUmemAllocationProp.

    Mirrors the C++ ``getAllocationProp`` helper — critically sets
    ``requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE`` so the driver does
    not try to create exportable handles (which can block for large allocs).
    """
    prop = cuda_drv.CUmemAllocationProp()
    prop.type = cuda_drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    if is_cpu:
        prop.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    else:
        prop.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = location_id
    prop.requestedHandleTypes = 0  # CU_MEM_HANDLE_TYPE_NONE
    return prop


def create_unified_buffer(cpu_bytes: int, gpu_bytes: int) -> torch.Tensor:
    """Allocate a unified GPU+CPU buffer using CUDA virtual memory management.

    Creates a contiguous virtual address range where the first ``gpu_bytes``
    are backed by GPU device memory and the following ``cpu_bytes`` are backed
    by pinned host (NUMA-local) memory.  Both regions are accessible from the
    GPU.  The returned tensor is a ``torch.uint8`` CUDA tensor covering the
    full ``gpu_bytes + cpu_bytes`` range.

    The returned tensor **automatically** frees the underlying VMM resources
    when it is garbage-collected (mirroring the C++ ``from_blob`` + custom
    deleter pattern).  You can also call :func:`free_unified_buffer` manually
    with ``tensor.data_ptr()`` for deterministic cleanup.

    Args:
        cpu_bytes: Size of the CPU-pinned portion (must be aligned to the
            recommended allocation granularity).
        gpu_bytes: Size of the GPU-device portion (must be aligned to the
            recommended allocation granularity).

    Returns:
        A ``torch.uint8`` CUDA tensor of size ``gpu_bytes + cpu_bytes``.
    """
    device_id = torch.cuda.current_device()

    # Set CPU affinity for NUMA-optimal host allocation (matches C++ impl).
    _set_optimal_cpu_affinity(device_id)

    # -- Allocation properties (fully initialized, matching C++ getAllocationProp) --
    gpu_prop = _make_alloc_prop(is_cpu=False, location_id=device_id)
    cpu_prop = _make_alloc_prop(is_cpu=True, location_id=0)

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

    # -- Wrap as a PyTorch tensor via internal APIs --
    # _construct_storage_from_data_pointer creates a non-owning Storage
    # (uses deleteNothing as deleter, so GC won't free the VMM memory).
    # _construct_CUDA_Tensor_From_Storage_And_Metadata wraps it as a tensor.
    # This replaces the __cuda_array_interface__ + torch.as_tensor workaround.
    storage = torch._C._construct_storage_from_data_pointer(
        ptr_int, torch.device(f"cuda:{device_id}"), total_bytes
    )
    tensor = torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
        {
            "dtype": torch.uint8,
            "size": (total_bytes,),
            "stride": (1,),
            "storage_offset": 0,
        },
        storage,
    )

    # Register allocation for later cleanup.
    alloc = _UnifiedBufferAllocation(
        gpu_handle=gpu_handle,
        cpu_handle=cpu_handle,
        gpu_bytes=gpu_bytes,
        cpu_bytes=cpu_bytes,
        device_ptr=device_ptr,
    )
    _g_allocated_cudacpu_buffers[ptr_int] = alloc

    # -- Automatic deleter via weak reference --
    # The storage created by _construct_storage_from_data_pointer is non-owning
    # (uses deleteNothing), so GC of the tensor/storage will NOT free the VMM
    # memory.  We use a weakref callback to trigger free_unified_buffer when the
    # tensor is garbage-collected, approximating the C++ from_blob + custom
    # deleter pattern.  The weakref object itself is kept alive in
    # _g_prevent_destructor_gc so it isn't collected before the tensor.
    #
    # If free_unified_buffer() was already called manually (deterministic
    # cleanup), the destructor is a no-op because the ptr will no longer be
    # in _g_allocated_cudacpu_buffers.
    captured_ptr = ptr_int

    def _destructor(ref: weakref.ref) -> None:
        if captured_ptr in _g_allocated_cudacpu_buffers:
            free_unified_buffer(captured_ptr)
        try:
            _g_prevent_destructor_gc.remove(ref)
        except ValueError:
            pass

    weak = weakref.ref(tensor, _destructor)
    _g_prevent_destructor_gc.append(weak)

    return tensor


class DeferredUnifiedBuffer:
    """Unified GPU+CPU buffer with deferred physical memory allocation.

    Separates virtual address reservation (cheap, no physical RAM) from
    physical memory allocation (expensive, real GPU HBM or CPU-pinned RAM).
    The full VA range is reserved at construction time, but physical pages
    are only allocated via ``cuMemCreate`` + ``cuMemMap`` when
    :meth:`ensure_mapped` is called.

    This avoids pinning tens of GB of host RAM at construction when the
    CPU overflow pool may never be needed (e.g. balanced routing).

    Args:
        gpu_capacity: Maximum bytes of GPU device memory.
        cpu_capacity: Maximum bytes of CPU-pinned memory (overflow pool).
    """

    def __init__(self, gpu_capacity: int, cpu_capacity: int):
        device_id = torch.cuda.current_device()
        _set_optimal_cpu_affinity(device_id)

        self._gpu_prop = _make_alloc_prop(is_cpu=False, location_id=device_id)
        self._cpu_prop = _make_alloc_prop(is_cpu=True, location_id=0)

        ctx = _check(
            cuda_drv.cuDevicePrimaryCtxRetain(device_id),
            "cuDevicePrimaryCtxRetain",
        )
        _check(cuda_drv.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

        numa_id = _check(
            cuda_drv.cuDeviceGetAttribute(
                cuda_drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                device_id,
            ),
            "cuDeviceGetAttribute(HOST_NUMA_ID)",
        )
        self._cpu_prop.location.id = numa_id

        self._page_size: int = _check(
            cuda_drv.cuMemGetAllocationGranularity(
                self._gpu_prop,
                cuda_drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            ),
            "cuMemGetAllocationGranularity",
        )

        self._gpu_capacity = self._align_up(gpu_capacity)
        self._cpu_capacity = self._align_up(cpu_capacity)
        total = self._gpu_capacity + self._cpu_capacity

        # Step 1: Reserve VA space (cheap — no physical memory).
        self._device_ptr = _check(
            cuda_drv.cuMemAddressReserve(total, self._page_size, 0, 0),
            "cuMemAddressReserve",
        )
        self._base_ptr: int = int(self._device_ptr)
        self._total_bytes: int = total
        self._device_id: int = device_id

        self._access_desc = cuda_drv.CUmemAccessDesc()
        self._access_desc.location.type = (
            cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        )
        self._access_desc.location.id = device_id
        self._access_desc.flags = (
            cuda_drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        )

        self._gpu_mapped: int = 0
        self._cpu_mapped: int = 0
        self._gpu_handles: list[tuple[object, int, int]] = []
        self._cpu_handles: list[tuple[object, int, int]] = []

        storage = torch._C._construct_storage_from_data_pointer(
            self._base_ptr, torch.device(f"cuda:{device_id}"), total
        )
        self._tensor = torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
            {
                "dtype": torch.uint8,
                "size": (total,),
                "stride": (1,),
                "storage_offset": 0,
            },
            storage,
        )

    # -- public API ----------------------------------------------------------

    @property
    def tensor(self) -> torch.Tensor:
        """``torch.uint8`` CUDA tensor spanning the full VA range."""
        return self._tensor

    @property
    def gpu_mapped(self) -> int:
        """Bytes of GPU physical memory currently mapped."""
        return self._gpu_mapped

    @property
    def cpu_mapped(self) -> int:
        """Bytes of CPU physical memory currently mapped."""
        return self._cpu_mapped

    @property
    def gpu_capacity(self) -> int:
        return self._gpu_capacity

    @property
    def cpu_capacity(self) -> int:
        return self._cpu_capacity

    @property
    def page_size(self) -> int:
        return self._page_size

    def ensure_mapped(self, total_bytes: int) -> None:
        """Ensure at least *total_bytes* of the buffer are physically backed.

        GPU pages are mapped first.  If *total_bytes* exceeds
        ``gpu_capacity``, CPU overflow pages are mapped for the remainder.
        Pages are allocated in ``page_size``-aligned chunks.
        """
        if total_bytes <= self._gpu_mapped + self._cpu_mapped:
            return

        # -- GPU region (bytes 0 .. gpu_capacity-1) --------------------------
        if self._gpu_mapped < self._gpu_capacity:
            target_gpu = self._align_up(min(total_bytes, self._gpu_capacity))
            new_gpu = target_gpu - self._gpu_mapped
            if new_gpu > 0:
                handle = _check(
                    cuda_drv.cuMemCreate(new_gpu, self._gpu_prop, 0),
                    f"cuMemCreate GPU ({new_gpu} bytes)",
                )
                _check(
                    cuda_drv.cuMemMap(
                        self._base_ptr + self._gpu_mapped,
                        new_gpu,
                        0,
                        handle,
                        0,
                    ),
                    "cuMemMap GPU",
                )
                _check(
                    cuda_drv.cuMemSetAccess(
                        self._base_ptr + self._gpu_mapped,
                        new_gpu,
                        [self._access_desc],
                        1,
                    ),
                    "cuMemSetAccess GPU",
                )
                self._gpu_handles.append((handle, self._gpu_mapped, new_gpu))
                self._gpu_mapped += new_gpu

        # -- CPU overflow region (bytes gpu_capacity .. total-1) --------------
        if total_bytes > self._gpu_capacity:
            cpu_needed = self._align_up(total_bytes - self._gpu_capacity)
            cpu_new = cpu_needed - self._cpu_mapped
            if cpu_new > 0:
                handle = _check(
                    cuda_drv.cuMemCreate(cpu_new, self._cpu_prop, 0),
                    f"cuMemCreate CPU ({cpu_new} bytes)",
                )
                _check(
                    cuda_drv.cuMemMap(
                        self._base_ptr + self._gpu_capacity + self._cpu_mapped,
                        cpu_new,
                        0,
                        handle,
                        0,
                    ),
                    "cuMemMap CPU",
                )
                _check(
                    cuda_drv.cuMemSetAccess(
                        self._base_ptr + self._gpu_capacity + self._cpu_mapped,
                        cpu_new,
                        [self._access_desc],
                        1,
                    ),
                    "cuMemSetAccess CPU",
                )
                self._cpu_handles.append((handle, self._cpu_mapped, cpu_new))
                self._cpu_mapped += cpu_new

    def release_physical(self) -> None:
        """Unmap and release ALL physical memory.

        The virtual address reservation is kept so the buffer can be
        re-mapped via :meth:`ensure_mapped`.
        """
        if not self._gpu_handles and not self._cpu_handles:
            return

        torch.cuda.synchronize()

        for handle, offset, size in self._gpu_handles:
            _check(
                cuda_drv.cuMemUnmap(self._base_ptr + offset, size),
                "cuMemUnmap GPU",
            )
            _check(cuda_drv.cuMemRelease(handle), "cuMemRelease GPU")

        for handle, offset, size in self._cpu_handles:
            _check(
                cuda_drv.cuMemUnmap(self._base_ptr + self._gpu_capacity + offset, size),
                "cuMemUnmap CPU",
            )
            _check(cuda_drv.cuMemRelease(handle), "cuMemRelease CPU")

        self._gpu_handles.clear()
        self._cpu_handles.clear()
        self._gpu_mapped = 0
        self._cpu_mapped = 0

    # -- private helpers -----------------------------------------------------

    def _align_up(self, nbytes: int) -> int:
        """Round *nbytes* up to ``self._page_size``."""
        ps = self._page_size
        return ((nbytes + ps - 1) // ps) * ps

    def __del__(self) -> None:
        try:
            self.release_physical()
        except Exception:
            pass
        try:
            if self._base_ptr:
                _check(
                    cuda_drv.cuMemAddressFree(self._base_ptr, self._total_bytes),
                    "cuMemAddressFree",
                )
                self._base_ptr = 0
        except Exception:
            pass
