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
    # GC anchor for the _CUDAArrayInterfaceWrapper that backs the tensor.
    #
    # torch.as_tensor() extracts the raw pointer from __cuda_array_interface__
    # and builds a Storage — it does NOT necessarily prevent the wrapper object
    # from being garbage-collected.  Storing it here keeps it alive for as long
    # as this allocation entry exists in _g_allocated_cudacpu_buffers.
    #
    # Ideally we'd use at::from_blob(..., deleter) which ties the tensor's
    # storage refcount directly to the custom deleter, but from_blob is a
    # C++-only API with no Python equivalent.  A small C++ extension calling
    # at::from_blob would eliminate both this field and the weakref destructor
    # machinery, but we avoid C++ extensions here to keep the implementation
    # pure-Python.
    _prevent_gc: object = None


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
            pass
    except (ImportError, Exception):
        pass


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

    Why this exists instead of at::from_blob
    -----------------------------------------
    The C++ implementation uses ``at::from_blob(ptr, sizes, deleter, opts)``
    which directly ties the tensor's storage lifetime to a custom deleter —
    no wrapper objects or weak references needed.  ``from_blob`` is C++-only;
    there is no ``torch.Tensor.from_blob`` in Python.

    From pure Python, the two viable ways to wrap a raw CUDA pointer as a
    PyTorch tensor are:

    1. ``__cuda_array_interface__`` + ``torch.as_tensor()`` (used here).
    2. DLPack (``__dlpack__`` / ``torch.from_dlpack()``) — requires building
       a ``DLManagedTensor`` struct via ctypes, which is more boilerplate
       than this approach.

    Both are zero-copy but neither supports an automatic destructor, so we
    pair this with a weak-reference destructor (see ``create_unified_buffer``)
    and a GC anchor (``_UnifiedBufferAllocation._prevent_gc``) to approximate
    the C++ from_blob lifecycle.

    Protocol spec: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    """

    def __init__(self, ptr_int: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",  # uint8
            "data": (ptr_int, False),  # (ptr, read_only)
            "version": 2,
        }


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
    assert (
        gpu_page_size == cpu_page_size
    ), f"GPU and CPU page sizes differ: {gpu_page_size} vs {cpu_page_size}"
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

    # -- Automatic deleter via weak reference --
    # The C++ version passes free_cudacpu_buffer as the deleter arg to
    # at::from_blob, so cleanup is driven by the tensor's storage refcount.
    # Python has no equivalent API (from_blob is C++-only), so we emulate
    # the same lifecycle with weakref: when the tensor (and all views/slices
    # sharing its storage) are garbage-collected, the weak-ref callback fires
    # and releases the VMM resources.  The weak-ref object itself is kept
    # alive in _g_prevent_destructor_gc so it isn't collected before the
    # tensor.
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
