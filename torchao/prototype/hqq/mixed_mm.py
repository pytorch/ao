import torch
from triton import cdiv
import triton.language as tl
from .kernels import mixed_mm_kernel_compute_bound, mixed_mm_kernel_max_autotune
#credit jlebar
def pack_2xint4(t):
    """
    The packing format is such that consecutive rows are packed into a lower / upper bits
    E.g.,
    Original, unpacked B (dtype i8):
    [
        [0, 1, 2, 3]
        [4, 5, 6, 7]
        [8, 9, 10, 11]
        [12, 13, 14, 15]
    ]
    Packed B:
    [
        [0|4, 1|5, 2|6, 3|7]
        [8|12, 9|13, 10|14, 11|15]
    ]
    (Note each entry in `Packed B` is shown lsb->msb)
    """
    assert t.dtype == torch.int8 or t.dtype == torch.uint8
    t = t.reshape(t.shape[0] // 2, 2, t.shape[1]).permute(1, 0, 2)
    return (t[0] & 0xF) | (t[1] << 4)

def triton_mixed_mm(
    a,
    b,
    scales,
    zeros,
    group_size,
    acc_dtype=None,
    input_precision="ieee",
    fp8_fast_accum=False,
    kernel_type="compute_bound",
):
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0] * 2, "incompatible dimensions"
    assert b.dtype == torch.int8 or b.dtype == torch.uint8, "b must be int8 or uint8"
    assert scales.ndim == 2
    assert kernel_type in ["max_autotune", "compute_bound"]
    
    M, K = a.shape
    _, N = b.shape
    assert scales.shape[1] == N
    assert scales.shape[0] == K // group_size
    assert scales.dtype == a.dtype
    assert scales.shape == zeros.shape
    assert zeros.dtype == a.dtype

    # Assumes c is same type as a
    c = torch.empty((M, N), device=device, dtype=a.dtype)
    if acc_dtype is None:
        acc_dtype = tl.float32

    grid = lambda META: (
        cdiv(M, META["BLOCK_M"]) * cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )

    if kernel_type == "max_autotune":
        kernel = mixed_mm_kernel_max_autotune
    else:
        kernel = mixed_mm_kernel_compute_bound
        
    kernel[grid](
            a,
            b,
            scales,
            zeros,
            c,
            M,
            N,
            K,  #
            a.stride(0),
            a.stride(1),  #
            b.stride(0),
            b.stride(1),  #
            c.stride(0),
            c.stride(1),
            scales.stride(0),
            scales.stride(1),
            IS_BFLOAT16=a.dtype == torch.bfloat16,
            QGROUP_SIZE=group_size,
            acc_dtype=acc_dtype,
            input_precision=input_precision,
            fp8_fast_accum=fp8_fast_accum,
        )
    return c
