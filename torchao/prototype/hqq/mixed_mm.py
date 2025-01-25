import torch
import triton.language as tl
from triton import cdiv

from .kernels import (
    mixed_mm_kernel_compute_bound,
    mixed_mm_kernel_max_autotune,
)


# h/t jlebar for the bit packing / unpacking logic (source: Triton Slack thread)
# https://gist.github.com/jlebar/3435b2c00deea53258887ce37231e5e2
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
    transposed=False,
    acc_dtype=None,
    input_precision="ieee",
    fp8_fast_accum=False,
    kernel_type="compute_bound",
    # For debugging only
    BLOCK_M=None,
    BLOCK_N=None,
    BLOCK_K=None,
):
    """Run fused int4 / fp16 dequant GEMM

    Args:
        a (torch.Tensor): M x K if not transposed, M x N if transposed
        b (torch.Tensor): (K // 2) x N, packed such that 2 int4's are packed into 1 uint8 (see pack_2xint4)
        scales (torch.Tensor): (num_groups x N), where num_groups = (N * K / group_size)
        zeros (torch.Tensor): same shape as scales
        group_size (torch.Tensor): size of group in groupwise quantization -- MUST be along axis 1 of an N x K matrix
        transposed (bool, optional): Whether to run a transposed matmul where shapes are (M x N) x (K x N) => (M x K)
        acc_dtype (_type_, optional): dtype of accumulator. Defaults to None, which corresponds to tl.float32.
        input_precision (str, optional): Only relevant when dtype of a is torch.float32. Defaults to "ieee".
        kernel_type (str, optional): Type of autoconfig to use. Either "max_autotune" or "compute_bound".
        BLOCK_M (int, optional): Only for debugging. Defaults to None.
        BLOCK_N (int, optional): Only for debugging. Defaults to None.
        BLOCK_K (int, optional): Only for debugging. Defaults to None.

    Returns:
        c (torch.Tensor): M x N
    """
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    if not transposed:
        assert a.shape[1] == b.shape[0] * 2, "incompatible dimensions"

    assert b.dtype == torch.int8 or b.dtype == torch.uint8, "b must be int8 or uint8"
    assert scales.ndim == 2
    if transposed:
        assert (
            a.shape[1] == b.shape[1]
        ), "transpose requires (M x N) x (K x N), where reduction dim is N"

    M, K = a.shape
    N = b.shape[1] if not transposed else b.shape[0] * 2
    # assert scales.shape[1] == N if not transposed else scales.shape[0] == N
    # assert scales.shape[0] == K // group_size if not transposed else scales.shape[1] == K // group_size
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
    elif kernel_type == "compute_bound":
        kernel = mixed_mm_kernel_compute_bound
    else:
        from .kernels import _mixed_mm_debug

        kernel = _mixed_mm_debug

    if kernel_type == "max_autotune" or kernel_type == "compute_bound":
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
            TRANSPOSED=transposed,
            IS_BFLOAT16=a.dtype == torch.bfloat16,
            QGROUP_SIZE=group_size,
            acc_dtype=acc_dtype,
            input_precision=input_precision,
            fp8_fast_accum=fp8_fast_accum,
        )
    else:
        assert all([BLOCK_M is not None, BLOCK_N is not None, BLOCK_K is not None])
        grid = (M // BLOCK_M * N // BLOCK_N, 1, 1)
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
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            SPLIT_K=1,
            EVEN_K=True,
            TRANSPOSED=transposed,
            IS_BFLOAT16=a.dtype == torch.bfloat16,
            QGROUP_SIZE=group_size,
            acc_dtype=acc_dtype,
            input_precision=input_precision,
            fp8_fast_accum=fp8_fast_accum,
            DEBUG=True,
        )

    return c
