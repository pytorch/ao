import os

import torch
import triton
import triton.language as tl

os.environ["ENABLE_TMA"] = "1"


@triton.jit
def grouped_launch(
    pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr
):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.jit
def gemm_split_k_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    scale_a,
    scale_b,
    m,
    n,
    k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    split_k: tl.constexpr,
    group_m: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k * split_k)

    pid_m, pid_n = grouped_launch(pid, m, n, block_m, block_n, group_m)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_ in range(0, grid_k):
        k_remaining = k - k_ * (block_k * split_k)

        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk

    acc = scale_a * scale_b * acc
    acc.to(tl.float16)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]

    tl.atomic_add(c_ptrs, acc, mask=mask)


def gemm_split_k(a, b, scale_a: float = 1.0, scale_b: float = 1.0):
    assert a.shape[1] == b.shape[0]
    m, k = a.shape
    _, n = b.shape

    block_m = 64
    block_n = 64
    block_k = 512
    num_stages = 3
    num_warps = 8
    split_k = 4
    group_m = 8

    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    total_programs_mn = total_blocks_m * total_blocks_n
    total_programs_k = split_k

    grid = (total_programs_mn, total_programs_k)

    c = torch.zeros((m, n), device=a.device, dtype=torch.float16)
    k = gemm_split_k_kernel[grid](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        scale_a,
        scale_b,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        split_k,
        group_m,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return c


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()
