import torch
from torchao.kernel.autotuner import get_best_config_fn

import triton
import triton.language as tl
import itertools
int8_powers_of_two = [32, 64, 128, 256]
int8_mm_kernel_configs = sum([
    # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
        [
            (i, j, k, 1, 1),

            (i, j, k, 1, 2),
            (i, j, k, 2, 2),

            (i, j, k, 1, 4),
            (i, j, k, 2, 4),
            (i, j, k, 3, 4),
            (i, j, k, 4, 4),

            (i, j, k, 1, 8),
            (i, j, k, 2, 8),
            (i, j, k, 3, 8),
            (i, j, k, 4, 8),
            (i, j, k, 5, 8),
            (i, j, k, 6, 8),
            (i, j, k, 7, 8),
            (i, j, k, 8, 8),
        ] for 
        (i, j, k) in 
        itertools.product(int8_powers_of_two,
                          int8_powers_of_two,
                          int8_powers_of_two)], [])

int8_mm_kernel_configs = [triton.Config({'BLOCK_SIZE_M': i, 'BLOCK_SIZE_N': j, 'BLOCK_SIZE_K': k, 'GROUP_SIZE_M': 8}, num_stages=s, num_warps = w) for (i, j, k, s, w) in int8_mm_kernel_configs]


@triton.jit
def matmul_kernel_with_block_pointers(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block.
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = accumulator #.to(tl.float16)

    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


@triton.jit
def scaled_matmul_kernel_with_block_pointers(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, s1_ptr, s2_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_s1m, stride_s1n,
        stride_s2m, stride_s2n,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block.
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        a = tl.load(a_block_ptr) #, boundary_check=(0, 1))
        b = tl.load(b_block_ptr) #, boundary_check=(0, 1))
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = accumulator #.to(tl.float16)

    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    s1_ptrs = s1_ptr + offs_m[:, None] * stride_s1m + offs_n[None, :] * stride_s1n
    s1 = tl.load(s1_ptrs)
    c = c * s1
    s2_ptrs = s2_ptr + offs_m[:, None] * stride_s2m + offs_n[None, :] * stride_s2n
    s2 = tl.load(s2_ptrs)
    c = c * s2
    c = c.to(tl.bfloat16)
    # Epilogue
    tl.store(c_block_ptr, c) #, boundary_check=(0, 1))

def int_matmul_kernel(a, b, c, config):
    M, K = a.shape
    K, N = b.shape
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_with_block_pointers[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        num_ctas=config.num_ctas,
        **config.kwargs,
    )
    return c

def int_scaled_matmul_kernel(a, b, scales1, scales2, c, config):
    M, K = a.shape
    K, N = b.shape
    # print("a.sizes(): ", a.size(), "a.strides(): ", a.stride(), "a.dtype: ", a.dtype)
    # print("b.sizes(): ", b.size(), "b.strides(): ", b.stride(), "b.dtype: ", b.dtype)
    # print("c.sizes(): ", c.size(), "c.strides(): ", c.stride(), "c.dtype: ", c.dtype)
    # print("scales1.sizes(): ", scales1.size(), "scales1.strides(): ", scales1.stride(), "scales1.dtype", scales1.dtype)
    # print("scales2.sizes(): ", scales2.size(), "scales2.strides(): ", scales2.stride(), "scales2.dtype", scales2.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    scaled_matmul_kernel_with_block_pointers[grid](
        a, b, c, scales1, scales2,
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),
        scales1.stride(0), scales1.stride(1),
        scales2.stride(0), scales2.stride(1),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        num_ctas=config.num_ctas,
        **config.kwargs,
    )
    return c


lib = torch.library.Library("torchao", "FRAGMENT")
lib.define("int_matmul(Tensor a, Tensor b) -> Tensor")
lib.define("int_scaled_matmul(Tensor a, Tensor b, Tensor scales1, Tensor scales2) -> Tensor")


@torch.library.impl(lib, "int_matmul", "Meta")
def int_matmul_meta(a, b):
    M, K = a.shape
    K, N = b.shape
    return torch.empty((M, N), device=a.device, dtype=torch.int32)


@torch.library.impl(lib, "int_matmul", "CUDA")
def int_matmul_cuda(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    # Allocates output.
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    # 1D launch kernel where each block gets its own program.
    best_config = get_best_config_fn(int_matmul_kernel,
                                     [a, b, c],
                                     int8_mm_kernel_configs)
    return int_matmul_kernel(a, b, c, best_config)


def int_matmul(a, b):
    return torch.ops.torchao.int_matmul(a, b)


@torch.library.impl(lib, "int_scaled_matmul", "Meta")
def int_scaled_matmul_meta(a, b, scales1, scales2):
    M, K = a.shape
    K, N = b.shape
    return torch.empty((M, N), device=a.device, dtype=scales1.dtype)


@torch.library.impl(lib, "int_scaled_matmul", "CUDA")
def int_scaled_matmul_cuda(a, b, scales1, scales2):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    # Allocates output.
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=scales1.dtype)
    # 1D launch kernel where each block gets its own program.
    best_config = get_best_config_fn(int_scaled_matmul_kernel,
                                     [a, b, scales1, scales2, c],
                                     int8_mm_kernel_configs)
    return int_scaled_matmul_kernel(a, b, scales1, scales2, c, best_config)

def int_scaled_matmul(a, b, scales1, scales2):
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.transpose(0, 1).is_contiguous(), "Matrix B must be transpose contiguous"
    M, K = a.shape
    K, N = b.shape
    assert M == scales1.size(0)
    assert 1 == scales1.size(1)
    assert scales1.is_contiguous()
    assert scales2.is_contiguous()
    assert scales1.dtype == torch.bfloat16
    scales1 = scales1.expand((M, N))
    scales2 = scales2.expand((M, N))
    assert scales1.dim() == 2
    assert scales2.dim() == 2
    return torch.ops.torchao.int_scaled_matmul(a, b, scales1, scales2)
