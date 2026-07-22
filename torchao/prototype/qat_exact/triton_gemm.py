import torch
import triton
import triton.language as tl


@triton.jit
def int8_matmul_kernel_precise(
    # Input pointers
    a_ptr,
    b_ptr,
    c_ptr,
    # Quantization parameters
    a_scale_ptr,
    a_zero_ptr,  # Per-token asymmetric scaling for activations
    b_scale_ptr,  # Per-group symmetric scaling for weights (no zero point)
    # Matrix dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    More precise version that handles per-group scaling correctly.
    This version processes one group at a time to apply correct per-group scales.
    """

    # Get program ID
    pid = tl.program_id(axis=0)

    # Calculate 2D grid coordinates
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Calculate block offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    # Load per-token quantization parameters for A
    a_scale = tl.load(a_scale_ptr + offs_am)  # [BLOCK_SIZE_M]
    a_zero = tl.load(a_zero_ptr + offs_am)  # [BLOCK_SIZE_M]

    # Initialize final accumulator
    final_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Process each group separately to apply correct per-group scaling
    num_groups = tl.cdiv(K, GROUP_SIZE)

    for group_idx in range(num_groups):
        # Calculate K range for this group
        k_start = group_idx * GROUP_SIZE
        k_end = tl.minimum(k_start + GROUP_SIZE, K)
        actual_group_size = k_end - k_start

        # Load B scale for this group
        b_scale_offs = group_idx * N + offs_bn
        b_scale_mask = offs_bn < N
        b_scale = tl.load(b_scale_ptr + b_scale_offs, mask=b_scale_mask, other=1.0)
        # f32 -> bf16 -> f32 to match xnnpack
        b_scale = b_scale.to(tl.bfloat16).to(tl.float32)

        # Process this group in sub-blocks
        group_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        group_zero_correction = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

        # Process the group in BLOCK_SIZE_K chunks
        for k_sub in range(0, actual_group_size, BLOCK_SIZE_K):
            k_offset = k_start + k_sub

            # Calculate offsets for this sub-block
            offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)

            # Load A block
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
            a_block = tl.load(a_ptrs, mask=a_mask, other=0)

            # Load B block
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )
            b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
            b_block = tl.load(b_ptrs, mask=b_mask, other=0)

            # Integer dot product using tensor cores
            group_accumulator = tl.dot(a_block, b_block, acc=group_accumulator)

            # Zero-point correction for this sub-block
            a_zero_expanded = a_zero[:, None].to(tl.int32)
            b_sum = tl.sum(b_block.to(tl.int32), axis=0, keep_dims=True)
            group_zero_correction += a_zero_expanded * b_sum

        # Apply zero-point correction for this group
        corrected_group_result = group_accumulator - group_zero_correction

        # Convert to float and apply scaling for this group
        group_result_float = corrected_group_result.to(tl.float32)

        # Apply combined scaling: A_scale * B_scale
        a_scale_expanded = a_scale[:, None]
        b_scale_expanded = b_scale[None, :]
        combined_scale = a_scale_expanded * b_scale_expanded

        # Add this group's contribution to final result
        final_accumulator += group_result_float * combined_scale

    # Store final result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, final_accumulator, mask=c_mask)


def int8_matmul_triton(
    a_int8: torch.Tensor,  # [M, K] int8 activations
    b_int8: torch.Tensor,  # [K, N] int8 weights
    a_scale: torch.Tensor,  # [M] fp32 per-token scales
    a_zero: torch.Tensor,  # [M] int8 per-token zero points
    b_scale: torch.Tensor,  # [K//32, N] fp16 per-group scales
    group_size: int = 32,
) -> torch.Tensor:
    """
    Wrapper function for the Triton int8 matmul kernel using integer tensor cores.

    Args:
        a_int8: Quantized activations [M, K] (int8)
        b_int8: Quantized weights [K, N] (int8)
        a_scale: Per-token scale factors for activations [M] (fp16)
        a_zero: Per-token zero points for activations [M] (int8)
        b_scale: Per-group scale factors for weights [K//group_size, N] (fp16)
        group_size: Group size for weight quantization (default: 32)

    Returns:
        Output tensor [M, N] (fp16)
    """
    M, K = a_int8.shape
    K_b, N = b_int8.shape

    assert K == K_b, f"K dimensions must match: {K} != {K_b}"
    assert K % group_size == 0, (
        f"K ({K}) must be divisible by group_size ({group_size})"
    )
    assert a_scale.shape == (M,), f"a_scale shape mismatch: {a_scale.shape} != {(M,)}"
    assert a_zero.shape == (M,), f"a_zero shape mismatch: {a_zero.shape} != {(M,)}"
    assert b_scale.shape == (K // group_size, N), (
        f"b_scale shape mismatch: {b_scale.shape} != {(K // group_size, N)}"
    )

    # Output tensor
    c = torch.empty((M, N), device=a_int8.device, dtype=torch.float32)

    # Flatten b_scale for easier indexing in kernel
    b_scale_flat = b_scale.reshape(-1)

    # Launch configuration - optimized for integer tensor cores
    # Use smaller blocks for better tensor core utilization
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32  # Should be multiple of 16 for tensor cores

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # Launch kernel
    int8_matmul_kernel_precise[grid](
        a_int8,
        b_int8,
        c,
        a_scale,
        a_zero,
        b_scale_flat,
        M,
        N,
        K,
        a_int8.stride(0),
        a_int8.stride(1),
        b_int8.stride(0),
        b_int8.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE=group_size,
    )

    return c


# Utility function to benchmark tensor core utilization
def benchmark_tensor_cores():
    """
    Benchmark function to demonstrate tensor core utilization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test different matrix sizes
    test_sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    for M, N, K in test_sizes:
        print(f"\nTesting {M}x{N}x{K} matrix multiplication")

        # Generate test data
        a_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
        b_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
        a_scale = torch.rand(M, device=device) * 0.1
        a_zero = torch.randint(0, 255, (M,), dtype=torch.uint8, device=device)
        b_scale = torch.rand(K // 32, N, device=device) * 0.1

        # Warmup
        for _ in range(5):
            _ = int8_matmul_triton(a_int8, b_int8, a_scale, a_zero, b_scale)

        # Benchmark
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(10):
            result = int8_matmul_triton(a_int8, b_int8, a_scale, a_zero, b_scale)
        end.record()
        torch.cuda.synchronize()

        elapsed_time = start.elapsed_time(end) / 10  # Average time per iteration

        # Calculate TOPS (Tera Operations Per Second)
        ops = 2 * M * N * K  # Multiply-add operations
        tops = (ops / (elapsed_time * 1e-3)) / 1e12

        print(f"  Time: {elapsed_time:.2f} ms")
        print(f"  TOPS: {tops:.2f}")
        print(f"  Result shape: {result.shape}")


# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Matrix dimensions
    M, N, K = 512, 512, 1024
    group_size = 32

    # Create test data
    torch.manual_seed(42)

    # Generate random int8 matrices
    a_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    b_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)

    # Create quantization parameters
    a_scale = torch.rand(M, device=device) * 0.1
    a_zero = torch.randint(0, 255, (M,), dtype=torch.uint8, device=device)
    b_scale = torch.rand(K // group_size, N, device=device) * 0.1

    print("Input shapes:")
    print(f"  a_int8: {a_int8.shape} ({a_int8.dtype})")
    print(f"  b_int8: {b_int8.shape} ({b_int8.dtype})")
    print(f"  a_scale: {a_scale.shape} ({a_scale.dtype})")
    print(f"  a_zero: {a_zero.shape} ({a_zero.dtype})")
    print(f"  b_scale: {b_scale.shape} ({b_scale.dtype})")

    # Test both kernels
    print("Testing precise kernel...")
    result_precise = int8_matmul_triton(
        a_int8, b_int8, a_scale, a_zero, b_scale, group_size
    )

    print("\nOutput shapes:")
    print(f"  Precise kernel result: {result_precise.shape} ({result_precise.dtype})")

    # Run benchmark
    if torch.cuda.is_available():
        print("\nRunning tensor core benchmark...")
        benchmark_tensor_cores()
