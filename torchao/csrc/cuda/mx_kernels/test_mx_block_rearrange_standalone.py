"""
Standalone test for mx_block_rearrange_2d_K_groups CUDA kernel.
Uses torch.utils.cpp_extension.load for quick compilation and iteration.

Usage:
    python test_mx_block_rearrange_standalone.py
"""

import os
import sys

import torch
from torch.utils.cpp_extension import load

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the CUDA extension
print("Compiling CUDA kernel...")
mx_block_rearrange = load(
    name="mx_block_rearrange_2d_K_groups",
    sources=[
        os.path.join(SCRIPT_DIR, "mxfp8_extension.cpp"),
        os.path.join(SCRIPT_DIR, "mxfp8_cuda.cu"),
        os.path.join(SCRIPT_DIR, "mx_block_rearrange_2d_K_groups.cu"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-gencode=arch=compute_100,code=sm_100",
    ],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=True,
)

print("✓ Compilation successful!")


def benchmark_kernel(kernel_fn, *args, warmup=10, iterations=100):
    """Benchmark a kernel function and return average time in microseconds."""
    # Warmup
    for _ in range(warmup):
        kernel_fn(*args)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        kernel_fn(*args)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    return (elapsed_ms / iterations) * 1000  # Convert to microseconds


def test_kernel():
    print("\n" + "=" * 80)
    print("Testing mx_block_rearrange_2d_K_groups kernel")
    print("=" * 80)

    # Try importing the Triton reference implementation
    try:
        ao_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
        sys.path.insert(0, ao_root)

        from torchao.prototype.moe_training.kernels.mxfp8.quant import (
            triton_mx_block_rearrange_2d_K_groups,
        )
        from torchao.prototype.moe_training.utils import generate_jagged_offs
        from torchao.prototype.mx_formats.mx_tensor import to_mx

        has_triton = True
        print("✓ Triton reference implementation available")
    except ImportError as e:
        print(f"⚠ Triton reference not available: {e}")
        has_triton = False

    # Test parameters - use larger size for meaningful benchmarks
    device = "cuda"
    m, total_k = 5120, 16384
    n_groups = 8
    block_size = 32

    print("\nTest configuration:")
    print(f"  Matrix size: {m} x {total_k}")
    print(f"  Number of groups: {n_groups}")

    # Generate test data
    print("\nGenerating test data...")
    torch.manual_seed(42)
    input_data = torch.randn(m, total_k, device=device)

    if has_triton:
        e8m0_scales, _ = to_mx(
            input_data, elem_dtype=torch.float8_e4m3fn, block_size=block_size
        )

        input_group_offsets = generate_jagged_offs(
            n_groups, total_k, multiple_of=block_size, device=device
        )
        scale_group_offsets = input_group_offsets // block_size

        print(f"  Scales shape: {e8m0_scales.shape}")
    else:
        return False

    # Calculate memory bandwidth metrics
    bytes_per_element = 1
    input_bytes = e8m0_scales.numel() * bytes_per_element

    # Test CUDA kernel
    print("\n" + "-" * 80)
    print("Running CUDA parallel kernel (optimized)...")
    cuda_parallel_out_scales = mx_block_rearrange.mx_block_rearrange_2d_K_groups(
        e8m0_scales.view(torch.uint8),
        scale_group_offsets,
    )
    print("✓ CUDA parallel kernel completed successfully")

    output_bytes = cuda_parallel_out_scales.numel() * bytes_per_element
    total_bytes = input_bytes + output_bytes

    # Compare with Triton reference
    print("\n" + "-" * 80)
    print("Running Triton reference kernel...")
    triton_out = triton_mx_block_rearrange_2d_K_groups(
        e8m0_scales,
        scale_group_offsets,
    )
    print("✓ Triton kernel completed successfully")

    # Verify correctness
    cuda_parallel_out_e8m0 = cuda_parallel_out_scales.view(torch.float8_e8m0fnu)

    print("\nVerifying correctness...")
    if not torch.equal(triton_out, cuda_parallel_out_e8m0):
        print("✗ CUDA parallel and Triton outputs differ!")
        return False
    print("✓ CUDA parallel matches Triton")

    print("\n✓ All outputs are IDENTICAL!")

    # Benchmark section
    print("\n" + "=" * 80)
    print("BENCHMARKING MEMORY BANDWIDTH")
    print("=" * 80)

    print("\nBenchmarking kernels (100 iterations each)...")

    # Benchmark Triton
    triton_time_us = benchmark_kernel(
        triton_mx_block_rearrange_2d_K_groups,
        e8m0_scales,
        scale_group_offsets,
    )
    triton_bw_gbps = (total_bytes / 1e9) / (triton_time_us / 1e6)

    # Benchmark CUDA parallel (optimized)
    cuda_parallel_time_us = benchmark_kernel(
        mx_block_rearrange.mx_block_rearrange_2d_K_groups,
        e8m0_scales.view(torch.uint8),
        scale_group_offsets,
    )
    cuda_parallel_bw_gbps = (total_bytes / 1e9) / (cuda_parallel_time_us / 1e6)

    # Print results
    print("\nResults:")
    print(f"  Input size:  {input_bytes / 1e6:.2f} MB")
    print(f"  Output size: {output_bytes / 1e6:.2f} MB")
    print(f"  Total I/O:   {total_bytes / 1e6:.2f} MB\n")
    print(f"{'Kernel':<25} {'Time (μs)':<15} {'Bandwidth (GB/s)':<20} {'Speedup':<10}")
    print("-" * 70)
    print(
        f"{'Triton':<25} {triton_time_us:<15.2f} {triton_bw_gbps:<20.2f} {'1.00x':<10}"
    )
    print(
        f"{'CUDA Parallel':<25} {cuda_parallel_time_us:<15.2f} {cuda_parallel_bw_gbps:<20.2f} {triton_time_us / cuda_parallel_time_us:<10.2f}x"
    )
    print()

    # Highlight best performer
    best_bw = max(triton_bw_gbps, cuda_parallel_bw_gbps)
    if cuda_parallel_bw_gbps == best_bw:
        print("🏆 CUDA parallel kernel achieves highest memory bandwidth!")
    else:
        print("🏆 Triton kernel achieves highest memory bandwidth!")

    return True


if __name__ == "__main__":
    success = test_kernel()

    print("\n" + "=" * 80)
    if success:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ TESTS FAILED")
        sys.exit(1)
