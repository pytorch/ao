"""
Standalone test for mx_block_rearrange_2d_K_groups CUDA kernel.
Tests both row-major and column-major kernel variants.
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
        "-Xptxas=-v",  # Show register usage per kernel
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
    print("Testing mx_block_rearrange_2d_K_groups kernels (row-major and column-major)")
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
    rows, cols = e8m0_scales.shape

    # Prepare row-major input (default contiguous)
    e8m0_scales_row_major = e8m0_scales.contiguous()
    assert e8m0_scales_row_major.is_contiguous(), "Row-major input should be contiguous"

    # Prepare column-major input (same shape, different memory layout)
    e8m0_scales_col_major = e8m0_scales.T.contiguous().T
    assert e8m0_scales_col_major.shape == e8m0_scales.shape, "Shape should be preserved"
    assert e8m0_scales_col_major.stride() == (
        1,
        rows,
    ), (
        f"Expected column-major strides (1, {rows}), got {e8m0_scales_col_major.stride()}"
    )

    # -------------------------------------------------------------------------
    # Test Row-Major CUDA Kernel
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Running CUDA row-major kernel...")
    print(
        f"  Input shape: {e8m0_scales_row_major.shape}, strides: {e8m0_scales_row_major.stride()}"
    )
    cuda_rowmajor_out = mx_block_rearrange.mx_block_rearrange_2d_K_groups_rowmajor(
        e8m0_scales_row_major.view(torch.uint8),
        scale_group_offsets,
    )
    print("✓ CUDA row-major kernel completed successfully")

    # -------------------------------------------------------------------------
    # Test Column-Major CUDA Kernel
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Running CUDA column-major kernel...")
    print(
        f"  Input shape: {e8m0_scales_col_major.shape}, strides: {e8m0_scales_col_major.stride()}"
    )
    cuda_colmajor_out = mx_block_rearrange.mx_block_rearrange_2d_K_groups_colmajor(
        e8m0_scales_col_major.view(torch.uint8),
        scale_group_offsets,
    )
    print("✓ CUDA column-major kernel completed successfully")

    output_bytes = cuda_rowmajor_out.numel() * bytes_per_element
    total_bytes = input_bytes + output_bytes

    # -------------------------------------------------------------------------
    # Test Triton Reference
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Running Triton reference kernel...")
    triton_out = triton_mx_block_rearrange_2d_K_groups(
        e8m0_scales,
        scale_group_offsets,
    )
    print("✓ Triton kernel completed successfully")

    # -------------------------------------------------------------------------
    # Verify Correctness
    # -------------------------------------------------------------------------
    print("\nVerifying correctness...")
    cuda_rowmajor_out_e8m0 = cuda_rowmajor_out.view(torch.float8_e8m0fnu)
    cuda_colmajor_out_e8m0 = cuda_colmajor_out.view(torch.float8_e8m0fnu)

    all_correct = True

    if not torch.equal(triton_out, cuda_rowmajor_out_e8m0):
        print("✗ CUDA row-major and Triton outputs differ!")
        all_correct = False
    else:
        print("✓ CUDA row-major matches Triton")

    if not torch.equal(triton_out, cuda_colmajor_out_e8m0):
        print("✗ CUDA column-major and Triton outputs differ!")
        all_correct = False
    else:
        print("✓ CUDA column-major matches Triton")

    if not all_correct:
        return False

    print("\n✓ All outputs are IDENTICAL!")

    # -------------------------------------------------------------------------
    # Benchmark Section
    # -------------------------------------------------------------------------
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

    # Benchmark CUDA row-major
    cuda_rowmajor_time_us = benchmark_kernel(
        mx_block_rearrange.mx_block_rearrange_2d_K_groups_rowmajor,
        e8m0_scales_row_major.view(torch.uint8),
        scale_group_offsets,
    )
    cuda_rowmajor_bw_gbps = (total_bytes / 1e9) / (cuda_rowmajor_time_us / 1e6)

    # Benchmark CUDA column-major
    cuda_colmajor_time_us = benchmark_kernel(
        mx_block_rearrange.mx_block_rearrange_2d_K_groups_colmajor,
        e8m0_scales_col_major.view(torch.uint8),
        scale_group_offsets,
    )
    cuda_colmajor_bw_gbps = (total_bytes / 1e9) / (cuda_colmajor_time_us / 1e6)

    # Print results
    print("\nResults:")
    print(f"  Input size:  {input_bytes / 1e6:.2f} MB")
    print(f"  Output size: {output_bytes / 1e6:.2f} MB")
    print(f"  Total I/O:   {total_bytes / 1e6:.2f} MB\n")
    print(
        f"{'Kernel':<25} {'Time (μs)':<15} {'Bandwidth (GB/s)':<20} {'Speedup vs Triton':<10}"
    )
    print("-" * 80)
    print(
        f"{'Triton':<25} {triton_time_us:<15.2f} {triton_bw_gbps:<20.2f} {'1.00x':<10}"
    )
    print(
        f"{'CUDA Row-Major':<25} {cuda_rowmajor_time_us:<15.2f} {cuda_rowmajor_bw_gbps:<20.2f} {triton_time_us / cuda_rowmajor_time_us:<10.2f}x"
    )
    print(
        f"{'CUDA Column-Major':<25} {cuda_colmajor_time_us:<15.2f} {cuda_colmajor_bw_gbps:<20.2f} {triton_time_us / cuda_colmajor_time_us:<10.2f}x"
    )
    print()

    # Compare row-major vs column-major
    print("-" * 80)
    if cuda_colmajor_time_us < cuda_rowmajor_time_us:
        speedup = cuda_rowmajor_time_us / cuda_colmajor_time_us
        print(f"🏆 Column-major kernel is {speedup:.2f}x faster than row-major!")
    else:
        speedup = cuda_colmajor_time_us / cuda_rowmajor_time_us
        print(f"🏆 Row-major kernel is {speedup:.2f}x faster than column-major!")

    # Highlight best overall performer
    best_bw = max(triton_bw_gbps, cuda_rowmajor_bw_gbps, cuda_colmajor_bw_gbps)
    if cuda_colmajor_bw_gbps == best_bw:
        print("🏆 CUDA column-major kernel achieves highest memory bandwidth!")
    elif cuda_rowmajor_bw_gbps == best_bw:
        print("🏆 CUDA row-major kernel achieves highest memory bandwidth!")
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
