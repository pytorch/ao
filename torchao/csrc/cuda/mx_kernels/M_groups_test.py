"""
Standalone test for mx_block_rearrange_2d_M_groups CUDA kernel.
Tests the pipelined kernel variant for groups along the M (row) dimension.
Uses torch.utils.cpp_extension.load for quick compilation and iteration.

Usage:
    python test_mx_block_rearrange_M_groups_standalone.py
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
    name="mx_block_rearrange_2d_M_groups",
    sources=[
        os.path.join(SCRIPT_DIR, "mxfp8_extension.cpp"),
        os.path.join(SCRIPT_DIR, "mx_block_rearrange_2d_M_groups.cu"),
        os.path.join(SCRIPT_DIR, "mx_block_rearrange_2d_K_groups.cu"),
        os.path.join(SCRIPT_DIR, "mxfp8_cuda.cu"),
    ],
    extra_cuda_cflags=[
        "-O0",
        "-g",
        "-DNDEBUG=0",  # Ensure NDEBUG is not defined so asserts are active
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
    ],
    extra_cflags=["-O0", "-g", "-std=c++17"],
    extra_ldflags=["-lcuda"],  # Link against CUDA driver API for cuGetErrorString
    verbose=True,
)

print("Compilation successful!")


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
    print("Testing mx_block_rearrange_2d_M_groups kernels (pipelined variants)")
    print("=" * 80)

    # Try importing the Triton reference implementation
    try:
        ao_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
        sys.path.insert(0, ao_root)

        from torchao.prototype.moe_training.kernels.mxfp8.quant import (
            triton_mx_block_rearrange_2d_M_groups,
        )
        from torchao.prototype.moe_training.utils import generate_jagged_offs
        from torchao.prototype.mx_formats.mx_tensor import to_mx

        has_triton = True
        print("Triton reference implementation available")
    except ImportError as e:
        print(f"WARNING: Triton reference not available: {e}")
        has_triton = False

    # Test parameters - use larger size for meaningful benchmarks
    # For M groups, groups are along rows (M dimension)
    device = "cuda"
    total_m, k = 131072, 4096  # Large M (rows), moderate K (cols)
    n_groups = 8
    block_size = 32

    print("\nTest configuration:")
    print(f"  Matrix size: {total_m} x {k}")
    print(f"  Number of groups: {n_groups}")
    print("  Groups are along M (row) dimension")

    # Generate test data
    print("\nGenerating test data...")
    torch.manual_seed(42)
    input_data = torch.randn(total_m, k, device=device)

    if has_triton:
        e8m0_scales, _ = to_mx(
            input_data, elem_dtype=torch.float8_e4m3fn, block_size=block_size
        )

        # For M groups, offsets are along the row dimension
        # Since MX scaling is along K (columns), scale rows = input rows
        # So input_group_offsets = input_group_offsets (no division by block_size)
        input_group_offsets = generate_jagged_offs(
            n_groups, total_m, multiple_of=block_size, device=device
        )
        print(f"  Scales shape: {e8m0_scales.shape}")
        print(f"  Group offsets: {input_group_offsets.tolist()}")
    else:
        return False

    rows, cols = e8m0_scales.shape

    # Prepare row-major input (default contiguous)
    e8m0_scales_row_major = e8m0_scales.contiguous()
    assert e8m0_scales_row_major.is_contiguous(), "Row-major input should be contiguous"

    # -------------------------------------------------------------------------
    # Test CUDA Pipelined Kernel (chunks_per_tb=4)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Running CUDA M-groups pipelined kernel (chunks_per_tb=4)...")
    print(
        f"  Input shape: {e8m0_scales_row_major.shape}, strides: {e8m0_scales_row_major.stride()}"
    )
    cuda_pipelined_4_out = (
        mx_block_rearrange.mx_block_rearrange_2d_M_groups_rowmajor_128x4_vec_pipelined(
            e8m0_scales_row_major.view(torch.uint8),
            input_group_offsets,
            64,  # max_cols
            4,  # chunks_per_tb
        )
    )
    print("CUDA M-groups pipelined kernel (chunks_per_tb=4) completed successfully")

    # -------------------------------------------------------------------------
    # Test CUDA Pipelined Kernel (chunks_per_tb=8)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Running CUDA M-groups pipelined kernel (chunks_per_tb=8)...")
    cuda_pipelined_8_out = (
        mx_block_rearrange.mx_block_rearrange_2d_M_groups_rowmajor_128x4_vec_pipelined(
            e8m0_scales_row_major.view(torch.uint8),
            input_group_offsets,
            64,  # max_cols
            8,  # chunks_per_tb
        )
    )
    print("CUDA M-groups pipelined kernel (chunks_per_tb=8) completed successfully")

    # -------------------------------------------------------------------------
    # Test CUDA Pipelined Kernel (chunks_per_tb=16)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Running CUDA M-groups pipelined kernel (chunks_per_tb=16)...")
    cuda_pipelined_16_out = (
        mx_block_rearrange.mx_block_rearrange_2d_M_groups_rowmajor_128x4_vec_pipelined(
            e8m0_scales_row_major.view(torch.uint8),
            input_group_offsets,
            64,  # max_cols
            16,  # chunks_per_tb
        )
    )
    print("CUDA M-groups pipelined kernel (chunks_per_tb=16) completed successfully")

    # -------------------------------------------------------------------------
    # Test Triton Reference
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Running Triton reference kernel...")
    triton_out = triton_mx_block_rearrange_2d_M_groups(
        e8m0_scales,
        input_group_offsets,
    )
    print("Triton kernel completed successfully")

    # -------------------------------------------------------------------------
    # Verify Correctness
    # -------------------------------------------------------------------------
    print("\nVerifying correctness...")

    all_correct = True

    # Convert CUDA outputs to e8m0 for comparison
    cuda_pipelined_4_out_e8m0 = cuda_pipelined_4_out.view(torch.float8_e8m0fnu)
    cuda_pipelined_8_out_e8m0 = cuda_pipelined_8_out.view(torch.float8_e8m0fnu)
    cuda_pipelined_16_out_e8m0 = cuda_pipelined_16_out.view(torch.float8_e8m0fnu)

    # Compare shapes first
    print(f"\n  Triton output shape: {triton_out.shape}")
    print(f"  CUDA output shape (chunks_per_tb=4): {cuda_pipelined_4_out_e8m0.shape}")

    # Triton may have different padding, so compare the valid region
    triton_rows, triton_cols = triton_out.shape
    cuda_rows, cuda_cols = cuda_pipelined_4_out_e8m0.shape

    compare_rows = min(triton_rows, cuda_rows)
    compare_cols = min(triton_cols, cuda_cols)

    # Check pipelined (chunks_per_tb=4) output
    triton_valid = triton_out[:compare_rows, :compare_cols]
    cuda_4_valid = cuda_pipelined_4_out_e8m0[:compare_rows, :compare_cols]

    if not torch.equal(triton_valid, cuda_4_valid):
        print(
            "FAILED: CUDA M-groups pipelined (chunks_per_tb=4) and Triton outputs differ!"
        )
        diff_mask = triton_valid != cuda_4_valid
        num_diffs = diff_mask.sum().item()
        print(f"  Number of differences: {num_diffs} / {triton_valid.numel()}")

        # Show first few differences for debugging
        diff_indices = torch.where(diff_mask)
        if len(diff_indices[0]) > 0:
            for i in range(min(5, len(diff_indices[0]))):
                r, c = diff_indices[0][i].item(), diff_indices[1][i].item()
                print(
                    f"    Diff at ({r}, {c}): Triton={triton_valid[r, c]}, CUDA={cuda_4_valid[r, c]}"
                )
        all_correct = False
    else:
        print("PASSED: CUDA M-groups pipelined (chunks_per_tb=4) matches Triton")

    # Check pipelined (chunks_per_tb=8) output
    cuda_8_valid = cuda_pipelined_8_out_e8m0[:compare_rows, :compare_cols]
    if not torch.equal(triton_valid, cuda_8_valid):
        print(
            "FAILED: CUDA M-groups pipelined (chunks_per_tb=8) and Triton outputs differ!"
        )
        diff_mask = triton_valid != cuda_8_valid
        num_diffs = diff_mask.sum().item()
        print(f"  Number of differences: {num_diffs} / {triton_valid.numel()}")
        all_correct = False
    else:
        print("PASSED: CUDA M-groups pipelined (chunks_per_tb=8) matches Triton")

    # Check pipelined (chunks_per_tb=16) output
    cuda_16_valid = cuda_pipelined_16_out_e8m0[:compare_rows, :compare_cols]
    if not torch.equal(triton_valid, cuda_16_valid):
        print(
            "FAILED: CUDA M-groups pipelined (chunks_per_tb=16) and Triton outputs differ!"
        )
        diff_mask = triton_valid != cuda_16_valid
        num_diffs = diff_mask.sum().item()
        print(f"  Number of differences: {num_diffs} / {triton_valid.numel()}")
        all_correct = False
    else:
        print("PASSED: CUDA M-groups pipelined (chunks_per_tb=16) matches Triton")

    if not all_correct:
        return False

    print("\nAll outputs are IDENTICAL!")

    # -------------------------------------------------------------------------
    # Benchmark
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Benchmarking kernels...")

    triton_time = benchmark_kernel(
        triton_mx_block_rearrange_2d_M_groups,
        e8m0_scales,
        input_group_offsets,
    )
    print(f"  Triton reference: {triton_time:.2f} us")

    cuda_4_time = benchmark_kernel(
        mx_block_rearrange.mx_block_rearrange_2d_M_groups_rowmajor_128x4_vec_pipelined,
        e8m0_scales_row_major.view(torch.uint8),
        input_group_offsets,
        64,
        4,
    )
    print(
        f"  CUDA pipelined (chunks_per_tb=4): {cuda_4_time:.2f} us ({triton_time / cuda_4_time:.2f}x vs Triton)"
    )

    cuda_8_time = benchmark_kernel(
        mx_block_rearrange.mx_block_rearrange_2d_M_groups_rowmajor_128x4_vec_pipelined,
        e8m0_scales_row_major.view(torch.uint8),
        input_group_offsets,
        64,
        8,
    )
    print(
        f"  CUDA pipelined (chunks_per_tb=8): {cuda_8_time:.2f} us ({triton_time / cuda_8_time:.2f}x vs Triton)"
    )

    cuda_16_time = benchmark_kernel(
        mx_block_rearrange.mx_block_rearrange_2d_M_groups_rowmajor_128x4_vec_pipelined,
        e8m0_scales_row_major.view(torch.uint8),
        input_group_offsets,
        64,
        16,
    )
    print(
        f"  CUDA pipelined (chunks_per_tb=16): {cuda_16_time:.2f} us ({triton_time / cuda_16_time:.2f}x vs Triton)"
    )

    return True


if __name__ == "__main__":
    success = test_kernel()

    print("\n" + "=" * 80)
    if success:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("TESTS FAILED")
        sys.exit(1)
