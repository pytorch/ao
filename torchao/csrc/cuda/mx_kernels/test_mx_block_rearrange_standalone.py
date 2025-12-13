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
    print(
        "Testing mx_block_rearrange_2d_K_groups kernels (row-major, column-major, vectorized)"
    )
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
        print("Triton reference implementation available")
    except ImportError as e:
        print(f"WARNING: Triton reference not available: {e}")
        has_triton = False

    # Test parameters - use larger size for meaningful benchmarks
    device = "cuda"
    m, total_k = 7168, 131072
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
    print("CUDA row-major kernel completed successfully")

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
    print("CUDA column-major kernel completed successfully")

    # -------------------------------------------------------------------------
    # Test Column-Major Vectorized CUDA Kernel
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Running CUDA column-major vectorized kernel...")
    print(
        f"  Input shape: {e8m0_scales_col_major.shape}, strides: {e8m0_scales_col_major.stride()}"
    )
    cuda_colmajor_vec_out = (
        mx_block_rearrange.mx_block_rearrange_2d_K_groups_colmajor_vectorized(
            e8m0_scales_col_major.view(torch.uint8),
            scale_group_offsets,
        )
    )
    print("CUDA column-major vectorized kernel completed successfully")

    # -------------------------------------------------------------------------
    # Test Column-Major Vectorized 16B CUDA Kernel
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Running CUDA column-major vectorized 16B kernel...")
    print(
        f"  Input shape: {e8m0_scales_col_major.shape}, strides: {e8m0_scales_col_major.stride()}"
    )
    cuda_colmajor_vec_16B_out = (
        mx_block_rearrange.mx_block_rearrange_2d_K_groups_colmajor_vectorized_16B(
            e8m0_scales_col_major.view(torch.uint8),
            scale_group_offsets,
        )
    )
    print("CUDA column-major vectorized 16B kernel completed successfully")

    # -------------------------------------------------------------------------
    # Test Row-Major Vectorized CUDA Kernel
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Running CUDA row-major vectorized kernel...")
    print(
        f"  Input shape: {e8m0_scales_row_major.shape}, strides: {e8m0_scales_row_major.stride()}"
    )
    cuda_rowmajor_vec_out = (
        mx_block_rearrange.mx_block_rearrange_2d_K_groups_rowmajor_vectorized(
            e8m0_scales_row_major.view(torch.uint8),
            scale_group_offsets,
        )
    )
    print("CUDA row-major vectorized kernel completed successfully")

    # -------------------------------------------------------------------------
    # Test Row-Major 128x4 Vectorized CUDA Kernel
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Running CUDA row-major 128x4 vectorized kernel...")
    print(
        f"  Input shape: {e8m0_scales_row_major.shape}, strides: {e8m0_scales_row_major.stride()}"
    )
    cuda_rowmajor_128x4_vec_out = (
        mx_block_rearrange.mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec(
            e8m0_scales_row_major.view(torch.uint8),
            scale_group_offsets,
        )
    )
    print("CUDA row-major 128x4 vectorized kernel completed successfully")

    # -------------------------------------------------------------------------
    # Test Row-Major 128x4 Vectorized Pipelined CUDA Kernel (default chunks_per_tb=4)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(
        "Running CUDA row-major 128x4 vectorized pipelined kernel (chunks_per_tb=4)..."
    )
    print(
        f"  Input shape: {e8m0_scales_row_major.shape}, strides: {e8m0_scales_row_major.stride()}"
    )
    cuda_rowmajor_128x4_vec_pipelined_out = (
        mx_block_rearrange.mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined(
            e8m0_scales_row_major.view(torch.uint8),
            scale_group_offsets,
            64,  # max_cols
            4,  # chunks_per_tb
        )
    )
    print(
        "CUDA row-major 128x4 vectorized pipelined kernel (chunks_per_tb=4) completed successfully"
    )

    # -------------------------------------------------------------------------
    # Test Row-Major 128x4 Vectorized Pipelined CUDA Kernel (chunks_per_tb=8)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(
        "Running CUDA row-major 128x4 vectorized pipelined kernel (chunks_per_tb=8)..."
    )
    cuda_rowmajor_128x4_vec_pipelined_8_out = (
        mx_block_rearrange.mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined(
            e8m0_scales_row_major.view(torch.uint8),
            scale_group_offsets,
            64,  # max_cols
            8,  # chunks_per_tb
        )
    )
    print(
        "CUDA row-major 128x4 vectorized pipelined kernel (chunks_per_tb=8) completed successfully"
    )

    # -------------------------------------------------------------------------
    # Test Row-Major 128x4 Vectorized Pipelined CUDA Kernel (chunks_per_tb=16)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(
        "Running CUDA row-major 128x4 vectorized pipelined kernel (chunks_per_tb=16)..."
    )
    cuda_rowmajor_128x4_vec_pipelined_16_out = (
        mx_block_rearrange.mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined(
            e8m0_scales_row_major.view(torch.uint8),
            scale_group_offsets,
            64,  # max_cols
            16,  # chunks_per_tb
        )
    )
    print(
        "CUDA row-major 128x4 vectorized pipelined kernel (chunks_per_tb=16) completed successfully"
    )

    # -------------------------------------------------------------------------
    # Test Triton Reference
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Running Triton reference kernel...")
    triton_out = triton_mx_block_rearrange_2d_K_groups(
        e8m0_scales,
        scale_group_offsets,
    )
    print("Triton kernel completed successfully")

    # -------------------------------------------------------------------------
    # Verify Correctness
    # -------------------------------------------------------------------------
    print("\nVerifying correctness...")
    cuda_rowmajor_out_e8m0 = cuda_rowmajor_out.view(torch.float8_e8m0fnu)
    cuda_colmajor_out_e8m0 = cuda_colmajor_out.view(torch.float8_e8m0fnu)
    cuda_colmajor_vec_out_e8m0 = cuda_colmajor_vec_out.view(torch.float8_e8m0fnu)

    all_correct = True

    if not torch.equal(triton_out, cuda_rowmajor_out_e8m0):
        print("FAILED: CUDA row-major and Triton outputs differ!")
        all_correct = False
    else:
        print("PASSED: CUDA row-major matches Triton")

    if not torch.equal(triton_out, cuda_colmajor_out_e8m0):
        print("FAILED: CUDA column-major and Triton outputs differ!")
        all_correct = False
    else:
        print("PASSED: CUDA column-major matches Triton")

    if not torch.equal(triton_out, cuda_colmajor_vec_out_e8m0):
        print("FAILED: CUDA column-major vectorized and Triton outputs differ!")
        all_correct = False
    else:
        print("PASSED: CUDA column-major vectorized matches Triton")

    cuda_colmajor_vec_16B_out_e8m0 = cuda_colmajor_vec_16B_out.view(
        torch.float8_e8m0fnu
    )
    if not torch.equal(triton_out, cuda_colmajor_vec_16B_out_e8m0):
        print("FAILED: CUDA column-major vectorized 16B and Triton outputs differ!")
        all_correct = False
    else:
        print("PASSED: CUDA column-major vectorized 16B matches Triton")

    cuda_rowmajor_vec_out_e8m0 = cuda_rowmajor_vec_out.view(torch.float8_e8m0fnu)
    if not torch.equal(triton_out, cuda_rowmajor_vec_out_e8m0):
        print("FAILED: CUDA row-major vectorized and Triton outputs differ!")
        all_correct = False
    else:
        print("PASSED: CUDA row-major vectorized matches Triton")

    cuda_rowmajor_128x4_vec_out_e8m0 = cuda_rowmajor_128x4_vec_out.view(
        torch.float8_e8m0fnu
    )
    if not torch.equal(triton_out, cuda_rowmajor_128x4_vec_out_e8m0):
        print("FAILED: CUDA row-major 128x4 vectorized and Triton outputs differ!")
        # Print debug info for differences
        diff_mask = triton_out != cuda_rowmajor_128x4_vec_out_e8m0
        num_diffs = diff_mask.sum().item()
        print(f"  Number of differences: {num_diffs} / {triton_out.numel()}")
        all_correct = False
    else:
        print("PASSED: CUDA row-major 128x4 vectorized matches Triton")

    cuda_rowmajor_128x4_vec_pipelined_out_e8m0 = (
        cuda_rowmajor_128x4_vec_pipelined_out.view(torch.float8_e8m0fnu)
    )
    if not torch.equal(triton_out, cuda_rowmajor_128x4_vec_pipelined_out_e8m0):
        print(
            "FAILED: CUDA row-major 128x4 vectorized pipelined (chunks_per_tb=4) and Triton outputs differ!"
        )
        diff_mask = triton_out != cuda_rowmajor_128x4_vec_pipelined_out_e8m0
        num_diffs = diff_mask.sum().item()
        print(f"  Number of differences: {num_diffs} / {triton_out.numel()}")
        all_correct = False
    else:
        print(
            "PASSED: CUDA row-major 128x4 vectorized pipelined (chunks_per_tb=4) matches Triton"
        )

    cuda_rowmajor_128x4_vec_pipelined_8_out_e8m0 = (
        cuda_rowmajor_128x4_vec_pipelined_8_out.view(torch.float8_e8m0fnu)
    )
    if not torch.equal(triton_out, cuda_rowmajor_128x4_vec_pipelined_8_out_e8m0):
        print(
            "FAILED: CUDA row-major 128x4 vectorized pipelined (chunks_per_tb=8) and Triton outputs differ!"
        )
        diff_mask = triton_out != cuda_rowmajor_128x4_vec_pipelined_8_out_e8m0
        num_diffs = diff_mask.sum().item()
        print(f"  Number of differences: {num_diffs} / {triton_out.numel()}")
        all_correct = False
    else:
        print(
            "PASSED: CUDA row-major 128x4 vectorized pipelined (chunks_per_tb=8) matches Triton"
        )

    cuda_rowmajor_128x4_vec_pipelined_16_out_e8m0 = (
        cuda_rowmajor_128x4_vec_pipelined_16_out.view(torch.float8_e8m0fnu)
    )
    if not torch.equal(triton_out, cuda_rowmajor_128x4_vec_pipelined_16_out_e8m0):
        print(
            "FAILED: CUDA row-major 128x4 vectorized pipelined (chunks_per_tb=16) and Triton outputs differ!"
        )
        diff_mask = triton_out != cuda_rowmajor_128x4_vec_pipelined_16_out_e8m0
        num_diffs = diff_mask.sum().item()
        print(f"  Number of differences: {num_diffs} / {triton_out.numel()}")
        all_correct = False
    else:
        print(
            "PASSED: CUDA row-major 128x4 vectorized pipelined (chunks_per_tb=16) matches Triton"
        )

    if not all_correct:
        return False

    print("\nAll outputs are IDENTICAL!")
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
