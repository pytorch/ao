"""
Standalone test for mx_block_rearrange_2d_M_groups CUDA kernel.
Tests the pipelined kernel variant for groups along the M (row) dimension.
Uses torch.utils.cpp_extension.load for quick compilation and iteration.

Usage:
    python M_groups_test.py                    # Run all test configurations
    python M_groups_test.py --quick            # Run quick sanity test only
    python M_groups_test.py --config 0         # Run specific config by index
    python M_groups_test.py --no-benchmark     # Skip benchmarking
"""

import argparse
import os
import sys
from itertools import product

import torch
from torch.utils.cpp_extension import load

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Parse arguments first
parser = argparse.ArgumentParser(
    description="Test mx_block_rearrange_2d_M_groups kernel"
)
parser.add_argument("--quick", action="store_true", help="Run quick sanity test only")
parser.add_argument(
    "--config", type=int, default=None, help="Run specific config by index"
)
parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmarking")
parser.add_argument(
    "--list-configs", action="store_true", help="List all test configurations"
)
args = parser.parse_args()

# Test configurations: (total_m, k, n_groups)
# Covers various cases:
#   - Small/medium/large M dimensions
#   - Different K dimensions (affects scale_cols)
#   - Different group counts
#   - Partial row chunks (M not divisible by 128)
#   - Partial column chunks (scale_cols not divisible by chunk_width)
TEST_CONFIGS = [
    # Basic cases - divisible by 128
    (1024, 2048, 2),  # Small, 2 groups
    (4096, 2048, 4),  # Medium, 4 groups
    (32768, 2048, 8),  # Large, 8 groups
    (131072, 2048, 8),  # Very large M
    (131072, 2048, 32),  # Very large M, many groups
    # Partial row chunks (M not divisible by 128)
    (1024 + 32, 2048, 2),  # 1000 % 128 = 104
    # Different K values (affects scale_cols = k // 32)
    (4096, 1024, 4),  # scale_cols = 32
    (4096, 4096, 4),  # scale_cols = 128 (exactly chunk_width)
    (4096, 7168, 4),  # scale_cols = 224 (partial last col chunk)
    (4096, 8192, 4),  # scale_cols = 256 (2 full chunks)
    # Edge cases for partial column chunks
    (4096, 2048, 4),  # scale_cols = 64 (less than chunk_width=128)
    (4096, 3072, 4),  # scale_cols = 96
    (4096, 5120, 4),  # scale_cols = 160 (128 + 32)
    # Stress tests
    (262144, 2048, 16),  # 256K rows
    (65536, 4096, 32),  # Many groups, large K
]

QUICK_CONFIG = [(131072, 2048, 4)]  # Quick sanity test

if args.list_configs:
    print("Available test configurations:")
    print("-" * 60)
    for i, (m, k, n) in enumerate(TEST_CONFIGS):
        scale_cols = k // 32
        print(
            f"  [{i:2d}] M={m:7d}, K={k:5d}, n_groups={n:2d}  (scale_cols={scale_cols})"
        )
    sys.exit(0)

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
        "-lineinfo",
        "-DNDEBUG=0",
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
    ],
    extra_cflags=["-O0", "-g", "-std=c++17"],
    extra_ldflags=["-lcuda"],
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


def test_single_config(total_m, k, n_groups, run_benchmark=True):
    """Test a single configuration and return (passed, timing_info)."""
    block_size = 32
    device = "cuda"

    # Import Triton reference
    try:
        ao_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
        if ao_root not in sys.path:
            sys.path.insert(0, ao_root)

        from torchao.prototype.moe_training.kernels.mxfp8.quant import (
            triton_mx_block_rearrange_2d_M_groups,
        )
        from torchao.prototype.moe_training.utils import generate_jagged_offs
        from torchao.prototype.mx_formats.mx_tensor import to_mx
    except ImportError as e:
        print(f"  ERROR: Triton reference not available: {e}")
        return False, None

    # Compute scale dimensions
    scale_cols = k // block_size

    print(f"\n  Config: M={total_m}, K={k}, n_groups={n_groups}")
    print(f"    scale_cols = {scale_cols}")
    print(
        f"    M % 128 = {total_m % 128} (partial row chunk: {'yes' if total_m % 128 != 0 else 'no'})"
    )
    print(
        f"    scale_cols % 128 = {scale_cols % 128} (partial col chunk: {'yes' if scale_cols % 128 != 0 else 'no'})"
    )

    # Generate test data
    torch.manual_seed(42)
    input_data = torch.randn(total_m, k, device=device)

    e8m0_scales, _ = to_mx(
        input_data, elem_dtype=torch.float8_e4m3fn, block_size=block_size
    )

    # For M groups, offsets are along the row dimension
    input_group_offsets = generate_jagged_offs(
        n_groups, total_m, multiple_of=block_size, device=device
    )
    print(
        f"    Group sizes: {[input_group_offsets[i].item() - (input_group_offsets[i-1].item() if i > 0 else 0) for i in range(len(input_group_offsets))]}"
    )

    # Prepare row-major input (default contiguous)
    e8m0_scales_row_major = e8m0_scales.contiguous()

    # Run CUDA kernel
    try:
        cuda_out = mx_block_rearrange.mx_block_rearrange_2d_M_groups_cuda(
            e8m0_scales_row_major.view(torch.uint8),
            input_group_offsets,
            64,  # chunk_width
            1,  # chunks_per_tb (1 for higher occupancy)
        )
    except Exception as e:
        print(f"    CUDA kernel FAILED: {e}")
        return False, None

    # Run Triton reference
    try:
        triton_out = triton_mx_block_rearrange_2d_M_groups(
            e8m0_scales,
            input_group_offsets,
        )
    except Exception as e:
        print(f"    Triton kernel FAILED: {e}")
        return False, None

    # Compare outputs
    cuda_out_e8m0 = cuda_out.view(torch.float8_e8m0fnu)
    triton_rows, triton_cols = triton_out.shape
    cuda_rows, cuda_cols = cuda_out_e8m0.shape

    compare_rows = min(triton_rows, cuda_rows)
    compare_cols = min(triton_cols, cuda_cols)
    triton_valid = triton_out[:compare_rows, :compare_cols]
    cuda_valid = cuda_out_e8m0[:compare_rows, :compare_cols]

    if not torch.equal(triton_valid, cuda_valid):
        diff_mask = triton_valid != cuda_valid
        num_diffs = diff_mask.sum().item()
        print(f"    FAILED: {num_diffs} / {triton_valid.numel()} elements differ")

        # Show first few differences
        diff_indices = torch.where(diff_mask)
        if len(diff_indices[0]) > 0:
            for i in range(min(3, len(diff_indices[0]))):
                r, c = diff_indices[0][i].item(), diff_indices[1][i].item()
                print(
                    f"      Diff at ({r}, {c}): Triton={triton_valid[r, c]}, CUDA={cuda_valid[r, c]}"
                )
        return False, None

    print(f"    PASSED: Outputs match")

    # Benchmark if requested
    timing_info = None
    if run_benchmark:
        triton_time = benchmark_kernel(
            triton_mx_block_rearrange_2d_M_groups,
            e8m0_scales,
            input_group_offsets,
        )
        cuda_time = benchmark_kernel(
            mx_block_rearrange.mx_block_rearrange_2d_M_groups_cuda,
            e8m0_scales_row_major.view(torch.uint8),
            input_group_offsets,
            64,
            1,  # chunks_per_tb (1 for higher occupancy)
        )
        speedup = triton_time / cuda_time
        timing_info = (triton_time, cuda_time, speedup)
        print(
            f"    Timing: Triton={triton_time:.1f}us, CUDA={cuda_time:.1f}us, Speedup={speedup:.2f}x"
        )

    return True, timing_info


def run_all_tests():
    """Run all test configurations."""
    print("\n" + "=" * 80)
    print("Testing mx_block_rearrange_2d_M_groups kernel")
    print("=" * 80)

    # Select configurations based on args
    if args.quick:
        configs = QUICK_CONFIG
        print("Running QUICK test only")
    elif args.config is not None:
        if 0 <= args.config < len(TEST_CONFIGS):
            configs = [TEST_CONFIGS[args.config]]
            print(f"Running config [{args.config}]")
        else:
            print(
                f"ERROR: Config index {args.config} out of range (0-{len(TEST_CONFIGS)-1})"
            )
            return False
    else:
        configs = TEST_CONFIGS
        print(f"Running all {len(configs)} test configurations")

    run_benchmark = not args.no_benchmark

    # Track results
    passed = 0
    failed = 0
    results = []

    for i, (total_m, k, n_groups) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]", end="")
        success, timing = test_single_config(
            total_m, k, n_groups, run_benchmark=run_benchmark
        )

        if success:
            passed += 1
            results.append((total_m, k, n_groups, "PASS", timing))
        else:
            failed += 1
            results.append((total_m, k, n_groups, "FAIL", None))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Passed: {passed}/{len(configs)}")
    print(f"  Failed: {failed}/{len(configs)}")

    if run_benchmark and passed > 0:
        print("\nBenchmark Results:")
        print("-" * 80)
        print(
            f"  {'M':>8} {'K':>6} {'Groups':>6} {'Triton (us)':>12} {'CUDA (us)':>12} {'Speedup':>8}"
        )
        print("-" * 80)
        for total_m, k, n_groups, status, timing in results:
            if timing:
                triton_time, cuda_time, speedup = timing
                print(
                    f"  {total_m:>8} {k:>6} {n_groups:>6} {triton_time:>12.1f} {cuda_time:>12.1f} {speedup:>8.2f}x"
                )

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()

    print("\n" + "=" * 80)
    if success:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
