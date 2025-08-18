# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Microbenchmark script to compare Triton kernels vs torch._scaled_mm native blockwise scaling
for blockwise fp8 GEMM operations.

This provides a proper 1:1 comparison between the Triton blockwise implementation
and the torch._scaled_mm native blockwise scaling with CUDA 12.9+, as recommended 
by danielvegamyhre to avoid uncoalesced memory access issues in Triton kernels.
"""

import torch
import pandas as pd
from typing import Tuple, Optional
from tqdm import tqdm

if torch.cuda.is_available():
    from triton.testing import do_bench
    from torchao.float8.float8_utils import compute_error
    from torchao.prototype.blockwise_fp8_training.kernels import (
        blockwise_fp8_gemm_1x128_128x128,
        blockwise_fp8_gemm_1x128_128x1,
        fp8_blockwise_act_quant_lhs,
        fp8_blockwise_act_quant_rhs,
        fp8_blockwise_act_quant_transposed_lhs,
        fp8_blockwise_weight_quant_rhs,
        fp8_blockwise_weight_quant_transposed_rhs,
    )
    from torchao.utils import is_sm_at_least_90
else:
    raise RuntimeError("This benchmark is only available on CUDA hardware")


def benchmark_microseconds(f, *args, warmup=25, rep=100):
    """Benchmark function in microseconds"""
    return (
        do_bench(lambda: f(*args), warmup=warmup, rep=rep, return_mode="median") * 1e3
    )




def blockwise_fp8_scaled_mm_1x128_128x128_reference(
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor, 
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation using native torch._scaled_mm with blockwise scaling.
    This uses the CUDA 12.9+ native blockwise scaling support to provide optimal
    performance through direct CUTLASS kernel usage, avoiding the uncoalesced
    memory access issues present in Triton kernels.
    """
    from torchao.prototype.blockwise_fp8_training.scaled_mm_kernels import (
        blockwise_fp8_gemm_scaled_mm_1x128_128x128
    )
    
    return blockwise_fp8_gemm_scaled_mm_1x128_128x128(
        a_fp8, 
        1.0 / a_scale, 
        b_fp8, 
        1.0 / b_scale, 
        block_size=128
    )


def create_test_tensors(
    m: int, k: int, n: int, block_size: int = 128, device="cuda"
) -> Tuple:
    """Create test tensors for benchmarking"""
    # Create high precision reference tensors
    a_ref = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b_ref = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    
    # Quantize activation (A) with 1x128 blockwise scaling
    a_fp8, a_scale = fp8_blockwise_act_quant_lhs(a_ref, block_size)
    
    # Quantize weight (B) with 128x128 blockwise scaling, transposed dims in column major
    b_fp8, b_scale = fp8_blockwise_weight_quant_transposed_rhs(b_ref, block_size)
    
    return a_ref, b_ref, a_fp8, a_scale, b_fp8, b_scale


def create_test_tensors_128x1(
    m: int, k: int, n: int, block_size: int = 128, device="cuda"
):
    """Create test tensors for 1x128 (LHS) x 128x1 (RHS) blockwise GEMM."""
    # High-precision reference tensors
    a_ref = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b_ref = torch.randn(k, n, device=device, dtype=torch.bfloat16)

    # LHS: use transposed-lhs quantization. Input to that kernel should be KxM
    a_t = a_ref.t().contiguous()
    a_fp8, a_scale = fp8_blockwise_act_quant_transposed_lhs(a_t, block_size)

    # RHS: 128x1 scaling along K
    b_fp8, b_scale = fp8_blockwise_act_quant_rhs(b_ref, block_size)

    return a_ref, b_ref, a_fp8, a_scale, b_fp8, b_scale


def benchmark_gemm_variants(
    m: int, k: int, n: int, block_size: int = 128, device="cuda"
) -> dict:
    """Benchmark different GEMM implementations"""
    
    # Create test tensors
    a_ref, b_ref, a_fp8, a_scale, b_fp8, b_scale = create_test_tensors(
        m, k, n, block_size, device
    )
    
    results = {
        "m": m, "k": k, "n": n, "block_size": block_size
    }
    
    # Benchmark reference bf16 GEMM
    bf16_time = benchmark_microseconds(torch.nn.functional.linear, a_ref, b_ref)
    results["bf16_time_us"] = bf16_time
    
    # Benchmark Triton blockwise fp8 GEMM
    triton_time = benchmark_microseconds(
        blockwise_fp8_gemm_1x128_128x128,
        a_fp8, 1.0 / a_scale, b_fp8, 1.0 / b_scale, block_size
    )
    results["triton_time_us"] = triton_time
    
    # Benchmark torch._scaled_mm (native blockwise scaling with CUDA 12.9+)
    try:
        scaled_mm_time = benchmark_microseconds(
            blockwise_fp8_scaled_mm_1x128_128x128_reference,
            a_fp8, a_scale, b_fp8, b_scale
        )
        results["scaled_mm_time_us"] = scaled_mm_time
    except Exception as e:
        print(f"Warning: torch._scaled_mm native blockwise benchmark failed: {e}")
        print(f"Note: Requires CUDA 12.9+ for native blockwise scaling support")
        results["scaled_mm_time_us"] = float('inf')
    
    # Calculate speedups
    results["triton_speedup"] = bf16_time / triton_time if triton_time > 0 else 0
    results["scaled_mm_speedup"] = (
        bf16_time / results["scaled_mm_time_us"] 
        if results["scaled_mm_time_us"] > 0 and results["scaled_mm_time_us"] != float('inf')
        else 0
    )
    
    return results


def blockwise_fp8_scaled_mm_1x128_128x1_reference(
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    from torchao.prototype.blockwise_fp8_training.scaled_mm_kernels import (
        blockwise_fp8_gemm_scaled_mm_1x128_128x1,
    )
    return blockwise_fp8_gemm_scaled_mm_1x128_128x1(
        a_fp8,
        1.0 / a_scale,
        b_fp8,
        1.0 / b_scale,
        block_size,
    )


def benchmark_gemm_variants_128x1(
    m: int, k: int, n: int, block_size: int = 128, device="cuda"
) -> dict:
    """Benchmark 1x128 (LHS) x 128x1 (RHS) blockwise GEMM variants."""
    a_ref, b_ref, a_fp8, a_scale, b_fp8, b_scale = create_test_tensors_128x1(
        m, k, n, block_size, device
    )

    results = {"m": m, "k": k, "n": n, "block_size": block_size, "case": "1x128_128x1"}

    # Reference bf16 GEMM
    bf16_time = benchmark_microseconds(torch.nn.functional.linear, a_ref, b_ref)
    results["bf16_time_us"] = bf16_time

    # Triton
    triton_time = benchmark_microseconds(
        blockwise_fp8_gemm_1x128_128x1,
        a_fp8, 1.0 / a_scale, b_fp8, 1.0 / b_scale, block_size
    )
    results["triton_time_us"] = triton_time

    # Native torch._scaled_mm
    try:
        scaled_mm_time = benchmark_microseconds(
            blockwise_fp8_scaled_mm_1x128_128x1_reference,
            a_fp8, a_scale, b_fp8, b_scale, block_size
        )
        results["scaled_mm_time_us"] = scaled_mm_time
    except Exception as e:
        print(f"Warning: torch._scaled_mm native blockwise 128x1 benchmark failed: {e}")
        results["scaled_mm_time_us"] = float('inf')

    results["triton_speedup"] = bf16_time / triton_time if triton_time > 0 else 0
    sm_time = results["scaled_mm_time_us"]
    results["scaled_mm_speedup"] = bf16_time / sm_time if sm_time not in (0, float('inf')) else 0
    return results


def benchmark_precision(
    m: int, k: int, n: int, block_size: int = 128, device="cuda"
) -> dict:
    """Benchmark numerical precision of different implementations"""
    
    # Create test tensors  
    a_ref, b_ref, a_fp8, a_scale, b_fp8, b_scale = create_test_tensors(
        m, k, n, block_size, device
    )
    
    # Reference computation
    ref_output = torch.nn.functional.linear(a_ref, b_ref)
    
    # Triton blockwise fp8 computation
    triton_output = blockwise_fp8_gemm_1x128_128x128(
        a_fp8, 1.0 / a_scale, b_fp8, 1.0 / b_scale, block_size
    )
    
    results = {
        "m": m, "k": k, "n": n, "block_size": block_size,
        "triton_error_db": compute_error(ref_output, triton_output),
    }
    
    # torch._scaled_mm precision (native blockwise scaling)
    try:
        scaled_mm_output = blockwise_fp8_scaled_mm_1x128_128x128_reference(
            a_fp8, a_scale, b_fp8, b_scale
        )
        results["scaled_mm_error_db"] = compute_error(ref_output, scaled_mm_output)
    except Exception as e:
        print(f"Warning: torch._scaled_mm native blockwise precision test failed: {e}")
        print(f"Note: Requires CUDA 12.9+ for native blockwise scaling support")
        results["scaled_mm_error_db"] = float('inf')
    
    return results


def benchmark_precision_128x1(
    m: int, k: int, n: int, block_size: int = 128, device="cuda"
) -> dict:
    """Precision benchmark for 1x128 x 128x1."""
    a_ref, b_ref, a_fp8, a_scale, b_fp8, b_scale = create_test_tensors_128x1(
        m, k, n, block_size, device
    )

    ref_output = torch.nn.functional.linear(a_ref, b_ref)

    results = {"m": m, "k": k, "n": n, "block_size": block_size, "case": "1x128_128x1"}

    # Triton
    triton_output = blockwise_fp8_gemm_1x128_128x1(
        a_fp8, 1.0 / a_scale, b_fp8, 1.0 / b_scale, block_size
    )

    from torchao.float8.float8_utils import compute_error
    results["triton_error_db"] = compute_error(ref_output, triton_output)

    # Native torch._scaled_mm
    try:
        scaled_mm_output = blockwise_fp8_scaled_mm_1x128_128x1_reference(
            a_fp8, a_scale, b_fp8, b_scale, block_size
        )
        results["scaled_mm_error_db"] = compute_error(ref_output, scaled_mm_output)
    except Exception as e:
        print(f"Warning: torch._scaled_mm native blockwise 128x1 precision failed: {e}")
        results["scaled_mm_error_db"] = float('inf')

    return results


def run_benchmarks():
    """Run comprehensive benchmarks"""
    if not is_sm_at_least_90():
        print("Warning: This benchmark requires SM90 or higher for optimal performance")
    
    # Test configurations - various matrix sizes commonly used in LLMs
    test_configs = [
        # (M, K, N) - batch_size x hidden_dim x output_dim
        (1, 4096, 4096),      # Single token
        (32, 4096, 4096),     # Small batch
        (128, 4096, 4096),    # Medium batch  
        (1, 4096, 11008),     # MLP up projection
        (32, 4096, 11008),    # MLP up projection, batched
        (1, 11008, 4096),     # MLP down projection
        (32, 11008, 4096),    # MLP down projection, batched
        (1, 4096, 128256),    # Vocab projection
        (32, 4096, 128256),   # Vocab projection, batched
    ]
    
    print("Running performance benchmarks...")
    perf_results = []
    for m, k, n in tqdm(test_configs):
        if k % 128 == 0 and n % 128 == 0:  # Ensure divisibility by block size
            try:
                result = benchmark_gemm_variants(m, k, n)
                perf_results.append(result)
            except Exception as e:
                print(f"Error benchmarking {m}x{k}x{n}: {e}")
            try:
                result_128x1 = benchmark_gemm_variants_128x1(m, k, n)
                perf_results.append(result_128x1)
            except Exception as e:
                print(f"Error benchmarking 128x1 {m}x{k}x{n}: {e}")
    
    print("Running precision benchmarks...")
    precision_results = []
    for m, k, n in tqdm(test_configs):
        if k % 128 == 0 and n % 128 == 0:
            try:
                result = benchmark_precision(m, k, n)
                precision_results.append(result)
            except Exception as e:
                print(f"Error in precision test {m}x{k}x{n}: {e}")
            try:
                result_128x1 = benchmark_precision_128x1(m, k, n)
                precision_results.append(result_128x1)
            except Exception as e:
                print(f"Error in 128x1 precision test {m}x{k}x{n}: {e}")
    
    # Save and display results
    if perf_results:
        perf_df = pd.DataFrame(perf_results)
        perf_df.to_csv("triton_vs_scaled_mm_performance.csv", index=False)
        print("\nPerformance Results:")
        print(perf_df.to_markdown(index=False))
    
    if precision_results:
        precision_df = pd.DataFrame(precision_results)
        precision_df.to_csv("triton_vs_scaled_mm_precision.csv", index=False)
        print("\nPrecision Results:")
        print(precision_df.to_markdown(index=False))


if __name__ == "__main__":
    if torch.cuda.is_available():
        run_benchmarks()
    else:
        print("CUDA not available. Skipping benchmarks.")
