# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Microbenchmark script to compare Triton kernels vs torch._scaled_mm
for blockwise fp8 GEMM operations.
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


def prepare_blockwise_scaled_mm_tensors(
    a_fp8: torch.Tensor, 
    a_scale: torch.Tensor, 
    b_fp8: torch.Tensor, 
    b_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare tensors for torch._scaled_mm with proper layout and scaling.
    """
    # torch._scaled_mm expects reciprocal scales
    a_scale_recip = 1.0 / a_scale
    b_scale_recip = 1.0 / b_scale
    
    # Ensure proper memory layout for torch._scaled_mm
    # A should be row-major, B should be column-major or properly strided
    a_mm = a_fp8.contiguous()
    b_mm = b_fp8.contiguous() if b_fp8.is_contiguous() else b_fp8.t().contiguous().t()
    
    return a_mm, a_scale_recip, b_mm, b_scale_recip


def blockwise_fp8_scaled_mm_1x128_128x128_reference(
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor, 
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation using torch._scaled_mm for blockwise fp8 GEMM.
    This is a simplified version - the actual implementation needs to handle
    blockwise scaling properly.
    """
    a_mm, a_scale_recip, b_mm, b_scale_recip = prepare_blockwise_scaled_mm_tensors(
        a_fp8, a_scale, b_fp8, b_scale
    )
    
    # For now, use tensorwise scaling as a baseline comparison
    # The full blockwise implementation will need custom logic
    a_scale_tensor = a_scale_recip.mean()
    b_scale_tensor = b_scale_recip.mean()
    
    return torch._scaled_mm(
        a_mm,
        b_mm,
        scale_a=a_scale_tensor,
        scale_b=b_scale_tensor,
        out_dtype=torch.bfloat16,
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
    
    # Benchmark torch._scaled_mm (simplified reference)
    try:
        scaled_mm_time = benchmark_microseconds(
            blockwise_fp8_scaled_mm_1x128_128x128_reference,
            a_fp8, a_scale, b_fp8, b_scale
        )
        results["scaled_mm_time_us"] = scaled_mm_time
    except Exception as e:
        print(f"Warning: torch._scaled_mm benchmark failed: {e}")
        results["scaled_mm_time_us"] = float('inf')
    
    # Calculate speedups
    results["triton_speedup"] = bf16_time / triton_time if triton_time > 0 else 0
    results["scaled_mm_speedup"] = (
        bf16_time / results["scaled_mm_time_us"] 
        if results["scaled_mm_time_us"] > 0 and results["scaled_mm_time_us"] != float('inf')
        else 0
    )
    
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
    
    # torch._scaled_mm precision (simplified reference)
    try:
        scaled_mm_output = blockwise_fp8_scaled_mm_1x128_128x128_reference(
            a_fp8, a_scale, b_fp8, b_scale
        )
        results["scaled_mm_error_db"] = compute_error(ref_output, scaled_mm_output)
    except Exception as e:
        print(f"Warning: torch._scaled_mm precision test failed: {e}")
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
    
    print("Running precision benchmarks...")
    precision_results = []
    for m, k, n in tqdm(test_configs):
        if k % 128 == 0 and n % 128 == 0:
            try:
                result = benchmark_precision(m, k, n)
                precision_results.append(result)
            except Exception as e:
                print(f"Error in precision test {m}x{k}x{n}: {e}")
    
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