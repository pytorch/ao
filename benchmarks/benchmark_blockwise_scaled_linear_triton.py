# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import torch

if torch.cuda.is_available():
    import pandas as pd
    from tqdm import tqdm
    from triton.testing import do_bench

    from torchao.float8.float8_utils import compute_error
    from torchao.prototype.blockwise_fp8_inference.blockwise_quantization import (
        blockwise_fp8_gemm,
        fp8_blockwise_act_quant,
        fp8_blockwise_weight_quant,
    )
    # Import training kernels for comparison
    from torchao.prototype.blockwise_fp8_training.kernels import (
        blockwise_fp8_gemm_1x128_128x128,
        fp8_blockwise_act_quant_lhs,
        fp8_blockwise_weight_quant_transposed_rhs,
    )
    from torchao.prototype.blockwise_fp8_training.scaled_mm_kernels import (
        blockwise_fp8_gemm_scaled_mm_1x128_128x128,
    )
    from torchao.prototype.blockwise_fp8_training.linear import (
        Float8BlockwiseLinear,
    )
    from torchao.utils import is_sm_at_least_89
else:
    raise RuntimeError("This benchmark is only avaible on CUDA hardware")


def benchmark_microseconds(f, *args, warmup=25, rep=100):
    return (
        do_bench(lambda: f(*args), warmup=warmup, rep=rep, return_mode="median") * 1e3
    )


def get_blockwise_problem(
    m: int, n: int, k: int, block_size: int, dtype: torch.dtype, device
):
    assert n % block_size == 0 and k % block_size == 0, (
        "N and K dims must be divisible by block_size"
    )
    assert dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ], "dtype must be torch.float8_e4m3fn or torch.float8_e5m2"
    dtype_max = torch.finfo(dtype).max
    A = (dtype_max * (2 * torch.rand(m, k, device=device) - 1)).to(dtype)
    A_scale = torch.randn((m, k // block_size), dtype=torch.half, device=device)
    B = (dtype_max * (2 * torch.rand(n, k, device=device) - 1)).to(dtype)
    B_scale = torch.randn(
        (n // block_size, k // block_size), dtype=torch.half, device=device
    )

    return A, A_scale, B, B_scale


def benchmark_latency(
    m: int, k: int, n: int, block_size: int, dtype: torch.dtype, device
):
    A_ref = torch.randn((m, k), dtype=torch.half, device=device)
    B_ref = torch.randn((n, k), dtype=torch.half, device=device)
    fp16_time = benchmark_microseconds(torch.nn.functional.linear, A_ref, B_ref)

    A, A_scale, B, B_scale = get_blockwise_problem(m, n, k, block_size, dtype, device)
    blockwise_time = benchmark_microseconds(
        blockwise_fp8_gemm, A, A_scale, B, B_scale, block_size
    )

    return {
        "m": m,
        "k": k,
        "n": n,
        "block_size": block_size,
        "dtype": dtype,
        "fp16_latency (ms)": fp16_time,
        "blockwise_latency (ms)": blockwise_time,
        "blockwise_speedup": fp16_time / blockwise_time,
    }


def benchmark_training_kernels_latency(
    m: int, k: int, n: int, block_size: int, dtype: torch.dtype, device
):
    """Benchmark training kernels: Triton vs torch._scaled_mm implementations."""
    # Create reference tensors
    A_ref = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    B_ref = torch.randn((k, n), dtype=torch.bfloat16, device=device) 
    fp16_time = benchmark_microseconds(torch.nn.functional.linear, A_ref, B_ref)

    # Create quantized inputs for training kernels
    A_fp8, A_scale = fp8_blockwise_act_quant_lhs(A_ref, block_size)
    B_fp8, B_scale = fp8_blockwise_weight_quant_transposed_rhs(B_ref, block_size)

    # Benchmark Triton training kernel
    try:
        triton_time = benchmark_microseconds(
            blockwise_fp8_gemm_1x128_128x128,
            A_fp8, 1.0 / A_scale, B_fp8, 1.0 / B_scale
        )
    except Exception as e:
        print(f"Triton kernel failed: {e}")
        triton_time = float('inf')

    # Benchmark torch._scaled_mm training kernel
    try:
        scaled_mm_time = benchmark_microseconds(
            blockwise_fp8_gemm_scaled_mm_1x128_128x128,
            A_fp8, 1.0 / A_scale, B_fp8, 1.0 / B_scale, block_size
        )
    except Exception as e:
        print(f"Scaled MM kernel failed: {e}")
        scaled_mm_time = float('inf')

    return {
        "m": m,
        "k": k, 
        "n": n,
        "block_size": block_size,
        "dtype": dtype,
        "fp16_latency (ms)": fp16_time,
        "triton_training_latency (ms)": triton_time,
        "scaled_mm_training_latency (ms)": scaled_mm_time,
        "triton_training_speedup": fp16_time / triton_time if triton_time != float('inf') else 0,
        "scaled_mm_training_speedup": fp16_time / scaled_mm_time if scaled_mm_time != float('inf') else 0,
        "scaled_mm_vs_triton_speedup": triton_time / scaled_mm_time if triton_time != float('inf') and scaled_mm_time != float('inf') else 0,
    }


def benchmark_precision(
    m: int, k: int, n: int, block_size: int, dtype: torch.dtype, device
):
    lin = torch.nn.Linear(k, n, False, device, torch.half)
    A = torch.randn((m, k), dtype=torch.half, device=device)
    W = lin.weight
    output = A @ W.T

    A_q, A_s = fp8_blockwise_act_quant(A, block_size, dtype)
    W_q, W_s = fp8_blockwise_weight_quant(W, block_size, dtype)
    output_blockwise = blockwise_fp8_gemm(A_q, A_s, W_q, W_s, block_size)

    return {
        "m": m,
        "k": k,
        "n": n,
        "block_size": block_size,
        "dtype": dtype,
        "error_blockwise (dB)": compute_error(output, output_blockwise),
    }


def benchmark_training_kernels_precision(
    m: int, k: int, n: int, block_size: int, dtype: torch.dtype, device
):
    """Benchmark precision of training kernels: Triton vs torch._scaled_mm."""
    # Create high precision reference
    A_ref = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    B_ref = torch.randn((k, n), dtype=torch.bfloat16, device=device)
    ref_output = torch.nn.functional.linear(A_ref, B_ref)

    # Create quantized inputs
    A_fp8, A_scale = fp8_blockwise_act_quant_lhs(A_ref, block_size)
    B_fp8, B_scale = fp8_blockwise_weight_quant_transposed_rhs(B_ref, block_size)

    results = {
        "m": m, "k": k, "n": n, "block_size": block_size, "dtype": dtype
    }

    # Test Triton kernel
    try:
        triton_output = blockwise_fp8_gemm_1x128_128x128(
            A_fp8, 1.0 / A_scale, B_fp8, 1.0 / B_scale
        )
        results["triton_error_db"] = compute_error(ref_output, triton_output)
    except Exception as e:
        print(f"Triton precision test failed: {e}")
        results["triton_error_db"] = float('inf')

    # Test torch._scaled_mm kernel
    try:
        scaled_mm_output = blockwise_fp8_gemm_scaled_mm_1x128_128x128(
            A_fp8, 1.0 / A_scale, B_fp8, 1.0 / B_scale, block_size
        )
        results["scaled_mm_error_db"] = compute_error(ref_output, scaled_mm_output)
    except Exception as e:
        print(f"Scaled MM precision test failed: {e}")
        results["scaled_mm_error_db"] = float('inf')

    # Compare the two implementations
    if results["triton_error_db"] != float('inf') and results["scaled_mm_error_db"] != float('inf'):
        try:
            triton_output = blockwise_fp8_gemm_1x128_128x128(
                A_fp8, 1.0 / A_scale, B_fp8, 1.0 / B_scale
            )
            scaled_mm_output = blockwise_fp8_gemm_scaled_mm_1x128_128x128(
                A_fp8, 1.0 / A_scale, B_fp8, 1.0 / B_scale, block_size
            )
            results["triton_vs_scaled_mm_error_db"] = compute_error(triton_output, scaled_mm_output)
        except Exception:
            results["triton_vs_scaled_mm_error_db"] = float('inf')
    else:
        results["triton_vs_scaled_mm_error_db"] = float('inf')

    return results


if __name__ == "__main__" and torch.cuda.is_available():
    device = torch.device("cuda")
    
    # Original inference benchmark configurations
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (8192, 10240, 57344, 8192)
    block_size_vals = (128, 128, 128, 128)
    
    # Training kernel benchmark configurations (smaller set for faster testing)
    training_configs = [
        (1, 4096, 4096),    # Single token
        (32, 4096, 4096),   # Small batch
        (8, 4096, 11008),   # MLP up projection  
        (8, 11008, 4096),   # MLP down projection
        (1, 4096, 128256),  # Vocab projection (if memory allows)
    ]

    latency_results = []
    precision_results = []
    training_latency_results = []
    training_precision_results = []

    available_dtypes = (
        [torch.float8_e4m3fn, torch.float8_e5m2]
        if is_sm_at_least_89()
        else [torch.float8_e5m2]
    )
    
    print("Running original inference benchmarks...")
    for m in tqdm([1 << i for i in range(14)]):
        for dtype in available_dtypes:
            for n, k, block_size in zip(n_vals, k_vals, block_size_vals):
                latency_results.append(
                    benchmark_latency(m, k, n, block_size, dtype, device)
                )
                precision_results.append(
                    benchmark_precision(m, k, n, block_size, dtype, device)
                )
    
    print("Running training kernel benchmarks...")
    for m, k, n in tqdm(training_configs):
        # Only test on fp8_e4m3fn for training (most common)
        if k % 128 == 0 and n % 128 == 0:  # Ensure divisibility
            try:
                training_latency_results.append(
                    benchmark_training_kernels_latency(m, k, n, 128, torch.float8_e4m3fn, device)
                )
                training_precision_results.append(
                    benchmark_training_kernels_precision(m, k, n, 128, torch.float8_e4m3fn, device)
                )
            except Exception as e:
                print(f"Skipping training config ({m}, {k}, {n}): {e}")

    # Save results
    if latency_results:
        df_latency = pd.DataFrame(latency_results)
        df_latency.to_csv("blockwise_triton_inference_latency_results.csv", index=False)
        print("\nInference Latency Results:")
        print(df_latency.to_markdown(index=False))

    if precision_results:
        df_precision = pd.DataFrame(precision_results)
        df_precision.to_csv("blockwise_triton_inference_precision_results.csv", index=False)
        print("\nInference Precision Results:")
        print(df_precision.to_markdown(index=False))
    
    if training_latency_results:
        df_training_latency = pd.DataFrame(training_latency_results)
        df_training_latency.to_csv("blockwise_training_kernels_latency_results.csv", index=False)
        print("\nTraining Kernels Latency Results:")
        print(df_training_latency.to_markdown(index=False))
    
    if training_precision_results:
        df_training_precision = pd.DataFrame(training_precision_results)
        df_training_precision.to_csv("blockwise_training_kernels_precision_results.csv", index=False)
        print("\nTraining Kernels Precision Results:")
        print(df_training_precision.to_markdown(index=False))
