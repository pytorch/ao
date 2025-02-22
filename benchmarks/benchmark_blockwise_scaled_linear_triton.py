import pandas as pd
import torch
from tqdm import tqdm
from triton.testing import do_bench

from torchao.float8.float8_utils import compute_error
from torchao.ops import rowwise_scaled_linear_cutlass_s8s4
from torchao.prototype.blockwise_fp8.blockwise_fp8_gemm_triton import blockwise_fp8_gemm
from torchao.prototype.blockwise_fp8.blockwise_quantization import (
    fp8_blockwise_act_quant,
    fp8_blockwise_weight_quant,
)
from torchao.quantization.quant_api import (
    int8_dynamic_activation_int4_weight,
    quantize_,
)


def benchmark_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3

def get_rowwise_problem(m: int, n: int, k: int):
    dev = torch.device("cuda")
    A = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=dev)
    A_scale = torch.randn((m,), dtype=torch.half, device=dev)
    B = torch.randint(
        -128, 127, size=(n, 4 * k // 8), dtype=torch.int8, device=dev
    )
    B_scale = torch.randn((n,), dtype=torch.half, device=dev)
    C = None

    return A, A_scale, B, B_scale, C

def get_blockwise_problem(m: int, n: int, k: int, block_size: int):
    assert n % block_size == 0 and k % block_size == 0, "N and K dims must be divisible by block_size"
    dev = torch.device("cuda")
    A = (448.0 * (2 * torch.rand(m, k, device=dev) - 1)).to(torch.float8_e4m3fn)
    A_scale = torch.randn((m, k // block_size), dtype=torch.half, device=dev)
    B = (448.0 * (2 * torch.rand(n, k, device=dev) - 1)).to(torch.float8_e4m3fn)
    B_scale = torch.randn((n // block_size, k // block_size), dtype=torch.half, device=dev)

    return A, A_scale, B, B_scale

def benchmark(m: int, k: int, n: int, block_size: int):
    # Speed benchmark
    dev = torch.device("cuda")
    A_ref = torch.randn((m, k), dtype=torch.half, device=dev)
    B_ref = torch.randn((n, k), dtype=torch.half, device=dev)
    fp16_time = benchmark_microseconds(torch.nn.functional.linear, A_ref, B_ref)

    A, A_scale, B, B_scale, C = get_rowwise_problem(m, n, k)
    rowwise_scaled_linear_cutlass_s8s4_time = benchmark_microseconds(
        rowwise_scaled_linear_cutlass_s8s4, A, A_scale, B, B_scale, C
    )

    A, A_scale, B, B_scale = get_blockwise_problem(m, n, k, block_size)
    blockwise_fp8_gemm_time = benchmark_microseconds(
        blockwise_fp8_gemm, A, A_scale, B, B_scale
    )

    # Precision benchmark
    lin = torch.nn.Linear(k, n, False, dev, torch.half)
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    W = lin.weight
    output = A @ W.T

    A_q, A_s = fp8_blockwise_act_quant(A, block_size)
    W_q, W_s = fp8_blockwise_weight_quant(W, block_size)
    output_blockwise_quant = blockwise_fp8_gemm(A_q, A_s, W_q, W_s)

    quantize_(lin, int8_dynamic_activation_int4_weight())
    output_rowwise_quant = lin(A)

    error_rowwise_quant = compute_error(output, output_rowwise_quant)
    error_blockwise_quant = compute_error(output, output_blockwise_quant)

    return {
        "m": m,
        "k": k,
        "n": n,
        "fp16_latency (ms)": fp16_time,
        "rowwise_scaled_linear_cutlass_s8s4 latency (ms)": rowwise_scaled_linear_cutlass_s8s4_time,
        "rowwise s8s4 speedup (d/s)": fp16_time / rowwise_scaled_linear_cutlass_s8s4_time,
        "blockwise_fp8_gemm latency (ms)": blockwise_fp8_gemm_time,
        "blockwise fp8 speedup (d/s)": fp16_time / blockwise_fp8_gemm_time,
        "error_rowwise_quant (dB)": error_rowwise_quant,
        "error_blockwise_quant (dB)": error_blockwise_quant
    }

if __name__ == "__main__":
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (8192, 10240, 57344, 8192)
    block_size_vals = (128, 128, 128, 128)

    results = []
    for m in tqdm([1 << i for i in range(10)]):
        for n, k, block_size in zip(n_vals, k_vals, block_size_vals):
            results.append(benchmark(m, k, n, block_size))

    df = pd.DataFrame(results)
    df.to_csv("blockwise_scaled_linear_triton_results.csv", index=False)
    print(df.to_markdown(index=False))