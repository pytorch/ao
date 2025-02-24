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
    B = torch.randint(-128, 127, size=(n, 4 * k // 8), dtype=torch.int8, device=dev)
    B_scale = torch.randn((n,), dtype=torch.half, device=dev)
    C = None

    return A, A_scale, B, B_scale, C


def get_blockwise_problem(m: int, n: int, k: int, block_size: int):
    assert (
        n % block_size == 0 and k % block_size == 0
    ), "N and K dims must be divisible by block_size"
    dev = torch.device("cuda")
    A = (448.0 * (2 * torch.rand(m, k, device=dev) - 1)).to(torch.float8_e4m3fn)
    A_scale = torch.randn((m, k // block_size), dtype=torch.half, device=dev)
    B = (448.0 * (2 * torch.rand(n, k, device=dev) - 1)).to(torch.float8_e4m3fn)
    B_scale = torch.randn(
        (n // block_size, k // block_size), dtype=torch.half, device=dev
    )

    return A, A_scale, B, B_scale


def benchmark_latency(m: int, k: int, n: int, block_size: int):
    dev = torch.device("cuda")
    A_ref = torch.randn((m, k), dtype=torch.half, device=dev)
    B_ref = torch.randn((n, k), dtype=torch.half, device=dev)
    fp16_time = benchmark_microseconds(torch.nn.functional.linear, A_ref, B_ref)

    A, A_scale, B, B_scale, C = get_rowwise_problem(m, n, k)
    rowwise_time = benchmark_microseconds(
        rowwise_scaled_linear_cutlass_s8s4, A, A_scale, B, B_scale, C
    )

    A, A_scale, B, B_scale = get_blockwise_problem(m, n, k, block_size)
    blockwise_time = benchmark_microseconds(blockwise_fp8_gemm, A, A_scale, B, B_scale)

    return {
        "m": m,
        "k": k,
        "n": n,
        "fp16_latency (ms)": fp16_time,
        "rowwise_latency (ms)": rowwise_time,
        "blockwise_latency (ms)": blockwise_time,
        "rowwise_speedup": fp16_time / rowwise_time,
        "blockwise_speedup": fp16_time / blockwise_time,
    }


def benchmark_precision(m: int, k: int, n: int, block_size: int):
    dev = torch.device("cuda")
    lin = torch.nn.Linear(k, n, False, dev, torch.half)
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    W = lin.weight
    output = A @ W.T

    A_q, A_s = fp8_blockwise_act_quant(A, block_size)
    W_q, W_s = fp8_blockwise_weight_quant(W, block_size)
    output_blockwise = blockwise_fp8_gemm(A_q, A_s, W_q, W_s)

    quantize_(lin, int8_dynamic_activation_int4_weight())
    output_rowwise = lin(A)

    return {
        "m": m,
        "k": k,
        "n": n,
        "error_rowwise (dB)": compute_error(output, output_rowwise),
        "error_blockwise (dB)": compute_error(output, output_blockwise),
    }


if __name__ == "__main__":
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (8192, 10240, 57344, 8192)
    block_size_vals = (128, 128, 128, 128)

    latency_results = []
    precision_results = []

    for m in tqdm([1 << i for i in range(10)]):
        for n, k, block_size in zip(n_vals, k_vals, block_size_vals):
            latency_results.append(benchmark_latency(m, k, n, block_size))
            precision_results.append(benchmark_precision(m, k, n, block_size))

    df_latency = pd.DataFrame(latency_results)
    df_precision = pd.DataFrame(precision_results)

    df_latency.to_csv("blockwise_triton_latency_results.csv", index=False)
    df_precision.to_csv("blockwise_triton_precision_results.csv", index=False)

    print(df_latency.to_markdown(index=False))
    print(df_precision.to_markdown(index=False))
