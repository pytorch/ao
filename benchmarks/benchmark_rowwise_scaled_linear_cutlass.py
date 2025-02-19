import pandas as pd
import torch
from tqdm import tqdm
from triton.testing import do_bench

from torchao.ops import (
    rowwise_scaled_linear_cutlass_s4s4,
    rowwise_scaled_linear_cutlass_s8s4,
)


def benchmark_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def get_problem(m: int, n: int, k: int, A_nbits: int, B_nbits: int):
    assert A_nbits in (4, 8) and B_nbits in (4, 8)

    dev = torch.device("cuda")
    A = torch.randint(-128, 127, (m, k * A_nbits // 8), dtype=torch.int8, device=dev)
    A_scale = torch.randn((m,), dtype=torch.half, device=dev)
    B = torch.randint(
        -128, 127, size=(n, k * B_nbits // 8), dtype=torch.int8, device=dev
    )
    B_scale = torch.randn((n,), dtype=torch.half, device=dev)
    C = None

    return A, A_scale, B, B_scale, C


def benchmark(m: int, k: int, n: int):
    dev = torch.device("cuda")
    A_ref = torch.randn((m, k), dtype=torch.half, device=dev)
    B_ref = torch.randn((n, k), dtype=torch.half, device=dev)
    fp16_time = benchmark_microseconds(torch.nn.functional.linear, A_ref, B_ref)

    A, A_scale, B, B_scale, C = get_problem(m, n, k, 8, 4)
    rowwise_scaled_linear_cutlass_s8s4_time = benchmark_microseconds(
        rowwise_scaled_linear_cutlass_s8s4, A, A_scale, B, B_scale, C
    )

    A, A_scale, B, B_scale, C = get_problem(m, n, k, 4, 4)
    rowwise_scaled_linear_cutlass_s4s4_time = benchmark_microseconds(
        rowwise_scaled_linear_cutlass_s4s4, A, A_scale, B, B_scale, C
    )

    return {
        "m": m,
        "k": k,
        "n": n,
        "fp16_latency (ms)": fp16_time,
        "rowwise_scaled_linear_cutlass_s8s4 latency (ms)": rowwise_scaled_linear_cutlass_s8s4_time,
        "s8s4 speedup (d/s)": fp16_time / rowwise_scaled_linear_cutlass_s8s4_time,
        "rowwise_scaled_linear_cutlass_s4s4 latency (ms)": rowwise_scaled_linear_cutlass_s4s4_time,
        "s4s4 speedup (d/s)": fp16_time / rowwise_scaled_linear_cutlass_s4s4_time,
    }


if __name__ == "__main__":
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (8192, 10240, 57344, 8192)

    results = []
    for m in tqdm([1 << i for i in range(10)]):
        for n, k in zip(n_vals, k_vals):
            results.append(benchmark(m, k, n))

    df = pd.DataFrame(results)
    df.to_csv("rowwise_scaled_linear_cutlass_time_results.csv", index=False)
    print(df.to_markdown(index=False))
