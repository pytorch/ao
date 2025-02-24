import pandas as pd
import torch
from tqdm import tqdm
from triton.testing import do_bench

from torchao.ops import (
    rowwise_scaled_linear_sparse_cutlass_f8f8,
    to_sparse_semi_structured_cutlass_sm9x_f8,
)


def benchmark_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def get_problem(m: int, n: int, k: int):
    dev = torch.device("cuda")

    A = torch.randn((m, k), dtype=torch.half, device=dev).to(torch.float8_e5m2)
    A_scale = torch.randn((m,), dtype=torch.half, device=dev)
    B = torch.randn((n, k), dtype=torch.half, device=dev).to(torch.float8_e4m3fn)
    B_sp, B_meta = to_sparse_semi_structured_cutlass_sm9x_f8(B)
    B_scale = torch.randn((n,), dtype=torch.half, device=dev)
    C = None

    return A, A_scale, B_sp, B_meta, B_scale, C


def benchmark(m: int, k: int, n: int):
    dev = torch.device("cuda")
    A_ref = torch.randn((m, k), dtype=torch.half, device=dev)
    B_ref = torch.randn((n, k), dtype=torch.half, device=dev)
    fp16_time = benchmark_microseconds(torch.nn.functional.linear, A_ref, B_ref)

    A, A_scale, B_sp, B_meta, B_scale, C = get_problem(m, n, k)
    rowwise_scaled_linear_sparse_cutlass_f8f8_time = benchmark_microseconds(
        rowwise_scaled_linear_sparse_cutlass_f8f8, A, A_scale, B_sp, B_meta, B_scale, C
    )

    return {
        "m": m,
        "k": k,
        "n": n,
        "fp16_latency (ms)": fp16_time,
        "rowwise_scaled_linear_sparse_cutlass_f8f8 latency (ms)": rowwise_scaled_linear_sparse_cutlass_f8f8_time,
        "f8f8 speedup (d/s)": fp16_time
        / rowwise_scaled_linear_sparse_cutlass_f8f8_time,
    }


if __name__ == "__main__":
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (8192, 10240, 57344, 8192)

    results = []
    for m in tqdm([1 << i for i in range(10)]):
        for n, k in zip(n_vals, k_vals):
            results.append(benchmark(m, k, n))

    df = pd.DataFrame(results)
    df.to_csv("rowwise_scaled_linear_sparse_cutlass_time_results.csv", index=False)
    print(df.to_markdown(index=False))
