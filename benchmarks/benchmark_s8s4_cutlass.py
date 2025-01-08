import pandas as pd
import torch
from tqdm import tqdm

from torchao.ops import s8s4_linear_cutlass
from torchao.utils import benchmark_torch_function_in_microseconds


def get_problem(m, n, k):
    dev = torch.device("cuda")
    A_ref = torch.randn((m, k), dtype=torch.half, device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)

    A = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=dev)
    A_scale = torch.randn((m,), dtype=torch.half, device=dev)
    B = torch.randint(-128, 127, size=(n, k // 2), dtype=torch.int8, device=dev)
    B_scale = torch.randn((n,), dtype=torch.half, device=dev)
    C = None

    return A_ref, B_ref, A, A_scale, B, B_scale, C


def benchmark(m: int, k: int, n: int):
    A_ref, B_ref, A, A_scale, B, B_scale, C = get_problem(m, n, k)

    fp16_time = benchmark_torch_function_in_microseconds(torch.matmul, A_ref, B_ref)
    s8s4_linear_cutlass_time = benchmark_torch_function_in_microseconds(
        s8s4_linear_cutlass, A, A_scale, B, B_scale, C
    )

    return {
        "m": m,
        "k": k,
        "n": n,
        "fp16_latency (ms)": fp16_time,
        "s8s4_linear_cutlass latency (ms)": s8s4_linear_cutlass_time,
        "speedup (d/s)": fp16_time / s8s4_linear_cutlass_time,
    }


if __name__ == "__main__":
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (8192, 10240, 57344, 8192)

    results = []
    for m in tqdm([1 << i for i in range(10)]):
        for n, k in zip(n_vals, k_vals):
            results.append(benchmark(m, k, n))

    df = pd.DataFrame(results)
    df.to_csv("s8s4_linear_cutlass_time_results.csv", index=False)
    print(df.to_markdown(index=False))
