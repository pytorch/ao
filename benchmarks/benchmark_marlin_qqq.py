# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import torch
from tqdm import tqdm

from torchao.ops import marlin_qqq_gemm
from torchao.quantization.marlin_qqq import marlin_qqq_workspace, pack_to_marlin_qqq
from torchao.utils import benchmark_torch_function_in_microseconds


def get_problem(m, n, k, groupsize=-1):
    if groupsize == -1:
        groupsize = k
    dev = torch.device("cuda")
    A_ref = torch.randn((m, k), dtype=torch.half, device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)

    A = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=dev)
    B = torch.randint(low=-(2**31), high=2**31, size=(k, n), device=dev)
    s_tok = torch.ones((m, 1), dtype=torch.float, device=dev)
    if groupsize == k:
        s_group = torch.tensor([], dtype=torch.half, device=dev)
    else:
        s_group = torch.ones((k // groupsize, n), dtype=torch.half, device=dev)
    s_channel = torch.ones((1, n), dtype=torch.float, device=dev)
    B, s_group, s_channel = pack_to_marlin_qqq(
        B, s_group, s_channel, num_bits=4, group_size=group_size
    )
    qqq_workspace = marlin_qqq_workspace(n)
    return A, B, A_ref, B_ref, s_tok, s_channel, s_group, qqq_workspace


def benchmark(m: int, k: int, n: int, group_size: int):
    A, B, A_ref, B_ref, s_tok, s_channel, s_group, qqq_workspace = get_problem(
        m, n, k, group_size
    )

    fp16_time = benchmark_torch_function_in_microseconds(torch.matmul, A_ref, B_ref)
    marlin_qqq_w4a8_time = benchmark_torch_function_in_microseconds(
        marlin_qqq_gemm, A, B, s_tok, s_channel, s_group, qqq_workspace, m, n, k
    )

    return {
        "m": m,
        "k": k,
        "n": n,
        "group_size": group_size,
        "fp16_latency (ms)": fp16_time,
        "marlin_qqq_w4a8_latency (ms)": marlin_qqq_w4a8_time,
        "speedup (d/s)": fp16_time / marlin_qqq_w4a8_time,
    }


if __name__ == "__main__":
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (8192, 10240, 57344, 8192)

    results = []
    for group_size in tqdm([-1, 128]):
        for m in tqdm([1 << i for i in range(10)]):
            for n, k in zip(n_vals, k_vals):
                results.append(benchmark(m, k, n, group_size))

    df = pd.DataFrame(results)
    df.to_csv("marlin_qqq_w4a8_llm_benchmark_results.csv", index=False)
    print(df.to_markdown(index=False))
