# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import torch
from tqdm import tqdm
from triton.testing import do_bench

from torchao.ops import (
    rowwise_scaled_linear_cutlass_s4s4,
    rowwise_scaled_linear_cutlass_s8s4,
)
from torchao.quantization.quant_api import (
    _int4_symm_cutlass_quant,
    _int8_symm_cutlass_quant,
)

dtype = torch.bfloat16
dtypeq = torch.int8
dtype_scale = torch.float32
device = torch.device("cuda")


def benchmark_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def get_problem(m: int, n: int, k: int, Xq_nbits: int):
    assert k % 2 == 0
    assert Xq_nbits in [4, 8]

    X_ref = torch.randn((m, k), dtype=dtype, device=device)
    W_ref = torch.rand((n, k), dtype=dtype, device=device)

    X_quant_func = (
        _int4_symm_cutlass_quant if Xq_nbits == 4 else _int8_symm_cutlass_quant
    )
    W_quant_func = _int4_symm_cutlass_quant
    X_aqt = X_quant_func(X_ref)
    W_aqt = W_quant_func(W_ref)

    Xq = X_aqt.tensor_impl.int_data
    X_scale = X_aqt.tensor_impl.scale
    Wq = W_aqt.tensor_impl.int_data
    W_scale = W_aqt.tensor_impl.scale
    bias = None
    out_dtype = dtype

    return (X_ref, W_ref), (Xq, X_scale, Wq, W_scale, bias, out_dtype)


def benchmark(m: int, k: int, n: int):
    ref_args, args = get_problem(m, n, k, 4)
    fp16_time = benchmark_microseconds(torch.nn.functional.linear, *ref_args)
    rowwise_scaled_linear_cutlass_s4s4_time = benchmark_microseconds(
        rowwise_scaled_linear_cutlass_s4s4, *args
    )

    _, args = get_problem(m, n, k, 8)
    rowwise_scaled_linear_cutlass_s8s4_time = benchmark_microseconds(
        rowwise_scaled_linear_cutlass_s8s4, *args
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
