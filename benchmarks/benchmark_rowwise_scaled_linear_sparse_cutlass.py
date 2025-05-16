# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import torch
from tqdm import tqdm
from triton.testing import do_bench

from torchao.ops import rowwise_scaled_linear_sparse_cutlass_f8f8
from torchao.quantization.quant_api import (
    _float8_cutlass_quant,
    _float8_cutlass_quant_sparse,
)
from torchao.sparsity.utils import create_semi_structured_tensor

dtype = torch.bfloat16
dtypeq_X = torch.float8_e4m3fn
dtypeq_W = torch.float8_e4m3fn
device = torch.device("cuda")


def benchmark_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def get_problem_cutlass(m: int, n: int, k: int):
    X_ref = torch.randn((m, k), dtype=dtype, device=device)
    W_ref = create_semi_structured_tensor(n, k, dtype=dtype).to(device)

    X_quant_func = _float8_cutlass_quant
    W_quant_func = _float8_cutlass_quant_sparse
    X_aqt = X_quant_func(X_ref, dtypeq_X)
    W_aqt = W_quant_func(W_ref, dtypeq_W)

    Xq = X_aqt.tensor_impl.float8_data
    X_scale = X_aqt.tensor_impl.scale
    Wq_sparse = W_aqt.tensor_impl.sparse
    W_meta = W_aqt.tensor_impl.meta
    W_scale = W_aqt.tensor_impl.scale
    bias = None
    out_dtype = dtype

    return (X_ref, W_ref), (Xq, X_scale, Wq_sparse, W_meta, W_scale, bias, out_dtype)


def get_problem_cusparselt(m: int, n: int, k: int):
    X_ref = torch.randn((m, k), dtype=dtype, device=device)
    W_ref = create_semi_structured_tensor(n, k, dtype=dtype).to(device)

    Xq = X_ref.to(dtypeq_W)
    Wq = W_ref.to(dtypeq_W)

    Wqs = torch._cslt_compress(Wq)

    alg_id, split_k, split_k_one_kernel, _ = torch._C._cusparselt.mm_search(
        Wqs, Xq.t(), None, None, None, False
    )

    return (Wqs, Xq.t(), None, None, dtype, False, alg_id, split_k, split_k_one_kernel)


def get_problem_scaled_mm(m: int, n: int, k: int):
    X_ref = torch.randn((m, k), dtype=dtype, device=device)
    W_ref = create_semi_structured_tensor(n, k, dtype=dtype).to(device)

    X_aqt = _float8_cutlass_quant(X_ref, dtypeq_W)
    W_aqt = _float8_cutlass_quant(W_ref, dtypeq_W)

    Xq = X_aqt.tensor_impl.float8_data
    Wq = W_aqt.tensor_impl.float8_data
    X_scale = X_aqt.tensor_impl.scale.unsqueeze(0)
    W_scale = W_aqt.tensor_impl.scale.unsqueeze(-1)

    return (Wq, Xq.t(), W_scale, X_scale, None, None, dtype)


def benchmark(m: int, k: int, n: int):
    ref_args, args = get_problem_cutlass(m, n, k)
    fp16_time = benchmark_microseconds(torch.nn.functional.linear, *ref_args)
    rowwise_scaled_linear_sparse_cutlass_f8f8_time = benchmark_microseconds(
        rowwise_scaled_linear_sparse_cutlass_f8f8, *args
    )

    cslt_args = get_problem_cusparselt(m, n, k)
    cusparselt_time = benchmark_microseconds(torch._cslt_sparse_mm, *cslt_args)

    fp8_args = get_problem_scaled_mm(m, n, k)
    fp8_time = benchmark_microseconds(torch._scaled_mm, *fp8_args)

    return {
        "m": m,
        "k": k,
        "n": n,
        "fp16_latency (ms)": fp16_time,
        "fp8_latency (ms)": fp8_time,
        "rowwise_scaled_linear_sparse_cutlass_f8f8 latency (ms)": rowwise_scaled_linear_sparse_cutlass_f8f8_time,
        "cusparselt latency (ms)": cusparselt_time,
        "f8f8 speedup (d/s)": fp8_time / rowwise_scaled_linear_sparse_cutlass_f8f8_time,
    }


if __name__ == "__main__":
    k_vals = (8192,)
    n_vals = (8192,)

    results = []
    for m in tqdm([2048, 4096, 8192]):
        for n, k in zip(n_vals, k_vals):
            results.append(benchmark(m, k, n))

    df = pd.DataFrame(results)
    df.to_csv("rowwise_scaled_linear_sparse_cutlass_time_results.csv", index=False)
    print(df.to_markdown(index=False))
