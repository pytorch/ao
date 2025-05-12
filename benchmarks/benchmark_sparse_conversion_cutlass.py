# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import torch
from triton.testing import do_bench

from torchao.ops import (
    to_sparse_semi_structured_cutlass_sm9x_f8,
)
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


def benchmark(m, k):
    torch.manual_seed(123)
    W_ref = create_semi_structured_tensor(m, k, dtype=torch.float8_e4m3fn).cuda()

    # packed, meta = torch.ops.torchao.sparse_semi_structured_tile.default(W_ref, "", True)
    cutlass_reference_args = (W_ref,)
    cutlass_custom_args = (W_ref, "", True)

    cutlass_reference_compression_time = benchmark_microseconds(
        to_sparse_semi_structured_cutlass_sm9x_f8, *cutlass_reference_args
    )
    cutlass_custom_compression_time = benchmark_microseconds(
        torch.ops.torchao.sparse_semi_structured_tile.default, *cutlass_custom_args
    )

    return {
        "cutlass_reference (ms)": cutlass_reference_compression_time,
        "cutlass_custom (ms)": cutlass_custom_compression_time,
    }


def profile():
    torch.manual_seed(123)
    W_ref = create_semi_structured_tensor(8192, 8192, dtype=torch.float8_e4m3fn).cuda()

    # clear cache
    new_val = torch.empty(10000, 10000, device="cuda")
    new_val[:, :] = 0

    packed, meta = torch.ops.torchao.sparse_semi_structured_tile.default(
        W_ref, "", True
    )


if __name__ == "__main__":
    results = []
    for m in (2048, 4096, 8192):
        results.append(benchmark(m, 8192))

    df = pd.DataFrame(results)
    df.to_csv("rowwise_scaled_linear_sparse_cutlass_time_results.csv", index=False)
    print(df.to_markdown(index=False))

    # print("PROFILING")
    # profile()
