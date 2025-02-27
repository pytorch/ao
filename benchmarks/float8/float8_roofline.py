# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a script to estimate the benefit from converting a `torch.nn.Linear`
layer to float8, by estimating the difference in e2e GPU kernel time between:
1. bf16 gemms in fwd and bwd, and
2. float8 gemms in fwd and bwd, and float8 overhead

The gemm times are estimated either from direct measurements via benchmarks,
or with a roofline estimation based on TOPS and peak compute bandwidth of an
NVIDIA H100.

The float8 overhead times are estimated by counting memory reads and writes
based on the specified float8 scaling, and estimating that we can achieve
a certain % of machine peak memory bandwidth when performing these reads and writes.

Additional context:
1. the formulas for fwd/bwd gemms in a linear layer, with corresponding input
   and output sizes:

  input @ weight_t = output
  MxK @ KxN => MxN

  grad_output @ weight = grad_input
  MxN @ NxK => MxK

  input_t @ grad_output = grad_weight
  KxM @ MxN => KxN

2. we properly model the worst-case of the current torch.compile limitations regarding
   float8 scaling
3. assume for float8 activations/gradients that torch.compile will fuse to the
preceding op. Note that this is not always true in practice.
4. assume no AC (TODO model it)
5. assume no float8 all-gather (TODO model it)
"""

import copy
import json
import os
from typing import Optional

import fire
import pandas as pd
import sympy
import torch
import torch.utils.benchmark as benchmark
import tqdm
from torch.profiler import ProfilerActivity, profile
from utils import (
    get_gpu_kernel_gemm_time_s,
    get_name_to_shapes_iter,
    profiler_output_to_filtered_time_by_kernel_name,
)

from torchao.float8 import (
    convert_to_float8_training,
)
from torchao.testing.float8.roofline_utils import (
    get_float8_mem_sympy,
    get_gemm_time_sympy,
)


class LNLinearSigmoid(torch.nn.Module):
    def __init__(self, fc_dim1, fc_dim2):
        super().__init__()
        self.ln = torch.nn.LayerNorm(fc_dim1, elementwise_affine=False)
        self.fc = torch.nn.Linear(fc_dim1, fc_dim2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


# TODO(next): hook this up


def benchmark_fn_in_sec(f, *args, **kwargs):
    # Manual warmup
    for _ in range(4):
        f(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    measurement = t0.blocked_autorange()
    return measurement.mean


def get_gpu_kernel_time(m, x):
    # warm up
    for _ in range(2):
        m(x).sum().backward()

    # capture a profiling run
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    n_iter = 5
    with profile(activities=activities) as prof:
        for _ in range(n_iter):
            m(x).sum().backward()
            torch.cuda.synchronize()
    # get the gpu kernel time and aggregate it
    num_leaf_tensors = 1 + len(list(m.parameters()))
    ref_times = profiler_output_to_filtered_time_by_kernel_name(
        prof, n_iter, num_leaf_tensors
    )
    total_time_s = sum(v for v in ref_times.values()) / 1e6 / n_iter
    return total_time_s


def get_gemm_times(M, K, N, fast_accum, cache_filename=None):
    # Note: this is definitely not the best way to build a cache,
    # but it will do for now.
    if cache_filename is not None:
        if os.path.isfile(cache_filename):
            # cache already exists, use it
            with open(cache_filename, "r") as f:
                cache = json.load(f)
        else:
            # cache does not exist yet, create it
            cache = dict()
    else:
        cache = dict()
    key = f"{M},{K},{N},{fast_accum}"
    if key in cache:
        return cache[key]

    device = torch.device("cuda")

    # bf16 time
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device).t().contiguous().t()
    bf16_time_s = get_gpu_kernel_gemm_time_s(torch.mm, x_bf16, w_bf16)

    # f8 time
    d1, d2, d3 = torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16
    A = torch.zeros(M, K, device=device, dtype=d1)
    B = torch.zeros(K, N, device=device, dtype=d2).t().contiguous().t()
    scale_a = torch.tensor([1.0], device=device)
    scale_b = torch.tensor([1.0], device=device)

    def do_matmul(A, B):
        return torch._scaled_mm(
            A, B, scale_a, scale_b, out_dtype=d3, use_fast_accum=fast_accum
        )

    f8_time_s = get_gpu_kernel_gemm_time_s(do_matmul, A, B)

    # save to cache if needed
    if cache_filename is not None:
        cache[key] = [bf16_time_s, f8_time_s]
        with open(cache_filename, "w") as f:
            json.dump(cache, f)

    return bf16_time_s, f8_time_s


def run(
    outfile: str,
    gemm_time_strategy: str = "benchmarks",
    model_torch_compile_limitations: bool = False,
    shape_gen_name: str = "square",
    gemm_cache_filename: Optional[str] = None,
    n_limit: Optional[int] = None,
):
    """
    Args:
    * `gemm_time_strategy`:
      - `benchmarks`: use benchmarks for gemm times (more accurate for all shapes)
      - `roofline`: use roofline model for gemm times (only accurate for large shapes)
    * `shape_gen_name`: `llama`, `square`, or `sweep`
    * `gemm_cache_filename (optional)`: file to cache gemm benchmark results
    * `n_limit (optional)`: if specified, only runs `n_limit` iterations
    """

    print(f"gemm_time_strategy: {gemm_time_strategy}")
    print(f"shape_gen_name: {shape_gen_name}")

    assert gemm_time_strategy in (
        "benchmarks",
        "roofline",
    ), "`gemm_time_strategy` must be 'benchmarks' or 'roofline'"

    M, K, N = sympy.symbols("M K N")

    fp8_mem_time_sympy_dyn_limit = get_float8_mem_sympy(
        M,
        K,
        N,
        model_torch_compile_limitations=True,
    )
    fp8_mem_time_sympy_dyn_nolimit = get_float8_mem_sympy(
        M,
        K,
        N,
        model_torch_compile_limitations=False,
    )

    if gemm_time_strategy == "roofline":
        bf16_gemm_time_sympy = get_gemm_time_sympy(M, K, N, torch.bfloat16)
        print("bf16_gemm_time_sympy", bf16_gemm_time_sympy)
        fp8_gemm_time_sympy = get_gemm_time_sympy(M, K, N, torch.float8_e4m3fn)
        print("fp8_gemm_time_sympy", fp8_gemm_time_sympy)
        print()
    else:
        print()

    headers = [
        "fwd_M",
        "fwd_K",
        "fwd_N",
        # gemm microbenchmarks
        "bf16_gemm_s",
        "fp8_gemm_s",
        # roofline memory overhead estimates
        "fp8_oh_estimated",
        "fp8_oh_ideal",
        # actual e2e measurements
        "bf16_s",
        "fp8_dyn_s",
        "fp8_dyn_sp",
    ]
    results = []

    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, None, None, None)

    for idx, (name, (M_val, K_val, N_val)) in enumerate(tqdm.tqdm(name_to_shapes)):
        if n_limit is not None and idx >= n_limit:
            break

        if gemm_time_strategy == "benchmarks":
            bf16_g1, f8_g1 = get_gemm_times(
                M_val, K_val, N_val, True, gemm_cache_filename
            )
            bf16_g2, f8_g2 = get_gemm_times(
                M_val, N_val, K_val, False, gemm_cache_filename
            )
            bf16_g3, f8_g3 = get_gemm_times(
                K_val, M_val, N_val, False, gemm_cache_filename
            )
            bf16_time_val = bf16_g1 + bf16_g2 + bf16_g3
            fp8_gemm_time_s = f8_g1 + f8_g2 + f8_g3
        else:
            assert gemm_time_strategy == "roofline", "unsupported"
            bf16_time_val = (
                bf16_gemm_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
            )
            fp8_gemm_time_s = (
                fp8_gemm_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
            )

        fp8_mem_time_dyn_limit_s = (
            fp8_mem_time_sympy_dyn_limit.subs(M, M_val).subs(K, K_val).subs(N, N_val)
        )
        fp8_mem_time_dyn_nolimit_s = (
            fp8_mem_time_sympy_dyn_nolimit.subs(M, M_val).subs(K, K_val).subs(N, N_val)
        )

        # create the model
        m_orig = LNLinearSigmoid(K_val, N_val).cuda().bfloat16()
        x = torch.randn(
            M_val, K_val, dtype=torch.bfloat16, device="cuda"
        ).requires_grad_()

        # get the bf16 gpu kernel time
        torch._dynamo.reset()
        m_bf16 = torch.compile(copy.deepcopy(m_orig))
        bf16_time_actual_s = get_gpu_kernel_time(m_bf16, x)

        # get the float8 dynamic scaling gpu kernel time

        torch._dynamo.reset()
        m_fp8_dyn = convert_to_float8_training(copy.deepcopy(m_orig))
        m_fp8_dyn = torch.compile(m_fp8_dyn)
        fp8_dyn_time_actual_s = get_gpu_kernel_time(m_fp8_dyn, x)

        results.append(
            [
                M_val,
                K_val,
                N_val,
                # gemm microbenchmarks
                bf16_time_val,
                fp8_gemm_time_s,
                # roofline overhead estimates
                fp8_mem_time_dyn_limit_s,
                fp8_mem_time_dyn_nolimit_s,
                # e2e numbers
                bf16_time_actual_s,
                fp8_dyn_time_actual_s,
                bf16_time_actual_s / fp8_dyn_time_actual_s,
            ]
        )

    df = pd.DataFrame(results, columns=headers)
    print(df)
    df.to_csv(outfile)
    print("done")


if __name__ == "__main__":
    fire.Fire(run)
