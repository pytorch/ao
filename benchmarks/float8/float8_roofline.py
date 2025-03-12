# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a script to estimate the benefit from converting a `torch.nn.Linear`
layer to float8 given a single saturated GPU, by estimating the difference
in e2e GPU kernel time between:
1. bf16 gemms in fwd and bwd, and
2. float8 gemms in fwd and bwd, and float8 overhead

The gemm times are estimated either from direct measurements via benchmarks,
or with a roofline estimation based on TOPS and peak compute bandwidth of an
NVIDIA H100 or B200.

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

2. assume for float8 activations/gradients that torch.compile will fuse to the
preceding op. Note that this is not always true in practice.
3. assume no AC (TODO model it)
4. assume no float8 all-gather (TODO model it)
"""

import copy
import json
import os
from typing import Optional

import fire
import pandas as pd
import sympy
import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import tqdm
from torch.profiler import ProfilerActivity, profile
from utils import (
    get_gpu_kernel_gemm_time_s,
    get_name_to_shapes_iter,
    profiler_output_to_filtered_time_by_kernel_name,
)

from torchao.float8 import (
    Float8LinearConfig,
    convert_to_float8_training,
)
from torchao.prototype.mx_formats.config import MXLinearConfig
from torchao.prototype.mx_formats.mx_linear import swap_linear_with_mx_linear
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


def get_gpu_kernel_time(m, x, grad_output):
    # warm up
    for _ in range(2):
        y = m(x)
        y.backward(grad_output)

    # capture a profiling run
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    n_iter = 5
    with profile(activities=activities) as prof:
        for _ in range(n_iter):
            y = m(x)
            y.backward(grad_output)
            torch.cuda.synchronize()
    # get the gpu kernel time and aggregate it
    num_leaf_tensors = 1 + len(list(m.parameters()))
    ref_times = profiler_output_to_filtered_time_by_kernel_name(
        prof, n_iter, num_leaf_tensors
    )
    total_time_s = sum(v for v in ref_times.values()) / 1e6 / n_iter
    return total_time_s


def get_gemm_times(
    gemm_role: str,
    M: int,
    K: int,
    N: int,
    fast_accum: bool,
    bf16_memory_formats: str,
    float8_recipe_name: Optional[str],
    mx_recipe_name: Optional[str],
    cache_filename=None,
):
    assert gemm_role in ("output", "grad_input", "grad_weight"), "unsupported"
    assert bf16_memory_formats in (
        "row_major:col_major",
        "row_major:row_major",
        "col_major:row_major",
    ), "unsupported"

    # Note: this is definitely not the best way to build a cache,
    # but it will do for now.
    if cache_filename is not None:
        assert False, "TODO retest this for new arguments"
        if os.path.isfile(cache_filename):
            # cache already exists, use it
            with open(cache_filename, "r") as f:
                cache = json.load(f)
        else:
            # cache does not exist yet, create it
            cache = dict()
    else:
        cache = dict()
    key = f"{M},{K},{N},{fast_accum},{bf16_memory_formats}"
    if key in cache:
        return cache[key]

    device = torch.device("cuda")

    # bf16 time
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    # w_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device).t().contiguous().t()
    w_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    if bf16_memory_formats == "row_major:col_major":
        w_bf16 = w_bf16.t().contiguous().t()
    elif bf16_memory_formats == "col_major:row_major":
        x_bf16 = x_bf16.t().contiguous().t()
    elif bf16_memory_formats == "col_major:row_major":
        x_bf16 = x_bf16.t().contiguous().t()

    bf16_time_s = get_gpu_kernel_gemm_time_s(torch.mm, x_bf16, w_bf16)

    # f8 time
    if float8_recipe_name == "rowwise_with_gw_hp" and gemm_role == "grad_weight":
        f8_time_s = bf16_time_s
    else:
        d1, d2, d3 = torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16
        A = torch.zeros(M, K, device=device, dtype=d1)
        B = torch.zeros(K, N, device=device, dtype=d2).t().contiguous().t()
        if float8_recipe_name == "tensorwise":
            scale_a = torch.tensor([1.0], device=device)
            scale_b = torch.tensor([1.0], device=device)
        elif float8_recipe_name in ("rowwise", "rowwise_with_gw_hp"):
            scale_a = torch.ones(M, 1, device=device)
            scale_b = torch.ones(1, N, device=device)
        elif mx_recipe_name == "mxfp8_cublas":
            scale_a = torch.ones(M, K // 32, device=device, dtype=torch.float8_e8m0fnu)
            scale_b = torch.ones(N, K // 32, device=device, dtype=torch.float8_e8m0fnu)
        else:
            assert False, "TODO add cutlass mx gemm here"

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
    do_benchmarks: bool = True,
    shape_gen_name: str = "pow2",
    gemm_cache_filename: Optional[str] = None,
    n_limit: Optional[int] = None,
    float8_recipe_name: Optional[str] = None,
    mx_recipe_name: Optional[str] = None,
    enable_fusion_modeling: bool = False,
):
    """
    Args:
    * `do_benchmarks`: if True, gemm and e2e fwd+bwd of LNLinearSigmoid are benchmarked
    * `shape_gen_name`: `llama`, `pow2`, `pow2_extended`, or `sweep`
    * `gemm_cache_filename (optional)`: file to cache gemm benchmark results
    * `n_limit (optional)`: if specified, only runs `n_limit` iterations
    * `enable_fusion_modeling`: if False uses Linear, if True uses LNLinearSigmoid and models the fusion of float8 overhead
    """

    assert not (
        (float8_recipe_name is not None) and (mx_recipe_name is not None)
    ), "unsupported"
    if float8_recipe_name is None and mx_recipe_name is None:
        float8_recipe_name = "tensorwise"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"do_benchmarks: {do_benchmarks}")
    print(f"shape_gen_name: {shape_gen_name}")
    print(f"float8_recipe_name: {float8_recipe_name}")
    print(f"mx_recipe_name: {mx_recipe_name}")
    print(f"enable_fusion_modeling: {enable_fusion_modeling}")

    M, K, N = sympy.symbols("M K N")

    fp8_ovhd_time_sympy = get_float8_mem_sympy(
        M,
        K,
        N,
        float8_recipe_name,
        mx_recipe_name,
        enable_fusion_modeling,
    )
    bf16_gemm_time_sympy = get_gemm_time_sympy(M, K, N, torch.bfloat16, None, None)
    fp8_gemm_time_sympy = get_gemm_time_sympy(
        M, K, N, torch.float8_e4m3fn, float8_recipe_name, mx_recipe_name
    )
    print("bf16_gemm_time_sympy", bf16_gemm_time_sympy)
    print("fp8_gemm_time_sympy", fp8_gemm_time_sympy)
    print("fp8_ovhd_time_sympy", fp8_ovhd_time_sympy)
    print()

    headers = [
        "fwd_M",
        "fwd_K",
        "fwd_N",
        # roofline - gemm time (fwd + bwd, 3 gemms)
        "r_bf16_gemm_s",
        "r_fp8_gemm_s",
        # roofline - fp8 overhead time (by counting reads/writes in the ideal case)
        "r_fp8_ovhd_s",
        # roofline - fp8 gemm + fp8 overhead time (does not include LN or sigmoid)
        "r_fp8_gemm_and_ovhd_s",
        "r_fp8_gemm_and_ovhd_spdp",
        # benchmarks - gemm time (fwd + bwd, 3 gemms)
        "b_bf16_gemm_s",
        "b_fp8_gemm_s",
        # benchmarks - e2e LNLinearSigmoid time fwd + bwd
        "b_bf16_e2e_s",
        "b_fp8_e2e_s",
        # note that e2e speedup is not the same as the roofline speedup:
        # 1. roofline speedup: (bf16_gemm_time) / (fp8_gemm_time + fp8_ovhd_time)
        # 2. e2e speedup: (ln + bf16_gemm_time + sigmoid) / (ln + fp8_gemm_time + fp8_ovhd_time + sigmoid)
        # the difference is the fwd+bwd ln and sigmoid terms, for now to keep things simple
        # we don't break them out and don't have a roofline for them.
        "b_fp8_e2e_spdp",
        # how well benchmarked gemms match roofline predicted gemms
        "rb_bf16_gemm_ratio",
        "rb_fp8_gemm_ratio",
    ]
    results = []

    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, None, None, None)

    for idx, (name, (M_val, K_val, N_val)) in enumerate(tqdm.tqdm(name_to_shapes)):
        if n_limit is not None and idx >= n_limit:
            break

        # use roofline model to estimate gemm time
        # note: cast from sympy.core.numbers.Float to float to make pandas formatting work
        r_bf16_gemm_time_s = float(
            bf16_gemm_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
        )
        r_fp8_gemm_time_s = float(
            fp8_gemm_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
        )

        # if enabled, also measured observed gemm time
        b_bf16_gemm_time_s, b_fp8_gemm_time_s = 0, 0
        rb_bf16_gemm_ratio = -1
        rb_fp8_gemm_ratio = -1

        if do_benchmarks:
            # TODO(future): make the bf16 gemm times exactly match the e2e
            # benchmarks, there is a slight deviation, probably related to gemm
            # operand memory formats/transpositions below not exactly matching
            # what PyTorch core is doing for `torch.mm`
            # input @ weight_t = output
            bf16_g1, f8_g1 = get_gemm_times(
                "output",
                M_val,
                K_val,
                N_val,
                True,
                "row_major:col_major",
                float8_recipe_name,
                mx_recipe_name,
                gemm_cache_filename,
            )
            # grad_output @ weight = grad_input
            bf16_g2, f8_g2 = get_gemm_times(
                "grad_input",
                M_val,
                N_val,
                K_val,
                False,
                "row_major:row_major",
                float8_recipe_name,
                mx_recipe_name,
                gemm_cache_filename,
            )
            # input_t @ grad_output = grad_weight
            bf16_g3, f8_g3 = get_gemm_times(
                "grad_weight",
                K_val,
                M_val,
                N_val,
                False,
                "col_major:row_major",
                float8_recipe_name,
                mx_recipe_name,
                gemm_cache_filename,
            )
            b_bf16_gemm_time_s = bf16_g1 + bf16_g2 + bf16_g3
            b_fp8_gemm_time_s = f8_g1 + f8_g2 + f8_g3
            rb_bf16_gemm_ratio = r_bf16_gemm_time_s / b_bf16_gemm_time_s
            rb_fp8_gemm_ratio = r_fp8_gemm_time_s / b_fp8_gemm_time_s

        # note: cast from sympy.core.numbers.Float to float to make pandas formatting work
        r_fp8_ovhd_time_s = float(
            fp8_ovhd_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
        )

        b_bf16_e2e_time_s, b_fp8_e2e_time_s = 0, 0
        if do_benchmarks:
            # create the model
            if enable_fusion_modeling:
                m_orig = LNLinearSigmoid(K_val, N_val).cuda().bfloat16()
            else:
                m_orig = (
                    nn.Sequential(nn.Linear(K_val, N_val, bias=False)).cuda().bfloat16()
                )
            x = torch.randn(
                M_val, K_val, dtype=torch.bfloat16, device="cuda"
            ).requires_grad_()

            # get the gradient of the right shape
            grad_output = torch.randn(N_val, K_val, dtype=torch.bfloat16, device="cuda")

            # get the bf16 gpu kernel time
            torch._dynamo.reset()
            m_bf16 = torch.compile(copy.deepcopy(m_orig))
            b_bf16_e2e_time_s = get_gpu_kernel_time(m_bf16, x, grad_output)

            # get the float8 dynamic scaling gpu kernel time

            torch._dynamo.reset()
            if float8_recipe_name is not None:
                config = Float8LinearConfig.from_recipe_name(float8_recipe_name)
                m_fp8_dyn = convert_to_float8_training(
                    copy.deepcopy(m_orig), config=config
                )
            else:
                assert mx_recipe_name is not None
                config = MXLinearConfig.from_recipe_name(mx_recipe_name)
                m_fp8_dyn = copy.deepcopy(m_orig)
                swap_linear_with_mx_linear(m_fp8_dyn, config=config)
            m_fp8_dyn = torch.compile(m_fp8_dyn)
            b_fp8_e2e_time_s = get_gpu_kernel_time(m_fp8_dyn, x, grad_output)

        results.append(
            [
                M_val,
                K_val,
                N_val,
                # roofline - gemm
                r_bf16_gemm_time_s,
                r_fp8_gemm_time_s,
                # roofline - fp8 overhead
                r_fp8_ovhd_time_s,
                # roofline - gemm + overhead, and speedup
                r_fp8_gemm_time_s + r_fp8_ovhd_time_s,
                r_bf16_gemm_time_s / (r_fp8_gemm_time_s + r_fp8_ovhd_time_s),
                # benchmarks - gemm
                b_bf16_gemm_time_s,
                b_fp8_gemm_time_s,
                # benchmarks - e2e, and speedup
                b_bf16_e2e_time_s,
                b_fp8_e2e_time_s,
                b_bf16_e2e_time_s / (b_fp8_e2e_time_s + 1e-20),
                # gemm ratios
                rb_bf16_gemm_ratio,
                rb_fp8_gemm_ratio,
            ]
        )

    pd.set_option("display.precision", 2)
    df = pd.DataFrame(results, columns=headers)
    print(df)
    df.to_csv(outfile)
    print("done")


if __name__ == "__main__":
    fire.Fire(run)
