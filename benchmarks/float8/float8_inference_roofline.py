# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a script to estimate the benefit from converting a `torch.nn.Linear`
layer to float8 given a single saturated GPU, by estimating the difference
in e2e GPU kernel time between:
1. bf16 gemms in fwd and
2. float8 gemms in fwd and float8 overhead

The gemm times are estimated either from direct measurements via benchmarks,
or with a roofline estimation based on TOPS and peak compute bandwidth of an
NVIDIA H100 or B200.

The float8 overhead times are estimated by counting memory reads and writes
based on the specified float8 scaling, and estimating that we can achieve
a certain % of machine peak memory bandwidth when performing these reads and writes.
"""

import copy
from typing import Optional

import fire
import pandas as pd
import sympy
import torch
import torch.nn as nn
import tqdm
from torch.profiler import ProfilerActivity, profile
from utils import (
    get_gpu_kernel_gemm_time_s,
    get_name_to_shapes_iter,
    profiler_output_to_filtered_time_by_kernel_name,
)

import torchao
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
    quantize_,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.testing.training.roofline_utils import (
    get_inference_float8_mem_sympy,
    get_inference_gemm_time_sympy,
)
from torchao.utils import is_MI300


@torch.no_grad()
def get_gpu_kernel_time(m, x):
    # warm up
    for _ in range(2):
        __ = m(x)

    # capture a profiling run
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    n_iter = 5
    with profile(activities=activities) as prof:
        for _ in range(n_iter):
            __ = m(x)
            torch.cuda.synchronize()
    # get the gpu kernel time and aggregate it
    num_leaf_tensors = 1 + len(list(m.parameters()))
    ref_times = profiler_output_to_filtered_time_by_kernel_name(
        prof, n_iter, num_leaf_tensors
    )
    total_time_s = sum(v for v in ref_times.values()) / 1e6 / n_iter
    return total_time_s


def get_gemm_times(
    M: int,
    K: int,
    N: int,
    fast_accum: bool,
    float8_recipe_name: Optional[str],
):
    device = torch.device("cuda")

    # bf16 time
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    # w_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device).t().contiguous().t()
    w_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    bf16_time_s = get_gpu_kernel_gemm_time_s(torch.mm, x_bf16, w_bf16)

    e4m3_dtype = torch.float8_e4m3fn
    if torch.version.hip and torch.cuda.is_available() and is_MI300():
        e4m3_dtype = torch.float8_e4m3fnuz
    d1, d2, d3 = e4m3_dtype, e4m3_dtype, torch.bfloat16
    A = torch.zeros(M, K, device=device, dtype=d1)
    B = torch.zeros(K, N, device=device, dtype=d2).t().contiguous().t()
    if float8_recipe_name in ("rowwise"):
        scale_a = torch.ones(M, 1, device=device)
        scale_b = torch.ones(1, N, device=device)
    else:
        assert False, "unsupported"

    def do_matmul(A, B):
        return torch._scaled_mm(
            A, B, scale_a, scale_b, out_dtype=d3, use_fast_accum=fast_accum
        )

    f8_time_s = get_gpu_kernel_gemm_time_s(do_matmul, A, B)

    return bf16_time_s, f8_time_s


def run(
    outfile: str,
    do_benchmarks: bool = True,
    shape_gen_name: str = "pow2",
    n_limit: Optional[int] = None,
    float8_recipe_name: Optional[str] = None,
):
    """
    Args:
    * `do_benchmarks`: if True, gemm and e2e fwd+bwd of LNLinearSigmoid are benchmarked
    * `shape_gen_name`: `llama`, `pow2`, `pow2_extended`, or `sweep`
    * `n_limit (optional)`: if specified, only runs `n_limit` iterations
    """

    assert float8_recipe_name is not None, "unsupported"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch version: {torch.__version__}")
    print(f"torchao version: {torchao.__version__}")
    print(f"do_benchmarks: {do_benchmarks}")
    print(f"shape_gen_name: {shape_gen_name}")
    print(f"float8_recipe_name: {float8_recipe_name}")

    M, K, N = sympy.symbols("M K N")

    fp8_ovhd_time_sympy = get_inference_float8_mem_sympy(
        M,
        K,
        N,
        float8_recipe_name,
    )
    bf16_gemm_time_sympy = get_inference_gemm_time_sympy(
        M, K, N, torch.bfloat16, None, None
    )
    fp8_gemm_time_sympy = get_inference_gemm_time_sympy(
        M, K, N, torch.float8_e4m3fn, float8_recipe_name, None
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
                M_val,
                K_val,
                N_val,
                True,
                float8_recipe_name,
            )
            b_bf16_gemm_time_s = bf16_g1
            b_fp8_gemm_time_s = f8_g1
            rb_bf16_gemm_ratio = r_bf16_gemm_time_s / b_bf16_gemm_time_s
            rb_fp8_gemm_ratio = r_fp8_gemm_time_s / b_fp8_gemm_time_s

        # note: cast from sympy.core.numbers.Float to float to make pandas formatting work
        r_fp8_ovhd_time_s = float(
            fp8_ovhd_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
        )

        b_bf16_e2e_time_s, b_fp8_e2e_time_s = 0, 0
        if do_benchmarks:
            # create the model
            m_orig = (
                nn.Sequential(nn.Linear(K_val, N_val, bias=False)).cuda().bfloat16()
            )
            x = torch.randn(
                M_val, K_val, dtype=torch.bfloat16, device="cuda"
            ).requires_grad_()

            # get the bf16 gpu kernel time
            torch._dynamo.reset()
            m_bf16 = torch.compile(copy.deepcopy(m_orig))
            b_bf16_e2e_time_s = get_gpu_kernel_time(m_bf16, x)

            # get the float8 dynamic scaling gpu kernel time
            torch._dynamo.reset()

            config = Float8DynamicActivationFloat8WeightConfig(
                granularity=PerRow(),
                # for now, use TORCH. In the future might be interesting
                # to benchmark AUTO and FBGEMM.
                kernel_preference=KernelPreference.TORCH,
            )
            m_fp8_dyn = copy.deepcopy(m_orig)
            quantize_(m_fp8_dyn, config)

            m_fp8_dyn = torch.compile(m_fp8_dyn)
            b_fp8_e2e_time_s = get_gpu_kernel_time(m_fp8_dyn, x)

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
