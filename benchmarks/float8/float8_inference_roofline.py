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
from tabulate import tabulate
from torch.profiler import ProfilerActivity, profile
from utils import (
    get_gpu_kernel_gemm_time_s,
    get_name_to_shapes_iter,
    profiler_output_to_filtered_time_by_kernel_name,
)

import torchao
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
    PerTensor,
    quantize_,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.testing.training.roofline_utils import (
    get_inference_float8_mem_sympy,
    get_inference_gemm_time_sympy,
)
from torchao.utils import is_MI300


@torch.no_grad()
def get_gpu_kernel_time(m, x, trace_filename=None):
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

    # save a trace, if requested
    if trace_filename is not None:
        print(f"exporting trace to {trace_filename}")
        prof.export_chrome_trace(trace_filename)

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
    recipe_name: Optional[str],
):
    device = torch.device("cuda")

    # bf16 time
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    bf16_time_s = get_gpu_kernel_gemm_time_s(torch.mm, x_bf16, w_bf16)

    if recipe_name in ("mxfp4_cutlass", "nvfp4"):
        d1, d2, d3 = torch.float4_e2m1fn_x2, torch.float4_e2m1fn_x2, torch.bfloat16
        A = torch.randint(0, 255, (M, K // 2), device=device, dtype=torch.uint8).view(
            d1
        )
        B = (
            torch.randint(0, 255, (K // 2, N), device=device, dtype=torch.uint8)
            .t()
            .contiguous()
            .t()
            .view(d2)
        )
    else:
        e4m3_dtype = torch.float8_e4m3fn
        if torch.version.hip and torch.cuda.is_available() and is_MI300():
            e4m3_dtype = torch.float8_e4m3fnuz
        d1, d2, d3 = e4m3_dtype, e4m3_dtype, torch.bfloat16
        A = torch.randint(0, 255, (M, K), device=device, dtype=torch.uint8).view(d1)
        B = (
            torch.randint(0, 255, (K, N), device=device, dtype=torch.uint8)
            .view(d2)
            .t()
            .contiguous()
            .t()
        )

    if recipe_name == "rowwise":
        scale_a = torch.ones(M, 1, device=device)
        scale_b = torch.ones(1, N, device=device)
    elif recipe_name == "mxfp8_cublas":
        scale_a = torch.ones(M, K // 32, device=device, dtype=torch.float8_e8m0fnu)
        scale_b = torch.ones(N, K // 32, device=device, dtype=torch.float8_e8m0fnu)
        scale_a = to_blocked(scale_a)
        scale_b = to_blocked(scale_b)
    elif recipe_name == "mxfp4_cutlass":
        scale_a = torch.ones(M, K // 32, device=device, dtype=torch.float8_e8m0fnu)
        scale_b = torch.ones(N, K // 32, device=device, dtype=torch.float8_e8m0fnu)
        scale_a = to_blocked(scale_a)
        scale_b = to_blocked(scale_b)
    elif recipe_name == "nvfp4":
        scale_a = torch.ones(M, K // 16, device=device, dtype=torch.float8_e4m3fn)
        scale_b = torch.ones(N, K // 16, device=device, dtype=torch.float8_e4m3fn)
        scale_a = to_blocked(scale_a)
        scale_b = to_blocked(scale_b)

    else:
        assert False, "unsupported"

    def do_matmul(A, B):
        if recipe_name == "mxfp4_cutlass":
            return torchao.ops.mx_fp4_bf16(A, B, scale_a, scale_b)
        if recipe_name == "nvfp4":
            return torch._scaled_mm(
                A, B, scale_a, scale_b, out_dtype=d3, use_fast_accum=False
            )
        else:
            return torch._scaled_mm(
                A, B, scale_a, scale_b, out_dtype=d3, use_fast_accum=fast_accum
            )

    f8_time_s = get_gpu_kernel_gemm_time_s(do_matmul, A, B)

    return bf16_time_s, f8_time_s


def run(
    outfile: str,
    recipe_name: str,
    do_benchmarks: bool = True,
    shape_gen_name: str = "pow2",
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    n_limit: Optional[int] = None,
    save_profile_traces: bool = False,
    enable_fusion_modeling: bool = False,
    op_name: str = "linear",
    D: Optional[int] = None,
    H: Optional[int] = None,
    W: Optional[int] = None,
    kernel_size: Optional[int] = None,
):
    """
    Args:
    * `recipe_name`: quantization recipe (tensorwise, rowwise, mxfp8*, mxfp4*, nvfp4*)
    * `do_benchmarks`: if True, gemm and e2e fwd+bwd of LNLinearSigmoid are benchmarked
    * `shape_gen_name`: `llama`, `pow2`, `pow2_extended`, `sweep`, or `custom`
    * `M|K|N`: if shape_gen_name is `custom`, then these values are used for MKN
    * `n_limit (optional)`: if specified, only runs `n_limit` iterations
    # `save_profile_traces (optional)`: if True, saves profiling traces
    # `enable_fusion_modeling`: if True, models activation -> gemm instead of just gemm
    # `op_name`: linear, conv2d or conv3d, decides which op to benchmark
    # `D`, `H`, `W`: spatial dimensiosn for conv3d / conv2d
    # `kernel_size`: kernel_size for conv3d / conv2d
    """
    _SUPPORTED_OPS = ["linear", "conv2d", "conv3d"]
    assert op_name in _SUPPORTED_OPS, (
        f"Unsupported op: {op_name}, supported are: {_SUPPORTED_OPS}"
    )
    if op_name == "conv2d":
        assert H is not None and W is not None, (
            "Expected D, H, W to be specified for conv2d"
        )
        assert kernel_size is not None, (
            "Expected kernel_size to be specified for conv2d"
        )
    elif op_name == "conv3d":
        assert D is not None and H is not None and W is not None, (
            "Expected D, H, W to be specified for conv3d"
        )
        assert kernel_size is not None, (
            "Expected kernel_size to be specified for conv3d"
        )

    config_table = [
        ["GPU", torch.cuda.get_device_name(0)],
        ["torch version", torch.__version__],
        ["torchao version", torchao.__version__],
        ["recipe_name", recipe_name],
        ["do_benchmarks", do_benchmarks],
        ["shape_gen_name", shape_gen_name],
        ["enable_fusion_modeling", enable_fusion_modeling],
        ["op_name", op_name],
        ["MKN", f"{M} {K} {N}"],
        ["DHW", f"{D} {H} {W}"],
        ["kernel_size", kernel_size],
    ]
    print(tabulate(config_table, headers=["Parameter", "Value"], tablefmt="simple"))

    # reassign user specified MKN, so we can use them for sympy
    user_M, user_K, user_N = M, K, N

    M, K, N = sympy.symbols("M K N")

    if op_name == "linear":
        fp8_ovhd_time_sympy = get_inference_float8_mem_sympy(
            M,
            K,
            N,
            recipe_name,
            # TODO(future): also enable fusion modeling here
        )
        bf16_gemm_time_sympy = get_inference_gemm_time_sympy(
            M, K, N, torch.bfloat16, None
        )

        if recipe_name and recipe_name.startswith(("nvfp4", "mxfp4")):
            fp8_gemm_time_sympy = get_inference_gemm_time_sympy(
                M, K, N, torch.float4_e2m1fn_x2, recipe_name
            )
        else:
            gemm_recipe_name = "mxfp8" if recipe_name.startswith("mxfp8") else None
            fp8_gemm_time_sympy = get_inference_gemm_time_sympy(
                M, K, N, torch.float8_e4m3fn, gemm_recipe_name
            )
        print("bf16_gemm_time_sympy", bf16_gemm_time_sympy)
        print("fp8_gemm_time_sympy", fp8_gemm_time_sympy)
        print("fp8_ovhd_time_sympy", fp8_ovhd_time_sympy)
        print()
    else:
        # TODO: enable roofline analysis for conv
        pass

    # Note: roofline for conv2d/conv3d is not added yet, so most of the
    # things for conv2d/conv3d we'll left out for now
    headers = [
        "fwd_M",
        "fwd_K",
        "fwd_N",
        "D",
        "H",
        "W",
        "kernel_size",
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

    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, user_M, user_K, user_N)

    for idx, (name, (M_val, K_val, N_val)) in enumerate(tqdm.tqdm(name_to_shapes)):
        if n_limit is not None and idx >= n_limit:
            break

        if op_name == "linear":
            # use roofline model to estimate gemm time
            # note: cast from sympy.core.numbers.Float to float to make pandas formatting work
            r_bf16_gemm_time_s = float(
                bf16_gemm_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
            )
            r_fp8_gemm_time_s = float(
                fp8_gemm_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
            )

            # note: cast from sympy.core.numbers.Float to float to make pandas formatting work
            r_fp8_ovhd_time_s = float(
                fp8_ovhd_time_sympy.subs(M, M_val).subs(K, K_val).subs(N, N_val)
            )
            r_fp8_gemm_and_ovhd_s = r_fp8_gemm_time_s + r_fp8_ovhd_time_s
            r_speedup = r_bf16_gemm_time_s / (r_fp8_gemm_time_s + r_fp8_ovhd_time_s)

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
                    recipe_name,
                )
                b_bf16_gemm_time_s = bf16_g1
                b_fp8_gemm_time_s = f8_g1
                rb_bf16_gemm_ratio = r_bf16_gemm_time_s / b_bf16_gemm_time_s
                rb_fp8_gemm_ratio = r_fp8_gemm_time_s / b_fp8_gemm_time_s

        else:
            # roofline analysis for conv2d/conv3d are not added yet
            r_bf16_gemm_time_s = None
            r_fp8_gemm_time_s = None

            r_fp8_ovhd_time_s = None
            r_fp8_gemm_and_ovhd_s = None
            r_speedup = None

            # real gemm benchmark time, also not added yet
            # if enabled, also measured observed gemm time
            b_bf16_gemm_time_s, b_fp8_gemm_time_s = 0, 0
            # gemm roofline ratio achieved in real benchmark
            rb_bf16_gemm_ratio = -1
            rb_fp8_gemm_ratio = -1

        b_bf16_e2e_time_s, b_fp8_e2e_time_s = 0, 0
        if do_benchmarks:
            # create the model
            if op_name == "conv2d":
                if not enable_fusion_modeling:
                    m_orig = nn.Sequential(
                        nn.Conv2d(K_val, N_val, kernel_size, bias=False)
                    ).to(memory_format=torch.channels_last)
                else:
                    m_orig = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(K_val, N_val, kernel_size, bias=False),
                        nn.ReLU(),
                    ).to(memory_format=torch.channels_last)
            elif op_name == "conv3d":
                if not enable_fusion_modeling:
                    m_orig = nn.Sequential(
                        nn.Conv3d(K_val, N_val, kernel_size, bias=False)
                    ).to(memory_format=torch.channels_last_3d)
                else:
                    m_orig = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv3d(K_val, N_val, kernel_size, bias=False),
                        nn.ReLU(),
                    ).to(memory_format=torch.channels_last_3d)
            else:
                if not enable_fusion_modeling:
                    m_orig = nn.Sequential(nn.Linear(K_val, N_val, bias=False))
                else:
                    m_orig = nn.Sequential(
                        nn.ReLU(), nn.Linear(K_val, N_val, bias=False)
                    )
            m_orig = m_orig.cuda().bfloat16()
            if op_name == "conv2d":
                x = torch.randn(
                    M_val, K_val, H, W, dtype=torch.bfloat16, device="cuda"
                ).to(memory_format=torch.channels_last)
            elif op_name == "conv3d":
                x = torch.randn(
                    M_val, K_val, D, H, W, dtype=torch.bfloat16, device="cuda"
                ).to(memory_format=torch.channels_last_3d)
            else:
                x = torch.randn(
                    M_val, K_val, dtype=torch.bfloat16, device="cuda"
                ).requires_grad_()

            # get the bf16 gpu kernel time
            torch._dynamo.reset()
            m_bf16 = torch.compile(copy.deepcopy(m_orig))

            bf16_trace_filename = None
            if save_profile_traces:
                bf16_trace_filename = f"{outfile}_{M_val}_{K_val}_{N_val}_bf16.json"
            b_bf16_e2e_time_s = get_gpu_kernel_time(m_bf16, x, bf16_trace_filename)

            # get the float8 dynamic scaling gpu kernel time
            torch._dynamo.reset()

            if recipe_name == "tensorwise":
                config = Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor(),
                )
            elif recipe_name == "rowwise":
                config = Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerRow(),
                    # for now, use TORCH. In the future might be interesting
                    # to benchmark AUTO and FBGEMM.
                    kernel_preference=KernelPreference.TORCH,
                )
            elif recipe_name == "mxfp8_cublas":
                config = MXDynamicActivationMXWeightConfig(
                    activation_dtype=torch.float8_e4m3fn,
                    weight_dtype=torch.float8_e4m3fn,
                    kernel_preference=KernelPreference.AUTO,
                )
            elif recipe_name == "mxfp4_cutlass":
                config = MXDynamicActivationMXWeightConfig(
                    activation_dtype=torch.float4_e2m1fn_x2,
                    weight_dtype=torch.float4_e2m1fn_x2,
                    kernel_preference=KernelPreference.AUTO,
                )
            elif recipe_name == "nvfp4":
                config = NVFP4DynamicActivationNVFP4WeightConfig(
                    use_dynamic_per_tensor_scale=False,
                )
            else:
                assert False, "unsupported"

            m_fp8_dyn = copy.deepcopy(m_orig)
            if op_name == "linear":
                quantize_(m_fp8_dyn, config)
            elif op_name == "conv2d":
                _is_conv2d = lambda m, fqn: isinstance(m, torch.nn.Conv2d)
                quantize_(m_fp8_dyn, config, filter_fn=_is_conv2d)
            else:
                _is_conv3d = lambda m, fqn: isinstance(m, torch.nn.Conv3d)
                quantize_(m_fp8_dyn, config, filter_fn=_is_conv3d)

            m_fp8_dyn = torch.compile(m_fp8_dyn)

            fp8_trace_filename = None
            if save_profile_traces:
                fp8_trace_filename = f"{outfile}_{M_val}_{K_val}_{N_val}_fp8.json"
            b_fp8_e2e_time_s = get_gpu_kernel_time(m_fp8_dyn, x, fp8_trace_filename)

        results.append(
            [
                M_val,
                K_val,
                N_val,
                D,
                H,
                W,
                kernel_size,
                # roofline - gemm
                r_bf16_gemm_time_s,
                r_fp8_gemm_time_s,
                # roofline - fp8 overhead
                r_fp8_ovhd_time_s,
                # roofline - gemm + overhead, and speedup
                r_fp8_gemm_and_ovhd_s,
                r_speedup,
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
