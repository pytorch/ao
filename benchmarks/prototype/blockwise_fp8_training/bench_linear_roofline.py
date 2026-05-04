# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import sympy
import torch
import torch.nn as nn
import tqdm

from benchmarks.float8.utils import (
    DSV3_16B_671B_DIM,
    DSV3_16B_671B_INTER_DIM,
    DSV3_16B_671B_SEQ_LEN,
    DSV3_16B_671B_SHAPE_GEN_NAME,
    get_name_to_shapes_iter,
)
from torchao.testing.training.roofline_utils import (
    get_blockwise_float8_mem_sympy,
    get_blockwise_gemm_time_sympy,
    get_gemm_time_sympy,
    get_roofline_gpu_name,
)
from torchao.utils import is_sm_at_least_90


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    M: int
    K: int
    N: int


def _get_shape_configs(
    shape_gen_name: str,
    dsv3_seq_len: int,
    dsv3_dim: int,
    dsv3_inter_dim: int,
    n_limit: Optional[int],
):
    if shape_gen_name == DSV3_16B_671B_SHAPE_GEN_NAME:
        name_to_shapes = get_name_to_shapes_iter(
            shape_gen_name,
            dsv3_seq_len,
            dsv3_dim,
            dsv3_inter_dim,
        )
    else:
        name_to_shapes = get_name_to_shapes_iter(shape_gen_name, None, None, None)

    configs = []
    for idx, (name, (M, K, N)) in enumerate(name_to_shapes):
        if n_limit is not None and idx >= n_limit:
            break
        configs.append(ExperimentConfig(name=str(name), M=M, K=K, N=N))
    return configs


def _set_blockwise_backend(model: nn.Module, use_triton: bool):
    from torchao.prototype.blockwise_fp8_training.linear import Float8BlockwiseLinear

    found_blockwise_linear = False
    for module in model.modules():
        if isinstance(module, Float8BlockwiseLinear):
            module.use_triton = use_triton
            found_blockwise_linear = True
    if not found_blockwise_linear:
        raise AssertionError("expected a Float8BlockwiseLinear")


def _make_models(K: int, N: int, use_triton: bool, compile: bool):
    from torchao.prototype.blockwise_fp8_training.linear import (
        Float8BlockwiseLinearConfig,
    )
    from torchao.quantization import quantize_

    bf16_model = nn.Sequential(nn.Linear(K, N, bias=False)).cuda().bfloat16()
    fp8_model = copy.deepcopy(bf16_model)
    quantize_(fp8_model, config=Float8BlockwiseLinearConfig())
    _set_blockwise_backend(fp8_model, use_triton)

    if compile:
        bf16_model = torch.compile(bf16_model)
        fp8_model = torch.compile(fp8_model)

    return bf16_model, fp8_model


def _benchmark_fwd_bwd_s(
    model: nn.Module,
    x: torch.Tensor,
    grad_output: torch.Tensor,
    warmup: int,
    iterations: int,
):
    def step():
        model.zero_grad(set_to_none=True)
        x.grad = None
        y = model(x)
        y.backward(grad_output)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iterations):
        step()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / 1000 / iterations


def _get_roofline_expressions(gpu_name: str, enable_fusion_modeling: bool):
    M, K, N = sympy.symbols("M K N")
    bf16_gemm_time_sympy = get_gemm_time_sympy(
        M, K, N, torch.bfloat16, None, None, gpu_name
    )
    fp8_gemm_time_sympy = get_blockwise_gemm_time_sympy(M, K, N, gpu_name)
    fp8_ovhd_time_sympy = get_blockwise_float8_mem_sympy(
        M, K, N, enable_fusion_modeling, gpu_name
    )
    return M, K, N, bf16_gemm_time_sympy, fp8_gemm_time_sympy, fp8_ovhd_time_sympy


def _eval_roofline_s(expr, M_sym, K_sym, N_sym, M_val: int, K_val: int, N_val: int):
    return float(expr.subs(M_sym, M_val).subs(K_sym, K_val).subs(N_sym, N_val))


def run(args: argparse.Namespace):
    torch.cuda.get_device_name(0)
    if not is_sm_at_least_90():
        raise RuntimeError("Float8BlockwiseLinear benchmarks require CUDA SM90+")

    torch.random.manual_seed(args.seed)
    roofline_gpu_name = get_roofline_gpu_name(args.roofline_gpu_name)
    backend = "blockwise_triton_gemm" if args.use_triton else "blockwise_scaled_mm"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch version: {torch.__version__}")
    print(f"shape_gen_name: {args.shape_gen_name}")
    print(f"roofline_gpu_name: {roofline_gpu_name}")
    print(f"backend: {backend}")
    print(f"compile: {args.compile}")

    configs = _get_shape_configs(
        args.shape_gen_name,
        args.dsv3_seq_len,
        args.dsv3_dim,
        args.dsv3_inter_dim,
        args.n_limit,
    )
    M_sym, K_sym, N_sym, r_bf16_expr, r_fp8_gemm_expr, r_fp8_ovhd_expr = (
        _get_roofline_expressions(roofline_gpu_name, args.enable_fusion_modeling)
    )

    rows = []
    for config in tqdm.tqdm(configs):
        if config.K % 128 != 0 or config.N % 128 != 0:
            raise AssertionError(
                f"{config.name}: K={config.K} and N={config.N} must be divisible by 128"
            )

        bf16_model, fp8_model = _make_models(
            config.K,
            config.N,
            args.use_triton,
            args.compile,
        )
        x = torch.randn(
            config.M,
            config.K,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        grad_output = torch.randn(
            config.M,
            config.N,
            dtype=torch.bfloat16,
            device="cuda",
        )

        b_bf16_e2e_s = _benchmark_fwd_bwd_s(
            bf16_model, x, grad_output, args.warmup, args.iterations
        )
        b_fp8_e2e_s = _benchmark_fwd_bwd_s(
            fp8_model, x, grad_output, args.warmup, args.iterations
        )

        r_bf16_gemm_s = _eval_roofline_s(
            r_bf16_expr, M_sym, K_sym, N_sym, config.M, config.K, config.N
        )
        r_fp8_gemm_s = _eval_roofline_s(
            r_fp8_gemm_expr, M_sym, K_sym, N_sym, config.M, config.K, config.N
        )
        r_fp8_ovhd_s = _eval_roofline_s(
            r_fp8_ovhd_expr, M_sym, K_sym, N_sym, config.M, config.K, config.N
        )
        r_fp8_gemm_and_ovhd_s = r_fp8_gemm_s + r_fp8_ovhd_s
        r_fp8_gemm_and_ovhd_spdp = r_bf16_gemm_s / r_fp8_gemm_and_ovhd_s
        b_fp8_e2e_spdp = b_bf16_e2e_s / b_fp8_e2e_s
        b_fp8_e2e_spdp_ratio_of_r = b_fp8_e2e_spdp / r_fp8_gemm_and_ovhd_spdp

        rows.append(
            {
                "name": config.name,
                "fp8_backend": backend,
                "compiled": args.compile,
                "fwd_M": config.M,
                "fwd_K": config.K,
                "fwd_N": config.N,
                "r_bf16_gemm_s": r_bf16_gemm_s,
                "r_fp8_gemm_s": r_fp8_gemm_s,
                "r_fp8_ovhd_s": r_fp8_ovhd_s,
                "r_fp8_gemm_and_ovhd_s": r_fp8_gemm_and_ovhd_s,
                "r_fp8_gemm_and_ovhd_spdp": r_fp8_gemm_and_ovhd_spdp,
                "b_bf16_e2e_s": b_bf16_e2e_s,
                "b_fp8_e2e_s": b_fp8_e2e_s,
                "b_fp8_e2e_spdp": b_fp8_e2e_spdp,
                "b_fp8_e2e_spdp_ratio_of_r": b_fp8_e2e_spdp_ratio_of_r,
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.precision", 3)
    print(df)
    if args.outfile:
        df.to_csv(args.outfile, index=False)
        print(f"wrote {args.outfile}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Float8BlockwiseLinear fwd/bwd and compare measured "
            "speedup against the shared blockwise FP8 roofline target."
        )
    )
    parser.add_argument("--outfile", default=None)
    parser.add_argument("--shape_gen_name", default=DSV3_16B_671B_SHAPE_GEN_NAME)
    parser.add_argument("--n_limit", type=int, default=None)
    parser.add_argument("--dsv3_seq_len", type=int, default=DSV3_16B_671B_SEQ_LEN)
    parser.add_argument("--dsv3_dim", type=int, default=DSV3_16B_671B_DIM)
    parser.add_argument("--dsv3_inter_dim", type=int, default=DSV3_16B_671B_INTER_DIM)
    parser.add_argument("--roofline_gpu_name", default=None)
    parser.add_argument("--enable_fusion_modeling", action="store_true")
    parser.add_argument("--use_triton", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
