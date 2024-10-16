# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from torchao.testing.float8.test_utils import get_test_float8_linear_config
from torchao.float8 import convert_to_float8_training
from torchao.float8.config import (
    CastConfig, 
    Float8LinearConfig, 
    ScalingType,
    ScalingGranularity,
    Float8LinearRecipeName,
    recipe_name_to_linear_config,
)
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.float8_linear_utils import (
    linear_requires_sync,
    sync_float8_amax_and_scale_history,
)
from utils import (
    get_name_to_shapes_iter,
    do_bench_using_profiling_wrapper_s,
    subgraph_name_to_subgraph_and_inputs,
)
from tqdm import tqdm


# prevent splitting columns when printing a data frame
pd.set_option("display.expand_frame_repr", False)
# print the entire data frame
pd_print_full_ctx = pd.option_context(
    "display.max_rows", None, "display.max_columns", None
)


def benchmark_torch_function_in_microseconds(
    func: Callable,
    *args,
    **kwargs,
) -> float:
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    return t0.blocked_autorange().median * 1e6


@dataclass
class Experiment:
    name: str
    shape: Tuple[int, int, int]
    ref_time_sec: float
    float8_time_sec: float
    dtype: torch.dtype


# TODO(future PR): add option to measure GPU kernel time, as in other
# scripts in this folder
def main(
    sweep_path: Optional[Path] = None,
    compile: bool = True,
    n_limit: Optional[int] = None,
    shape_name_filter: Optional[str] = None,
    *,
    shape_gen_name: str = 'llama',
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    scaling_type_input: str = "dynamic",
    scaling_type_weight: str = "dynamic",
    scaling_type_grad_output: str = "dynamic",
    recipe_name: Optional[str] = None,
    use_e2e_time: bool = True,
    model_type: str = "linear",
):
    device = "cuda"
    print(f"model_type: {model_type}")
    print(f"use_e2e_time: {use_e2e_time}")

    # TODO(future PR): switch to fire.Fire

    scaling_type_input = ScalingType(scaling_type_input)
    scaling_type_weight = ScalingType(scaling_type_weight)
    scaling_type_grad_output = ScalingType(scaling_type_grad_output)

    if recipe_name is None:
        config = get_test_float8_linear_config(
            scaling_type_input,
            scaling_type_weight,
            scaling_type_grad_output,
            emulate=False,
        )
    elif recipe_name is not None:
        recipe_name = Float8LinearRecipeName(recipe_name)
        config = recipe_name_to_linear_config(recipe_name)
    print(f"config: {config}")

    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, M, K, N)
    input_bias = False
    if shape_name_filter is not None:
        k = shape_name_filter
        name_to_shapes = ((k, v) for (k, v) in name_to_shapes if k == shape_name_filter)
    experiment_list: List[Experiment] = []
    dtype = torch.bfloat16
    for idx, (name, (M, K, N)) in enumerate(tqdm(name_to_shapes)):
        if n_limit is not None and idx >= n_limit:
            break
        m_ref, input_tensors = subgraph_name_to_subgraph_and_inputs(
            model_type, mkn_override=(M, K, N))

        # linear_ref = nn.Sequential(nn.Linear(K, N, bias=input_bias)).to(
        #     device=device, dtype=dtype
        # )
        # input_tensor = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)

        m_float8 = copy.deepcopy(m_ref)
        convert_to_float8_training(m_float8, config=config)

        def ref_forw_backward(*input_tensors):
            return m_ref(*input_tensors).sum().backward()

        def float8_forw_backward(*input_tensors):
            if linear_requires_sync(config):
                sync_float8_amax_and_scale_history(m_float8)
            return m_float8(*input_tensors).sum().backward()

        if compile:
            ref_forw_backward = torch.compile(ref_forw_backward)
            float8_forw_backward = torch.compile(float8_forw_backward)

        n_warmup = 2
        for _ in range(n_warmup):
            ref_forw_backward(*input_tensors)
        if not use_e2e_time:
            ref_time = do_bench_using_profiling_wrapper_s(ref_forw_backward, *input_tensors)
        else:
            ref_time = (
                benchmark_torch_function_in_microseconds(ref_forw_backward, *input_tensors)
                * 1e-6
            )

        for _ in range(n_warmup):
            float8_forw_backward(*input_tensors)
        if not use_e2e_time:
            float8_time = do_bench_using_profiling_wrapper_s(float8_forw_backward, *input_tensors)
        else:
            float8_time = (
                benchmark_torch_function_in_microseconds(float8_forw_backward, *input_tensors)
                * 1e-6
            )

        experiment = Experiment(
            name,
            (M, K, N),
            ref_time,
            float8_time,
            dtype,
        )
        experiment_list.append(experiment)
        torch._dynamo.reset()

    headers = [
        "name",
        "M",
        "K",
        "N",
        "ref_dtype",
        "ref_time_sec",
        "pt_fp8_time_sec",
    ]
    data = []
    for experiment in experiment_list:
        data.append(
            [
                experiment.name,
                experiment.shape[0],
                experiment.shape[1],
                experiment.shape[2],
                experiment.dtype,
                experiment.ref_time_sec,
                experiment.float8_time_sec,
            ]
        )

    data_pd = pd.DataFrame(data, columns=headers)
    data_pd["pt_fp8_speedup"] = data_pd["ref_time_sec"] / data_pd["pt_fp8_time_sec"]
    data_pd["shape"] = (
        "("
        + data_pd["M"].astype(str)
        + ", "
        + data_pd["K"].astype(str)
        + ", "
        + data_pd["N"].astype(str)
        + ")"
    )

    data_pd_simple = data_pd[
        [
            "name",
            "shape",
            "ref_time_sec",
            "pt_fp8_time_sec",
            "pt_fp8_speedup",
        ]
    ]
    with pd_print_full_ctx:
        print(data_pd_simple)

    if sweep_path is not None:
        sweep_path = sweep_path.with_suffix(".csv")
        data_pd.to_csv(sweep_path)


def invoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", type=str, required=False)
    parser.add_argument("--disable_compile", action="store_true")
    parser.add_argument("-n", "--n_limit", type=int, required=False)
    parser.add_argument("--shape_gen_name", type=str, required=False)
    parser.add_argument("--M", type=int, required=False)
    parser.add_argument("--K", type=int, required=False)
    parser.add_argument("--N", type=int, required=False)
    parser.add_argument("--shape_name_filter", type=str, required=False)
    parser.add_argument("--scaling_type_input", type=str, required=False)
    parser.add_argument("--scaling_type_weight", type=str, required=False)
    parser.add_argument("--scaling_type_grad_output", type=str, required=False)
    parser.add_argument("--recipe_name", type=str, required=False)
    # TODO(future PR): align this arg name with profiling script after
    # switching to fire.Fire
    parser.add_argument("--use_e2e_time", action="store_true", required=False)
    parser.add_argument("--model_type", type=str, required=False)
    args = parser.parse_args()
    output_path = Path(args.output_path) if args.output_path is not None else None
    kwargs = {}
    if args.shape_gen_name is not None:
        kwargs["shape_gen_name"] = args.shape_gen_name
    if args.M is not None:
        kwargs["M"] = args.M,
    if args.K is not None:
        kwargs["K"] = args.K,
    if args.N is not None:
        kwargs["N"] = args.N,
    if args.scaling_type_input is not None:
        kwargs["scaling_type_input"] = args.scaling_type_input
    if args.scaling_type_weight is not None:
        kwargs["scaling_type_weight"] = args.scaling_type_weight
    if args.scaling_type_grad_output is not None:
        kwargs["scaling_type_grad_output"] = args.scaling_type_grad_output
    if args.recipe_name is not None:
        kwargs["recipe_name"] = args.recipe_name
    if args.use_e2e_time is not None:
        kwargs["use_e2e_time"] = args.use_e2e_time
    if args.model_type is not None:
        kwargs["model_type"] = args.model_type
    main(
        output_path,
        not args.disable_compile,
        args.n_limit,
        args.shape_name_filter,
        **kwargs,
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
