# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import copy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd

import torch
import torch.utils.benchmark as benchmark
from torchao.float8.config import (
    CastConfig, 
    Float8LinearConfig, 
    ScalingType,
    ScalingGranularity,
)
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.float8_linear_utils import (
    linear_requires_sync,
    sync_float8_amax_and_scale_history,
)
from torchao.float8.float8_tensor import ScaledMMConfig
from utils import get_name_to_shapes_iter
from tqdm import tqdm

# estimating TOPs for matmuls in fp32, fp16, fp8
# assuming A * B = C, with A being M * K, B being K * N, C being M * N

# H100 SXM specs: bottom of https://www.nvidia.com/en-us/data-center/h100/
h100_peak_flops_float32 = 67e12
h100_peak_flops_fp16_tc = 1979e12
h100_peak_tops_float8_tc = 3958e12

dtype_to_peak_tops = {
    torch.float32: h100_peak_flops_float32,
    torch.float16: h100_peak_flops_fp16_tc,
    torch.bfloat16: h100_peak_flops_fp16_tc,
    torch.float8_e4m3fn: h100_peak_tops_float8_tc,
    torch.float8_e5m2: h100_peak_tops_float8_tc,
}

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
    compiled: bool
    use_fast_accum: bool
    scaling_repr: str

    # 3 Times since we are calculating forward backward
    @property
    def ref_tops_sec(self):
        M, K, N = self.shape
        return float(3 * (2 * M * K * N)) / self.ref_time_sec

    @property
    def ref_pct_top_peak(self):
        return self.ref_tops_sec / dtype_to_peak_tops[self.dtype]

    @property
    def float8_tops_sec(self):
        M, K, N = self.shape
        return float(3 * (2 * M * K * N)) / self.float8_time_sec

    @property
    def float8_pct_top_peak(self):
        return self.float8_tops_sec / dtype_to_peak_tops[torch.float8_e4m3fn]


# TODO(future PR): add option to measure GPU kernel time, as in other
# scripts in this folder
def main(
    sweep_path: Optional[Path] = None,
    compile: bool = True,
    n_limit: Optional[int] = None,
    fast_accum_filter: Optional[bool] = None,
    shape_name_filter: Optional[str] = None,
    *,
    shape_gen_name: str = 'llama',
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    scaling_type_input: str = "dynamic",
    scaling_type_weight: str = "dynamic",
    scaling_type_grad_output: str = "dynamic",
    scaling_granularity: str = "tensorwise",
):
    device = "cuda"
    print(f"Compile is set to             | {compile}")

    scaling_type_input = ScalingType(scaling_type_input)
    scaling_type_weight = ScalingType(scaling_type_weight)
    scaling_type_grad_output = ScalingType(scaling_type_grad_output)
    scaling_granularity = ScalingGranularity(scaling_granularity)

    if scaling_type_input is ScalingType.STATIC:
        cast_config_input=CastConfig(
            scaling_type=scaling_type_input,
            static_scale=torch.tensor([1.0], device="cuda"),
            scaling_granularity=scaling_granularity,
        )
    else:
        cast_config_input=CastConfig(
            scaling_type=scaling_type_input,
            scaling_granularity=scaling_granularity,
        )
    if scaling_type_weight is ScalingType.STATIC:
        cast_config_weight=CastConfig(
            scaling_type=scaling_type_weight,
            static_scale=torch.tensor([1.0], device="cuda"),
            scaling_granularity=scaling_granularity,
        )
    else:
        cast_config_weight=CastConfig(
            scaling_type=scaling_type_weight,
            scaling_granularity=scaling_granularity,
        )
    if scaling_type_grad_output is ScalingType.STATIC:
        cast_config_grad_output=CastConfig(
            scaling_type=scaling_type_grad_output,
            static_scale=torch.tensor([1.0], device="cuda"),
            scaling_granularity=scaling_granularity,
        )
    else:
        cast_config_grad_output=CastConfig(
            scaling_type=scaling_type_grad_output,
            scaling_granularity=scaling_granularity,
        )

    config = Float8LinearConfig(
        cast_config_input=cast_config_input,
        cast_config_weight=cast_config_weight,
        cast_config_grad_output=cast_config_grad_output,
    )

    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, M, K, N)
    input_bias = False
    if fast_accum_filter is not None:
        use_fast_accum = [fast_accum_filter]
    else:
        use_fast_accum = [True, False]
    if shape_name_filter is not None:
        k = shape_name_filter
        name_to_shapes = ((k, v) for (k, v) in name_to_shapes if k == shape_name_filter)
    experiment_list: List[Experiment] = []
    dtype = torch.bfloat16
    for idx, (fast_accum, (name, (M, K, N))) in enumerate(
        tqdm(list(product(use_fast_accum, name_to_shapes)))
    ):
        if n_limit is not None and idx >= n_limit:
            break
        linear_ref = torch.nn.Linear(K, N, bias=input_bias).to(
            device=device, dtype=dtype
        )

        linear_float8 = Float8Linear.from_float(
            copy.deepcopy(linear_ref),
            config=config,
        )
        scaling_repr = f"{linear_float8.scaling_type_repr()},{linear_float8.scaling_granularity_repr()}"

        if fast_accum:
            linear_float8.forward_config = ScaledMMConfig(False, True, False)
        else:
            linear_float8.forward_config = ScaledMMConfig(False, False, False)

        input_tensor = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
        ref_forw_backward = lambda: linear_ref(input_tensor).sum().backward()

        def float8_forw_backward():
            if linear_requires_sync(config):
                sync_float8_amax_and_scale_history(linear_float8)
            linear_float8(input_tensor).sum().backward()

        def n_times(n, fn, *args, **kwargs):
            def wrapper(*args, **kwargs):
                for _ in range(n):
                    fn(*args, **kwargs)

            return wrapper

        REPEAT_N = 100

        ref_forw_backward = n_times(REPEAT_N, ref_forw_backward)
        float8_forw_backward = n_times(REPEAT_N, float8_forw_backward)

        if compile:
            ref_forw_backward = torch.compile(ref_forw_backward)
            float8_forw_backward = torch.compile(float8_forw_backward)

        for _ in range(5):
            ref_forw_backward()
            float8_forw_backward()

        ref_time = (
            benchmark_torch_function_in_microseconds(ref_forw_backward)
            * 1e-6
            / REPEAT_N
        )
        float8_time = (
            benchmark_torch_function_in_microseconds(float8_forw_backward)
            * 1e-6
            / REPEAT_N
        )
        experiment = Experiment(
            name,
            (M, K, N),
            ref_time,
            float8_time,
            dtype,
            compile,
            use_fast_accum=fast_accum,
            scaling_repr=scaling_repr,
        )
        print(experiment)
        print("float8 speedup", experiment.ref_time_sec / experiment.float8_time_sec)
        experiment_list.append(experiment)
        torch._dynamo.reset()

    headers = [
        "name",
        "M",
        "K",
        "N",
        "scaling_repr",
        "ref_dtype",
        "compiled",
        "use_fast_accum",
        "ref_time_sec",
        "pt_fp8_time_sec",
        "ref_tops_sec",
        "ref_pct_top_peak",
        "pt_fp8_tops_sec",
        "pt_fp8_pct_top_peak",
    ]
    data = []
    for experiment in experiment_list:
        data.append(
            [
                experiment.name,
                experiment.shape[0],
                experiment.shape[1],
                experiment.shape[2],
                experiment.scaling_repr,
                experiment.dtype,
                experiment.compiled,
                experiment.use_fast_accum,
                experiment.ref_time_sec,
                experiment.float8_time_sec,
                experiment.ref_tops_sec,
                experiment.ref_pct_top_peak,
                experiment.float8_tops_sec,
                experiment.float8_pct_top_peak,
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
            "scaling_repr",
            "compiled",
            "use_fast_accum",
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
    parser.add_argument("--fast_accum_filter", type=bool, required=False)
    parser.add_argument("--shape_name_filter", type=str, required=False)
    parser.add_argument("--scaling_type_input", type=str, required=False)
    parser.add_argument("--scaling_type_weight", type=str, required=False)
    parser.add_argument("--scaling_type_grad_output", type=str, required=False)
    parser.add_argument("--scaling_granularity", type=str, required=False)
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
    if args.scaling_granularity is not None:
        kwargs["scaling_granularity"] = args.scaling_granularity
    main(
        output_path,
        not args.disable_compile,
        args.n_limit,
        args.fast_accum_filter,
        args.shape_name_filter,
        **kwargs,
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
