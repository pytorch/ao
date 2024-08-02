# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import io
import random
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import fire
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.float8.config import CastConfig, Float8LinearConfig, ScalingType
from torchao.float8.float8_linear_utils import (
    convert_to_float8_training,
    linear_requires_sync,
    sync_float8_amax_and_scale_history,
)
from torch.profiler import profile, ProfilerActivity, record_function
from utils import (
    kernel_name_to_category,
    parse_bw_and_kernel_name,
    profiler_output_to_gpu_time_for_key,
    profiler_output_to_time_by_kernel_name,
)

# don't truncate long kernel names
pd.options.display.max_colwidth = 100
# display 3 trailing decimal points for floats
pd.set_option("display.float_format", "{:.3f}".format)


class LNLinear(torch.nn.Module):
    def __init__(self, fc_dim1, fc_dim2):
        super().__init__()
        self.ln = torch.nn.LayerNorm(fc_dim1, elementwise_affine=False)
        self.fc = torch.nn.Linear(fc_dim1, fc_dim2, bias=False)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        return x


# copied from https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/norms.py
class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


# copied from https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama/model.py
class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class NormFFNResidualNorm(nn.Module):
    """
    A fragment representing the end of TransformerBlock n and the start
    of TransformerBlock n + 1, intended to include the fusions relevant
    to float8 gemms in the FFN module in forward and backward.
    """

    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__()
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
        self.attn_norm = RMSNorm(dim)

    def forward(self, h):
        # end of transformer block n
        x = self.ffn_norm(h)
        x = self.ffn(x)
        x = h + x
        # start of transformer block n + 1
        x = self.attn_norm(x)
        return x


@dataclass
class ProfileConfig:
    file_path: Optional[str] = None
    name: Optional[str] = None
    cuda: bool = True
    iters: int = 0
    warmup_iters: int = 0
    sync: bool = False
    extra_kwargs: dict = field(default_factory=dict)
    memory_profile_path: Optional[str] = None


def profile_function(
    config: ProfileConfig, func: Callable, *args, **kwargs
) -> torch.profiler.profile:
    """Profile a torch function and save the result to a file"""
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)

    activities = [ProfilerActivity.CPU]
    if config.cuda:
        activities.append(ProfilerActivity.CUDA)

    if config.warmup_iters >= 0:
        for _ in range(config.warmup_iters):
            func(*args, **kwargs)
    if config.sync:
        torch.cuda.synchronize()
    name_context = (
        nullcontext() if config.name is None else record_function(config.name)
    )
    profile_memory = config.memory_profile_path is not None
    with profile(
        activities=activities,
        profile_memory=profile_memory,
        record_shapes=profile_memory,
        with_stack=profile_memory,
        **config.extra_kwargs,
    ) as prof:
        for _ in range(config.iters):
            with name_context:
                func(*args, **kwargs)
                if config.sync:
                    torch.cuda.synchronize()

    if config.file_path is not None:
        prof.export_chrome_trace(config.file_path)

    return prof


def main(
    profile_path_prefix: Path,
    compile: bool = True,
    scaling_type_input: str = "dynamic",
    scaling_type_weight: str = "dynamic",
    scaling_type_grad_output: str = "dynamic",
    model_type: str = "linear",
    dtype_filter: str = "both",
):
    assert model_type in ("linear", "ln_linear", "norm_ffn_norm"), "unsupported"
    assert dtype_filter in ("both", "float8", "bfloat16")

    scaling_type_input = ScalingType(scaling_type_input)
    scaling_type_weight = ScalingType(scaling_type_weight)
    scaling_type_grad_output = ScalingType(scaling_type_grad_output)
    config = Float8LinearConfig(
        cast_config_input=CastConfig(scaling_type=scaling_type_input),
        cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
        cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
    )
    scaling_repr = "_".join(
        [
            s.short_str()
            for s in (scaling_type_input, scaling_type_weight, scaling_type_grad_output)
        ]
    )

    print(f"Compile is set to       | {compile}")
    print(f"model_type is set to    | {model_type}")
    print(f"scaling_repr is set to  | {scaling_repr}")

    device = "cuda"
    ref_dtype = torch.bfloat16
    if model_type == "ln_linear":
        M, K, N = 4 * 4096, 8192, 7168
        m_ref = LNLinear(K, N)
        input_tensor = torch.randn(
            M, K, device=device, dtype=ref_dtype, requires_grad=True
        )
    elif model_type == "norm_ffn_norm":
        m_ref = NormFFNResidualNorm(
            dim=4096,
            hidden_dim=16384,
            multiple_of=1024,
            ffn_dim_multiplier=1.3,
        )
        input_tensor = torch.randn(
            1, 8192, 4096, device=device, dtype=ref_dtype
        ).requires_grad_()
    else:
        M, K, N = 4 * 4096, 8192, 7168
        m_ref = torch.nn.Sequential(
            torch.nn.Linear(K, N, bias=False),
        )
        input_tensor = torch.randn(
            M, K, device=device, dtype=ref_dtype, requires_grad=True
        )

    m_ref = m_ref.to(device).to(ref_dtype)

    m_float8 = copy.deepcopy(m_ref)
    convert_to_float8_training(m_float8, config=config)

    def ref_forw_backward(x):
        out = m_ref(x)
        out.sum().backward()

    def float8_forw(x):
        out = m_float8(x)
        return out

    sync_amax_history = sync_float8_amax_and_scale_history

    def float8_forw_backward_wrapper(x):
        # sync_float8_amax_and_scale_history is not full graph torch
        # compile friendly, so we add a high level wrapper to allow
        # inspection of the fw+bw torch.compile without the scale
        # syncing code
        # TODO(future): make this better
        if linear_requires_sync(config):
            with record_function("scale_amax_and_scales"):
                sync_amax_history(m_float8)
        out = float8_forw(x)

        # out.sum().backward() is also not torch.compile fullgraph
        # friendly
        with record_function("backward"):
            out.sum().backward()

    if compile:
        m_ref = torch.compile(m_ref, fullgraph=True)
        float8_forw = torch.compile(float8_forw, fullgraph=True)
        # Note: it's faster to compile the combination of sync_amax_history wit
        # forward because we only look up from dynamo cache once.
        # However, compiling the sync function separately makes it more
        # convenient to analyze the total time spent on it.
        sync_amax_history = torch.compile(sync_amax_history)

    # if the `TORCHINDUCTOR_PROFILE` env var is enabled, parse its output
    # to populate triton kernel bandwidth further down in the script
    f = io.StringIO()
    with redirect_stdout(f):
        # warm up
        for _ in range(1):
            if dtype_filter != "float8":
                ref_forw_backward(input_tensor)
            if dtype_filter != "bfloat16":
                float8_forw_backward_wrapper(input_tensor)

        profile_iters = 5
        ref_times, float8_times = None, None
        data = []

        if dtype_filter != "float8":
            # Profile Reference Model
            print("profiling ref")
            ref_suffix = f"_{model_type}_ref_compile_{compile}.json"
            ref_path = profile_path_prefix + ref_suffix
            profile_config = ProfileConfig(
                ref_path, ref_suffix, iters=profile_iters, warmup_iters=2, sync=True
            )
            p = profile_function(profile_config, ref_forw_backward, input_tensor)
            print(f"saved {ref_path}")
            ref_times = profiler_output_to_time_by_kernel_name(p)
            total_time_ms = sum(v for v in ref_times.values()) / 1e3 / profile_iters
            for k, v in ref_times.items():
                v_ms = v / 1e3 / profile_iters
                data.append(
                    [
                        "0_ref",
                        k,
                        kernel_name_to_category(k),
                        v_ms,
                        v_ms / total_time_ms,
                        None,
                    ]
                )

        if dtype_filter != "bfloat16":
            # Profile Float8 Model
            print("profiling float8")
            float8_suffix = (
                f"_{model_type}_float8_compile_{compile}_{scaling_repr}.json"
            )
            float8_path = profile_path_prefix + float8_suffix
            profile_config = ProfileConfig(
                float8_path,
                float8_suffix,
                iters=profile_iters,
                warmup_iters=2,
                sync=True,
            )
            p = profile_function(
                profile_config, float8_forw_backward_wrapper, input_tensor
            )
            print(f"saved {float8_path}")
            float8_times = profiler_output_to_time_by_kernel_name(p)
            total_time_ms = sum(v for v in float8_times.values()) / 1e3 / profile_iters
            for k, v in float8_times.items():
                v_ms = v / 1e3 / profile_iters
                data.append(
                    [
                        "1_float8",
                        k,
                        kernel_name_to_category(k),
                        v / 1e3 / profile_iters,
                        v_ms / total_time_ms,
                        None,
                    ]
                )

            # get the time spent per user annotation
            sync_time_us = profiler_output_to_gpu_time_for_key(
                p, "scale_amax_and_scales"
            )
            sync_time_ms = sync_time_us / profile_iters / 1e3
            print(f"Sync time ms: {sync_time_ms}")

    # print the redirected stdout back to regular stdout
    print(f.getvalue())

    # populate the triton kernel bandwidth
    for line in f.getvalue().split("\n"):
        maybe_bw, maybe_kernel_name = parse_bw_and_kernel_name(line)
        if maybe_kernel_name is not None:
            # O(N) search, but it's ok since lists are small
            for datum in data:
                if datum[1] == maybe_kernel_name:
                    datum[-1] = maybe_bw

    df = pd.DataFrame(
        data,
        columns=[
            "experiment",
            "kernel",
            "category",
            "time_ms",
            "pct_gpu_time",
            "bw_gpbs",
        ],
    )
    df.sort_values(
        ["experiment", "category", "pct_gpu_time"],
        ascending=[True, True, False],
        inplace=True,
    )
    print("\nSummary of GPU time by CPU kernel\n\n", df)

    # compare gemm and overhead time
    df_p = df.pivot_table(
        columns=["category"],
        index="experiment",
        values="time_ms",
        aggfunc="sum",
        fill_value=0,
        margins=True,
    )
    # drop last row, which has totals across ref + float8 which does not make sense
    df_p = df_p[:-1]
    df_p = df_p.transpose()

    if dtype_filter == "both":
        df_p["f8_div_ref"] = df_p["1_float8"] / df_p["0_ref"]
        df_p["ref_div_f8"] = df_p["0_ref"] / df_p["1_float8"]

        # calculate sync time as pct of total float time
        # note: this time is not useful if TORCHINDUCTOR_PROFILE is on
        total_float8_ms = df_p.iloc[3]["1_float8"]
        sync_approx_ratio = sync_time_ms / total_float8_ms
        print(
            f"\nFloat8 amax/scale sync approx ratio of total time: {sync_approx_ratio:.3f}"
        )

    print("\nSummary of time (ms) by kernel category\n\n", df_p)


def invoke_main() -> None:
    # Example usage: python benchmarks/profile_linear_float8.py benchmarks/data/profiles/current_profile --compile=True --linear_type="dynamic"
    # You can set TORCHINDUCTOR_PROFILE=1 to also capture triton kernel bandwidth
    fire.Fire(main)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
