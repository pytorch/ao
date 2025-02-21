# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import io
import os
import pathlib
import random
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass, field
from typing import Callable, Optional

import fire
import pandas as pd

# disable inductor FX cache, so we can can always study the inductor output logs
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)
from utils import (
    kernel_name_to_category,
    parse_bw_and_kernel_name,
    profiler_output_to_filtered_time_by_kernel_name,
    update_triton_kernels_in_prof_chome_trace_with_torch_logs,
)

from torchao.float8.config import (
    Float8LinearConfig,
    ScalingType,
)
from torchao.float8.float8_linear_utils import (
    convert_to_float8_training,
)
from torchao.testing.float8.test_utils import get_test_float8_linear_config

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
    trace_file_path: Optional[str] = None
    logs_file_path: Optional[str] = None
    trace_modified_file_path: Optional[str] = None
    name: Optional[str] = None
    cuda: bool = True
    iters: int = 0
    warmup_iters: int = 0
    sync: bool = False
    extra_kwargs: dict = field(default_factory=dict)
    memory_profile_path: Optional[str] = None


def profile_function(
    config: ProfileConfig,
    func: Callable,
    add_inductor_metadata_to_trace: bool,
    *args,
    **kwargs,
) -> torch.profiler.profile:
    """Profile a torch function and save the result to a file"""
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)

    if add_inductor_metadata_to_trace:
        # ensure we aren't interfering with other torch_log settings
        if os.environ.get("TORCH_LOGS", "") != "":
            raise AssertionError(
                "using TORCH_LOGS together with add_inductor_metadata_to_trace is not supported yet"
            )

        # save torch.compile logs to a file specific to this benchmark run
        # TODO(future): can we hack torch.compile to print to file only and not stdout?
        # or maybe just use tlparse?
        torch._logging.set_logs(output_code=True)
        # by default torch.compile appends to log_file_name, so we delete it
        # if it exists
        if os.path.isfile(config.logs_file_path):
            pathlib.Path(config.logs_file_path).unlink()
        torch._logging._init_logs(log_file_name=config.logs_file_path)

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

    # warm up
    func(*args, **kwargs)

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

    if config.trace_file_path is not None:
        prof.export_chrome_trace(config.trace_file_path)

    if add_inductor_metadata_to_trace:
        # modify the trace to have the triton kernel metadata and code
        # visible inline
        update_triton_kernels_in_prof_chome_trace_with_torch_logs(
            config.trace_file_path,
            config.logs_file_path,
            config.trace_modified_file_path,
        )

        # undo custom log settings
        torch._logging.set_logs(output_code=False)
        torch._logging._init_logs(log_file_name=None)

    return prof


# set up AC for max(abs(tensor))
# context: https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.create_selective_checkpoint_contexts
ops_to_save = [
    torch.ops.aten.abs.default,
    torch.ops.aten.max.default,
]


def policy_fn(ctx, op, *args, **kwargs):
    if op in ops_to_save:
        return CheckpointPolicy.MUST_SAVE
    else:
        return CheckpointPolicy.PREFER_RECOMPUTE


context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)


def main(
    profile_path_prefix: pathlib.Path,
    compile: bool = True,
    scaling_type_input: str = "dynamic",
    scaling_type_weight: str = "dynamic",
    scaling_type_grad_output: str = "dynamic",
    recipe_name: Optional[str] = None,
    model_type: str = "linear",
    dtype_filter: str = "both",
    add_inductor_metadata_to_trace: bool = True,
    enable_activation_checkpointing: bool = False,
):
    assert model_type in (
        "linear",
        "ln_linear",
        "norm_ffn_norm",
        "norm_ffn_norm_small",
    ), "unsupported"
    assert dtype_filter in ("both", "float8", "bfloat16")

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
        config = Float8LinearConfig.from_recipe_name(recipe_name)

    scaling_repr = "_".join(
        [
            s.short_str()
            for s in (scaling_type_input, scaling_type_weight, scaling_type_grad_output)
        ]
    )

    print(f"Compile is set to       | {compile}")
    print(f"model_type is set to    | {model_type}")
    print(f"scaling_repr is set to  | {scaling_repr}")
    print(
        f"enable_activation_checkpointing is set to {enable_activation_checkpointing}"
    )

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
    elif model_type == "norm_ffn_norm_small":
        m_ref = NormFFNResidualNorm(
            dim=4096,
            hidden_dim=4096,
            multiple_of=1024,
            ffn_dim_multiplier=1.0,
        )
        input_tensor = torch.randn(
            1, 2048, 4096, device=device, dtype=ref_dtype
        ).requires_grad_()
    else:
        M, K, N = 2048, 4096, 8192
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
        if enable_activation_checkpointing:
            out = checkpoint(m_ref, x, use_reentrant=False, context_fn=context_fn)
        else:
            out = m_ref(x)
        out.sum().backward()

    def float8_forw(x):
        if enable_activation_checkpointing:
            out = checkpoint(m_float8, x, use_reentrant=False, context_fn=context_fn)
        else:
            out = m_float8(x)
        return out

    def float8_forw_backward_wrapper(x):
        # TODO(future PR): this wrapper is for delayed scaling, we can clean it
        # up now that delayed scaling is deprecated.
        out = float8_forw(x)

        # out.sum().backward() is also not torch.compile fullgraph
        # friendly
        with record_function("backward"):
            out.sum().backward()

    if compile:
        m_ref = torch.compile(m_ref, fullgraph=True)
        float8_forw = torch.compile(float8_forw, fullgraph=True)

    # if the `TORCHINDUCTOR_PROFILE` env var is enabled, parse its output
    # to populate triton kernel bandwidth further down in the script
    if os.environ.get("TORCHINDUCTOR_PROFILE", "") == "":
        context = nullcontext()
        f = None
    else:
        f = io.StringIO()
        context = redirect_stdout(f)
    try:
        with context:
            profile_iters = 5
            ref_times, float8_times = None, None
            data = []

            num_leaf_tensors = 1 + len(list(m_ref.parameters()))

            if dtype_filter != "float8":
                # Profile Reference Model
                print("profiling ref")
                ref_trace_suffix = f"_{model_type}_ref_compile_{compile}.json"
                ref_logs_suffix = f"_{model_type}_ref_compile_{compile}.txt"
                trace_ref_path = profile_path_prefix + ref_trace_suffix
                log_ref_path = profile_path_prefix + ref_logs_suffix
                trace_ref_modified_path = trace_ref_path.replace(
                    ".json", "_modified.json"
                )
                profile_config = ProfileConfig(
                    trace_ref_path,
                    log_ref_path,
                    trace_ref_modified_path,
                    ref_trace_suffix,
                    iters=profile_iters,
                    warmup_iters=2,
                    sync=True,
                )
                p = profile_function(
                    profile_config,
                    ref_forw_backward,
                    add_inductor_metadata_to_trace,
                    input_tensor,
                )
                print(f"saved profiling trace to {trace_ref_path}")
                if add_inductor_metadata_to_trace:
                    print(f"saved torch logs to {log_ref_path}")
                    print(f"saved modified trace to {trace_ref_modified_path}")
                ref_times = profiler_output_to_filtered_time_by_kernel_name(
                    p, profile_iters, num_leaf_tensors
                )
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
                float8_trace_suffix = (
                    f"_{model_type}_float8_compile_{compile}_{scaling_repr}.json"
                )
                float8_log_suffix = (
                    f"_{model_type}_float8_compile_{compile}_{scaling_repr}.txt"
                )
                trace_float8_path = profile_path_prefix + float8_trace_suffix
                log_float8_path = profile_path_prefix + float8_log_suffix
                trace_float8_modified_path = trace_float8_path.replace(
                    ".json", "_modified.json"
                )
                profile_config = ProfileConfig(
                    trace_float8_path,
                    log_float8_path,
                    trace_float8_modified_path,
                    float8_trace_suffix,
                    iters=profile_iters,
                    warmup_iters=2,
                    sync=True,
                )
                p = profile_function(
                    profile_config,
                    float8_forw_backward_wrapper,
                    add_inductor_metadata_to_trace,
                    input_tensor,
                )
                print(f"saved profiling trace to {trace_float8_path}")
                if add_inductor_metadata_to_trace:
                    print(f"saved torch logs to {log_float8_path}")
                    print(f"saved modified trace to {trace_float8_modified_path}")
                float8_times = profiler_output_to_filtered_time_by_kernel_name(
                    p, profile_iters, num_leaf_tensors
                )
                total_time_ms = (
                    sum(v for v in float8_times.values()) / 1e3 / profile_iters
                )
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

    finally:
        if f is not None:
            # print the redirected stdout back to regular stdout
            print(f.getvalue())

    if os.environ.get("TORCHINDUCTOR_PROFILE", "") != "":
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

    print("\nSummary of time (ms) by kernel category\n\n", df_p)


def invoke_main() -> None:
    # Example usage: python benchmarks/profile_linear_float8.py benchmarks/data/profiles/current_profile --compile=True --linear_type="dynamic"
    # You can set TORCHINDUCTOR_PROFILE=1 to also capture triton kernel bandwidth
    fire.Fire(main)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
