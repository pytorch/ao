# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from triton.testing import do_bench

from torchao.prototype.sparsity.activation.srelu_linear import (
    SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig,
)
from torchao.sparsity.sparse_api import (
    Float8DynamicSemiSparseActivationFloat8WeightConfig
)
from torchao.prototype.sparsity.activation.utils import SquaredReLU
from torchao.quantization import (
    Float8DynamicActivationFloat8SemiSparseWeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
    Float8MMConfig,
    PerRow,
    quantize_,
)

PROFILE = False


def benchmark_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def benchmark(num_tokens, hidden_size=4096, intermediate_size=16384):
    ffn_ref = (
        nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            SquaredReLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        .to(torch.bfloat16)
        .cuda()
    )

    input_tensor = torch.randn(num_tokens, hidden_size).to(torch.bfloat16).cuda()
    fp16_time = benchmark_microseconds(ffn_ref, input_tensor)

    # bf16
    ffn_clone = (
        nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            SquaredReLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        .to(torch.bfloat16)
        .cuda()
    )
    ffn_clone.forward = torch.compile(ffn_clone.forward, fullgraph=True)
    fp16_c_time = benchmark_microseconds(ffn_clone, input_tensor)

    # fp8
    ffn_clone = (
        nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            SquaredReLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        .to(torch.bfloat16)
        .cuda()
    )
    quantize_(
        ffn_clone,
        Float8DynamicActivationFloat8WeightConfig(
            granularity=PerRow(), mm_config=Float8MMConfig(use_fast_accum=True)
        ),
    )
    ffn_clone.forward = torch.compile(ffn_clone.forward, fullgraph=True)
    fp8_c_time = benchmark_microseconds(ffn_clone, input_tensor)

    # fp8 sparse
    # ffn_clone = (
    #     nn.Sequential(
    #         nn.Linear(hidden_size, intermediate_size, bias=False),
    #         SquaredReLU(),
    #         nn.Linear(intermediate_size, hidden_size, bias=False),
    #     )
    #     .to(torch.bfloat16)
    #     .cuda()
    # )
    # quantize_(ffn_clone, Float8DynamicActivationFloat8SemiSparseWeightConfig())
    # ffn_clone.forward = torch.compile(ffn_clone.forward, fullgraph=True)
    # fp8_c_sparse_time = benchmark_microseconds(ffn_clone, input_tensor)

    if PROFILE:
        print("PROFILING FP8")
        from torchao.prototype.sparsity.activation.utils import profiler_runner
        inputs = (ffn_clone, input_tensor)
        profiler_runner(None, benchmark_microseconds, *inputs)

    # activation fp8 sparse
    ffn_clone = (
        nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            # no Squared RELU since it will be fused into the second linear
            SquaredReLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        .to(torch.bfloat16)
        .cuda()
    )
    quantize_(
        ffn_clone[0],
        Float8DynamicActivationFloat8WeightConfig(
            granularity=PerRow(), mm_config=Float8MMConfig(use_fast_accum=True)
        ),
    )
    quantize_(
        ffn_clone,
        Float8DynamicSemiSparseActivationFloat8WeightConfig(
            granularity=PerRow(), mm_config=Float8MMConfig(use_fast_accum=True)
        ),
        filter_fn=lambda mod, fqn: "2" in fqn,
    )
    ffn_clone.forward = torch.compile(ffn_clone.forward, fullgraph=True)
    fp8_c_activation_sparse_time = benchmark_microseconds(ffn_clone, input_tensor)

    if PROFILE:
        print("PROFILING 24")
        from torchao.prototype.sparsity.activation.utils import profiler_runner
        inputs = (ffn_clone, input_tensor)
        profiler_runner(None, benchmark_microseconds, *inputs)

    return {
        "num_tokens": num_tokens,
        "bf16_latency (us)": fp16_time,
        "bf16_c_latency (us)": fp16_c_time,
        "fp8_c_time (us)": fp8_c_time,
        # "fp8_c_sparse_time (us)": fp8_c_sparse_time,
        "fp8_c_activation_sparse_time (us)": fp8_c_activation_sparse_time,
        "speedup": fp8_c_time / fp8_c_activation_sparse_time,
    }


if __name__ == "__main__":
    with torch.no_grad():
        results = []
        # for num_tokens in tqdm([64, 128, 256, 512, 1024, 2048, 4096]):
        for num_tokens in tqdm([512, 1024, 2048, 4096, 8192]):
            results.append(benchmark(num_tokens))
            torch.compiler.reset()

        df = pd.DataFrame(results)
        df.to_csv("e2e_fp8_sparse.csv", index=False)
        print(df.to_markdown(index=False))
