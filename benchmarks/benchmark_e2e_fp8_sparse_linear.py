# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from triton.testing import do_bench

from torchao.prototype.sparsity.activation.srelu_linear import (
    ActivationSparseLinearConfig,
)
from torchao.prototype.sparsity.activation.utils import SquaredReLU
from torchao.quantization import (
    Float8DynamicActivationFloat8SemiSparseWeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
    Float8MMConfig,
    PerRow,
    quantize_,
)
from torchao.sparsity.utils import create_binary_tensor


def benchmark_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def benchmark(num_tokens, hidden_size=4096, intermediate_size=16384):

    target_sparsity_output = create_binary_tensor((1, num_tokens, intermediate_size), 0.9).cuda().to(torch.bfloat16)
    target_sparsity_output = torch.randn(num_tokens, intermediate_size).cuda().to(torch.bfloat16) * target_sparsity_output
    print(target_sparsity_output)


    ffn_ref = (
        nn.Sequential(
            # nn.Linear(hidden_size, intermediate_size, bias=False),
            SquaredReLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        .to(torch.bfloat16)
        .cuda()
    )


    # input_tensor = torch.randn(num_tokens, hidden_size).to(torch.bfloat16).cuda()
    input_tensor = target_sparsity_output
    # ffn_ref[0].weight.data = torch.linalg.solve(input_tensor, target_sparsity_output).T
    # ffn_ref[0].weight.data = torch.load("/data/users/jessecai/ao/checkpoints/meta-llama/ffn_up.pt")

    fp16_time = benchmark_microseconds(ffn_ref, input_tensor)
    # breakpoint()

    # bf16
    ffn_clone = copy.deepcopy(ffn_ref)
    ffn_clone.forward = torch.compile(ffn_clone.forward, fullgraph=True)
    fp16_c_time = benchmark_microseconds(ffn_clone, input_tensor)

    # fp8
    ffn_clone = copy.deepcopy(ffn_ref)
    quantize_(
        ffn_clone,
        Float8DynamicActivationFloat8WeightConfig(
            granularity=PerRow(), mm_config=Float8MMConfig(use_fast_accum=True)
        ),
    )
    ffn_clone.forward = torch.compile(ffn_clone.forward, fullgraph=True)
    fp8_c_time = benchmark_microseconds(ffn_clone, input_tensor)

    # fp8 sparse
    ffn_clone = copy.deepcopy(ffn_ref)
    quantize_(ffn_clone, Float8DynamicActivationFloat8SemiSparseWeightConfig())
    ffn_clone.forward = torch.compile(ffn_clone.forward, fullgraph=True)
    fp8_c_sparse_time = benchmark_microseconds(ffn_clone, input_tensor)

    # activation fp8 sparse
    ffn_clone = copy.deepcopy(ffn_ref)
    # quantize_(
    #     ffn_clone[0],
    #     Float8DynamicActivationFloat8WeightConfig(
    #         granularity=PerRow(), mm_config=Float8MMConfig(use_fast_accum=True)
    #     ),
    # )
    # quantize_(
    #     ffn_clone,
    #     SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig(),
    #     filter_fn=lambda mod, fqn: "1" in fqn,
    # )
    quantize_(
        ffn_clone,
        ActivationSparseLinearConfig(),
        filter_fn=lambda mod, fqn: "1" in fqn,
    )
    ffn_clone.forward = torch.compile(ffn_clone.forward, fullgraph=True)
    fp8_c_activation_sparse_time = benchmark_microseconds(ffn_clone, input_tensor)

    return {
        "num_tokens": num_tokens,
        "bf16_latency (us)": fp16_time,
        "bf16_c_latency (us)": fp16_c_time,
        "fp8_c_time (us)": fp8_c_time,
        "fp8_c_sparse_time (us)": fp8_c_sparse_time,
        "fp8_c_activation_sparse_time (us)": fp8_c_activation_sparse_time,
        "speedup": fp8_c_time / fp8_c_activation_sparse_time,
    }


if __name__ == "__main__":
    with torch.no_grad():
        results = []
        for num_tokens in tqdm([1]):
            results.append(benchmark(num_tokens))
            torch.compiler.reset()

        df = pd.DataFrame(results)
        df.to_csv("e2e_fp8_sparse.csv", index=False)
        print(df.to_markdown(index=False))
