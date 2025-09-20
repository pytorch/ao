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
from torchao.prototype.sparsity.activation.utils import SquaredReLU
from torchao.quantization import (
    Float8DynamicActivationFloat8SemiSparseWeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
    Float8MMConfig,
    PerRow,
    quantize_,
)


def benchmark_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def benchmark(num_tokens, hidden_size=8192, intermediate_size=8192):
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

    # Sparsify-only benchmarks
    ao_fast_sparsification_time = benchmark_microseconds(
        torch.ops.torchao.sparse24_sm90_sparsify(
            input_tensor,
            "cutlass",
            "srelu",
            "largest",
            dtype=torch.float8_e4m3fn,
        )
    )
    cusparse_time = benchmark_microseconds(torch._cslt_compress, input_tensor)

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
    ffn_clone = (
        nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            SquaredReLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        .to(torch.bfloat16)
        .cuda()
    )
    quantize_(ffn_clone, Float8DynamicActivationFloat8SemiSparseWeightConfig())
    ffn_clone.forward = torch.compile(ffn_clone.forward, fullgraph=True)
    fp8_c_sparse_time = benchmark_microseconds(ffn_clone, input_tensor)

    # activation fp8 sparse
    ffn_clone = (
        nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            # no Squared RELU since it will be fused into the second linear
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
        SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig(),
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
        "ao_fast_sparsification_time (us)": ao_fast_sparsification_time,
        "cusparse*_compress_time (us)": cusparse_time,
        "speedup": fp8_c_time / fp8_c_activation_sparse_time,
        "sparsify_speedup": cusparse_time / ao_fast_sparsification_time,
    }


if __name__ == "__main__":
    with torch.no_grad():
        results = []
        for num_tokens in tqdm([64, 128, 256, 512, 1024, 2048, 4096]):
            results.append(benchmark(num_tokens))
            torch.compiler.reset()

        df = pd.DataFrame(results)
        df.to_csv("e2e_fp8_sparse.csv", index=False)
        print(df.to_markdown(index=False))
