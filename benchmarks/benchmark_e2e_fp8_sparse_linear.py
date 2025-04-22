# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import torch
from tqdm import tqdm
from triton.testing import do_bench
from torch import nn
import torch.nn.functional as F
import copy

from torchao.ops import rowwise_scaled_linear_sparse_cutlass_f8f8, sparse_semi_structured_tile
from torchao.quantization.quant_api import (
    _float8_cutlass_quant,
    _float8_cutlass_quant_sparse,
)
from torchao.quantization import quantize_, PerRow, Float8DynamicActivationFloat8WeightConfig, Float8DynamicActivationFloat8SemiSparseWeightConfig
from torchao.sparsity.utils import create_semi_structured_tensor

from typing import Optional, Callable

import random
from torch.profiler import ProfilerActivity, profile, record_function
import json

from datetime import datetime
import os

from torch._inductor import config as inductorconfig

inductorconfig.triton.unique_kernel_names = True
inductorconfig.coordinate_descent_tuning = True
inductorconfig.coordinate_descent_check_all_directions = True

def profiler_runner(path, fn, *args, **kwargs):
    if path is None:
        path = os.path.join(
            os.path.expanduser("~/traces"),
            f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json.gz',
        )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    print(f"Exported trace to {path}")
    return result

dtype = torch.bfloat16
dtypeq_X = torch.float8_e4m3fn
dtypeq_W = torch.float8_e4m3fn
device = torch.device("cuda")

def get_problem_cutlass(m: int, n: int, k: int):
    X_ref = torch.randn((m, k), dtype=dtype, device=device)
    W_ref = create_semi_structured_tensor(n, k, dtype=dtype).to(device)

    bias = None
    out_dtype = dtype

    return (X_ref, W_ref), (Xq, X_scale, Wq_sparse, W_meta, W_scale, bias, out_dtype)
    cutlass_custom_compression_time = benchmark_microseconds(torch.ops.torchao.sparse_semi_structured_tile.default, *cutlass_custom_args)

class FP8SemiSparseActivationLinear(torch.nn.Module):
    """
    Replacement nn.Linear that supports runtime fp8 activation sparsity
    """
    def __init__(self, weight) -> None:
        super().__init__()
        W_quant_func = _float8_cutlass_quant
        W_aqt = W_quant_func(weight, dtypeq_W)
        # breakpoint()
        self.Wq = W_aqt.tensor_impl.float8_data
        self.W_scale= W_aqt.tensor_impl.scale

    def forward(self, x):
        X_scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device='cuda')

        Xq_sparse, X_meta = torch.ops.torchao.sparse24_sm90_sparsify(
            x,
            "cutlass",
            "identity",
            sp_selection_algo="largest",
            dtype=torch.float8_e4m3fn,
            scale=X_scale
        )
        return rowwise_scaled_linear_sparse_cutlass_f8f8(self.Wq, self.W_scale, Xq_sparse, X_meta, X_scale, bias=None, out_dtype=dtype)

    @classmethod
    def from_dense(cls, linear):
        mod = cls(linear.weight.data)
        return mod



def benchmark_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def benchmark(num_tokens, ffn):
    # need to copy before compile
    ffn_ref = copy.deepcopy(ffn) 

    input_tensor = torch.randn(num_tokens, ffn.hidden_size).to(torch.bfloat16).cuda()
    fp16_time = benchmark_microseconds(ffn, input_tensor)

    ffn_clone = copy.deepcopy(ffn_ref)
    ffn_clone.forward = torch.compile(ffn_clone.forward)
    fp16_c_time = benchmark_microseconds(ffn_clone, input_tensor)

    # both of them are fp8
    ffn_clone = copy.deepcopy(ffn_ref)
    quantize_(ffn_clone, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    ffn_clone.forward = torch.compile(ffn_clone.forward)
    fp8_c_time = benchmark_microseconds(ffn_clone, input_tensor)

    # both fp8 sparse
    ffn_clone = copy.deepcopy(ffn_ref)
    quantize_(ffn_clone, Float8DynamicActivationFloat8SemiSparseWeightConfig())
    ffn_clone.forward = torch.compile(ffn_clone.forward)
    fp8_c_sparse_time = benchmark_microseconds(ffn_clone, input_tensor)

    # activation sparsity config
    ffn_clone = copy.deepcopy(ffn_ref)
    quantize_(ffn_clone.w1, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    ffn_clone.w2 = FP8SemiSparseActivationLinear.from_dense(ffn_clone.w2)
    # quantize_(ffn_clone.w2, Float8DynamicActivationFloat8SemiSparseWeightConfig())
    ffn_clone.forward = torch.compile(ffn_clone.forward)
    fp8_c_activation_sparse_time = benchmark_microseconds(ffn_clone, input_tensor)


    return {
        "num_tokens": num_tokens,
        "fp16_latency (ms)": fp16_time,
        "fp16_c_latency (ms)": fp16_c_time,
        "fp8_c_time (ms)": fp8_c_time,
        "fp8_c_sparse_time (ms)": fp8_c_sparse_time,
        "fp8_c_activation_sparse_time (ms)": fp8_c_activation_sparse_time,
        "speedup": fp8_c_time / fp8_c_activation_sparse_time,
    }



if __name__ == "__main__":
    results = []
    hidden = 8192
    intermediate = 8192
    test_ffn = FFNSRelu(
        hidden_size=8192,
        intermediate_size=8192,
    ).to(torch.bfloat16).cuda()

    for num_tokens in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        results.append(benchmark(num_tokens, test_ffn))


    # test_ffn = LlamaMLP(
    #     hidden_size=4096,
    #     intermediate_size=14336,
    # ).to(torch.bfloat16).cuda()

    df = pd.DataFrame(results)
    df.to_csv("e2e_fp8_sparse.csv", index=False)
    print(df.to_markdown(index=False))


    # input = create_semi_structured_tensor(4096, 8192, dtype=torch.bfloat16).to(device)
    # print(input)

    # ffn_clone = copy.deepcopy(test_ffn)
    # quantize_(ffn_clone.w1, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    # ffn_clone.w2 = FP8SemiSparseActivationLinear.from_dense(ffn_clone.w2)
    # # quantize_(ffn_clone.w2, Float8DynamicActivationFloat8SemiSparseWeightConfig())
    # ffn_clone.forward = torch.compile(ffn_clone.forward, mode="max-autotune", fullgraph=True)
    # # warmup
    # def test():
    #     for i in range(10):
    #         ffn_clone(input)
    # test()
    # fp8_c_activation_sparse_time = benchmark_microseconds(test)
    # print(fp8_c_activation_sparse_time / 10)

    

    # profiler_runner(None, test)

    # test_linear = nn.Linear(8192, 8192).cuda().to(torch.bfloat16)
    # test_linear.weight.data = torch.ones(8192, 8192).cuda().to(torch.bfloat16)
    # print(test_linear(input))
    # sparse_fp8_linear = FP8SemiSparseActivationLinear.from_dense(test_linear)
    # print(sparse_fp8_linear(input))
