# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import fire
import torch
import torch.nn as nn
from torch._inductor.utils import do_bench_using_profiling

from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
    quantize_,
)


def benchmark_fn_in_usec(f, *args, **kwargs):
    no_args = lambda: f(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3


def run(torch_compile_mode: str = "default"):
    M, K, N = 1024, 2048, 4096
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    m = nn.Sequential(nn.Linear(K, N, device="cuda", dtype=torch.bfloat16))
    quantize_(m, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    m = torch.compile(m, mode=torch_compile_mode)
    # warm up
    with torch.no_grad():
        _ = m(x)
    # measure
    with torch.no_grad():
        time_us = benchmark_fn_in_usec(m, x)
    print("time_us", time_us)


if __name__ == "__main__":
    fire.Fire(run)
