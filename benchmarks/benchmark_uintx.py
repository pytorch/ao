# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy

import torch

from torchao.prototype.uintx import (
    uintx_affine_weight_only,
    unpack_cpu,
)
from torchao.quantization.quant_api import quantize_


class Linear16(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(scale * 2, scale, bias=True, dtype=torch.float16).cuda(),
            torch.nn.Linear(scale, scale, bias=True, dtype=torch.float16).cuda(),
            torch.nn.Linear(scale, scale // 2, bias=True, dtype=torch.float16).cuda(),
        )

    def forward(self, x):
        return self.net(x)


def benchmark(function, args, num_runs):
    # warmup
    torch._dynamo.reset()
    for i in range(100):
        function(*args)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(num_runs):
        function(*args)

    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs


def profile_bitpack():
    from torch.profiler import ProfilerActivity, profile

    fake_tensor = [torch.randint(2**8, (512, 512), dtype=torch.uint8).cuda()]
    func = torch.compile(unpack_cpu, fullgraph=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(1000):
            func(fake_tensor, 4)

    # Print a summary
    with open("profile-bitpack.txt", "a") as f:
        print(f"{func}", file=f)
        print(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=10), file=f
        )
    prof.export_chrome_trace("trace.json")
    """
    CPU perf:
        unpack_gpu
        Self CPU time total: 602.501ms

        unpack_cpu
        Self CPU time total: 415.469ms
    GPU perf:
        unpack_gpu  on gpu:
        Self CPU time total: 58.512ms
        Self CUDA time total: 5.083ms

        unpack_cpu:
        Self CPU time total: 96.947ms
        Self CUDA time total: 5.253ms
    """


def uintx_vs_fp16(nbits=[1, 2, 3, 4, 5, 6, 7], scales=[256, 512, 1024], repeats=30):
    results = []
    nbits.sort()
    scales.sort()
    for scale in scales:
        test_input = torch.randn(scale * 2, dtype=torch.float16).cuda()
        forward_args = [test_input]
        times = [scale]

        fp16 = Linear16(scale)
        fp16c = torch.compile(fp16, fullgraph=True)
        fp16_time = benchmark(fp16c.forward, forward_args, repeats)
        times.append(fp16_time)
        for bit_size in nbits:
            m = deepcopy(fp16)
            quantize_(m, uintx_affine_weight_only(bit_size))
            m = torch.compile(m, fullgraph=True)
            uintx_time = benchmark(m.forward, forward_args, repeats)
            times.append(uintx_time)
        print(f"scale={scale} done")

        results.append(times)
    print("----------- benchmark results -----------")
    for result in results:
        print(f"scale: {result[0]} fp16 time:{result[1]: .2f}ms speedups:")
        for i in range(2, len(result)):
            print(f"int{nbits[i-2]}: {result[1]/result[i]: .2f}x")


if __name__ == "__main__":
    uintx_vs_fp16(nbits=[4, 7])
