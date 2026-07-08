# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import time

import fire
import torch

from torchao.prototype.gptq import GPTQConfig, gptq_quantize
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig,
)


def run(
    K: int = 2048,
    N: int = 4096,
    profile_fname: str = None,
):
    print(f"K={K}, N={N}")

    A = torch.randn(K, K, dtype=torch.float32, device="cuda")
    H = A.t() @ A

    W_t = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    config = GPTQConfig(
        step="convert",
        base_config=NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True,
            use_triton_kernel=True,
        ),
    )

    # Warmup
    print("Warmup...")
    gptq_quantize(H.clone(), W_t.clone(), config)
    torch.cuda.synchronize()

    num_runs = 5
    if profile_fname is not None:
        print("Profiling run...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            torch.cuda.synchronize()
            start = time.time()
            gptq_quantize(H.clone(), W_t.clone(), config)
            torch.cuda.synchronize()
            elapsed = time.time() - start
        print(f"gptq_quantize time: {elapsed:.3f}s")
        prof.export_chrome_trace(profile_fname)
        print(f"Saved: {profile_fname}")
    else:
        print(f"Timed run ({num_runs} iterations)...")
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            gptq_quantize(H.clone(), W_t.clone(), config)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        avg = sum(times) / len(times)
        print(f"gptq_quantize avg time: {avg:.3f}s")


if __name__ == "__main__":
    fire.Fire(run)
