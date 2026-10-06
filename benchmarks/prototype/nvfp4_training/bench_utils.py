# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the NVFP4 training kernel benchmarks.

The benchmarks compare the Triton and CuteDSL backends of each quantize op on the
same shapes, reporting **device kernel time** (CUDA self-time) rather than wall clock:
NVFP4 training runs the quantizes under CUDA graphs / ``torch.compile``, so host launch
overhead is amortized and the kernel's device time is the metric that matters.
"""

import torch
from torch.profiler import ProfilerActivity, profile


def kernel_time_us(fn, warmup: int = 15, iters: int = 50) -> float:
    """Device kernel time per call (us): summed CUDA self-time averaged over ``iters``.

    Excludes host overhead (so the custom-op dispatch each backend pays is not counted)
    and memcpy/memset, isolating the kernels' device time for an apples-to-apples compare.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = sum(
        (
            getattr(e, "self_device_time_total", 0)
            or getattr(e, "self_cuda_time_total", 0)
        )
        for e in prof.key_averages()
        if "memcpy" not in e.key.lower() and "memset" not in e.key.lower()
    )
    return total / iters
