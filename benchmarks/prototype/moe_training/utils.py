import statistics
from time import perf_counter_ns

import torch
from torch.nn import functional as F


def bench_fwd_bwd_microseconds(fn, *args, labels=None, use_compile=False, **kwargs):
    assert labels is not None
    fn = torch.compile(fn, fullgraph=False) if use_compile else fn
    times = []
    for _ in range(10):
        start_ns = perf_counter_ns()
        out = fn(*args, **kwargs)
        loss = F.mse_loss(out, labels)
        loss.backward()
        torch.cuda.synchronize()
        end_ns = perf_counter_ns()
        duration_us = (end_ns - start_ns) / 1000
        times.append(duration_us)
    return statistics.median(times)
