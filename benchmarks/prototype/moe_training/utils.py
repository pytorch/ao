import statistics
from time import perf_counter_ns

import torch
from torch.nn import functional as F


def bench_fwd_bwd_microseconds(
    fn, *args, labels=None, use_compile=False, fullgraph=True, **kwargs
):
    assert labels is not None
    fn = torch.compile(fn, fullgraph=fullgraph) if use_compile else fn
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


def profile_fn(
    fn,
    *args,
    labels=None,
    use_compile=False,
    fullgraph=True,
    profile_name="profile",
    **kwargs,
):
    assert labels is not None
    fn = torch.compile(fn, fullgraph=fullgraph) if use_compile else fn
    wait, warmup, active = 1, 3, 1
    total_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=0
        ),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(total_steps):
            out = fn(*args, **kwargs)
            loss = F.mse_loss(out, labels)
            loss.backward()
            prof.step()

    # Save profiler results
    prof.export_chrome_trace(f"{profile_name}.json")
    print(f"Saved: {profile_name}.json")
