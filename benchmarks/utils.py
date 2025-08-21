import torch
from torch.nn import functional as F
from triton.testing import do_bench


def bench_fwd_bwd_microseconds(
    fn, *args, labels=None, use_compile=False, fullgraph=True, **kwargs
):
    assert labels is not None

    def fwd_bwd():
        out = fn(*args, **kwargs)
        loss = F.mse_loss(out, labels)
        loss.backward()

    fwd_bwd_compiled = (
        torch.compile(fwd_bwd, fullgraph=fullgraph) if use_compile else fwd_bwd
    )
    return benchmark_cuda_function_in_microseconds(fwd_bwd_compiled)


def profile_fwd_bwd(
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


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3
