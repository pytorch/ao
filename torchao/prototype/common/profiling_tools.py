# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import os
import types
from datetime import datetime
from functools import partial

import pandas as pd
import torch
import torch.autograd.profiler_util
from tabulate import tabulate
from torch.autograd.profiler import record_function
from torch.cuda.nvtx import range as nvtx_range
from triton.testing import do_bench

# from torch.cuda.nvtx import range_pop, range_push

TIME_FORMAT_STR: str = "%m_%d"
PROFILE_DIR = "./profiles"


def simple_bench(fn, *args, **kwargs):
    t = do_bench(lambda: fn(*args, **kwargs))
    return t


def check(expected, actual, atol=1e-3):
    diff = (expected - actual).abs().max()
    print(f"diff: {diff}")
    # assert diff < atol


def benchmark_mm(
    test_fn, xs, weight, ref_fn=torch.matmul, headers=["M", "K", "N", "test", "ref"]
):
    timings = []
    for x in xs:
        M, K = x.shape
        _, N = weight.shape
        assert x.shape[1] == weight.shape[0]
        print(f"Benchmarking {(M, K, N)}")
        test_times = do_bench(lambda: test_fn(x, weight))
        ref_times = do_bench(lambda: ref_fn(x, weight))
        timings.append([M, K, N, test_times, ref_times])
    return pd.DataFrame(timings, columns=headers)


def run_bench(xs, weight):
    df = benchmark_mm(xs, weight)
    print(tabulate(df, headers="keys", floatfmt=".4f"))
    return df


class CudaProfilerCtx:
    def __enter__(self):
        print("Starting cuda profiler")
        torch.cuda.cudart().cudaProfilerStart()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        print("Stopping cuda profiler")
        torch.cuda.cudart().cudaProfilerStop()
        if exc_type is not None:
            print(f"Exception occurred: {exc_type}, {exc_value}")
        # Return True to suppress the exception
        return True

    def step(self):
        pass


def trace_handler(
    prof: torch.profiler.profile,
    group_by_stack: int = 5,
    group_by_input_shapes: bool = False,
    prefix="",
    out_dir=None,
    export_events=False,
    export_trace=True,
    export_memory_timeline=False,
):
    # Prefix for file names.
    out_dir = out_dir or PROFILE_DIR
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = os.path.join(out_dir, f"{prefix}-{timestamp}")

    if export_events:
        evt_list = prof.key_averages(
            group_by_stack_n=group_by_stack, group_by_input_shape=group_by_input_shapes
        )
        torch.save(evt_list, f"{file_prefix}-key_averages.pt")

    # Construct the trace file.
    if export_trace:
        prof.export_chrome_trace(f"{file_prefix}-chrome-trace.json")

    # Construct the memory timeline file.
    if export_memory_timeline:
        prof.export_memory_timeline(
            f"{file_prefix}-memory-timeline.html", device="cuda:0"
        )
        prof.export_memory_timeline(
            f"{file_prefix}-memory-timeline.json", device="cuda:0"
        )


# print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))


def get_torch_profiler(
    name,
    with_stack=True,
    with_flops=True,
    with_modules=True,
    record_shapes=False,
    export_events=False,
    export_trace=True,
    export_memory_timeline=False,
    out_dir=None,
    warmup=1,
    active=5,
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    callback = partial(
        trace_handler,
        prefix=name,
        out_dir=out_dir,
        group_by_input_shapes=record_shapes,
        group_by_stack=5 if export_events else None,
        export_events=export_events,
        export_trace=export_trace,
        export_memory_timeline=export_memory_timeline,
    )
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        with_stack=with_stack,
        with_flops=with_flops,
        with_modules=with_modules,
        profile_memory=export_memory_timeline,
        schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=active),
        on_trace_ready=callback,
    )


class TorchProfilerCtx:
    @staticmethod
    def profiler(
        name,
        out_dir,
        warmup=1,
        active=5,
        record_shapes=False,
        with_stack=True,
        export_events=False,
        export_trace=True,
        export_memory_timeline=False,
    ):
        return get_torch_profiler(
            name,
            with_stack=with_stack,
            record_shapes=export_memory_timeline or record_shapes,
            export_events=export_events,
            export_trace=export_trace,
            export_memory_timeline=export_memory_timeline,
            out_dir=out_dir,
            warmup=warmup,
            active=active,
        )


def get_annotation_ctx(profiler_type):
    assert profiler_type in ["nsys", "torch"]
    if profiler_type == "nsys":
        return nvtx_range
    else:
        return record_function


_PERF_COLUMNS = [
    "key",
    "count",
    "cpu_children",
    "cpu_parent",
    "self_device_time_total",
    "cuda_time",
    "flops",
    "self_cpu_time",
    "self_cpu_time_total",
    "cpu_time",
    "cpu_time_total" "self_device_memory_usage",
    "device_memory_usage",
    "self_cpu_memory_usage",
    "cpu_memory_usage",
]
PERF_COLS_SELECT = [
    "key",
    "cpu_parent",
    "cpu_children",
    # "self_cpu_time",
    # "self_cpu_time_total",
    "cpu_time",
    "cpu_time_total",
    "cuda_time",
    "self_device_time_total",
]


# cuda_time, cpu_time are avg times -- corresponds to CUDA time avg and CPU time avg in table() above
# "self" times is not meaningful for annotated regions, since they only have child regions
def is_function(obj):
    return isinstance(obj, types.FunctionType)


def is_method(obj):
    return isinstance(obj, types.MethodType)


def is_private(prop):
    return prop.startswith("_")


def should_exclude(obj, prop):
    return (
        is_function(getattr(obj, prop))
        or is_method(getattr(obj, prop))
        or is_private(prop)
    )


def _get_event_props(event: torch.autograd.profiler_util.FunctionEvent):
    props = [p for p in dir(event) if not should_exclude(event, p)]
    return props


def get_events_df(events: torch.autograd.profiler_util.EventList):
    event_props = _get_event_props(events[0])
    data = [{p: getattr(e, p) for p in event_props} for e in events]
    return pd.DataFrame(data)


def get_perf_df(events: torch.autograd.profiler_util.EventList, sort=True):
    df = get_events_df(events).filter(PERF_COLS_SELECT)
    if sort:
        df = df.sort_values(["cpu_time", "cuda_time"], ascending=False)
    return df


def pivot_df(
    df,
    id_cols: str | list[str],
    columns: str | list[str],
    values: str | list[str],
    column_order: list[str] = None,
    show: bool = True,
):
    df = df.pivot_table(
        index=id_cols,
        columns=columns,
        values=values,
    ).reset_index()
    if column_order is not None:
        df = df[column_order]
    if show:
        print(df.to_string(index=False))
    return df
