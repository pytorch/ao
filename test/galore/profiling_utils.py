import gc
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from functools import partial

import torch

logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%m-%d-%H"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


def flush_cuda_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@contextmanager
def cuda_max_memory():
    try:
        flush_cuda_mem()
        yield

    finally:
        mem_miB = torch.cuda.max_memory_allocated() // (1024 * 1024)
        print(f"{mem_miB} MB of CUDA memory allocated")
        flush_cuda_mem()
    return mem_miB


def export_memory_snapshot(prefix) -> None:

    # Prefix for file names.
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{prefix}_{timestamp}"

    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return


@contextmanager
def memory_recorder(file_name="cuda_memory_snapshot") -> None:
    assert (
        torch.cuda.is_available()
    ), "Memory profiler requires GPU, check torch.cuda.is_available()"
    try:
        logger.info("Starting snapshot record_memory_history")
        torch.cuda.memory._record_memory_history(
            max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
        )
        yield
    finally:
        logger.info("Stopping snapshot record_memory_history")
        torch.cuda.memory._record_memory_history(enabled=None)
        export_memory_snapshot(file_name)


def trace_handler(
    prof: torch.profiler.profile,
    prefix: str = "profile",
    output_dir="./",
    sort_key="cuda_time_total",
    export_trace=True,
    export_memory_timeline=True,
    print_table=True,
):

    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = os.path.join(output_dir, f"{prefix}_{timestamp}")

    if export_trace:
        prof.export_chrome_trace(f"{file_prefix}-trace.json.gz")

    if export_memory_timeline:
        prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
        prof.export_memory_timeline(
            f"{file_prefix}-memory-timeline.json", device="cuda:0"
        )
    if print_table:
        print(prof.key_averages().table(sort_by=sort_key, row_limit=10))


def get_torch_profiler(
    name: str = "profile",
    output_dir: str = "./profiler_out",
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    wait_steps=1,
    warmup_steps=1,
    active_steps=10,
    repeat=1,
    # options for profiler outputs
    on_trace_ready=trace_handler,
    export_trace=True,
    export_memory_timeline=True,
    print_table=True,
):
    """
    Args:
        name: name of the profiler, used for output files
        table_key: key to sort profiler table by: one of `cpu_time`, `cuda_time`, `cpu_time_total`,
                `cuda_time_total`, `cpu_memory_usage`, `cuda_memory_usage`,
                `self_cpu_memory_usage`, `self_cuda_memory_usage`, `count`.

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=repeat
        ),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        on_trace_ready=partial(
            on_trace_ready,
            prefix=name,
            output_dir=output_dir,
            export_trace=export_trace,
            export_memory_timeline=export_memory_timeline,
            print_table=print_table,
        ),
    )


@contextmanager
def nsys_profiler():
    try:
        torch.cuda.cudart().cudaProfilerStart()
        free, total = torch.cuda.mem_get_info()
        print(f"Start, Memory Usage: Free {free:.2e}, Used {(total - free):.2e}")
        yield "nsys"
    finally:
        free, total = torch.cuda.mem_get_info()
        print(f"End, Memory Usage: Free {free:.2e}, Used {(total - free):.2e}")
        torch.cuda.cudart().cudaProfilerStop()
