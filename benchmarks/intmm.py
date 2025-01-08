import argparse
import csv
import itertools
import pathlib
import sys

import torch

# Check if CUDA is available, if not, exit the script
if not torch.cuda.is_available():
    print("CUDA is not available. Exiting the script.")
    sys.exit(0)

from torchao.kernel.intmm import int_matmul, int_scaled_matmul

torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 128

dtype = torch.float16
device = "cuda"


def benchmark_in_ms(warmup, iters, f, *args, **kwargs):
    for _ in range(warmup):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(iters):
        f(*args, **kwargs)

    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / float(iters)


@torch.compile(mode="max-autotune")
def compiled_mm(x, w):
    return torch.mm(x, w)


@torch.compile(mode="max-autotune")
def compiled_int_mm(x, w):
    return torch._int_mm(x, w)


def run_int_mm_benchmark(x, w, b):
    fp_time = benchmark_in_ms(10, 100, torch.mm, x, w)
    x_int = x.to(dtype=torch.int8)
    w_int = w.to(dtype=torch.int8)
    int_mm_time = benchmark_in_ms(10, 100, int_matmul, x_int, w_int)
    return fp_time, int_mm_time


def run_int_scaled_mm_benchmark(x, w, b):
    scales = x.sum(-1, keepdim=True)
    fp_time = benchmark_in_ms(10, 100, lambda x, w, s: torch.mm(x, w) * s, x, w, scales)
    x_int = x.to(dtype=torch.int8)
    w_int = w.to(dtype=torch.int8)
    int_scaled_mm_time = benchmark_in_ms(
        10, 100, int_scaled_matmul, x_int, w_int, scales
    )
    return fp_time, int_scaled_mm_time


def run_benchmarks(shapes):
    print("fn,m,k,n,fp_time,int_mm_time,ratio")
    dtype = torch.bfloat16
    device = "cuda"
    for fn, (m, k, n) in itertools.product(
        [run_int_mm_benchmark, run_int_scaled_mm_benchmark], shapes
    ):
        x = torch.randn(m, k, dtype=dtype, device=device)
        w = torch.randn(n, k, dtype=dtype, device=device).t()
        b = torch.randn(m, n, dtype=dtype, device=device)

        fp_time, int_mm_time = fn(x, w, b)
        ratio = fp_time / int_mm_time
        result = ",".join(map(str, [fn, m, k, n, fp_time, int_mm_time, ratio]))
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="integer matmul benchmarks")
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to csv file with shapes"
    )
    args = parser.parse_args()
    # Access the file path provided as an argument
    file_path = args.file_path
    file_path = pathlib.Path(file_path)
    assert file_path.is_file()

    # Format is (m, k, n)
    shapes = list(csv.reader(open(file_path, "r")))[1:]
    # Turn into list of int tuples
    shapes = list(map(lambda x: tuple(map(int, x)), shapes))

    run_benchmarks(shapes)
