import torch
import torch.nn.functional as F
import itertools
import torch.utils.benchmark as benchmark
import math
import csv
from intmm_triton import int_matmul

torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 128

dtype = torch.float16
device = "cuda"

# Format is (m, k, n)
shapes = list(csv.reader(open('intmm_shapes.csv', 'r')))[1:]
# Turn into list of int tuples
shapes = list(map(lambda x: tuple(map(int, x)), shapes))


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


@torch.compile(mode='max-autotune')
def compiled_mm(x, w):
    return torch.mm(x, w)


@torch.compile(mode='max-autotune')
def compiled_int_mm(x, w):
    return torch._int_mm(x, w)


def run_benchmark(x, w, b):
    m, k = x.size()
    k1, n = w.size()
    assert k == k1  # sanity check
    fp_time = benchmark_in_ms(10, 100, torch.mm, x, w)

    x_int = x.to(dtype=torch.int8)
    w_int = w.to(dtype=torch.int8)
    # print(f"w: {w.stride()} w_int: {w_int.stride()}")

    int_mm_time = benchmark_in_ms(10, 100, int_matmul, x_int, w_int)
    ratio = fp_time / int_mm_time

    return (",".join(map(str, [m, k, n, fp_time, int_mm_time, ratio]))), ratio


positives = []
dtype = torch.float16
device = 'cuda'
for (m, k, n) in shapes:
    x = torch.randn(m, k, dtype=dtype, device=device)
    # w = torch.randn(k, n, dtype=dtype, device=device)
    w = torch.randn(n, k, dtype=dtype, device=device).t()

    b = torch.randn(m, n, dtype=dtype, device=device)
    result, ratio = run_benchmark(x, w, b)
    print(result)
    if ratio is not None and ratio > 1.0:
        positives += [result]


# print(",".join(["weightsize", "batchsize", "blocksize", "seqlen", "sparsity", "dense_time", "sparse_time", "ratio"]))
# print("\n".join(positives))
