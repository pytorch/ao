import torch
from typing import Tuple
from functools import reduce
from importlib.metadata import version
from math import gcd
import torch.nn.utils.parametrize as parametrize
import itertools

__all__ = [
    "benchmark_model",
    "profiler_runner",
    "get_compute_capability",
    "skip_if_compute_capability_less_than",
    "benchmark_torch_function_in_microseconds",
    "find_multiple",
    "get_model_size_in_bytes",
    "TORCH_VERSION_AFTER_2_2",
    "TORCH_VERSION_AFTER_2_3",
    "TORCH_VERSION_AFTER_2_4",
    "TORCH_VERSION_AFTER_2_5",
]


def benchmark_model(model, num_runs, input_tensor):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # benchmark
    for _ in range(num_runs):
        with torch.autograd.profiler.record_function("timed region"):
            model(input_tensor)

    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs

def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result

def get_compute_capability():
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        return float(f"{capability[0]}.{capability[1]}")
    return 0.0

def skip_if_compute_capability_less_than(min_capability):
    import unittest
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            if get_compute_capability() < min_capability:
                raise unittest.SkipTest(f"Compute capability is less than {min_capability}")
            return test_func(*args, **kwargs)
        return wrapper
    return decorator


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    import torch.utils.benchmark as benchmark # this avoids importing numpy when torchao module is loaded

    # Manual warmup
    f(*args, **kwargs)
    f(*args, **kwargs)

    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},  # noqa: E501
    )
    measurement = t0.blocked_autorange()
    return measurement.mean * 1e6


def find_multiple(n: int, *args: Tuple[int]) -> int:
    k: int = reduce(lambda x, y: x * y // gcd(x, y), args + (1,))  # type: ignore[9]
    if n % k == 0:
        return n
    return n + k - (n % k)

def get_model_size_in_bytes(model, ignore_embeddings=False):
    """
    Returns the model size in bytes. The option to ignore embeddings
    is useful for models with disproportionately large embeddings compared
    to other model parameters that get quantized/sparsified.
    """
    def flat_size(tensor):
        if hasattr(tensor, "__tensor_flatten__"):
            size = 0
            # 0th element is a list of attributes that
            # hold tensors
            for attr_name in tensor.__tensor_flatten__()[0]:
                sub_tensor = getattr(tensor, attr_name)
                size += flat_size(sub_tensor)
            return size
        else:
            return tensor.numel() * tensor.element_size()

    model_size = 0
    for name, child in model.named_children():
        if not (isinstance(child, torch.nn.Embedding) and ignore_embeddings):
            for p in itertools.chain(child.parameters(recurse=False), child.buffers(recurse=False)):
                model_size += flat_size(p)
            model_size += get_model_size_in_bytes(child, ignore_embeddings)
    return model_size

def torch_version_at_least(min_version):
    return version("torch") >= min_version

TORCH_VERSION_AFTER_2_5 = torch_version_at_least("2.5.0.dev")
TORCH_VERSION_AFTER_2_4 = torch_version_at_least("2.4.0.dev")
TORCH_VERSION_AFTER_2_3 = torch_version_at_least("2.3.0.dev")
TORCH_VERSION_AFTER_2_2 = torch_version_at_least("2.2.0.dev")

def is_fbcode():
    return not hasattr(torch.version, "git_version")
