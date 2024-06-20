import torch
import torch.utils.benchmark as benchmark
from typing import Tuple
from functools import reduce
from math import gcd
from packaging import version
import torch.nn.utils.parametrize as parametrize
from torch.ao.quantization.utils import _assert_and_get_unique_device
import time

__all__ = [
    "benchmark_model",
    "profiler_runner",
    "get_compute_capability",
    "skip_if_compute_capability_less_than",
    "benchmark_torch_function_in_microseconds",
    "find_multiple",
    "get_model_size_in_bytes",
    "unwrap_tensor_subclass",
    "TORCH_VERSION_AFTER_2_2",
    "TORCH_VERSION_AFTER_2_3",
    "TORCH_VERSION_AFTER_2_4",
]


def benchmark_model(model, num_runs, input_tensor):
    print(_assert_and_get_unique_device(model).type)
    if _assert_and_get_unique_device(model).type == "cuda":
        print("Running on GPU")
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
    
    elif _assert_and_get_unique_device(model).type == "mps":
        print("Running on MPS")
        torch.mps.event.synchronize()
        start_event = torch.mps.event.Event(enable_timing=True)
        end_event = torch.mps.event.Event(enable_timing=True)
        start_event.record()

        # benchmark
        for _ in range(num_runs):
            with torch.autograd.profiler.record_function("timed region"):
                model(input_tensor)

        end_event.record()
        torch.mps.event.synchronize()
        return start_event.elapsed_time(end_event) / num_runs
    
    elif _assert_and_get_unique_device(model).type == "cpu":
        print("Running on CPU")
        torch.cpu.synchronize()
        start_time = time.time()
        # Benchmark
        for _ in range(num_runs):
            with torch.autograd.profiler.record_function("timed region"):
                model(input_tensor)
        # End timing
        end_time = time.time()
        torch.cpu.synchronize()
        average_time_per_run = (end_time - start_time) / num_runs
        return average_time_per_run

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

# https://discuss.pytorch.org/t/finding-model-size/130275
def get_model_size_in_bytes(model):
    s = 0
    for p in model.parameters():
        s += p.nelement() * p.element_size()
    for b in model.buffers():
        s += b.nelement() * b.element_size()
    return s

class UnwrapTensorSubclass(torch.nn.Module):
    def forward(self, *tensors):
        todo = list(tensors)
        for tp, meta, inner_tensors in reversed(self.rebuild_stack):
            nb_tensor = len(inner_tensors)
            inner_tensors = {a: b for a, b in zip(inner_tensors, todo[-nb_tensor:])}
            todo = todo[nb_tensor:]
            rebuilt = tp.__tensor_unflatten__(inner_tensors, meta, None, None)
            todo.append(rebuilt)

        assert len(todo) == 1
        return todo[0]

    def right_inverse(self, tensor):
        assert type(tensor) is not torch.Tensor
        rebuild_stack = []
        plain_tensors = []
        todo = [tensor]
        while todo:
            obj = todo.pop()
            inner_tensors, metadata = obj.__tensor_flatten__()
            rebuild_stack.append((type(obj), metadata, inner_tensors))
            for attr_name in inner_tensors:
                val = getattr(obj, attr_name)
                if type(val) is torch.Tensor:
                    plain_tensors.append(val)
                else:
                    assert isinstance(val, torch.Tensor)
                    todo.append(val)

        self.rebuild_stack = rebuild_stack

        return plain_tensors

def unwrap_tensor_subclass(model, filter_fn=None):
    for name, child in model.named_children():
        # make sure child.weight is a tensor subclass
        if (
            isinstance(child, torch.nn.Linear) and
            hasattr(child, "weight") and
            type(child.weight) is not torch.Tensor and
            type(child.weight) is not torch.nn.Parameter and
            isinstance(child.weight, torch.Tensor) and
            issubclass(type(child.weight), torch.Tensor)
        ):
            parametrize.register_parametrization(child, "weight", UnwrapTensorSubclass())
        unwrap_tensor_subclass(child)
    return model

if version.parse(torch.__version__) >= version.parse("2.4.0.dev"):
    TORCH_VERSION_AFTER_2_4 = True
else:
    TORCH_VERSION_AFTER_2_4 = False

if version.parse(torch.__version__) >= version.parse("2.3.0.dev"):
    TORCH_VERSION_AFTER_2_3 = True
else:
    TORCH_VERSION_AFTER_2_3 = False

if version.parse(torch.__version__) >= version.parse("2.2.0.dev"):
    TORCH_VERSION_AFTER_2_2 = True
else:
    TORCH_VERSION_AFTER_2_2 = False

def is_fbcode():
    return not hasattr(torch.version, "git_version")
