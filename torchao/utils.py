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
    "unwrap_tensor_subclass",
    "TORCH_VERSION_AFTER_2_2",
    "TORCH_VERSION_AFTER_2_3",
    "TORCH_VERSION_AFTER_2_4",
    "TORCH_VERSION_AFTER_2_5",
]


def benchmark_model(model, num_runs, input_tensor):
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    start_event.record()
    start_event.synchronize()

    # benchmark
    for _ in range(num_runs):
        with torch.autograd.profiler.record_function("timed region"):
            model(input_tensor)

    end_event.synchronize()
    end_event.record()
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
    """Unwraps (nested) tensor subclass in the model to plain tensors
    This is a workaround to make a model with tensor subclass to work with `torch.export.export`
    and `torch.aot_compile`, we hope this can be integrated into compile stack soon
    tracking issue: https://github.com/pytorch/ao/issues/345
    """
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


def torch_version_at_least(min_version):
    return version("torch") >= min_version

TORCH_VERSION_AFTER_2_5 = torch_version_at_least("2.5.0.dev")
TORCH_VERSION_AFTER_2_4 = torch_version_at_least("2.4.0.dev")
TORCH_VERSION_AFTER_2_3 = torch_version_at_least("2.3.0.dev")
TORCH_VERSION_AFTER_2_2 = torch_version_at_least("2.2.0.dev")

def is_fbcode():
    return not hasattr(torch.version, "git_version")
