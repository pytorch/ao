import torch
from typing import Tuple, Any
from functools import reduce
from importlib.metadata import version
from math import gcd
import torch.nn.utils.parametrize as parametrize
import itertools
import time
import warnings
import re


__all__ = [
    "benchmark_model",
    "profiler_runner",
    "get_compute_capability",
    "skip_if_compute_capability_less_than",
    "benchmark_torch_function_in_microseconds",
    "find_multiple",
    "_register_custom_op",
    "get_model_size_in_bytes",
    "unwrap_tensor_subclass",
    "TorchAOBaseTensor",
    "TORCH_VERSION_AT_LEAST_2_2",
    "TORCH_VERSION_AT_LEAST_2_3",
    "TORCH_VERSION_AT_LEAST_2_4",
    "TORCH_VERSION_AT_LEAST_2_5",

    # Needs to be deprecated in the future
    "TORCH_VERSION_AFTER_2_2",
    "TORCH_VERSION_AFTER_2_3",
    "TORCH_VERSION_AFTER_2_4",
    "TORCH_VERSION_AFTER_2_5",
]


# Referenced from: https://github.com/pytorch/pytorch/blob/9105d54c6b37099575c0059ef274c86c4dc80c57/torch/ao/quantization/utils.py#L711
def _assert_and_get_unique_device(module: torch.nn.Module) -> Any:
    """
    Returns the unique device for a module, or None if no device is found.
    Throws an error if multiple devices are detected.
    """
    devices = {p.device for p in module.parameters()} | \
        {p.device for p in module.buffers()}

    assert len(devices) <= 1, (
        "prepare only works with cpu or single-device CUDA modules, "
        f"but got devices {devices}"
    )
    device = next(iter(devices)) if len(devices) > 0 else None
    return device


def benchmark_model(model, num_runs, args=(), kwargs=None, device_type=None):
    """Benchmark model runs with `args` and `kwargs` both are optional
    """
    if kwargs is None:
        kwargs = {}

    if device_type is None:
        assert isinstance(model, torch.nn.Module), "Expecting `model` to be torch.nn.Module if device_type is not provided"
        device_type = _assert_and_get_unique_device(model).type

    if device_type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # benchmark
        for _ in range(num_runs):
            with torch.autograd.profiler.record_function("timed region"):
                model(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / num_runs

    elif device_type == "mps":
        torch.mps.synchronize()
        start_event = torch.mps.event.Event(enable_timing=True)
        end_event = torch.mps.event.Event(enable_timing=True)
        start_event.record()

        # benchmark
        for _ in range(num_runs):
            with torch.autograd.profiler.record_function("timed region"):
                model(*args, **kwargs)

        end_event.record()
        torch.mps.synchronize()
        return start_event.elapsed_time(end_event) / num_runs

    elif device_type == "cpu":
        torch.cpu.synchronize()
        start_time = time.time()

        # benchmark
        for _ in range(num_runs):
            with torch.autograd.profiler.record_function("timed region"):
                model(*args, **kwargs)

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

def compute_max_diff(output: torch.Tensor, output_ref: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref))

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

def _register_custom_op(lib):
    """This decorator is used to preserve some high level operators for torch.export.export
    while still allow them to be decomposed for inductor path

    requirement: make sure `fn.__name__[1:]` is the operator name you want to register

    NOTE: This should be applied at the top, after all other decorators have been applied
    NOTE: We haven't tested the case when `fn` accepts tensor subclass instance as input,
    e.g. uint4 tensor subclass instance, and we'll probably need to figure out what would make
    sense for downstream system (like executorch) to accept as well

    Example:
        lib = torch.library.Library("my_namespace', "FRAGMENT")

        register_custom_op = _register_custom_op(lib)

        @register_custom_op
        def _the_op_that_needs_to_be_preserved(...)
            ...

        # after this, `_the_op_that_needs_to_be_preserved` will be preserved as
        # torch.ops.my_namespace.the_op_that_needs_to_be_preserved operator after
        # torch.export.export / torch._export.capture_pre_autograd_graph

    """
    from torch._inductor.decomposition import register_decomposition

    def decorator(fn):
        if TORCH_VERSION_AT_LEAST_2_5:
            from torch._library.infer_schema import infer_schema

            # expecting fn.__name__ starts with `_` and we want to take the rest
            # to be the name of the custom op
            assert fn.__name__[0] == "_", f"Expecting function name starts with `_`, got {fn.__name__}"
            assert not any(c in fn.__name__ for c in ".<>"), f"Expecting op to be defined in normal functions, not lambda or local: {fn.__name__}"
            op_name = fn.__name__[1:]
            schema = op_name + infer_schema(fn, mutates_args={})
            lib.define(schema)
            lib.impl(op_name, fn, "CompositeImplicitAutograd")

            lib_namespace = lib.ns
            op = getattr(getattr(torch.ops, lib_namespace), op_name)
            register_decomposition([op])(fn)
            return op
        else:
            return fn

    return decorator

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

class TorchAOBaseTensor(torch.Tensor):
    """A util tensor subclass that provides commonly used functions
    """
    def _get_to_kwargs(self, *args, **kwargs):
        # `torch._C._nn._parse_to` can't handle `layout` argument
        for arg in args:
            if isinstance(arg, torch.layout):
                args.remove(arg)
        if "layout" in kwargs:
            kwargs.pop("layout")
        # ignoring `non_blocking` and `memory_format` args since these are not
        # very useful for most of the tensor subclasses
        # if in the future there are use cases that need these, we'd recommend
        # to override `_get_to_kwargs` and return these args
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        kwargs = {
            "device": device,
            "dtype": dtype,
        }
        return kwargs



def parse_version(version_string):
    # Extract just the X.Y.Z part from the version string
    match = re.match(r'(\d+\.\d+\.\d+)', version_string)
    if match:
        version = match.group(1)
        return [int(x) for x in version.split('.')]
    else:
        raise ValueError(f"Invalid version string format: {version_string}")

def compare_versions(v1, v2):
    v1_parts = parse_version(v1)
    v2_parts = parse_version(v2)
    return (v1_parts > v2_parts) - (v1_parts < v2_parts)

def is_fbcode():
    return not hasattr(torch.version, "git_version")

def torch_version_at_least(min_version):
    return is_fbcode() or compare_versions(torch.__version__, min_version) >= 0

TORCH_VERSION_AT_LEAST_2_5 = torch_version_at_least("2.5.0")
TORCH_VERSION_AT_LEAST_2_4 = torch_version_at_least("2.4.0")
TORCH_VERSION_AT_LEAST_2_3 = torch_version_at_least("2.3.0")
TORCH_VERSION_AT_LEAST_2_2 = torch_version_at_least("2.2.0")


## Deprecated, will be deleted in the future
def _torch_version_at_least(min_version):
    return is_fbcode() or version("torch") >= min_version

TORCH_VERSION_AFTER_2_5 = _torch_version_at_least("2.5.0.dev")
TORCH_VERSION_AFTER_2_4 = _torch_version_at_least("2.4.0.dev")
TORCH_VERSION_AFTER_2_3 = _torch_version_at_least("2.3.0.dev")
TORCH_VERSION_AFTER_2_2 = _torch_version_at_least("2.2.0.dev")
