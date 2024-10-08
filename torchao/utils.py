import torch
from typing import Tuple, Any, Callable
from functools import reduce
import functools
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
    "TORCH_VERSION_AT_LEAST_2_6",

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
        # torch.export.export / torch._export.export_for_training

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


def _is_float8_type(dtype: torch.dtype) -> bool:
    fp8_types = {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }
    return dtype in fp8_types


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

TORCH_VERSION_AT_LEAST_2_6 = torch_version_at_least("2.6.0")
TORCH_VERSION_AT_LEAST_2_5 = torch_version_at_least("2.5.0")
TORCH_VERSION_AT_LEAST_2_4 = torch_version_at_least("2.4.0")
TORCH_VERSION_AT_LEAST_2_3 = torch_version_at_least("2.3.0")
TORCH_VERSION_AT_LEAST_2_2 = torch_version_at_least("2.2.0")


"""
Helper function for implementing aten op or torch function dispatch
and dispatching to these implementations.
"""
def _implements(cls, aten_ops_or_torch_fns):
    """Use this decorator to implement a function for an aten ops in __torch_dispatch__
    (if user passed in a list of ops)
    or torch function in __torch_function__ (if user passed in a single object)

    class MyTensor(torch.Tensor):
        ...
        implements = classmethod(_implements)

    implements = MyTensor.implements

    @implements(torch.nn.functional.linear):
    def _(func, types, args, kwargs):
        ...

    """
    if not hasattr(cls, "_ATEN_OP_OR_TORCH_FN_TABLE"):
        cls._ATEN_OP_OR_TORCH_FN_TABLE = {}

    if not isinstance(aten_ops_or_torch_fns, (list, tuple)):
        aten_ops_or_torch_fns = [aten_ops_or_torch_fns]
    def decorator(func):
        for op in aten_ops_or_torch_fns:
            @functools.wraps(op)
            def wrapper(f, types, args, kwargs):
                return func(f, types, args, kwargs)

            cls._ATEN_OP_OR_TORCH_FN_TABLE[op] = wrapper
        return func
    return decorator

def _dispatch__torch_function__(cls, func, types, args=(), kwargs=None):
    """Use this util function for a common `__torch_function__` implementation
    that dispatches to ops/functions registered with `_implements`

    class MyTensor(torch.Tensor):
        ...
        __torch_function__ = classmethod(_dispatch__torch_function__)
    """
    kwargs = {} if kwargs is None else kwargs
    if hasattr(cls, "_ATEN_OP_OR_TORCH_FN_TABLE") and \
       func in cls._ATEN_OP_OR_TORCH_FN_TABLE:
        return cls._ATEN_OP_OR_TORCH_FN_TABLE[func](func, types, args, kwargs)

    with torch._C.DisableTorchFunctionSubclass():
        return func(*args, **kwargs)

def _dispatch__torch_dispatch__(cls, func, types, args, kwargs):
    """Use this util function for a common `__torch_dispatch__` implementation
    that dispatches to ops/functions registered with `_implements`

    class MyTensor(torch.Tensor):
        ...
        __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)
    """
    if hasattr(cls, "_ATEN_OP_OR_TORCH_FN_TABLE") and \
       func in cls._ATEN_OP_OR_TORCH_FN_TABLE:
        return cls._ATEN_OP_OR_TORCH_FN_TABLE[func](func, types, args, kwargs)

    arg_types = tuple(type(arg) for arg in args)
    kwarg_types = {k: type(arg) for k, arg in kwargs}
    raise NotImplementedError(f"{cls.__name__} dispatch: attempting to run unimplemented operator/function: {func=}, {types=}, {arg_types=}, {kwarg_types=}")

def _register_layout_cls(cls: Callable, layout_type_class: Callable):
    """Helper function for layout registrations, this is used to implement
    register_layout_cls decorator for each tensor subclass, see aqt.py for example usage

    Args:
        cls: Tensor subclass type
        layout_type_class: the class type of subclass of `LayoutType`, e.g. `PlainLayoutType`

    Returns:
        a decorator that registers the layout tensor constructor in the table
    """

    # cls._LAYOUT_CONSTRUCTOR_TABLE is a map from layout_type_class like TensorCoreTiledLayout
    # to layout class constructor like TensorCoreTiledAQTLayout.from_plain that can construct a layout_tensor
    # from plain data like (quantized, unpacked) `data`, `scale`, `zero_point`
    if not hasattr(cls, "_LAYOUT_CONSTRUCTOR_TABLE"):
        cls._LAYOUT_CONSTRUCTOR_TABLE = {}

    def decorator(layout_class):
        cls._LAYOUT_CONSTRUCTOR_TABLE[layout_type_class] = layout_class.from_plain
        if TORCH_VERSION_AT_LEAST_2_5:
            # Allow serialization to work for models uses this layout tensor subclass
            torch.serialization.add_safe_globals([layout_type_class, layout_class])
        return layout_class
    return decorator

def _get_layout_tensor_constructor(cls: Callable, layout_type_class: Callable) -> Callable:
    """Get Layout class constructor (LayoutClass.from_plain) for `cls` based on `layout_type_class`
    `layout_type_class` means the class type of subclass of `LayoutType`, e.g. `PlainLayoutType`

    Args:
        cls: Tensor subclass type
        layout_type_class: the class type of subclass of `LayoutType`, e.g. `PlainLayoutType`

    Returns:
        layout tensor subclass constructor for the layout_type_class
    """
    if not hasattr(cls, "_LAYOUT_CONSTRUCTOR_TABLE"):
        raise ValueError(f"no registered layout class constructor for: {cls}")
    if layout_type_class not in cls._LAYOUT_CONSTRUCTOR_TABLE:
        raise ValueError(f"layout_name: {layout_type_class} is not supported yet for {cls}")

    return cls._LAYOUT_CONSTRUCTOR_TABLE[layout_type_class]


class TorchAOBaseTensor(torch.Tensor):
    """A util tensor subclass that provides commonly used functions
       new tensor subclass can inherit it to get all the utility functions

       class MyTensor(TorchAOBaseTensor):
           pass

    This includes:
       `_get_to_kwargs` that can get the kwargs for `to`
            class MyTensor(TorchAOBaseTensor):
                def to(self, *args, **kwargs):
                    kwargs = _get_to_kwargs(*args, **kwargs)
                    ...
        `implements`:
            implements = MyTensor.implements

            @implements(torch.nn.functional.linear):
            def _(func, types, args, kwargs):
                ...

        `register_layout_cls`:
            register_layout_cls = MyTensor.register_layout_cls

            @register_layout_cls(PlainLayoutType)
            class PlainAQTLayout(...):
                ...

         `get_layout_tensor_constructor`:
            get_layout_tensor_constructor = MyTensor.get_layout_tensor_constructor
            # in constructor of MyTensor:
            layout_tensor_ctr = get_layout_tensor_constructor(type(layout_type))
            layout_tensor = layout_tensor_ctr(data, scale, zero_point, layout_type)

    """
    implements = classmethod(_implements)
    __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)
    __torch_function__ = classmethod(_dispatch__torch_function__)
    register_layout_cls = classmethod(_register_layout_cls)
    get_layout_tensor_constructor = classmethod(_get_layout_tensor_constructor)

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

def fill_defaults(args, n, defaults_tail):
    """
    __torch_dispatch__ doesn't guarantee the number of arguments you are
    passed (e.g., defaulted arguments are not passed); but usually it is
    convenient to pad out the arguments list with defaults.  This function
    helps you do that.
    Args:
        args: the list of positional arguments passed to __torch_dispatch__
        n: the number of arguments you are expecting to get
        defaults_tail: default values for the arguments, starting from the
            end of the list
    Example:
        >>> fill_defaults([1, 2, 3], 5, [3, 4, 5])
        [1, 2, 3, 4, 5]
        >>> fill_defaults([1, 2, 3], 5, [None, None, None])
        [1, 2, 3, None, None]]
    """
    if n - len(defaults_tail) > len(args):
        raise RuntimeError("not enough defaults to fill arguments")
    r = list(args)
    for i in range(len(args), n):
        r.append(defaults_tail[i - n + len(defaults_tail)])
    return r


## Deprecated, will be deleted in the future
def _torch_version_at_least(min_version):
    return is_fbcode() or version("torch") >= min_version

TORCH_VERSION_AFTER_2_5 = _torch_version_at_least("2.5.0.dev")
TORCH_VERSION_AFTER_2_4 = _torch_version_at_least("2.4.0.dev")
TORCH_VERSION_AFTER_2_3 = _torch_version_at_least("2.3.0.dev")
TORCH_VERSION_AFTER_2_2 = _torch_version_at_least("2.2.0.dev")
