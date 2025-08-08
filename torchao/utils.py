# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import functools
import importlib
import itertools
import re
import time
import warnings
from functools import reduce
from importlib.metadata import version
from math import gcd
from typing import Any, Callable, Optional

import torch
import torch.nn.utils.parametrize as parametrize
from torch.utils._python_dispatch import return_and_correct_aliasing

__all__ = [
    "benchmark_model",
    "profiler_runner",
    "get_available_devices",
    "get_compute_capability",
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
    "TORCH_VERSION_AT_LEAST_2_7",
    # Needs to be deprecated in the future
    "TORCH_VERSION_AFTER_2_2",
    "TORCH_VERSION_AFTER_2_3",
    "TORCH_VERSION_AFTER_2_4",
    "TORCH_VERSION_AFTER_2_5",
    "is_MI300",
    "is_sm_at_least_89",
    "is_sm_at_least_90",
    "is_package_at_least",
    "DummyModule",
]


# Referenced from: https://github.com/pytorch/pytorch/blob/9105d54c6b37099575c0059ef274c86c4dc80c57/torch/ao/quantization/utils.py#L711
def _assert_and_get_unique_device(module: torch.nn.Module) -> Any:
    """
    Returns the unique device for a module, or None if no device is found.
    Throws an error if multiple devices are detected.
    """
    devices = {p.device for p in module.parameters()} | {
        p.device for p in module.buffers()
    }

    assert len(devices) <= 1, (
        "prepare only works with cpu or single-device CUDA modules, "
        f"but got devices {devices}"
    )
    device = next(iter(devices)) if len(devices) > 0 else None
    return device


def benchmark_model(model, num_runs, args=(), kwargs=None, device_type=None):
    """Benchmark model runs with `args` and `kwargs` both are optional"""
    if kwargs is None:
        kwargs = {}

    if device_type is None:
        assert isinstance(model, torch.nn.Module), (
            "Expecting `model` to be torch.nn.Module if device_type is not provided"
        )
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
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result


def get_available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    elif torch.xpu.is_available():
        devices.append("xpu")
    if TORCH_VERSION_AT_LEAST_2_5:
        if torch.mps.is_available():
            devices.append("mps")
    return devices


def get_compute_capability():
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        return float(f"{capability[0]}.{capability[1]}")
    return 0.0


def compute_max_diff(output: torch.Tensor, output_ref: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref)
    )


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    import torch.utils.benchmark as benchmark  # this avoids importing numpy when torchao module is loaded

    # Manual warmup
    f(*args, **kwargs)
    f(*args, **kwargs)

    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},  # noqa: E501
    )
    measurement = t0.blocked_autorange()
    return measurement.mean * 1e6


def find_multiple(n: int, *args: int) -> int:
    k: int = reduce(lambda x, y: x * y // gcd(x, y), args + (1,))  # type: ignore[9]
    if n % k == 0:
        return n
    return n + k - (n % k)


def _register_custom_op(lib, inductor_decomposed=True):
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

    dispatch_key = (
        "CompositeImplicitAutograd"
        if inductor_decomposed
        else "CompositeExplicitAutograd"
    )

    def decorator(fn):
        if TORCH_VERSION_AT_LEAST_2_5:
            from torch._library.infer_schema import infer_schema

            assert not any(c in fn.__name__ for c in ".<>"), (
                f"Expecting op to be defined in normal functions, not lambda or local: {fn.__name__}"
            )
            op_name = fn.__name__
            if op_name[0] == "_":
                op_name = op_name[1:]
            schema = op_name + infer_schema(fn, mutates_args={})
            lib.define(schema)
            lib.impl(op_name, fn, dispatch_key)

            lib_namespace = lib.ns
            op = getattr(getattr(torch.ops, lib_namespace), op_name)
            if inductor_decomposed:
                register_decomposition([op])(fn)
            return op
        else:
            return fn

    return decorator


def _register_meta_op(lib, op_name):
    def decorator(fn):
        if TORCH_VERSION_AT_LEAST_2_5:
            op = lib.impl(op_name, fn, "Meta")
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
            for p in itertools.chain(
                child.parameters(recurse=False), child.buffers(recurse=False)
            ):
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
            (
                isinstance(child, torch.nn.Linear)
                or isinstance(child, torch.nn.Embedding)
            )
            and hasattr(child, "weight")
            and type(child.weight) is not torch.Tensor
            and type(child.weight) is not torch.nn.Parameter
            and isinstance(child.weight, torch.Tensor)
            and issubclass(type(child.weight), torch.Tensor)
            and isinstance(child.weight, TorchAOBaseTensor)
            and not parametrize.is_parametrized(child)
        ):
            parametrize.register_parametrization(
                child, "weight", UnwrapTensorSubclass()
            )
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
    match = re.match(r"(\d+\.\d+\.\d+)", version_string)
    if match:
        version = match.group(1)
        return [int(x) for x in version.split(".")]
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


def _deprecated_torch_version_at_least(version_str: str) -> str:
    version_str_var_name = "_".join(version_str.split(".")[:2])
    deprecation_msg = f"TORCH_VERSION_AT_LEAST_{version_str_var_name} is deprecated and will be removed in torchao 0.14.0"
    return _BoolDeprecationWrapper(
        torch_version_at_least(version_str),
        deprecation_msg,
    )


def _deprecated_torch_version_after(version_str: str) -> str:
    bool_value = is_fbcode() or version("torch") >= version_str
    version_str_var_name = "_".join(version_str.split(".")[:2])
    deprecation_msg = f"TORCH_VERSION_AFTER_{version_str_var_name} is deprecated and will be removed in torchao 0.14.0"
    return _BoolDeprecationWrapper(bool_value, deprecation_msg)


class _BoolDeprecationWrapper:
    """
    A deprecation wrapper that logs a warning when the given bool value is accessed.
    """

    def __init__(self, bool_value: bool, msg: str):
        self.bool_value = bool_value
        self.msg = msg

    def __bool__(self):
        warnings.warn(self.msg)
        return self.bool_value

    def __eq__(self, other):
        return bool(self) == bool(other)


TORCH_VERSION_AT_LEAST_2_8 = torch_version_at_least("2.8.0")
TORCH_VERSION_AT_LEAST_2_7 = torch_version_at_least("2.7.0")

# Deprecated
TORCH_VERSION_AT_LEAST_2_6 = _deprecated_torch_version_at_least("2.6.0")
TORCH_VERSION_AT_LEAST_2_5 = _deprecated_torch_version_at_least("2.5.0")
TORCH_VERSION_AT_LEAST_2_4 = _deprecated_torch_version_at_least("2.4.0")
TORCH_VERSION_AT_LEAST_2_3 = _deprecated_torch_version_at_least("2.3.0")
TORCH_VERSION_AT_LEAST_2_2 = _deprecated_torch_version_at_least("2.2.0")
TORCH_VERSION_AFTER_2_5 = _deprecated_torch_version_after("2.5.0.dev")
TORCH_VERSION_AFTER_2_4 = _deprecated_torch_version_after("2.4.0.dev")
TORCH_VERSION_AFTER_2_3 = _deprecated_torch_version_after("2.3.0.dev")
TORCH_VERSION_AFTER_2_2 = _deprecated_torch_version_after("2.2.0.dev")


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

    if cls not in cls._ATEN_OP_OR_TORCH_FN_TABLE:
        cls._ATEN_OP_OR_TORCH_FN_TABLE[cls] = {}

    if not isinstance(aten_ops_or_torch_fns, (list, tuple)):
        aten_ops_or_torch_fns = [aten_ops_or_torch_fns]

    def decorator(func):
        for op in aten_ops_or_torch_fns:

            @functools.wraps(op)
            def wrapper(f, types, args, kwargs):
                return func(f, types, args, kwargs)

            cls._ATEN_OP_OR_TORCH_FN_TABLE[cls][op] = wrapper
        return func

    return decorator


def _implements_common_tensor_ops(cls):
    implements = cls.implements
    aten = torch.ops.aten

    @implements(
        [aten.detach.default, aten.clone.default, aten.alias.default, aten.contiguous]
    )
    def _(func, types, args, kwargs):
        return return_and_correct_aliasing(
            func,
            args,
            kwargs,
            args[0]._apply_fn_to_data(lambda x: func(x, *args[1:], **kwargs)),
        )

    def _same_metadata(self: TorchAOBaseTensor, src: TorchAOBaseTensor) -> bool:
        _tensor_shape_match = all(
            getattr(self, t_name).shape == getattr(src, t_name).shape
            for t_name in self.tensor_data_names
        )
        _attr_match = all(
            getattr(self, a_name) == getattr(src, a_name)
            for a_name in self.tensor_attribute_names
        )
        return (
            type(self) == type(src)
            and self.shape == src.shape
            and _tensor_shape_match
            and _attr_match
        )

    @implements(aten.copy_.default)
    def _(func, types, args, kwargs):
        self = args[0]
        src = args[1]
        if _same_metadata(self, src):
            self_tensors = self.__tensor_flatten__()[0]
            for tensor_name in self_tensors:
                getattr(self, tensor_name).copy_(getattr(src, tensor_name))
            return
        raise ValueError(
            f"Not supported args for copy_ due to metadata mismatch: {args[0], args[1]}"
        )

    @implements(aten._to_copy.default)
    def _(func, types, args, kwargs):
        self = args[0]
        if hasattr(self, "tensor_data_names") and hasattr(
            self, "tensor_attribute_names"
        ):
            kwargs = self._get_to_kwargs(*args[1:], **kwargs)
            device = kwargs.pop("device")
            tensors = [
                getattr(self, name).to(device) for name in self.tensor_data_names
            ]
            # change device
            tensor_attributes = [
                getattr(self, attr_name) if attr_name != "device" else device
                for attr_name in self.tensor_attribute_names
            ]
            t = self.__class__(
                *tensors,
                *tensor_attributes,
            )
            return return_and_correct_aliasing(func, args, kwargs, t)

        raise NotImplementedError(
            "Subclasses must implement `aten._to_copy.default` or specify `tensor_data_names` and `tensor_attribute_names` for tensor class or tensor instance before using it"
        )


def _dispatch__torch_function__(cls, func, types, args=(), kwargs=None):
    """Use this util function for a common `__torch_function__` implementation
    that dispatches to ops/functions registered with `_implements`

    class MyTensor(torch.Tensor):
        ...
        __torch_function__ = classmethod(_dispatch__torch_function__)
    """
    kwargs = {} if kwargs is None else kwargs
    if (
        hasattr(cls, "_ATEN_OP_OR_TORCH_FN_TABLE")
        and cls in cls._ATEN_OP_OR_TORCH_FN_TABLE
        and func in cls._ATEN_OP_OR_TORCH_FN_TABLE[cls]
    ):
        return cls._ATEN_OP_OR_TORCH_FN_TABLE[cls][func](func, types, args, kwargs)

    with torch._C.DisableTorchFunctionSubclass():
        return func(*args, **kwargs)


def _dispatch__torch_dispatch__(cls, func, types, args, kwargs):
    """Use this util function for a common `__torch_dispatch__` implementation
    that dispatches to ops/functions registered with `_implements`

    class MyTensor(torch.Tensor):
        ...
        __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)
    """
    if (
        hasattr(cls, "_ATEN_OP_OR_TORCH_FN_TABLE")
        and cls in cls._ATEN_OP_OR_TORCH_FN_TABLE
        and func in cls._ATEN_OP_OR_TORCH_FN_TABLE[cls]
    ):
        return cls._ATEN_OP_OR_TORCH_FN_TABLE[cls][func](func, types, args, kwargs)

    arg_types = tuple(type(arg) for arg in args)
    kwarg_types = {k: type(arg) for k, arg in kwargs.items()}
    raise NotImplementedError(
        f"{cls.__name__} dispatch: attempting to run unimplemented operator/function: {func=}, {types=}, {arg_types=}, {kwarg_types=}"
    )


def _register_layout(tensor_class: Callable, layout_class: Callable):
    """Helper function for layout registrations, this is used to implement
    register_layout decorator for each tensor subclass, see aqt.py for example usage

    Args:
        tensor_class: Tensor subclass type
        layout_class: the class type of subclass of `Layout`, e.g. `PlainLayout`

    Returns:
        a decorator that registers the tensor impl constructor in the table
    """

    # tensor_class._LAYOUT_CONSTRUCTOR_TABLE is a map from layout_class like TensorCoreTiledLayout
    # to tensor_impl class constructor like TensorCoreTiledAQTTensorImpl.from_plain that can construct a tensor_impl
    # from plain data like (quantized, unpacked) `data`, `scale`, `zero_point`
    if not hasattr(tensor_class, "_LAYOUT_CONSTRUCTOR_TABLE"):
        tensor_class._LAYOUT_CONSTRUCTOR_TABLE = {}

    def decorator(tensor_impl_class):
        tensor_class._LAYOUT_CONSTRUCTOR_TABLE[layout_class] = (
            tensor_impl_class.from_plain
        )
        if TORCH_VERSION_AT_LEAST_2_5:
            # Allow serialization to work for models uses this tensor impl subclass
            torch.serialization.add_safe_globals([layout_class, tensor_impl_class])
        return tensor_impl_class

    return decorator


def _get_tensor_impl_constructor(
    tensor_class: Callable, layout_class: Callable
) -> Callable:
    """Get TensorImpl class constructor (TensorImplClass.from_plain) for `tensor_class` based on `layout_class`
    `layout_class` means the class type of subclass of `Layout`, e.g. `PlainLayout`

    Args:
        tensor_class: Tensor subclass type
        layout_class: the class type of subclass of `Layout`, e.g. `PlainLayout`

    Returns:
        tensor impl subclass constructor for the layout_class
    """
    if not hasattr(tensor_class, "_LAYOUT_CONSTRUCTOR_TABLE"):
        raise ValueError(
            f"no registered tensor_impl class constructor for: {tensor_class}"
        )
    if layout_class not in tensor_class._LAYOUT_CONSTRUCTOR_TABLE:
        raise ValueError(
            f"layout_name: {layout_class} is not supported yet for {tensor_class}"
        )

    return tensor_class._LAYOUT_CONSTRUCTOR_TABLE[layout_class]


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

        `register_layout`:
            register_layout = MyTensor.register_layout

            @register_layout(PlainLayout)
            class PlainAQTTensorImpl(...):
                ...

         `get_tensor_impl_constructor`:
            get_tensor_impl_constructor = MyTensor.get_tensor_impl_constructor
            # in constructor of MyTensor:
            tensor_impl_ctr = get_tensor_impl_constructor(type(_layout))
            tensor_impl = tensor_impl_ctr(data, scale, zero_point, _layout)

    """

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, "_ATEN_OP_OR_TORCH_FN_TABLE"):
            cls._ATEN_OP_OR_TORCH_FN_TABLE = {}

        if cls not in cls._ATEN_OP_OR_TORCH_FN_TABLE:
            cls._ATEN_OP_OR_TORCH_FN_TABLE[cls] = {}

        # define the common ops if the tensor_data_names and tensor_attribute_names are defined
        if hasattr(cls, "tensor_data_names") and hasattr(cls, "tensor_attribute_names"):
            cls._implements_common_tensor_ops()

        # inherit the torch function and dispatch implementations from direct parent classes
        # e.g. for `class C(B, A)`, C.__bases__ == (B, A)
        for parent in cls.__bases__:
            if parent in cls._ATEN_OP_OR_TORCH_FN_TABLE:
                cls._ATEN_OP_OR_TORCH_FN_TABLE[cls].update(
                    cls._ATEN_OP_OR_TORCH_FN_TABLE[parent]
                )

    implements = classmethod(_implements)
    _implements_common_tensor_ops = classmethod(_implements_common_tensor_ops)
    __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)
    __torch_function__ = classmethod(_dispatch__torch_function__)
    register_layout = classmethod(_register_layout)
    get_tensor_impl_constructor = classmethod(_get_tensor_impl_constructor)
    _get_to_kwargs = _get_to_kwargs

    def __tensor_flatten__(self):
        if hasattr(self, "tensor_data_names") and hasattr(
            self, "tensor_attribute_names"
        ):
            return self.tensor_data_names, [
                getattr(self, attr) for attr in self.tensor_attribute_names
            ]
        raise NotImplementedError(
            "Subclasses should implement __tensor_flatten__ or specify `tensor_data_names` and `tensor_attribute_names` for tensor class or tensor instance before using it"
        )

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        tensors = [tensor_data_dict[name] for name in cls.tensor_data_names]
        return cls(*tensors, *tensor_attributes)

    def _apply_fn_to_data(self, fn):
        if hasattr(self, "tensor_data_names") and hasattr(
            self, "tensor_attribute_names"
        ):
            tensors = [fn(getattr(self, attr)) for attr in self.tensor_data_names]
            tensor_attributes = [
                getattr(self, attr) for attr in self.tensor_attribute_names
            ]
            return self.__class__(
                *tensors,
                *tensor_attributes,
            )

        raise NotImplementedError(
            "Subclasses should implement _apply_fn_to_data or specify `tensor_data_names` and `tensor_attribute_names` for tensor class or tensor instance before using it"
        )

    def __repr__(self):
        if hasattr(self, "tensor_data_names") and hasattr(
            self, "tensor_attribute_names"
        ):
            repr_str = ""
            repr_str += f"{self.tensor_data_names[0]}={getattr(self, self.tensor_data_names[0])}"
            for tensor_data_name in self.tensor_data_names[1:]:
                repr_str += f", {tensor_data_name}={getattr(self, tensor_data_name)}"
            for tensor_attribute_name in self.tensor_attribute_names:
                repr_str += (
                    f", {tensor_attribute_name}={getattr(self, tensor_attribute_name)}"
                )
            return f"{self.__class__.__name__}({repr_str})"

        raise NotImplementedError(
            "Subclasses must implement __repr__ or specify `tensor_data_names` and `tensor_attribute_names` for tensor class or tensor instance before using it"
        )

    def get_layout(self):
        if not hasattr(self, "_layout"):
            return None
        return self._layout


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


# Supported AMD GPU Models and their LLVM gfx Codes:
#
# | AMD GPU Model | LLVM gfx Code          |
# |---------------|------------------------|
# | Navi4         | gfx1200, gfx1201       |
# | MI300X        | gfx940, gfx941, gfx942 |
# | MI350         | gfx950                 |


def is_ROCM():
    return torch.cuda.is_available() and torch.version.hip


def is_MI300():
    if is_ROCM():
        mxArchName = ["gfx940", "gfx941", "gfx942"]
        archName = torch.cuda.get_device_properties(0).gcnArchName
        for arch in mxArchName:
            if arch in archName:
                return True
    return False


def is_MI350():
    if is_ROCM():
        archName = torch.cuda.get_device_properties(0).gcnArchName
        if "gfx950" in archName:
            return True
    return False


def is_Navi4():
    if is_ROCM():
        archName = torch.cuda.get_device_properties(0).gcnArchName
        if "gfx1200" or "gfx1201" in archName:
            return True
    return False


def is_sm_version(major: int, minor: int) -> bool:
    """Check if the CUDA version is exactly major.minor"""
    is_cuda = torch.cuda.is_available() and torch.version.cuda
    return torch.cuda.get_device_capability() == (major, minor) if is_cuda else False


def is_sm_at_least_89():
    return (
        torch.cuda.is_available()
        and torch.version.cuda
        and torch.cuda.get_device_capability() >= (8, 9)
    )


def is_sm_at_least_90():
    return (
        torch.cuda.is_available()
        and torch.version.cuda
        and torch.cuda.get_device_capability() >= (9, 0)
    )


# TODO(future PR): rename to 8_9, 9_0, 10_0 instead of 89, 10, 100
def is_sm_at_least_100():
    return (
        torch.cuda.is_available()
        and torch.version.cuda
        and torch.cuda.get_device_capability() >= (10, 0)
    )


def check_cpu_version(device, version="2.6.0"):
    if isinstance(device, torch.device):
        device = device.type
    return device == "cpu" and compare_versions(torch.__version__, version) >= 0


def check_xpu_version(device, version="2.8.0"):
    if isinstance(device, torch.device):
        device = device.type
    return device == "xpu" and compare_versions(torch.__version__, version) >= 0


def ceil_div(a, b):
    return (a + b - 1) // b


def is_package_at_least(package_name: str, min_version: str):
    package_exists = importlib.util.find_spec(package_name) is not None
    if not package_exists:
        return False

    return version(package_name) >= min_version


def _is_fbgemm_genai_gpu_available():
    # TODO: use is_package_at_least("fbgemm_gpu", "1.2.0") when
    # https://github.com/pytorch/FBGEMM/issues/4198 is fixed
    if importlib.util.find_spec("fbgemm_gpu") is None:
        return False

    import fbgemm_gpu.experimental.gen_ai  # noqa: F401

    if not is_fbcode() and fbgemm_gpu.__version__ < "1.2.0":
        return False

    return True


class DummyModule(torch.nn.Module):
    """This is used because the TorchAO quantization functions tend to operate on modules so to apply the transform to a tensor, we can load a
    DummyModule with the target tensor and then apply the transformation to the module and then extract the transformed tensor.
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight = weight
        self.bias = bias
