import torch
from typing import Dict, Callable, Union
from collections import defaultdict
import functools
from dataclasses import dataclass
from torchao.utils import TORCH_VERSION_AFTER_2_5

"""
Helper function for implementing aten op or torch function dispatch
and dispatching to these implementations.
"""
def _implements(cls, aten_ops_or_torch_fns, dispatch_condition=None):
    """Use this decorator to implement a function for an aten ops in __torch_dispatch__
    (if user passed in a list of ops)
    or torch function in __torch_function__ (if user passed in a single object)

    class MyTensor(torch.Tensor):
        ...
        implements = classmethod(_implements)

    implements = MyTensor.implements

    # NOTE: implements with `dispatch_condition` has precedence over the ones
    # without `dispatch_condition`
    # and the `dispatch_condition` is checked in the order that they are registered
    def dispatch_condition(func, types, args, kwargs):
        return (
            isinstance(args[0], AffineQuantizedTensor) and
            isinstance(args[1], AffineQuantizedTensor)
        )

    @implements(torch.nn.functional.linear, dispatch_condition):
    def _(func, types, args, kwargs):
        ...

    # fallback implementation
    @implements(torch.nn.functional.linear):
    def _(func, types, args, kwargs):
        ...

    """
    # fallback implementation dispatch table
    if not hasattr(cls, "_DISPATCH_TABLE"):
        cls._DISPATCH_TABLE = {}

    # dispatch table with an extra dispatch_condition check for a more specific use case
    if not hasattr(cls, "_DISPATCH_TABLE_WITH_CONDITION"):
        cls._DISPATCH_TABLE_WITH_CONDITION = defaultdict(list)

    # The reason we separate the fallback implementation dispatch table and dispatch table with condition
    # is to make sure we always have correct precedence between the two, dispatch table with condition
    # should take precedence over the fallback implementation dispatch table

    if not isinstance(aten_ops_or_torch_fns, (list, tuple)):
        aten_ops_or_torch_fns = [aten_ops_or_torch_fns]

    def decorator(func):
        for op in aten_ops_or_torch_fns:
            @functools.wraps(op)
            def wrapper(f, types, args, kwargs):
                return func(f, types, args, kwargs)

            if dispatch_condition is None:
                cls._DISPATCH_TABLE[op] = wrapper
            else:
                cls._DISPATCH_TABLE_WITH_CONDITION[op].append((dispatch_condition, wrapper))

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

    if hasattr(cls, "_DISPATCH_TABLE_WITH_CONDITION") and \
       func in cls._DISPATCH_TABLE_WITH_CONDITION:
        condition_and_impls = cls._DISPATCH_TABLE_WITH_CONDITION[func]
        for dispatch_condition, impl in condition_and_impls:
            if dispatch_condition(func, types, args, kwargs):
                return impl(func, types, args, kwargs)

    # fallback implementation for op
    if hasattr(cls, "_DISPATCH_TABLE") and func in cls._DISPATCH_TABLE:
        return cls._DISPATCH_TABLE[func](func, types, args, kwargs)

    with torch._C.DisableTorchFunctionSubclass():
        return func(*args, **kwargs)

def _dispatch__torch_dispatch__(cls, func, types, args, kwargs):
    """Use this util function for a common `__torch_dispatch__` implementation
    that dispatches to ops/functions registered with `_implements`

    class MyTensor(torch.Tensor):
        ...
        __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)
    """
    if hasattr(cls, "_DISPATCH_TABLE_WITH_CONDITION") and \
       func in cls._DISPATCH_TABLE_WITH_CONDITION:
        condition_and_impls = cls._DISPATCH_TABLE_WITH_CONDITION[func]
        for dispatch_condition, impl in condition_and_impls:
            if dispatch_condition(func, types, args, kwargs):
                return impl(func, types, args, kwargs)

    # fallback implementation for op
    if hasattr(cls, "_DISPATCH_TABLE") and func in cls._DISPATCH_TABLE:
        return cls._DISPATCH_TABLE[func](func, types, args, kwargs)

    raise NotImplementedError(f"{cls.__name__} dispatch: attempting to run unimplemented operator/function: {func}")


"""
Base class for different LayoutType, should not be instantiated directly
"""
@dataclass(frozen=True)
class LayoutType:
    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def post_process(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def extra_repr(self) -> str:
        return ""

"""
Plain LayoutType, the most basic LayoutType, also has no extra metadata, will typically be the default
"""
@dataclass(frozen=True)
class PlainLayoutType(LayoutType):
    pass

"""
layout tensor constructor registration for different tensor subclassesa

first key is a tensor subclass type like AffineQuantizedTensor
second key is an extended layout string, like tensor_core_tiled
value is a constructor for the LayoutTensor class, e.g. TensorCoreTiledAQTLayout.from_plain
"""
_LAYOUT_CONSTRUCTOR_TABLE: Dict[Callable, Dict[type(LayoutType), Callable]] = defaultdict(dict)

def _register_layout_cls(cls: Callable, layout_type_class: type(LayoutType)):
    """Helper function for layout registrations, this is used to implement
    register_layout_cls decorator for each tensor subclass, see aqt.py for example usage

    Args:
        cls: Tensor subclass type
        layout_type_class: the class type of subclass of `LayoutType`, e.g. `PlainLayoutType`

    Returns:
        a decorator that registers the layout tensor constructor in the table
    """
    def decorator(layout_cls):
        _LAYOUT_CONSTRUCTOR_TABLE[cls][layout_type_class] = layout_cls.from_plain
        if TORCH_VERSION_AFTER_2_5:
            # Allow serialization to work for models uses this layout tensor subclass
            torch.serialization.add_safe_globals([layout_type_class, layout_cls])
        return layout_cls
    return decorator

def _get_layout_tensor_constructor(cls: Callable, layout_type_class: type(LayoutType)) -> Callable:
    """Get Layout class constructor (LayoutClass.from_plain) for `cls` based on `layout_type_class`
    `layout_type_class` means the class type of subclass of `LayoutType`, e.g. `PlainLayoutType`

    Args:
        cls: Tensor subclass type
        layout_type_class: the class type of subclass of `LayoutType`, e.g. `PlainLayoutType`

    Returns:
        layout tensor subclass constructor for the layout_type_class
    """
    if cls not in _LAYOUT_CONSTRUCTOR_TABLE:
        raise ValueError(f"no registered layout class constructor for: {cls}")
    if layout_type_class not in _LAYOUT_CONSTRUCTOR_TABLE[cls]:
        raise ValueError(f"layout_name: {layout_type_class} is not supported yet for {cls}")

    return _LAYOUT_CONSTRUCTOR_TABLE[cls][layout_type_class]

def is_device(target_device_str: str, device: Union[str, torch.device]):
    return torch.device(device).type == target_device_str
