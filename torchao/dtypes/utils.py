import torch
from typing import Dict, Callable
from collections import defaultdict
import functools
from dataclasses import dataclass

"""
torch_function and torch_dispatch operator dispatch registrations

first key is a tensor subclass type like AffineQuantizedTensor,
second key is a `func` in __torhc_function__ or __torch_dispatch__,
value is a function that implements the dispatch
"""
_ATEN_OP_OR_TORCH_FN_TABLE: Dict[Callable, Dict[Callable, Callable]] = defaultdict(dict)

def _implements(cls, aten_ops_or_torch_fns):
    """Use this decorator to implement a function for an aten ops in __torch_dispatch__
    (if user passed in a list of ops)
    or torch function in __torch_function__ (if user passed in a single object)
    """
    if not isinstance(aten_ops_or_torch_fns, (list, tuple)):
        aten_ops_or_torch_fns = [aten_ops_or_torch_fns]
    def decorator(func):
        for op in aten_ops_or_torch_fns:
            @functools.wraps(op)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            _ATEN_OP_OR_TORCH_FN_TABLE[cls][op] = wrapper
        return func
    return decorator

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
