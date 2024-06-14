from typing import Dict, Callable
from collections import defaultdict
import functools

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
layout tensor constructor registration for different tensor subclassesa

first key is a tensor subclass type like AffineQuantizedTensor
second key is an extended layout string, like tensor_core_tiled
value is a constructor for the LayoutTensor class, e.g. TensorCoreTiledAQTLayout.from_plain
"""
_LAYOUT_CONSTRUCTOR_TABLE: Dict[Callable, Dict[str, Callable]] = defaultdict(dict)

def _register_layout_cls(cls: Callable, extended_layout: str):
    """Helper function for layout registrations, this is used to implement
    register_layout_cls decorator for each tensor subclass, see aqt.py for example usage

    Args:
        cls: Tensor subclass type
        extended_layout: string name for the layout type

    Returns:
        a decorator that registers the layout tensor constructor in the table
    """
    def decorator(layout_cls):
        layout_cls.extended_layout = extended_layout
        _LAYOUT_CONSTRUCTOR_TABLE[cls][extended_layout] = layout_cls.from_plain
        return layout_cls
    return decorator

def _get_layout_tensor_constructor(cls: Callable, extended_layout: str) -> Callable:
    """Get Layout class constructor (LayoutClass.from_plain) for `cls` based on `extended_layout`
    """
    if cls not in _LAYOUT_CONSTRUCTOR_TABLE:
        raise ValueError(f"no registered layout class constructor for: {cls}")
    if extended_layout not in _LAYOUT_CONSTRUCTOR_TABLE[cls]:
        raise ValueError(f"extended_layout: {extended_layout} is not supported yet for {cls}")

    return _LAYOUT_CONSTRUCTOR_TABLE[cls][extended_layout]
