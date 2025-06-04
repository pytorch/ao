# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# mypy: allow-untyped-defs
"""
Utils shared by different modes of quantization (eager/graph)
"""

import functools

# mypy: allow-untyped-defs
import operator
import types
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F

# Makes sure that quantized_decomposed ops are registered
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx import Graph, GraphModule, Node
from torch.nn.utils.fusion import fuse_conv_bn_weights
from torch.nn.utils.parametrize import is_parametrized
from torch.utils._pytree import LeafSpec

from torchao.utils import _assert_and_get_unique_device

__all__ = [
    "is_per_tensor",
    "is_per_channel",
    "getattr_from_fqn",
    "get_qparam_dict",
    "check_min_max_valid",
    "calculate_qmin_qmax",
    "has_no_children_ignoring_parametrizations",
    "get_fqn_to_example_inputs",
    "to_underlying_dtype",
    "determine_qparams",
    "validate_qmin_qmax",
    "get_new_attr_name_with_prefix",
    "create_getattr_from_value",
    "_get_aten_graph_module_for_pattern",
    "_is_conv_node",
    "_is_conv_transpose_node",
    "_is_sym_size_node",
    "_filter_sym_size_users",
]


# TODO: remove unused
def is_per_tensor(qscheme):
    return qscheme == torch.per_tensor_affine or qscheme == torch.per_tensor_symmetric


def is_per_channel(qscheme):
    return qscheme in [
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
        torch.per_channel_symmetric,
    ]


def getattr_from_fqn(obj: Any, fqn: str) -> Any:
    """
    Given an obj and a fqn such as "foo.bar.baz", returns gm.foo.bar.baz.
    """
    return functools.reduce(getattr, fqn.split("."), obj)


def to_underlying_dtype(qdtype):
    DTYPE_MAPPING = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
        torch.quint4x2: torch.uint8,
        torch.quint2x4: torch.uint8,
        torch.uint8: torch.uint8,
        torch.int8: torch.int8,
        torch.uint16: torch.uint16,
        torch.int16: torch.int16,
        torch.int32: torch.int32,
        torch.float8_e5m2: torch.float8_e5m2,
        torch.float8_e4m3fn: torch.float8_e4m3fn,
    }
    assert qdtype in DTYPE_MAPPING, "Unsupported dtype: " + str(qdtype)
    return DTYPE_MAPPING[qdtype]


def get_qparam_dict(observer_or_fake_quant):
    from torchao.quantization.pt2e.observer import PlaceholderObserver

    qscheme = getattr(observer_or_fake_quant, "qscheme", None)
    dtype = observer_or_fake_quant.dtype
    qparams = {"qscheme": qscheme, "dtype": dtype}

    if not qscheme or isinstance(observer_or_fake_quant, PlaceholderObserver):
        return {"qscheme": None, "dtype": dtype}

    if is_per_tensor(qscheme):
        qscheme = torch.per_tensor_affine
    elif is_per_channel(qscheme):
        # change symmetric to affine since we do not have symmetric
        # quantized Tensor
        if qscheme == torch.per_channel_symmetric:
            qscheme = torch.per_channel_affine
        qparams["axis"] = observer_or_fake_quant.ch_axis
    else:
        raise RuntimeError(f"Unrecognized qscheme: {qscheme}")
    # update qscheme, since we don't have symmetric quant qscheme
    # in quantized Tensor
    qparams["qscheme"] = qscheme

    scale, zero_point = observer_or_fake_quant.calculate_qparams()
    qparams["scale"] = scale
    qparams["zero_point"] = zero_point

    if hasattr(observer_or_fake_quant, "quant_min"):
        qparams["quant_min"] = observer_or_fake_quant.quant_min
    if hasattr(observer_or_fake_quant, "quant_max"):
        qparams["quant_max"] = observer_or_fake_quant.quant_max

    return qparams


def check_min_max_valid(min_val: torch.Tensor, max_val: torch.Tensor) -> bool:
    """Checks if the given minimum and maximum values are valid, meaning that
    they exist and the min value is less than the max value.
    """
    if min_val.numel() == 0 or max_val.numel() == 0:
        warnings.warn(
            "must run observer before calling calculate_qparams. "
            + "Returning default values."
        )
        return False

    if min_val.dim() == 0 or max_val.dim() == 0:
        if min_val == float("inf") and max_val == float("-inf"):
            warnings.warn(
                "must run observer before calling calculate_qparams. "
                + "Returning default values."
            )

            return False

        assert min_val <= max_val, f"min {min_val} should be less than max {max_val}"
    else:
        assert torch.all(min_val <= max_val), (
            f"min {min_val} should be less than max {max_val}"
        )

    return True


def calculate_qmin_qmax(
    quant_min: int,
    quant_max: int,
    has_customized_qrange: bool,
    dtype: torch.dtype,
    reduce_range: bool,
) -> tuple[int, int]:
    r"""Calculates actual qmin and qmax based on the quantization range,
    observer datatype and if range is reduced.
    """
    # TODO(jerryzh): Figure out why custom quant_min/quant_max are still adjusted.
    if has_customized_qrange:
        # This initialization here is to be resolve TorchScript compilation issues and allow
        # using of refinement to decouple initial_qmin and initial_qmax from quantization range.
        # The actual values of initial_qmin and initial_qmax will be reset below.
        if dtype in [torch.qint32, torch.int32]:
            initial_quant_min, initial_quant_max = 0, 2**32 - 1
        else:
            initial_quant_min, initial_quant_max = 0, 255
        # The following assignment of self.qmin and self.qmax to the local variables and the if check refine the
        # attribute from Optional valid integers for use, based on TorchScript's requirements.
        custom_quant_min, custom_quant_max = quant_min, quant_max
        if custom_quant_min is not None and custom_quant_max is not None:
            initial_quant_min, initial_quant_max = (
                custom_quant_min,
                custom_quant_max,
            )

        qrange_len = initial_quant_max - initial_quant_min + 1
        if dtype in [torch.qint8, torch.int8]:
            assert 0 < qrange_len <= 256, (
                "quantization range should be positive and not exceed the maximum bit range (=256)."
            )
        elif dtype in [torch.qint32, torch.int32]:
            assert 0 < qrange_len <= 2**32, (
                "quantization range should be positive and not exceed the maximum bit range (=4294967296)."
            )
        if reduce_range:
            quant_min, quant_max = quant_min // 2, quant_max // 2
    else:
        # Fallback onto default 8-bit qmin and qmax calculation if dynamic range is not used.
        if dtype in [torch.qint8, torch.int8]:
            if reduce_range:
                quant_min, quant_max = -64, 63
            else:
                quant_min, quant_max = -128, 127
        elif dtype in [torch.quint8, torch.uint8]:
            if reduce_range:
                quant_min, quant_max = 0, 127
            else:
                quant_min, quant_max = 0, 255
        elif dtype in [torch.qint32, torch.int32]:
            quant_min, quant_max = -1 * (2**31), (2**31) - 1
        elif dtype in [torch.uint16]:
            quant_min, quant_max = 0, 2**16 - 1
        elif dtype in [torch.int16]:
            quant_min, quant_max = -(2**15), 2**15 - 1
        else:
            quant_min, quant_max = 0, 15
    return quant_min, quant_max


def _parent_name(target):
    """
    Turn 'foo.bar' into ['foo', 'bar']
    """
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]


def has_no_children_ignoring_parametrizations(module):
    """
    Checks if module._modules is empty or
    if module is a parametrization, checks that module._modules only has
    the 'parametrizations' module
    """
    if len(module._modules) == 0:
        return True
    elif is_parametrized(module):
        return len(module._modules) == 1 and "parametrizations" in module._modules
    else:
        return False


def _get_path_of_module(
    root: torch.nn.Module, submodule: torch.nn.Module
) -> Optional[str]:
    """Get the path (fully qualified name) of a submodule

    Example::

    >> class M(torch.nn.Module):
           def __init__(self) -> None:
               self.linear = torch.nn.Linear(5, 5)
           def forward(self, x):
               return self.linear(x)

    >> m = M()
    >> l = m.linear
    >> _get_path_of_module(m, l)
    "linear"
    """
    for n, p in root.named_modules():
        if submodule is p:
            return n
    return None


def _get_signature_locals(f: Callable, loc: dict[str, Any]) -> dict[str, Any]:
    """Get local keyword arguments

    Example::

    >> def f(self, a, b=9):
           pass
    >> loc = {"a": 6, "c": 7}
    >> _get_signature_locals(f, loc)
    {"a": 6}
    """
    return {k: v for k, v in loc.items() if k in signature(f).parameters}


def _get_default_kwargs(f: Callable) -> "OrderedDict[str, Any]":
    """Get all default keyword arguments from function signature

    Example::

    >> def f(self, a, b=9):
           pass
    >> _get_default_kwargs(f)
    {"b": 9}
    """
    kwargs = {}
    for name, param in signature(f).parameters.items():
        if param.default is not param.empty:
            kwargs[name] = param.default
        elif param.kind is param.VAR_POSITIONAL:
            kwargs[name] = ()
        elif param.kind is param.VAR_KEYWORD:
            kwargs[name] = {}
    return OrderedDict(kwargs)


def _normalize_kwargs(func: Callable, loc: dict[str, Any]) -> "OrderedDict[str, Any]":
    """Given a function and local function arguments, normalize the keyword
    arguments by filling in default arguments from function signature

    Example::

    >> def f(self, key1=3, key2=3):
           pass
    >> loc = {"key2": 6}
    >> _normalize_kwargs(f, loc)
    {"key1": 3, "key2": 6}
    """
    default_kwargs = _get_default_kwargs(func)
    local_kwargs = _get_signature_locals(func, loc)
    normalized_kwargs = default_kwargs.copy()
    for attr, val in local_kwargs.items():
        if attr in normalized_kwargs:
            # override the default keyword arguments
            normalized_kwargs[attr] = val
    return normalized_kwargs


def validate_qmin_qmax(quant_min: int, quant_max: int) -> None:
    r"""Validates that the user-specified quantization range is properly initialized
    and within the given bound supported by the observer dtype.

    To accommodate lower-bit quantization with respect to the existing torch.qint8 and
    torch.quint8 datatypes, the user can choose to use dynamic quantization range by passing
    in a tuple of initial qmin and qmax values. One use case is these customized qmin and qmax
    values are used to calculate static estimates of the scale and zero point for aggressive lower-bit
    fake quantization. These estimates are compared against parameters learned through backpropagation.
    The related literatures for scale and zero point via backpropagation are as follows:

    Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
    Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
    """
    # The variable names are prefixed with "initial" because their values (qmin and qmax) might be adjusted
    # based on whether quantization range is reduced and the datatype (signed/unsigned) used by the observer.
    assert quant_min <= 0 <= quant_max, (
        "Used-specified quantization range must include 0."
    )
    assert quant_min < quant_max, (
        "qmin must be strictly less than qmax for user-specified quantization range."
    )


# Functionally equivalent to '_calculate_qparams' in observer.py. Observers must be torchscriptable however and qscheme
# as far as I can tell is not allowed to passed as a parameter in torchscript functions. This makes refactoring observer
# to use this utility a massive pain and very gross. For now Im opting just to duplicate as this code seems unlikey to change
# (last update over 1 year ago) and when torchscript is fully deprecated we can refactor. TODO(jakeszwe, jerryzh168)
def determine_qparams(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    eps: torch.Tensor,
    has_customized_qrange: bool,
    qscheme: torch.qscheme = torch.per_tensor_affine,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Calculates the quantization parameters, given min and max
    value tensors. Works for both per tensor and per channel cases

    Args:
        min_val: Minimum values per channel
        max_val: Maximum values per channel

    Returns:
        scales: Scales tensor of shape (#channels,)
        zero_points: Zero points tensor of shape (#channels,)
    """
    if not check_min_max_valid(min_val, max_val):
        return torch.tensor([1.0], device=min_val.device.type), torch.tensor(
            [0], device=min_val.device.type
        )

    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    device = min_val_neg.device
    scale = torch.ones(min_val_neg.size(), dtype=torch.double, device=device)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)
    eps = eps.to(device)

    if qscheme == torch.per_tensor_symmetric or qscheme == torch.per_channel_symmetric:
        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        scale = torch.max(scale, eps)
        if dtype in [torch.uint8, torch.quint8]:
            if has_customized_qrange:
                # When customized quantization range is used, down-rounded midpoint of the range is chosen.
                zero_point = zero_point.new_full(
                    zero_point.size(), (quant_min + quant_max) // 2
                )
            else:
                zero_point = zero_point.new_full(zero_point.size(), 128)
    elif qscheme == torch.per_channel_affine_float_qparams:
        scale = (max_val - min_val) / float(quant_max - quant_min)
        scale = torch.where(scale > eps, scale, torch.ones_like(scale))
        # We use the quantize function
        # xq = Round(Xf * inv_scale + zero_point),
        # setting zero_point to (-1 * min *inv_scale) we get
        # Xq = Round((Xf - min) * inv_scale)
        zero_point = -1 * min_val / scale
    else:
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, eps)
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)

    # For scalar values, cast them to Tensors of size 1 to keep the shape
    # consistent with default values in FakeQuantize.
    if len(scale.shape) == 0:
        # TODO: switch to scale.item() after adding JIT support
        scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
    if len(zero_point.shape) == 0:
        # TODO: switch to zero_point.item() after adding JIT support
        zero_point = torch.tensor(
            [int(zero_point)], dtype=zero_point.dtype, device=device
        )
        if qscheme == torch.per_channel_affine_float_qparams:
            zero_point = torch.tensor(
                [float(zero_point)], dtype=zero_point.dtype, device=device
            )

    return scale.to(torch.double), zero_point.to(torch.int64)


def _get_num_pos_args(f: Callable) -> int:
    """Get number of positional args for a function

    Example::

    >> def f(self, key1=3, key2=3):
           pass
    >> _get_num_pos_args(f)
    3
    """
    return len(getfullargspec(f).args)


def get_fqn_to_example_inputs(
    model: torch.nn.Module, example_inputs: tuple[Any, ...]
) -> dict[str, tuple[Any, ...]]:
    """Given a model and its example inputs, return a dictionary from
    fully qualified name of submodules to example_inputs for that submodule,
    e.g. {"linear1": (tensor1,), "linear2": (tensor2,), "sub": (tensor3,),
          "sub.linear1": (tensor4,), ...}

    Used to make quantizing submodules easier now that FX Graph Mode Quantization requires
    example inputs.

    Also works for keyword arguments with default values, we would flatten keyword
    arguments as positional arguments and fill in the missing keyword args with default
    values, e.g. if we have a forward function:
    def forward(self, x, key1=3, key2=3):
        ...

    and we call it with self.submodule(x, key2=6)
    we'll get example_inputs: (x, 3, 6)

    user can also override `key1` with positional arguments as well:
    for self.submodule(x, 5, key2=6)
    we'll get: (x, 5, 6)

    variable positional arguments and variable positional keyword arguments in forward
    function are not supported currently, so please make sure no submodules is using
    them.
    """
    root = model
    fqn_to_example_inputs = {}

    def _patched_module_call(self, *args, **kwargs):
        submodule_example_inputs = list(args).copy()
        normalized_kwargs = _normalize_kwargs(self.forward, kwargs)
        # minus 1 to skipping counting `self`
        num_args = _get_num_pos_args(self.forward) - 1
        num_to_pop = num_args - len(submodule_example_inputs)
        while num_to_pop and normalized_kwargs:
            normalized_kwargs.popitem(last=False)
            num_to_pop -= 1
        submodule_example_inputs.extend(normalized_kwargs.values())
        submodule_example_inputs_tuple = tuple(submodule_example_inputs)
        fqn = _get_path_of_module(root, self)
        if fqn is not None:
            fqn_to_example_inputs[fqn] = submodule_example_inputs_tuple
        return orig_module_call(self, *args, **kwargs)

    orig_module_call = torch.nn.Module.__call__
    torch.nn.Module.__call__ = _patched_module_call  # type: ignore[method-assign]
    try:
        model(*example_inputs)
    finally:
        # restore the module call even if there is an exception
        torch.nn.Module.__call__ = orig_module_call  # type: ignore[method-assign]
    return fqn_to_example_inputs


# Returns a function that can get a new attribute name for module with given
# prefix, for example,
# >> get_new_observer_name = get_new_attr_name_with_prefix('_observer')
# >> new_name = get_new_observer_name(module)
# new_name will be an unused attribute name on module, e.g. `_observer_1`
def get_new_attr_name_with_prefix(prefix: str) -> Callable:
    prefix = prefix.replace(".", "_")

    def get_new_attr_name(module: torch.nn.Module):
        def get_attr_name(i: int):
            return prefix + str(i)

        i = 0
        attr_name = get_attr_name(i)
        while hasattr(module, attr_name):
            i += 1
            attr_name = get_attr_name(i)
        return attr_name

    return get_new_attr_name


def create_getattr_from_value(
    module: torch.nn.Module, graph: Graph, prefix: str, value: Any
) -> Node:
    """
    Given a value of any type, creates a getattr node corresponding to the value and
    registers the value as a buffer to the module.
    """
    get_new_attr_name = get_new_attr_name_with_prefix(prefix)
    attr_name = get_new_attr_name(module)
    device = _assert_and_get_unique_device(module)
    new_value = (
        value.detach().clone()
        if isinstance(value, torch.Tensor)
        else torch.tensor(value, device=device)
    )
    module.register_buffer(attr_name, new_value)
    # Create get_attr with value
    attr_node = graph.create_node("get_attr", attr_name)
    return attr_node


_QUANTIZE_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
]


_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]


def _is_connected(source: torch.fx.Node, dest: torch.fx.Node) -> bool:
    """
    Assuming dest is one of the ops inserted by quant workflow, this function
    finds if source and dest are connected. Assumption is that only quant workflow
    inserted ops exist between source and dest
    """
    quant_workflow_ops = _QUANTIZE_OPS + _DEQUANTIZE_OPS
    quant_workflow_ops.append(torch.ops.quantized_decomposed.choose_qparams.tensor)
    while dest.target in quant_workflow_ops:
        if not isinstance(dest.args[0], torch.fx.Node):
            raise ValueError(
                f"expected arg[0] of quant workflow ops to be a node but found {dest.args[0]}"
            )
        dest = dest.args[0]
    return dest == source


def _get_tensor_constant_from_node(node, m):
    if node is None:
        return None
    assert node.op == "get_attr"
    target_atoms = node.target.split(".")
    attr_itr = m
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def _get_all_arguments(orig_args, orig_kwargs, args_schema):
    all_args = []
    for i, schema in enumerate(args_schema):
        if schema.name in orig_kwargs:
            all_args.append(orig_kwargs[schema.name])
        elif not schema.kwarg_only and i < len(orig_args):
            all_args.append(orig_args[i])
        else:
            all_args.append(schema.default_value)
    return all_args


def _is_supported_batch_norm_for_training(node: Node):
    """
    Return True if the given node refers to an aten batch norm op QAT supports.
    """
    supported_ops = [
        torch.ops.aten.batch_norm.default,
        torch.ops.aten._native_batch_norm_legit.default,
        # Note: we won't need this op anymore after batch norm consolidation
        # For now, we need to continue to support it because it gives better
        # training numerics than `_native_batch_norm_legit`
        torch.ops.aten.cudnn_batch_norm.default,
        torch.ops.aten.miopen_batch_norm.default,
    ]
    return node.target in supported_ops


def _is_conv_node(n: Node):
    """
    Return whether the node refers to an aten conv op.
    """
    return n.op == "call_function" and n.target in [
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv1d.padding,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv2d.padding,
        torch.ops.aten.conv3d.default,
        torch.ops.aten.conv3d.padding,
    ]


def _is_conv_transpose_node(n: Node):
    """
    Return whether the node refers to an aten conv_transpose op.
    """
    return n.op == "call_function" and n.target in [
        torch.ops.aten.conv_transpose1d,
        torch.ops.aten.conv_transpose1d.default,
        torch.ops.aten.conv_transpose2d,
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.conv_transpose3d,
        torch.ops.aten.conv_transpose3d.input,
    ]


def _is_conv_or_conv_transpose_node(n: Node):
    """
    Return whether the node refers to an aten conv or conv transpose op.
    """
    return _is_conv_node(n) or _is_conv_transpose_node(n)


def _is_conv_transpose_fn(conv_fn: Callable):
    return conv_fn in [F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d]


def _is_bn_node(n: Node):
    return (
        _is_supported_batch_norm_for_training(n)
        or n.target == torch.ops.aten._native_batch_norm_legit_no_training.default
    )


def fold_bn_weights_into_conv_node(
    conv_node: Node,
    conv_weight_node: Node,
    conv_bias_node: Optional[Node],
    bn_node: Node,
    m: GraphModule,
) -> None:
    # conv args: input, weight, bias, stride, padding, dilation, ...
    conv_w = _get_tensor_constant_from_node(conv_weight_node, m)
    conv_b = _get_tensor_constant_from_node(conv_bias_node, m)
    transpose = _is_conv_transpose_node(conv_node)

    # eval bn args: input, weight, bias, running mean, running var, momentum, eps
    # train bn args: input, weight, bias, running mean, running var, training, momentum, eps
    bn_args_schema = bn_node.target._schema.arguments  # type: ignore[union-attr]
    bn_args = _get_all_arguments(bn_node.args, bn_node.kwargs, bn_args_schema)
    bn_w = _get_tensor_constant_from_node(bn_args[1], m)
    bn_b = _get_tensor_constant_from_node(bn_args[2], m)
    bn_rm = _get_tensor_constant_from_node(bn_args[3], m)
    bn_rv = _get_tensor_constant_from_node(bn_args[4], m)
    if bn_node.target == torch.ops.aten._native_batch_norm_legit_no_training.default:
        eps_arg_index = 6
    elif _is_supported_batch_norm_for_training(bn_node):
        eps_arg_index = 7
    else:
        raise ValueError("BN node target is unexpected ", bn_node.target)
    bn_eps = bn_args[eps_arg_index]

    fused_weight, fused_bias = fuse_conv_bn_weights(
        conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=transpose
    )

    # update the weight and bias for conv
    conv_args = list(conv_node.args)
    # filling in the default bias argument
    if len(conv_args) == 2:
        conv_args.append(None)

    # calling data since the fused_weight and fused_bias are nn.Parameter
    weight_attr_name = conv_weight_node.target
    assert isinstance(weight_attr_name, str)
    _assign_attr(fused_weight, m, weight_attr_name, _AttrKind.PARAMETER)
    if conv_bias_node is not None:
        bias_attr_name = conv_bias_node.target
        _assign_attr(fused_bias, m, str(bias_attr_name), _AttrKind.PARAMETER)
    else:
        bias_attr_name = weight_attr_name + "_bias"
        _assign_attr(fused_bias, m, bias_attr_name, _AttrKind.PARAMETER)
        with m.graph.inserting_before(conv_node):
            get_bias_node = m.graph.get_attr(bias_attr_name)
        # NOTE: here we assume the bias of conv is not quantized!
        conv_args[2] = get_bias_node
    conv_node.args = tuple(conv_args)

    # native_batch_norm has 3 outputs, we expect getitem calls on the output
    # and we want to replace the uses of getitem 0 with the output of conv
    #
    if bn_node.target == torch.ops.aten.batch_norm.default:
        # With the new training ir, instead of batch_norm + getitem,
        # we only have the batch_norm node.
        #
        # Before:
        # conv -> bn -> users
        # After:
        # conv -> users
        #       bn has no users now
        bn_node.replace_all_uses_with(conv_node)
    else:
        # Before:
        # conv -> bn - (first output) -> users1
        #          \ - (second output) -> users2
        #          \ - (third output) -> users3
        # After:
        # conv -> (first output) -> users1
        #       bn -
        #          \ - (second output) -> users2
        #          \ - (third output) -> users3
        # if users2 and users3 are empty then bn will be removed through dead code elimination
        for user in bn_node.users:
            if (
                user.op != "call_function"
                or user.target != operator.getitem
                or user.args[1] != 0
            ):
                continue
            user.replace_all_uses_with(conv_node)

    # If the BN node does not have users, erase it from the graph
    # Note: we need to do this manually because the model can still be in train
    # mode at this point, in which case DCE won't erase the BN node automatically
    # since the node refers to a mutating op. Here we still need to call DCE first
    # to get rid of the unused getitem nodes that consume the BN node.
    m.graph.eliminate_dead_code()
    if len(bn_node.users) == 0:
        m.graph.erase_node(bn_node)


# fuse conv bn weights, inplace modification of the graph_module and graph
def _fuse_conv_bn_(m: GraphModule) -> None:
    has_bn = any(_is_bn_node(n) for n in m.graph.nodes)
    if not has_bn:
        return
    for n in m.graph.nodes:
        if n.op != "call_function" or n.target not in (
            torch.ops.aten._native_batch_norm_legit_no_training.default,
            torch.ops.aten.batch_norm.default,
        ):
            continue
        bn_node = n
        n = bn_node.args[0]
        if not _is_conv_or_conv_transpose_node(n):
            continue
        conv_node = n
        conv_weight_node = conv_node.args[1]
        conv_bias_node = conv_node.args[2] if len(conv_node.args) > 2 else None
        fold_bn_weights_into_conv_node(
            conv_node, conv_weight_node, conv_bias_node, bn_node, m
        )

    m.graph.eliminate_dead_code()
    m.recompile()


def _get_node_name_to_scope(model: GraphModule) -> dict[str, tuple[str, type]]:
    # TODO: move this information to fx node itself
    node_name_to_scope: dict[str, tuple[str, type]] = {}
    for n in model.graph.nodes:
        nn_module_stack = n.meta.get("nn_module_stack", None)
        current_scope = ("", type(None))
        if nn_module_stack:
            bt = list(nn_module_stack.values())[-1]
            current_scope = (bt[0].split(".")[-1], bt[1])
        node_name_to_scope[n.name] = current_scope
    return node_name_to_scope


def _get_aten_graph_module_for_pattern(
    pattern: Callable,
    example_inputs: tuple[Any, ...],
    is_cuda: bool = False,
    **kwargs,
) -> GraphModule:
    """
    Convert the pattern to an FX graph with decomposed aten ops.
    """
    if is_cuda:
        example_inputs = tuple(
            [x.cuda() if isinstance(x, torch.Tensor) else x for x in example_inputs]
        )

    aten_pattern = torch.export.export_for_training(
        pattern,  # type: ignore[arg-type]
        example_inputs,
        kwargs,
        strict=True,
    ).module()

    aten_pattern.graph.eliminate_dead_code()  # type: ignore[operator, union-attr]
    aten_pattern.recompile()  # type: ignore[operator]

    # ep.module() adds copy_ nodes for the mutated inputs.
    # For patterns, it doesn't matter
    for node in aten_pattern.graph.nodes:  # type: ignore[union-attr]
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.copy_.default
            and len(node.users) == 0
        ):
            aten_pattern.graph.erase_node(node)  # type: ignore[operator, union-attr]

    aten_pattern.graph.eliminate_dead_code()  # type: ignore[operator, union-attr]
    aten_pattern.recompile()  # type: ignore[operator]

    return aten_pattern  # type: ignore[return-value]


def remove_tensor_overload_for_qdq_ops(match_pattern: GraphModule) -> None:
    """Remove .tensor overload for quantize/dequantize ops so that we can
    use the match_pattern that we get from torchdynamo export to match the output of convert_pt2e
    """
    _MAP = {
        torch.ops.quantized_decomposed.quantize_per_tensor.default: torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.default: torch.ops.quantized_decomposed.dequantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor: torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: torch.ops.quantized_decomposed.dequantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor2: torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor2: torch.ops.quantized_decomposed.dequantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_channel.default: torch.ops.quantized_decomposed.quantize_per_channel,
        torch.ops.quantized_decomposed.dequantize_per_channel.default: torch.ops.quantized_decomposed.dequantize_per_channel,
        torch.ops.aten.clamp.Tensor: torch.ops.aten.clamp,
    }
    for n in match_pattern.graph.nodes:
        if n.op != "call_function":
            continue
        if n.target in _MAP:
            n.target = _MAP[n.target]


def _is_literal(arg):
    if isinstance(arg, (int, float)):
        return True
    if isinstance(arg, (tuple, list)):
        return all(map(_is_literal, arg))
    return False


def _replace_literals_with_new_placeholders(
    gm: torch.fx.GraphModule,
    merge_dup: bool = False,
    exclude_literals: Optional[list[Any]] = None,
):
    """Replace the literals in the graph with placeholder nodes that's created on the fly while we
    traverse the graph, so that the literal arguments in the graph can be matched and replaced

    To use this, the pattern and replacement graph should have the exact same number of literal args
    and they should be used in the exact same order in the pattern and replacement graph.

    If the literal arguments are not used in the same order in pattern and replacement graph, please
    use `_replace_literals_with_existing_placeholders` instead

    Args:
        `gm`: input GraphModule that we'll transform
        `merge_dup`: boolean flag to indicate that if the same literal appears multiple times in
         the graph, whether they should correspond to the same placeholder or not
        `exclude_literals`: a list of literals that will not be replaced with placeholders

    Example:

    # 1. Original Graph
    def pattern(self, x):
        return x + 3

    def replacement(self, x):
        return x - 3

    example_inputs = (torch.randn(1, 3, 3, 3),)
    pattern_gm = _get_aten_graph_module_for_pattern(pattern, example_inputs)
    replacement_gm = _get_aten_graph_module_for_pattern(pattern, example_inptus)

    # 2. Before calling replace literals we'll see the following graph:
    def pattern(self, x):
        return x + 3

    def replacement(self, x):
        return x - 3

    pattern_gm = _replace_literals_with_new_placeholders(pattern_gm)
    replacement_gm = _replace_literals_with_new_placeholders(replacement_gm)

    # 3. After replacing literals with new placeholder nodes

    def pattern(self, x, new_ph):
        return x + new_ph

    def pattern(self, x, new_ph):
        return x - new_ph

    """
    last_ph = None
    cnt = 0
    literal_to_ph: dict[Union[float, bool, int, torch.dtype], Node] = {}
    if exclude_literals is None:
        exclude_literals = []

    in_spec = gm._in_spec
    args_spec = in_spec.children_specs[0]
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            last_ph = node
            cnt += 1
            continue
        with gm.graph.inserting_after(last_ph):
            new_args = []
            for arg in node.args:
                if _is_literal(arg) and arg not in exclude_literals:
                    if merge_dup and arg in literal_to_ph:
                        new_args.append(literal_to_ph[arg])
                    else:
                        ph_node = gm.graph.placeholder("arg" + str(cnt))
                        new_args.append(ph_node)
                        args_spec.children_specs.append(LeafSpec())
                        cnt += 1
                        if merge_dup:
                            literal_to_ph[arg] = ph_node
                else:
                    new_args.append(arg)
            new_args = tuple(new_args)

        node.args = new_args

    # Update `num_nodes`, `num_leaves`, `num_children`.
    args_spec.__post_init__()
    in_spec.__post_init__()
    return gm


def _replace_literals_with_existing_placeholders(
    gm: torch.fx.GraphModule,
    exclude_literals: Optional[list[Any]] = None,
    literal_to_ph_idx: Optional[dict[Union[float, int, bool, torch.dtype], int]] = None,
):
    """Replace the literals in the graph with **existing** placeholder nodes, so that the literal arguments
    in the graph can be matched and replaced

    To use this, all literal args in the graph should be unique and each of them should correspond
    to exactly one placeholder node

    # 1. Original Graph
    def pattern(self, x_i8, scale, zero_point, quant_min, quant_max):
        return torch.dequantize_per_tensor(x_i8, scale, zero_point, quant_min, quant_max)

    def replacement(x_i8, scale, zero_point, quant_min, quant_max):
        x_i8 = torch.clamp(x_i8, quant_min, quant_max)
        return ((x_i8.to(torch.float32) - zero_point) * scale).to(dtype=torch.float32)

    example_inputs = (
        torch.randn(1, 3, 3, 3),
        1.0,
        0,
        -128,
        127,
    )
    pattern_gm = _get_aten_graph_module_for_pattern(pattern, example_inputs)
    replacement_gm = _get_aten_graph_module_for_pattern(pattern, example_inptus)

    # 2. Before calling replace literals we'll see the following graph:
    def pattern(self, x_i8, scale, zero_point, quant_min, quant_max):
        # scale/zero_point/quant_min/quant_max are burnt in since they are scalar values
        return torch.dequantize_per_tensor(x_i8, 1.0, 0, -128, 127)

    def replacement(x_i8, scale, zero_point, quant_min, quant_max):
        # scale/zero_point/quant_min/quant_max are burnt in since they are scalar values
        x_i8 = torch.clamp(x_i8, -128, 127)
        return ((x_i8.to(torch.float32) - 0) * 1.0).to(dtype=torch.float32)

    # Note that literal args appear in different order in pattern and replacement graph, so
    # we can't use _replace_literals_with_new_placeholders

    literal_to_ph_idx = {1.0: 1, 0: 2, -128: 3, 127: 4}
    pattern_gm = _replace_literals_with_existing_placeholders(pattern_gm, literal_to_ph_idx)
    replacement_gm = _replace_literals_with_existing_placeholders(replacement_gm, literal_to_ph_idx)

    # 3. After replacing literals with existing placeholder nodes

    def pattern(self, x_i8, scale, zero_point, quant_min, quant_max):
        # scale/zero_point/quant_min/quant_max are burnt in since they are scalar values
        return torch.dequantize_per_tensor(x_i8, scale, zero_point, quant_min, quant_max)

    def replacement(x_i8, scale, zero_point, quant_min, quant_max):
        # scale/zero_point/quant_min/quant_max are burnt in since they are scalar values
        x_i8 = torch.clamp(x_i8, quant_min, quant_max)
        return ((x_i8.to(torch.float32) - zero_point) * scale).to(dtype=torch.float32)
    """
    if exclude_literals is None:
        exclude_literals = []

    if literal_to_ph_idx is None:
        literal_to_ph_idx = {}

    phs = [node for node in gm.graph.nodes if node.op == "placeholder"]

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        new_args = []
        for arg in node.args:
            if (
                _is_literal(arg)
                and arg not in exclude_literals
                and arg in literal_to_ph_idx
            ):
                ph_idx = literal_to_ph_idx[arg]
                ph_node = phs[ph_idx]
                new_args.append(ph_node)
            else:
                new_args.append(arg)
        new_args = tuple(new_args)
        node.args = new_args
    return gm


# TODO: Handle this in export itself and don't wrap the model in another GraphModule
# in prepare and convert
def _disallow_eval_train(model: GraphModule):
    """
    Disallow calling `model.train()` or `model.eval()` on the given GraphModule.
    This is useful for exported models, where these methods don't actually behave as expected.
    """
    error_message = """
        Calling train() or eval() is not supported for exported models.
        Please call `torchao.quantization.pt2e.move_exported_model_to_train(model)` (or eval) instead.

        If you cannot replace the calls to `model.train()` and `model.eval()`, you may override
        the behavior for these methods by calling `torchao.quantization.pt2e.allow_exported_model_train_eval(model)`,
        which does the above automatically for you. Note that this has limited effect on switching
        behavior between train and eval modes, and should be used only for special ops such as dropout
        and batchnorm.
        """

    def _train(self, mode: bool = True):
        raise NotImplementedError(error_message)

    def _eval(self, mode: bool = True):
        raise NotImplementedError(error_message)

    model.train = types.MethodType(_train, model)  # type: ignore[method-assign]
    model.eval = types.MethodType(_eval, model)  # type: ignore[method-assign]
    return model


def _is_sym_size_node(node: Node):
    return (
        node.op == "call_function"
        and node.target == torch.ops.aten.sym_size.default
        or node.target == torch.ops.aten.sym_numel.default
        or node.target == torch.ops.aten.sym_numel
        or node.target == torch.ops.aten.sym_size
    )


def _filter_sym_size_users(node: torch.fx.Node) -> list[torch.fx.Node]:
    node_users = list(filter((lambda x: (_is_sym_size_node(x) is False)), node.users))
    return node_users
