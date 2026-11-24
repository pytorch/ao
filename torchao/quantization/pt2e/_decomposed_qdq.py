# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors

"""Insertion of ``quantized_decomposed`` quantize/dequantize nodes for observers
and fake quantizers that do not build their own (see ``ObserverBase.convert``)."""

import operator
from typing import Callable, Optional

import torch

# importing the lib so that the quantized_decomposed ops are registered
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.utils import is_per_channel, to_underlying_dtype
from torch.fx import Node
from torch.fx.traceback import NodeSource, NodeSourceAction

from torchao.quantization.pt2e import FROM_NODE_KEY
from torchao.quantization.pt2e.fake_quantize import FakeQuantizeBase
from torchao.quantization.pt2e.observer import ObserverBase
from torchao.quantization.pt2e.utils import create_getattr_from_value

SUPPORTED_QDTYPES = [
    torch.quint8,
    torch.qint8,
    torch.qint32,
    torch.uint8,
    torch.int8,
    torch.uint16,
    torch.int16,
    torch.int32,
    torch.float8_e5m2,
    torch.float8_e4m3fn,
]

_QSCHEME_TO_CHOOSE_QPARAMS_OP = {
    torch.per_tensor_affine: torch.ops.quantized_decomposed.choose_qparams.tensor,
    torch.per_tensor_symmetric: torch.ops.quantized_decomposed.choose_qparams_symmetric.tensor,
}


def _uses_default_convert(observer: torch.nn.Module) -> bool:
    """True when the observer's class does not override the base ``convert``."""
    convert = getattr(type(observer), "convert", None)
    return convert is ObserverBase.convert or convert is FakeQuantizeBase.convert


def _insert_decomposed_qdq(
    observer: torch.nn.Module,
    model: torch.fx.GraphModule,
    node: Node,
    *,
    attr_prefix: str = "",
    model_device: Optional[torch.device] = None,
) -> None:
    """Replace the observer call ``node`` with ``quantized_decomposed``
    quantize/dequantize nodes (or an fp16 round trip), using the observer's
    dtype, qscheme and calculated qparams. Non-scalar scale and zero_point are
    registered on ``model`` as attributes named ``attr_prefix + "_scale_"`` and
    ``attr_prefix + "_zero_point_"``."""
    graph = model.graph
    # 1. extract the information from observer module for generating
    # the quantize and dequantize operator
    dtype = observer.dtype  # type: ignore[attr-defined]

    is_dynamic = False
    if hasattr(observer, "is_dynamic"):
        is_dynamic = observer.is_dynamic  # type: ignore[assignment]

    def add_dequantize_op_kwargs(dequantize_op, input_node):
        dequantize_op_kwargs = {}
        if "val" in input_node.meta:
            dq_out_dtype = input_node.meta["val"].dtype
            if dq_out_dtype != torch.float32:
                dequantize_op_kwargs = {"out_dtype": dq_out_dtype}
        return dequantize_op_kwargs

    def add_quantize_dequantize_node_info(qdq_node, original_node):
        # propagate from_node info from observer/fake_quant node to quantize/dequantize node
        qdq_node.meta[FROM_NODE_KEY] = [
            NodeSource(
                original_node,
                "replace_observer_with_quantize_dequantize_node",
                [NodeSourceAction.CREATE],
            )
        ]

    if dtype in SUPPORTED_QDTYPES and (not is_dynamic):
        # TODO: probably should cleanup this condition check, it's hard
        # to reason about this if and the following elif

        # uint8/int8/int32 static quantization branch

        # 1. extract information for inserting q/dq node from observer
        node_type = "call_function"
        quantize_op: Optional[Callable] = None
        scale, zero_point = observer.calculate_qparams()  # type: ignore[attr-defined, operator]
        if is_per_channel(observer.qscheme):  # type: ignore[attr-defined]
            ch_axis = int(observer.ch_axis)  # type: ignore[attr-defined, arg-type]
            quantize_op = torch.ops.quantized_decomposed.quantize_per_channel.default
            dequantize_op = (
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            )
            quant_min = observer.quant_min
            quant_max = observer.quant_max
            dtype_ = to_underlying_dtype(dtype)
            qparams = {
                "_scale_": scale,
                "_zero_point_": zero_point,
                "_axis_": ch_axis,
                "_quant_min_": quant_min,
                "_quant_max_": quant_max,
                "_dtype_": dtype_,
            }
        else:
            quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            scale = float(scale)
            zero_point = int(zero_point)
            quant_min = observer.quant_min  # type: ignore[attr-defined]
            quant_max = observer.quant_max  # type: ignore[attr-defined]
            dtype_ = to_underlying_dtype(dtype)
            qparams = {
                "_scale_": scale,
                "_zero_point_": zero_point,
                "_quant_min_": quant_min,
                "_quant_max_": quant_max,
                "_dtype_": dtype_,
            }

        # 2. replace observer node with quantize and dequantize
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value_or_node in qparams.items():
                # TODO: we can add the information of whether a value needs to
                # be registered as an attribute in qparams dict itself
                if key in ["_scale_", "_zero_point_"] and (
                    not isinstance(value_or_node, (float, int))
                ):
                    # For scale and zero_point values we register them as buffers in the root module.
                    # However, note that when the values are not tensors, as in the case of
                    # per_tensor quantization, they will be treated as literals.
                    # However, registering them as a node seems to cause issue with dynamo
                    # tracing where it may consider tensor overload as opposed to default.
                    # With extra check of scale and zero_point being scalar, it makes
                    # sure that the default overload can be used.
                    # TODO: maybe need more complex attr name here
                    qparam_node = create_getattr_from_value(
                        model,
                        graph,
                        attr_prefix + key,
                        value_or_node,
                        model_device,
                    )
                    quantize_op_inputs.append(qparam_node)
                else:
                    # for qparams that are not scale/zero_point (like axis, dtype) we store them as literals in the graph.
                    quantize_op_inputs.append(value_or_node)

            quantized_node = graph.create_node(
                node_type, quantize_op, tuple(quantize_op_inputs), {}
            )
            add_quantize_dequantize_node_info(quantized_node, node)

            # use the same qparams from quantize op
            dq_inputs = [quantized_node] + quantize_op_inputs[1:]
            dequantized_node = graph.call_function(
                dequantize_op,
                tuple(dq_inputs),
                add_dequantize_op_kwargs(dequantize_op, input_node),
            )

            node.replace_all_uses_with(dequantized_node)

            add_quantize_dequantize_node_info(dequantized_node, node)
            graph.erase_node(node)
    elif is_dynamic:
        # uint8/int8/fp16 dynamic quantization

        # 1. extract information for inserting q/dq node from observer
        node_type = "call_function"
        quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.tensor
        # we only use choose_qparams for is_decomposed now,
        # but we should probably align the non-decomposed path with this as well,
        # and that can be done after we remove reduce_range flag
        # 1. extract qparams from observer module
        dtype_ = to_underlying_dtype(dtype)
        assert dtype_ in [torch.uint8, torch.int8], (
            "only uint8 and int8 are supported in reference flow for "
            "dynamic quantization right now"
        )
        quant_min = observer.quant_min  # type: ignore[attr-defined]
        quant_max = observer.quant_max  # type: ignore[attr-defined]
        qscheme = getattr(observer, "qscheme", torch.per_tensor_affine)  # type: ignore[attr-defined]
        eps = getattr(observer, "eps", torch.finfo(torch.float32).eps)  # type: ignore[attr-defined]
        # note: scale and zero_point are missing for quantize_per_tensor op
        # we'll need to get this from choose_qparams op, which we'll add after
        # this step
        qparams = {
            "_quant_min_": quant_min,
            "_quant_max_": quant_max,
            "_eps_": eps,
            "_dtype_": dtype_,
        }

        choose_qparams_op = _QSCHEME_TO_CHOOSE_QPARAMS_OP[qscheme]
        # 2. insert choose_qparams op and update the qparams list
        with graph.inserting_before(node):
            input_node = node.args[0]
            choose_qparams_op_inputs = [node.args[0]]
            for key, value in qparams.items():
                # we have quant_min, quant_max and dtype, all should be stored
                # as literals
                choose_qparams_op_inputs.append(value)
            choose_qparams_node = graph.create_node(
                "call_function", choose_qparams_op, tuple(choose_qparams_op_inputs), {}
            )
            # choose_qparms returns (scale, zero_point)
            scale_node = graph.create_node(
                "call_function", operator.getitem, (choose_qparams_node, 0), {}
            )
            zero_point_node = graph.create_node(
                "call_function", operator.getitem, (choose_qparams_node, 1), {}
            )
            quant_min = qparams["_quant_min_"]
            quant_max = qparams["_quant_max_"]
            dtype = qparams["_dtype_"]
            qparams = {
                "_scale_": scale_node,
                "_zero_point_": zero_point_node,
                "_quant_min_": quant_min,
                "_quant_max_": quant_max,
                "_dtype_": dtype,
            }

        # 3. replace observer node to quantize and dequantize node
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value_or_node in qparams.items():
                # TODO: we can add the information of whether a value needs to
                # be registered as an attribute in qparams dict itself
                if key in ["_scale_", "_zero_point_"]:
                    # in this case we have a node in the graph since it's dynamically
                    # computed from the input, with choose_qparams op
                    qparam_node = value_or_node
                    quantize_op_inputs.append(qparam_node)
                else:
                    # for qparams that are not scale/zero_point (like axis, dtype) we
                    # store them as literals in the graph.
                    quantize_op_inputs.append(value_or_node)

            quantized_node = graph.create_node(
                node_type, quantize_op, tuple(quantize_op_inputs), {}
            )

            add_quantize_dequantize_node_info(quantized_node, node)

            # use the same qparams from quantize op
            dq_inputs = [quantized_node] + quantize_op_inputs[1:]
            # need to use the tensor variant of this op, since scale and zero_point
            # from choose_qparam are Tensors, instead of float/int, this is to
            # prevent these nodes being traced away by downstream systems
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor
            dequantized_node = graph.call_function(
                dequantize_op,
                tuple(dq_inputs),
                add_dequantize_op_kwargs(dequantize_op, input_node),
            )

            node.replace_all_uses_with(dequantized_node)

            add_quantize_dequantize_node_info(dequantized_node, node)

            graph.erase_node(node)
    elif dtype == torch.float16:
        # Insert to_fp16 -> to_fp32 node
        dtype_convert_op = torch.ops.quantized_decomposed.convert_element_type.no_fuse
        with graph.inserting_before(node):
            input_node = node.args[0]
            convert_fp16_node = graph.create_node(
                "call_function", dtype_convert_op, (input_node, torch.float16), {}
            )
            convert_fp32_node = graph.create_node(
                "call_function", dtype_convert_op, (convert_fp16_node, torch.float), {}
            )
            node.replace_all_uses_with(convert_fp32_node)
            graph.erase_node(node)

    # should not reach since we have checks in the beginning to make sure the
    # observer is supported
