# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# mypy: allow-untyped-defs
import logging
from typing import Any, Optional

import torch
from torch._export.error import InternalError
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY
from torchao.quantization.pt2e.utils import _filter_sym_size_users
from torchao.quantization.quant_primitives import quant_lib  # noqa: F401
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

from .quantizer import QuantizationSpecBase
from .utils import is_valid_annotation

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

__all__ = ["PortNodeMetaForQDQ"]

_METADATA_TO_PORT = [
    "stack_trace",
    "quantization_tag",
]

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

_CHOOSE_QPARAMS_OPS = [
    torch.ops.quantized_decomposed.choose_qparams.tensor,
    torch.ops.quantized_decomposed.choose_qparams_symmetric.tensor,
]


# ops are only registered after 2.5
if TORCH_VERSION_AT_LEAST_2_5:
    _QUANTIZE_OPS += [torch.ops.torchao.quantize_affine]
    _DEQUANTIZE_OPS += [torch.ops.torchao.dequantize_affine]
    _CHOOSE_QPARAMS_OPS += [torch.ops.torchao.choose_qparams_affine]


def _add_metadata(to_node: torch.fx.Node, from_node: torch.fx.Node) -> None:
    from_meta = from_node.meta
    for meta_name in _METADATA_TO_PORT:
        if meta_name in from_meta:
            to_node.meta[meta_name] = from_meta[meta_name]


def _has_quant_annotation(node: torch.fx.Node) -> bool:
    return Q_ANNOTATION_KEY in node.meta


def _find_choose_qparams_node(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    # BFS to look for choose qparams
    from collections import deque

    queue = deque(list(node.users.keys()))
    while len(queue):
        n = queue.popleft()
        if n.op == "output":
            continue
        if n.op == "call_function" and n.target in _CHOOSE_QPARAMS_OPS:
            return n
        for k in n.users.keys():
            queue.append(k)
    return None


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


def _find_q_dq_node_for_user(
    produer: torch.fx.Node, user: torch.fx.Node
) -> tuple[Any, Any]:
    """
    Find q, dq pair corresponding to [producer -> q -> dq -> user]
    Utils works by finding dq arg of user and ensuring it is connected to
    producer
    """
    dq_node = None
    for n in user.args:
        if (
            isinstance(n, torch.fx.Node)
            and n.op == "call_function"
            and n.target in _DEQUANTIZE_OPS
        ):
            if _is_connected(produer, n):
                dq_node = n
                break
    if dq_node is None:
        for n in user.kwargs:
            if (
                isinstance(n, torch.fx.Node)
                and n.op == "call_function"
                and n.target in _DEQUANTIZE_OPS
            ):
                if _is_connected(produer, n):
                    dq_node = n
                    break
    if dq_node is None:
        return (None, None)

    q_node = None
    if (
        dq_node.args[0].op == "call_function"  # type: ignore[union-attr]
        and dq_node.args[0].target in _QUANTIZE_OPS  # type: ignore[union-attr]
    ):
        q_node = dq_node.args[0]
    return (q_node, dq_node)


def _port_metadata_for_input_quant_nodes(
    input_node: torch.fx.Node,
    node: torch.fx.Node,
    qspec: Optional[QuantizationSpecBase],
):
    if qspec is None:
        return

    is_dynamic_quant = getattr(qspec, "is_dynamic", None)
    if is_dynamic_quant is not None and is_dynamic_quant is True:
        choose_qparams_node = _find_choose_qparams_node(input_node)
        if choose_qparams_node is None:
            raise ValueError(f"No chose qparams node found for {node}")
        choose_qparam_users = _filter_sym_size_users(choose_qparams_node)
        if len(choose_qparam_users) != 2:
            raise InternalError(f"Expecting exactly two user for {choose_qparams_node}")
        scale_node = choose_qparam_users.pop()
        dynamic_q_node = next(iter(scale_node.users.keys()))
        dynamic_q_node_users = _filter_sym_size_users(dynamic_q_node)
        if len(dynamic_q_node_users) > 1:
            raise InternalError(f"Expecting single user for {dynamic_q_node}")
        dynamic_dq_node = dynamic_q_node_users.pop()
        _add_metadata(choose_qparams_node, node)
        _add_metadata(dynamic_q_node, node)
        _add_metadata(dynamic_dq_node, node)
    else:
        q_node, dq_node = _find_q_dq_node_for_user(input_node, node)
        if q_node is None or dq_node is None:
            return
        # add metadata for all the node between q_node and get_attr node
        # if the q_node can be traced back to get_attr node
        q_to_get_attr_nodes = [q_node]
        q_node_input = q_node.args[0]
        while (
            isinstance(q_node_input, torch.fx.Node)
            and q_node_input.op == "call_function"
            and q_node_input.target
            in [
                torch.ops.aten.flatten.using_ints,
                torch.ops.aten.permute.default,
                torch.ops.aten.permute_copy.default,
                torch.ops.aten.slice_copy.Tensor,
                torch.ops.aten.squeeze.dim,
                torch.ops.aten.squeeze_copy.dim,
                torch.ops.aten.transpose.Dimname,
                torch.ops.aten.transpose.int,
                torch.ops.aten.transpose_,
                torch.ops.aten.view_copy.default,
                torch.ops.aten.view.default,
                torch.ops.aten._mkldnn_transpose,
            ]
        ):
            q_to_get_attr_nodes.append(q_node_input)
            q_node_input = q_node_input.args[0]
        if isinstance(q_node_input, torch.fx.Node) and q_node_input.op == "get_attr":
            for n in q_to_get_attr_nodes:
                _add_metadata(n, q_node_input)
        _add_metadata(dq_node, node)


def _port_metadata_for_output_quant_nodes(
    node: torch.fx.Node, qspec: Optional[QuantizationSpecBase]
):
    if qspec is None:
        return

    node_users = _filter_sym_size_users(node)
    if len(node.users) == 0:
        return
    if len(node_users) != 1:
        logger.warning(f"Expecting {node} to have single user")  # noqa: G004
    q_node = node_users.pop()
    if q_node.op != "call_function" or q_node.target not in _QUANTIZE_OPS:
        logger.warning(
            f"Expecting {node} user to be a quantized op but got {q_node}"  # noqa: G004
        )  # noqa: G004
        return

    _add_metadata(q_node, node)


class PortNodeMetaForQDQ(PassBase):
    """
    Port metadata for nodes added by quantization flow.
    For static quant these are:
    - quantizer_per_tensor.default, dequantize_per_tensor.default
    - quantizer_per_channel.default, dequantize_per_channel.default
    For dynamic quant these are:
    - choose_qparams.tensor
    - quantizer_per_tensor.tensor, dequantize_per_tensor.tensor
    - quantizer_per_channel.default, dequantize_per_channel.default

    Rules of porting metadata:
    - Metadata to be ported:
      - nn_module_stack
      - stack_trace
      - quantization_tag
    - Metadata to NOT be ported:
      - Everything else
    - Rules:
      - Statically quantized patterns:
        - Dequantize nodes on the inputs to be quantized inherit metadata of the consumer node.
        - Quantize nodes on the outputs inherit metadata of the producer node.
        - Example 1:
          - Original: [Conv -> AvgPool -> Linear]
          - Quantized [Q-> DQ -> Conv -> Q -> DQ -> AvgPool -> Q -> DQ -> Linear -> Q -> DQ]
          - Inner brackets specify which nodes Q/DQ inherit metdata from
          - [Q-> [DQ -> Conv -> Q] -> [DQ -> AvgPool -> Q] -> [DQ -> Linear -> Q] -> DQ]
          - Note first Q and last DQ do not inherit metadata from any nodes
        - Example 2:
          - Original: [Conv -> AvgPool -> Linear]
          - AvgPool is not quantized
          - Quantized [Q-> DQ -> Conv -> Q -> DQ -> AvgPool -> Q -> DQ -> Linear -> Q -> DQ]
          - Inner brackets specify which nodes Q/DQ inherit metdata from
          - [Q-> [DQ -> Conv -> Q] -> DQ -> [AvgPool] -> Q -> [DQ -> Linear -> Q] -> DQ]
          - Note DQ and Q nodes around AvgPool do not inherit metadata from AvgPool because
            AvgPool was not supposed to be quantized. Metadata porting relies on quantization_annotation
            on the nodes (in this case AvgPool node) to conclude if the node or patter was
            supposed to be quantized. And subsequntly decide if the preceding Q, if any, should
            inherit metadata from AvgPool.
      - Dynamically quantized patterns:
        - Input that are dynamically quantized have choose_qparams, quantize and dequantize nodes
        - For example, below linear is dynamically quantized while rest statically:
          - Original: [Conv -> AvgPool -> Linear]
          - Quantized [Q-> DQ -> Conv -> Q -> DQ -> AvgPool -> Q -> DQ -> choose_params -> Q -> DQ -> Linear]
          - Quantized [Q-> [DQ -> Conv -> Q] -> [DQ -> AvgPool -> Q] -> DQ -> [choose_params -> Q -> DQ -> Linear]]
          - Note first Q does not inherit metadata from any nodes
    NB:
    - The best place for porting metadata is during observer conversion to q/dq. This is because it precisely
      knows which quantization spec is converted to q/dq and thus from where the metadata should be ported.
      However, since FX and PT2E quant workflow are on a common code-base, this hurts readability quite a bit.
      Doing it via a separate pass, helps readability of the code. Once we are able to refactor PT2E quant
      code, this pass should like to be integrated in the refactored variant of "convert" step.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            annotation = node.meta.get(Q_ANNOTATION_KEY, None)
            if is_valid_annotation(annotation):
                input_qspec_map = node.meta[Q_ANNOTATION_KEY].input_qspec_map
                output_qspec = node.meta[Q_ANNOTATION_KEY].output_qspec
                for input_node, qspec in input_qspec_map.items():
                    _port_metadata_for_input_quant_nodes(input_node, node, qspec)
                _port_metadata_for_output_quant_nodes(node, output_qspec)
        return PassResult(graph_module, True)
